import sys
import typing
from typing import Callable, Optional, Union
import typing_extensions

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from transformers.utils import auto_docstring, can_return_tuple
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin

from .configuration import EnsembleConfig

if sys.version_info >= (3, 11):
    Unpack = typing.Unpack
else:
    Unpack = typing_extensions.Unpack

def WrapBaseModelOutputWithPast():
    pass


def UpackEnsembleCache(ensemble_cache):
    return ensemble_cache.draft_cache, ensemble_cache.target_cache

def UpackEnsembleHiddenStates(hidden_states):
    pass
    

class EnsembleCache(Cache):

    def __init__(self, draft_cache: Cache, target_cache: Cache):

        self.draft_cache = draft_cache
        self.target_cache = target_cache


class EnsembleBaseModelOutputWithPast(BaseModelOutputWithPast):

    last_hidden_state: Optional[torch.FloatTensor] = None
    draft_logits: Optional[torch.FloatTensor] = None
    target_logtis: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class EnsembleModel(PreTrainedModel):
    config_class = EnsembleConfig
    _supports_attention_backend = True
    _supports_flash_attn_3 = False
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True


    def __init__(self, config: EnsembleConfig):
        super().__init__(config)
        
        # The target_model and draft_model are ForCausalLM models
        raw_model = AutoModelForCausalLM.from_pretrained(config.target_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
        self.target_model = raw_model.model

        raw_model = AutoModelForCausalLM.from_pretrained(config.draft_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
        self.draft_model = raw_model.model

        del raw_model
    
        self.target_moodel_config = AutoConfig.from_pretrained(config.target_model_path, trust_remote_code=True)
        self.draft_model_config = AutoConfig.from_pretrained(config.draft_model_path, trust_remote_code=True)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> EnsembleBaseModelOutputWithPast:
        
        # We Unpack the model input here.
        draft_cache, target_cache = UpackEnsembleCache(past_key_values)

        
        # The draft_output is CausalLMOutputWithPast
        draft_output = self.draft_model(input_ids, 
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        past_key_values=draft_cache,
                                        inputs_embeds=inputs_embeds,
                                        use_cache=use_cache,
                                        output_attentions=output_attentions,
                                        output_hidden_states=True,
                                        cache_position=cache_position)  
        
        # The target_output is CausalLMOutputWithPast
        target_output = self.target_model(input_ids, 
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        past_key_values=target_cache,
                                        inputs_embeds=inputs_embeds,
                                        use_cache=use_cache,
                                        output_attentions=output_attentions,
                                        output_hidden_states=True,
                                        cache_position=cache_position)
        
        draft_hidden_states = draft_output.hidden_states[-1]
        target_hidden_states = target_output.hidden_states[-1]

        if use_cache:
            ensemble_cache = EnsembleCache()
            ensemble_cache.draft_cache = draft_output.past_key_values
            ensemble_cache.target_cache = target_output.past_key_values
        
        return EnsembleBaseModelOutputWithPast(
                                       last_hidden_state=torch.cat([draft_hidden_states, target_hidden_states], dim=-1),
                                       past_key_values=ensemble_cache, 
                                       hidden_states=target_output.hidden_states, 
                                       attentions=target_output.attentions,
                                       draft_logits=draft_output.logits,
                                       target_logits=target_output.logits)

    def get_input_embeddings(self):
        return self.target_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.embed_tokens = value

@auto_docstring
class EnsembleModelForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = EnsembleConfig
    _supports_attention_backend = True
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    _tied_weights_keys = ["ensemble_head.weight"]
    _tp_plan = {"ensemble_head": "colwise_rep"}
    _pp_plan = {"ensemble_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = EnsembleModel(config)
        self.vocab_size = config.vocab_size
        self.ensemble_head = nn.Linear(config.ensemble_hidden_size, 2, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.target_model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.target_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.ensemble_head

    def set_output_embeddings(self, new_embeddings):
        self.ensemble_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        
        draft_logits = outputs.draft_logits
        target_logits = outputs.target_logits

        weights = F.softmax(self.ensemble_head(hidden_states), dim=-1)

        # Expand weights to match logits shape
        w_draft = weights[..., 0].unsqueeze(-1)  # [B, T, 1]
        w_target = weights[..., 1].unsqueeze(-1)
        
        logits = w_draft * draft_logits + w_target * target_logits

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )