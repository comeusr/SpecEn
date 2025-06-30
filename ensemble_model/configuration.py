import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers  
from transformers import PretrainedConfig

class EnsembleConfig(PretrainedConfig):
    model_type = "customize_ensemble"
    
    def __init__(
        self,
        hidden_size=4096,
        vocab_size=151936,
        target_model_path=None,
        draft_model_path=None,
        trust_remote_code=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.ensemble_hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.target_model_path = target_model_path
        self.draft_model_path = draft_model_path
        self.trust_remote_code=trust_remote_code
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.intermediate_size = kwargs.get("intermediate_size", 11008)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 8196)
        self.rope_theta = kwargs.get("rope_theta", 1000000.0)
        self.num_key_value_heads=8


# class EnsembleConfig(PretrainedConfig):
#     model_type = "customize_ensemble"

#     keys_to_ignore_at_inference = ["past_key_values"]

#     # Default tensor parallel plan for base model `Qwen3`
#     base_model_tp_plan = {
#         "layers.*.self_attn.q_proj": "colwise",
#         "layers.*.self_attn.k_proj": "colwise",
#         "layers.*.self_attn.v_proj": "colwise",
#         "layers.*.self_attn.o_proj": "rowwise",
#         "layers.*.mlp.gate_proj": "colwise",
#         "layers.*.mlp.up_proj": "colwise",
#         "layers.*.mlp.down_proj": "rowwise",
#         "lm_head": 
#     }
    
#     base_model_pp_plan = {
#         "embed_tokens": (["input_ids"], ["inputs_embeds"]),
#         "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
#         "norm": (["hidden_states"], ["hidden_states"]),
#     }
    
#     def __init__(
#         self,
#         hidden_size=4096,
#         ensemble_hidden_size=4096,
#         vocab_size=151936,
#         target_model_path=None,
#         draft_model_path=None,
#         trust_remote_code=True,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.hidden_size = hidden_size
#         self.ensemble_hidden_size = ensemble_hidden_size
#         self.vocab_size = vocab_size
#         self.target_model_path = target_model_path
#         self.draft_model_path = draft_model_path
#         self.trust_remote_code=trust_remote_code
#         self.num_attention_heads = kwargs.get("num_attention_heads", 32)
#         self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
#         self.intermediate_size = kwargs.get("intermediate_size", 11008)
#         self.max_position_embeddings = kwargs.get("max_position_embeddings", 8196)
#         self.rope_theta = kwargs.get("rope_theta", 1000000.0)