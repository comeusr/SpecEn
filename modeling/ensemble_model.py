import vllm
import torch 
import troch.nn as nn

from vllm import LLM

from transformers import AutoModelForCausalLM, AutoTokenizer

def create_ensemble_model(cfg):

    target_model = AutoModelForCausalLM(cfg.method.model)
    draft_model = AutoModelForCausalLM(cfg.method.draft_model)

    
    
    pass

    

class EnsembleModel(nn.Module):

    def __init__(self, target_model, draft_model, hidden_size, head_hidden):

        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.ensemble_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 2)
        )

    def forward(self):
        pass
        

    