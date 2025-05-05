# coding=utf-8
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model, GPT2Attention, GPT2Block
from transformers.activations import ACT2FN
import math
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MengziGPTConfig(GPT2Config):
    model_type = "mengzi-gpt"
    
    def __init__(
        self,
        cross_attention_reduce_factor=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cross_attention_reduce_factor = cross_attention_reduce_factor
        # 中文模型特定参数
        self.activation_function = "gelu_new"
        self.resid_pdrop = 0.1
        self.embd_pdrop = 0.1

class MengziGPTAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)
        
        self.cross_attention_reduce_factor = config.cross_attention_reduce_factor
        
        if self.is_cross_attention:
            # 调整跨注意力层的维度
            self.c_attn = nn.Linear(config.hidden_size, int(2 * config.hidden_size / self.cross_attention_reduce_factor))
            self.q_attn = nn.Linear(config.hidden_size, int(config.hidden_size / self.cross_attention_reduce_factor))
            self.c_proj = nn.Linear(int(config.hidden_size / self.cross_attention_reduce_factor), config.hidden_size)
        else:
            self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
            self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # 中文注意力优化
        self.scale_factor = math.sqrt(config.hidden_size / config.num_attention_heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / self.scale_factor
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights

class MengziGPTBlock(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.attn = MengziGPTAttention(config, layer_idx=layer_idx)
        if config.add_cross_attention:
            self.crossattention = MengziGPTAttention(config, is_cross_attention=True, layer_idx=layer_idx)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            ACT2FN[config.activation_function],
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.Dropout(config.resid_pdrop),
        )

class MengziGPTModel(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([MengziGPTBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        
    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None):
        # 中文文本的特殊处理
        if attention_mask is None and input_ids is not None:
            attention_mask = (input_ids != self.config.pad_token_id).float()
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states
        )

class MengziGPTLMHeadModel(GPT2LMHeadModel):
    config_class = MengziGPTConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = MengziGPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
            "past_key_values": past
        }
