# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modified for Chinese Mengzi-GPT support

import math
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers.modeling_utils import PreTrainedModel

class ThisGPT2Config(GPT2Config):
    model_type = "mengzi-gpt"
    
    def __init__(self, cross_attention_reduce_factor=1, **kwargs):
        super().__init__(**kwargs)
        self.cross_attention_reduce_factor = cross_attention_reduce_factor

class ThisGPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.cross_attention_reduce_factor = config.cross_attention_reduce_factor
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if is_cross_attention:
            self.head_dim = int(self.head_dim / self.cross_attention_reduce_factor)

        if is_cross_attention:
            self.q_attn = nn.Linear(self.embed_dim, self.embed_dim // self.cross_attention_reduce_factor)
            self.c_attn = nn.Linear(self.embed_dim, 2 * self.embed_dim // self.cross_attention_reduce_factor)
            self.c_proj = nn.Linear(self.embed_dim // self.cross_attention_reduce_factor, self.embed_dim)
        else:
            self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
            self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.scale_factor = math.sqrt(self.head_dim)

    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / self.scale_factor
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, value)

class ThisGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.attn = ThisGPT2Attention(config, layer_idx=layer_idx)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.Dropout(config.resid_pdrop),
        )
        if config.add_cross_attention:
            self.crossattention = ThisGPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)

class ThisGPT2Model(GPT2Model):
    config_class = ThisGPT2Config  # 关键：添加config_class关联
    
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([ThisGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        
    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None):
        if attention_mask is None and input_ids is not None:
            attention_mask = (input_ids != self.config.pad_token_id).float()
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states
        )

class ThisGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = ThisGPT2Config  # 保持配置类一致
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = ThisGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
            "past_key_values": past
        }
