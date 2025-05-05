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
# 在src/__init__.py中添加（如果不存在则创建）
from transformers import AutoConfig
from .gpt2 import ThisGPT2Config

# 修改get_model_and_auxiliaries函数
def get_model_and_auxiliaries(args):
    # 1. 加载分词器
    from sentencepiece import SentencePieceProcessor
    tokenizer = SentencePieceProcessor()
    model_path = "/kaggle/working/smallcap_cn/src/mengzi_gpt.model"
    
    if not tokenizer.load(model_path):
        raise RuntimeError(f"无法加载分词器: {model_path}")

    # 2. 特殊token处理
    PAD_TOKEN = "</s>"
    EOS_TOKEN = "</s>"
    pad_id = eos_id = tokenizer.piece_to_id(PAD_TOKEN)
    
    # 3. 初始化模型组件
    encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    decoder_config = ThisGPT2Config(
        vocab_size=tokenizer.get_piece_size(),
        n_positions=CAPTION_LENGTH,
        n_embd=768,
        n_layer=6,
        n_head=12,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        cross_attention_reduce_factor=PARAMS2REDUCE_FACTOR[args.attention_size]
    )

    # 4. 构建模型（简化初始化）
    model = SmallCap(
        encoder=encoder,
        decoder=ThisGPT2LMHeadModel(decoder_config)
    )
    
    # 5. 设置额外参数
    model.config.decoder_start_token_id = tokenizer.bos_id()  # 使用<s>作为解码起始
    model.config.cross_attention_reduce_factor = PARAMS2REDUCE_FACTOR[args.attention_size]
    
    # 冻结编码器
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    return model, tokenizer, None
    
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
