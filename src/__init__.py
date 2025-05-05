from transformers import AutoConfig
from .gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
from .vision_encoder_decoder import SmallCapConfig

# 注册自定义配置
AutoConfig.register("this_gpt2", ThisGPT2Config)
AutoConfig.register("smallcap", SmallCapConfig)
