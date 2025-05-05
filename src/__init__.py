from transformers import AutoConfig, AutoModel
from .gpt2 import ThisGPT2Config, ThisGPT2Model, ThisGPT2LMHeadModel
from .vision_encoder_decoder import SmallCapConfig, SmallCap

# 注册配置类
AutoConfig.register("this_gpt2", ThisGPT2Config)
AutoConfig.register("smallcap", SmallCapConfig)

# 注册模型类
AutoModel.register(ThisGPT2Config, ThisGPT2Model)
