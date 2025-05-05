from transformers import AutoConfig, AutoModel
from .gpt2 import ThisGPT2Config, ThisGPT2Model, ThisGPT2LMHeadModel
from .vision_encoder_decoder import SmallCapConfig, SmallCap

# 注册配置类
# 修改注册代码（确保和 ThisGPT2Config.model_type 一致）
AutoConfig.register("mengzi-gpt", ThisGPT2Config)  # 而不是 "this_gpt2"
AutoConfig.register("smallcap", SmallCapConfig)

# 注册模型类
AutoModel.register(ThisGPT2Config, ThisGPT2Model)
