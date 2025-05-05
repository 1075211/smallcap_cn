from .gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel

# 修改注册名称以区分原版
AutoConfig.register("mengzi-gpt", ThisGPT2Config)
AutoModel.register(ThisGPT2Config, ThisGPT2Model)
AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
