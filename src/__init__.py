from .mengzi_gpt import (
    MengziGPTConfig,
    MengziGPTModel,
    MengziGPTLMHeadModel
)

AutoConfig.register("mengzi-gpt", MengziGPTConfig)
AutoModel.register(MengziGPTConfig, MengziGPTModel)
AutoModelForCausalLM.register(MengziGPTConfig, MengziGPTLMHeadModel)
