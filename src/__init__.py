from transformers import AutoConfig
from .gpt2 import ThisGPT2Config

AutoConfig.register("mengzi-gpt", ThisGPT2Config)
