from .gpt import GPT, GPTConfig
from .llama import Llama, LlamaConfig

MODEL_TYPES = ('gpt2', 'llama')


def build_model(model_type, **model_args):
    """Construct a model from its type name and constructor arguments."""
    if model_type == 'gpt2':
        return GPT(GPTConfig(**model_args))
    if model_type == 'llama':
        return Llama(LlamaConfig(**model_args))
    raise ValueError(f"unknown model_type {model_type!r}; choose from {MODEL_TYPES}")
