from .registry import available_optimizers, build_optimizer, register, split_params

# Importing a module runs its @register decorators. Add your own optimizer module here.
from . import adamw, muon, template  # noqa: E402,F401

__all__ = ['available_optimizers', 'build_optimizer', 'register', 'split_params']
