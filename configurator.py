"""
Minimal configuration system in the style of nanoGPT / Sophia.

    python train.py config/gpt2_small_adamw.py --dataset=fineweb --batch_size=16

Arguments are applied left to right on top of the defaults defined in train.py:
  * an argument without a leading `--` is a config file: a plain Python file whose
    top-level assignments override config keys;
  * `--key=value` overrides a single key.

Unlike the original configurator, unknown keys and type mismatches are errors, so a
typo in a hyperparameter name cannot silently fall back to the default.
"""

import ast
import sys
from types import ModuleType


def _coerce(key, value, default, raw=None):
    """Check `value` against the type of `default` and return the value to store."""
    if default is None or value is None:
        if default is not None:
            raise ValueError(f"config key {key!r} cannot be None")
        return value
    if isinstance(default, str):
        if isinstance(value, str):
            return value
        if raw is not None:  # e.g. --run_name=123 should stay the string '123'
            return raw
    elif isinstance(default, bool):
        if isinstance(value, bool):
            return value
    elif isinstance(default, int):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif isinstance(default, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif isinstance(value, type(default)):
        return value
    raise ValueError(f"config key {key!r} expects {type(default).__name__}, "
                     f"got {type(value).__name__} ({value!r})")


def load_config(defaults, argv=None):
    """Return a new dict: `defaults` updated by config files and --key=value overrides."""
    config = dict(defaults)
    argv = sys.argv[1:] if argv is None else argv
    for arg in argv:
        if not arg.startswith('--'):
            with open(arg) as f:
                source = f.read()
            namespace = {}
            exec(compile(source, arg, 'exec'), namespace)
            for key, value in namespace.items():
                if key.startswith('_') or isinstance(value, ModuleType):
                    continue
                if key not in defaults:
                    raise ValueError(f"unknown config key {key!r} in {arg}")
                config[key] = _coerce(key, value, defaults[key])
        else:
            if '=' not in arg:
                raise ValueError(f"expected --key=value, got {arg!r}")
            key, raw = arg[2:].split('=', 1)
            if key not in defaults:
                raise ValueError(f"unknown config key {key!r}")
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                value = raw  # a bare string such as --dataset=fineweb
            config[key] = _coerce(key, value, defaults[key], raw=raw)
    return config
