import os
import sys

# make `models`, `optimizers`, `configurator` and `train` importable from the repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
