import os
import importlib.util

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import torch

torch.set_num_threads(1)

# Some Linux TensorFlow/Triton builds abort in native code when TensorFlow is
# imported first. Load the optional Triton runtime before collecting TF tests.
if (importlib.util.find_spec('tensorflow') is not None
        and importlib.util.find_spec('triton') is not None):
    importlib.import_module('triton')
