"""Compatibility import; implementation now uses PyTorch.

The unchanged TensorFlow reference is in tests/reference/tensorflow_transformer.py.
"""
from .pytorch_transformer import (
    Seq2SeqTransformer, TreePositionEncodingPostProcessing,
    CustomSchedule, masked_loss, masked_accuracy,
)
