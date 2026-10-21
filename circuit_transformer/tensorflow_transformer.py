"""Compatibility import; implementation now uses PyTorch.

The unchanged TensorFlow reference is in tests/reference/tensorflow_transformer.py.
"""
from .pytorch_transformer import (
    Seq2SeqTransformer, TreePositionEncodingPostProcessing,
    CustomSchedule, masked_loss, masked_accuracy,
)

__all__ = [
    'Seq2SeqTransformer', 'TreePositionEncodingPostProcessing',
    'CustomSchedule', 'masked_loss', 'masked_accuracy',
]

# 当__init__.py写 `from xxx import *` 的时候，只会导入列表里这几个名字，隐藏内部其他代码。