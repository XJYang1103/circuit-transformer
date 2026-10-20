# TensorFlow reference

`model.py` and `tensorflow_transformer.py` are unchanged copies from repository
commit `0fc85a5798575334cbd18dab32545c4fcbcc2b1d`. They are test fixtures, not a
runtime backend. Original copyright and license notices are retained.

`loader.py` loads the actual installed TF Models layers, bypassing only package
initializers that eagerly import unrelated optional vision/text dependencies.
It does not replace attention, normalization, loss, or optimizer computations.

Reference environment: Python 3.12, TensorFlow 2.18.0, tf-keras 2.18.0,
tf-models-official 2.18.0. PyTorch runtime tested: 2.14.0+cpu.

Install in a separate environment (TensorFlow is not needed by native .pt users):

```bash
python -m pip install -e '.[test]' tensorflow==2.18.0 tf-keras==2.18.0 gin-config
python -m pip install tf-models-official==2.18.0 --no-deps
python -m pytest tests/test_tensorflow_parity.py -q -s
```

The reference version is pinned so future TF Models defaults cannot silently
change the baseline. It is not a TensorFlow requirement on the Linux training
server. Tests skip with an explicit reason when the reference dependencies are
absent; a skipped comparison is not evidence of parity.
