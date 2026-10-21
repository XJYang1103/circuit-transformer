"""Opt-in full 12-layer upstream checkpoint acceptance test."""
import ast
import os
from pathlib import Path

import numpy as np
import pytest
from test_pytorch_migration import SearchOptions

PREFIX = os.environ.get('CT_TF_CHECKPOINT')
pytestmark = pytest.mark.skipif(not PREFIX, reason='Set CT_TF_CHECKPOINT to the upstream TF prefix')


@pytest.mark.parametrize('mixed', [False, True])
def test_pretrained_greedy_mcts_and_truth_tables(mixed, tmp_path):
    import tensorflow as tf
    import tf_keras
    import torch
    from reference.loader import load_reference_circuit
    from circuit_transformer import CircuitTransformer, read_aiger, compute_tts, count_num_ands

    device = os.environ.get('CT_TEST_DEVICE', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('CT_TEST_DEVICE must be cpu or cuda')
    if device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA acceptance requires a CUDA-enabled PyTorch installation'
        gpus = tf.config.list_physical_devices('GPU')
        assert gpus, 'CUDA acceptance requires a GPU-enabled TensorFlow reference'
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Compare FP32 kernels, without TensorFloat-32 rounding.
        tf.config.experimental.enable_tensor_float_32_execution(False)
        torch.backends.cuda.matmul.allow_tf32 = False
    reference_device = '/GPU:0' if device == 'cuda' else '/CPU:0'
    tf_keras.mixed_precision.set_global_policy('mixed_float16' if mixed else 'float32')
    reference = load_reference_circuit()
    source = ast.parse(Path(__file__).with_name('test_circuit_transformer.py').read_text())
    aiger = next(node.value for node in ast.walk(source)
                 if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith('aag 33'))
    original, _ = read_aiger(aiger_str=aiger)
    with tf.device(reference_device):
        old = reference.CircuitTransformer(mixed_precision=mixed, ckpt_path=PREFIX)
    new = CircuitTransformer(mixed_precision=mixed, ckpt_path=PREFIX, device=device)
    max_error, calls = 0., 0
    old_inference = old._transformer_inference
    new_inference = new._transformer_inference
    # Compare full encoder/decoder predictions through real optimization paths.
    # Each side runs separately to preserve each framework's own KV cache.
    old_scores = []
    def record(inputs, **kwargs):
        result = old_inference(inputs, **kwargs)
        old_scores.append(result[0])
        return result
    def compare(inputs, return_kv_cache=False, return_last_token=False):
        nonlocal max_error, calls
        result = new_inference(inputs, return_kv_cache=return_kv_cache,
                               return_last_token=return_last_token)
        expected = old_scores[calls]
        legal = expected > np.finfo(np.float32).min
        error = np.abs(result[0][legal] - expected[legal]).max(initial=0.)
        max_error = max(max_error, float(error))
        calls += 1
        # Mixed precision accumulates rounding through 24 blocks; actions below
        # must still match exactly. FP32 uses a stricter numerical bound.
        np.testing.assert_allclose(result[0][legal], expected[legal],
                                   atol=0.15 if mixed else 5e-4, rtol=0.03 if mixed else 5e-5)
        np.testing.assert_array_equal(np.argmax(result[0], axis=-1), np.argmax(expected, axis=-1))
        return result
    old._transformer_inference = record
    new._transformer_inference = compare
    for mcts in (0, 1):
        old_scores.clear()
        calls = 0
        options = SearchOptions(max_inference_seq_length=100, num_mcts_steps=mcts,
                                num_leaf_parallelization=2, num_mcts_playouts_per_step=2,
                                return_envs=True)
        with tf.device(reference_device):
            before = old.optimize_batch([original], **options)
        after = new.optimize_batch([original], **options)
        assert calls == len(old_scores)
        for expected, actual in zip(before, after):
            assert actual.success and expected.success
            assert actual.tokens == expected.tokens
            assert actual.rewards == expected.rewards
            assert compute_tts(actual.roots, 8) == compute_tts(original, 8)
            assert count_num_ands(actual.roots) == count_num_ands(expected.roots)
            print(f'pretrained mixed={mixed} mcts={mcts} ANDs={count_num_ands(actual.roots)} '
                  f'calls={calls} max_logit_abs_error={max_error:.9g}')
    # Native conversion must round-trip every parameter exactly.
    native = tmp_path / 'pretrained.pt'
    new.save(native)
    original_state = {k: v.clone() for k, v in new._transformer.state_dict().items()}
    new.load(native)
    for name, value in new._transformer.state_dict().items():
        assert torch.equal(value, original_state[name])
