"""Behavioral coverage for defects found during the repository-wide review."""
import copy
import json
import subprocess
import sys

import bitarray
import numpy as np
import pytest
import torch

from circuit_transformer import (
    LogicNetworkEnv, Node, NodeWithInv, cec, compute_tt, compute_tts,
    count_seq_length, get_aig_info, read_aiger, sequential_synthesis,
    structural_hashing, write_aiger,
)
from circuit_transformer.checkpoints import FORMAT
from circuit_transformer.model import encode_aig, int_to_node
from circuit_transformer.pytorch_transformer import Seq2SeqTransformer, masked_accuracy, masked_loss
from circuit_transformer.training import ReplicaDataParallel
from test_pytorch_migration import aig_pair, small_circuit


def test_environment_copy_preserves_source_and_shared_inputs():
    model = small_circuit()
    env = LogicNetworkEnv(compute_tts(aig_pair(), 2), 2, max_length=12, context_hash=set())
    before = dict(vars(env))
    cloned = model._copy_env(env)
    assert cloned is not env
    assert vars(env).keys() == before.keys()
    for name, value in before.items():
        assert getattr(env, name) is value
    for name in ('context_hash', 'tts_bitarray', 'input_tt_bitarray',
                 'input_tt_bitarray_compressed', 'ffw'):
        assert getattr(cloned, name) is getattr(env, name)
    cloned.tokens.append(2)
    assert env.tokens == []
    assert model._copy_env([env])[0] is not env

    class CannotCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError('copy failure')

    setattr(env, 'extra', CannotCopy())
    with pytest.raises(RuntimeError, match='copy failure'):
        model._copy_env(env)
    for name, value in before.items():
        assert getattr(env, name) is value


@pytest.mark.parametrize('binary', [False, True])
def test_aiger_paths_symbols_and_missing_final_newline(tmp_path, binary):
    root = NodeWithInv(Node(0, None, None), False, output_symbol='output with spaces')
    root.input_symbol = 'input with spaces'
    directory = tmp_path / 'directory with spaces'
    directory.mkdir()
    path = directory / ('circuit.aig' if binary else 'circuit.aag')
    text = write_aiger(root, str(path))
    parsed, info = read_aiger(path)
    assert isinstance(parsed, NodeWithInv)
    assert parsed.output_symbol == root.output_symbol
    assert parsed.input_symbol == root.input_symbol
    assert info == [1, 1, 0, 1, 0]
    parsed_without_newline, _ = read_aiger(aiger_str=text.rstrip())
    assert compute_tts(parsed_without_newline, 1) == compute_tts(root, 1)


@pytest.mark.parametrize('inverted', [False, True])
def test_constant_aiger_roundtrip(inverted):
    root = NodeWithInv(Node(-1, None, None), inverted)
    parsed, info = read_aiger(aiger_str=write_aiger(root))
    assert isinstance(parsed, NodeWithInv)
    assert info == [0, 0, 0, 1, 0]
    assert compute_tt(parsed, 0) == bitarray.bitarray('1' if inverted else '0')


def test_external_synthesis_and_equivalence():
    original = aig_pair()
    for synthesize in (sequential_synthesis, structural_hashing):
        optimized, _ = synthesize(original)
        assert compute_tts(optimized, 2) == compute_tts(original, 2)
        assert cec(original, optimized)
    x = NodeWithInv(Node(0, None, None), False)
    assert not cec(x, NodeWithInv(x.parent, True))
    with pytest.raises(ValueError, match='Unsupported'):
        sequential_synthesis(original, command='missing-command')


def test_aiger_missing_input_and_empty_header_fail_promptly(tmp_path):
    with pytest.raises(ValueError, match='Provide'):
        read_aiger()
    path = tmp_path / 'empty.aig'
    path.touch()
    with pytest.raises(ValueError, match='Empty'):
        get_aig_info(path)
    assert count_seq_length(aig_pair()) > 0
    assert count_seq_length(aig_pair()[0].parent) > 0


def test_single_output_training_and_filename_optimization(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    original = aig_pair()[0]
    text = write_aiger(original)
    for index in range(4):
        (data / f'{index}.json').write_text(json.dumps([text, 1, text, 1]))
    model = small_circuit()
    model.train(str(data), str(tmp_path / 'weights'), epochs=1, batch_size=2, num_workers=0)
    assert (tmp_path / 'weights' / 'model-0001.pt').is_file()
    path = tmp_path / 'input.aag'
    path.write_text(text)
    result = model.optimize_batch([str(path)], max_inference_seq_length=12)[0]
    assert isinstance(result, list)
    assert compute_tts(result, 2) == compute_tts(original, 2)

    def unexpected_download():
        raise AssertionError('Training weights must not be replaced by pretrained weights.')

    monkeypatch.setattr(model, 'load_from_hf', unexpected_download)
    result = model.optimize([[original]], max_inference_seq_length=12)[0]
    assert compute_tts(result, 2) == compute_tts(original, 2)


@pytest.mark.parametrize('candidate', [False, True])
def test_disabling_kv_cache_preserves_decode(candidate):
    torch.manual_seed(15)
    model = small_circuit(add_action_mask_to_input=candidate)
    cached = model.optimize_batch([aig_pair()], max_inference_seq_length=12, return_envs=True)
    model.use_kv_cache = False
    uncached = model.optimize_batch([aig_pair()], max_inference_seq_length=12, return_envs=True)
    assert isinstance(cached[0], LogicNetworkEnv) and isinstance(uncached[0], LogicNetworkEnv)
    assert cached[0].tokens == uncached[0].tokens


def test_optimize_return_options_and_empty_inputs():
    model = small_circuit()
    assert model.optimize([]) == []
    with pytest.raises(ValueError, match='at least one'):
        model.optimize_batch([], max_inference_seq_length=12)
    with pytest.raises(ValueError, match='positive'):
        model.optimize_batch([aig_pair()], max_inference_seq_length=0)
    with pytest.raises(ValueError, match='positive'):
        model.optimize_batch([aig_pair()], max_inference_seq_length=12,
                             num_mcts_steps=1, num_mcts_playouts_per_step=0)
    result = model.optimize_batch([aig_pair()], max_inference_seq_length=12,
                                  return_envs=True, return_mcts_roots=True,
                                  return_input_encodings=True)
    assert isinstance(result, tuple) and len(result) == 3
    envs, roots, encodings = result
    assert isinstance(envs[0], LogicNetworkEnv)
    assert len(roots) == 1
    assert encodings['inputs'].shape[0] == 1


def test_token_and_incomplete_circuit_validation():
    with pytest.raises(ValueError):
        int_to_node(-1, 2)
    with pytest.raises(ValueError, match='one or two'):
        encode_aig([], 2)
    with pytest.raises(ValueError, match='incomplete'):
        encode_aig([NodeWithInv(Node(None, None, None), False)], 2)
    env = LogicNetworkEnv([bitarray.bitarray('0101')], 2)
    assert env.step(2)[1]


@pytest.mark.parametrize('fault', ['payload', 'missing', 'mapping', 'tensor', 'shape', 'nan'])
def test_bad_checkpoint_does_not_change_model(tmp_path, fault):
    model = small_circuit()
    state = copy.deepcopy(model._transformer.state_dict())
    payload = {'format': FORMAT, 'config': model._transformer.config, 'state_dict': copy.deepcopy(state)}
    name = next(iter(state))
    if fault == 'payload':
        payload = []
    elif fault == 'missing':
        del payload['state_dict']
    elif fault == 'mapping':
        payload['state_dict'] = []
    elif fault == 'tensor':
        payload['state_dict'][name] = 'bad parameter'
    elif fault == 'shape':
        payload['state_dict'][name] = torch.zeros(1)
    else:
        payload['state_dict'][name].fill_(float('nan'))
    path = tmp_path / 'invalid.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError):
        model.load(path)
    for key, value in model._transformer.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_checkpoint_save_is_atomic(tmp_path, monkeypatch):
    model = small_circuit()
    path = tmp_path / 'model.pt'
    model.save(path)
    before = path.read_bytes()

    def fail_save(*args, **kwargs):
        raise OSError('disk full')

    with monkeypatch.context() as context:
        context.setattr(torch, 'save', fail_save)
        with pytest.raises(OSError, match='disk full'):
            model.save(path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))
    # An unrelated TF index must not shadow an existing native checkpoint.
    (tmp_path / 'model.pt.index').touch()
    model.load(path)


def test_partial_batch_scatter_does_not_create_empty_replicas():
    parallel = ReplicaDataParallel(torch.nn.Identity())
    inputs = {'inputs': torch.tensor([[1, 2], [3, 4]])}
    labels = inputs['inputs'].clone()
    scattered, kwargs = parallel.scatter((inputs, labels), None, [torch.device('cpu')] * 4)
    assert len(scattered) == len(kwargs) == 2
    assert all(part[1].shape[0] == 1 for part in scattered)
    torch.testing.assert_close(torch.cat([part[1] for part in scattered]), labels)


def test_training_failure_restores_model_mode(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'broken.json').write_text('invalid json')
    model = small_circuit()
    with pytest.raises(json.JSONDecodeError):
        model.train(str(data), epochs=1, batch_size=1, num_workers=0, validation_split=0.)
    assert model._transformer.return_cache
    assert not model._transformer.training


def test_worker_failure_propagates_and_kwargs_are_used():
    # Run with a deadline so a broken producer cannot hang the test suite.
    script = '''
from circuit_transformer.utils import MPDataset
def mapping(value, increment=0):
    if increment < 0:
        raise ValueError('broken sample')
    return value + increment
dataset = MPDataset(list(range(4)), mapping, kwargs={'increment': 5}, num_processes=2, validation_split=0.)
try:
    assert list(dataset.train_generator()) == [5, 6, 7]
    assert list(dataset.train_generator()) == [8, 5, 6]
finally:
    dataset.close()
assert not dataset.process.is_alive()
dataset = MPDataset(list(range(4)), mapping, kwargs={'increment': -1}, num_processes=2, validation_split=0.)
try:
    try:
        next(dataset.train_generator())
    except RuntimeError as exc:
        assert 'broken sample' in str(exc)
    else:
        raise AssertionError('Expected worker error')
finally:
    dataset.close()
assert not dataset.process.is_alive()
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=20, capture_output=True, text=True)


def test_cached_sinusoidal_positions_match_full_decode():
    model = Seq2SeqTransformer(enc_vocab_size=8, dec_vocab_size=8, embedding_width=8,
                               num_layers=1, num_attention_heads=2, intermediate_size=16,
                               max_tree_depth=None).eval()
    model.return_cache = True
    source = torch.tensor([[2, 3, 1], [4, 5, 1]])
    target = torch.tensor([[2, 4, 1], [3, 5, 1]])
    mask = torch.ones(2, 3, 8, dtype=torch.bool)
    features = {'inputs': source, 'targets': target, 'dec_action_mask': mask}
    full, _ = model(features)
    cache = None
    for step in range(3):
        tokens = torch.zeros(2, 1, dtype=torch.long) if step == 0 else target[:, step-1:step]
        actual, cache = model({'inputs': source, 'targets': tokens,
                               'dec_action_mask': mask[:, step:step+1], 'cache': cache},
                              return_kv_cache=True, return_last_token=True)
        torch.testing.assert_close(actual, full[:, step], rtol=1e-5, atol=1e-6)


def test_all_padding_metrics_and_empty_candidate_masks_are_finite():
    logits = torch.randn(2, 3, 8, requires_grad=True)
    labels = torch.zeros(2, 3, dtype=torch.long)
    loss = masked_loss(labels, logits)
    assert loss.item() == masked_accuracy(labels, logits).item() == 0.
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    model = Seq2SeqTransformer(embedding_width=8, num_layers=1, num_attention_heads=2,
                               intermediate_size=16, max_tree_depth=None)
    result = model._candidates(np.zeros((2, 3, 20)), model.enc_embedding_lookup, torch.tensor(0.1))
    assert torch.all(result == 0)


@pytest.mark.parametrize('make_model,option', [
    (lambda: Seq2SeqTransformer(num_attention_heads=0), 'num_attention_heads'),
    (lambda: Seq2SeqTransformer(max_tree_depth=0), 'max_tree_depth'),
    (lambda: Seq2SeqTransformer(num_layers=0), 'num_layers'),
    (lambda: Seq2SeqTransformer(beam_size=0), 'beam_size'),
])
def test_invalid_model_sizes_have_clear_errors(make_model, option):
    with pytest.raises(ValueError, match=option):
        make_model()
