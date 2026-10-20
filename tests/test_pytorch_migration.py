import copy
import json
from pathlib import Path
import subprocess
import sys

import bitarray
import numpy as np
import pytest
import torch

from circuit_transformer import CircuitTransformer, Node, NodeWithInv, compute_tts, write_aiger
from circuit_transformer.pytorch_transformer import Seq2SeqTransformer, masked_loss, masked_accuracy, CustomSchedule


def small_circuit(**kwargs):
    return CircuitTransformer(num_inputs=2, embedding_width=64, num_layers=2,
                              num_attention_heads=2, intermediate_size=24,
                              max_tree_depth=32, max_seq_length=12,
                              mixed_precision=False, **kwargs)


def features(candidate=False):
    rng = np.random.default_rng(18)
    source = np.array([[6, 2, 4, 1, 0], [7, 3, 5, 1, 0]], np.int32)
    target = np.array([[6, 2, 4, 1, 0], [7, 3, 5, 1, 0]], np.int32)
    masks = rng.random((2, 5, 8)) > 0.4
    np.put_along_axis(masks, target[..., None], True, axis=-1)
    return dict(inputs=source, targets=target,
                enc_pos_encoding=rng.integers(0, 2, (2, 5, 64)).astype(np.float32),
                dec_pos_encoding=rng.integers(0, 2, (2, 5, 64)).astype(np.float32),
                enc_action_mask=masks.copy(), dec_action_mask=masks)


def aig_pair():
    x, y = Node(0, None, None), Node(1, None, None)
    return [NodeWithInv(Node(None, NodeWithInv(x, True), NodeWithInv(y, False)), False),
            NodeWithInv(x, False)]


def test_native_save_load_and_no_tensorflow_import(tmp_path):
    original = small_circuit()
    original.save(tmp_path / 'model.pt')
    restored = small_circuit(ckpt_path=tmp_path / 'model.pt')
    x = features()
    np.testing.assert_array_equal(original._transformer_inference(x)[0], restored._transformer_inference(x)[0])
    script = ('import sys; from circuit_transformer import CircuitTransformer; '
              'c=CircuitTransformer(num_inputs=2,embedding_width=64,num_layers=2,'
              'num_attention_heads=2,intermediate_size=24,max_tree_depth=32,'
              f'ckpt_path={str(tmp_path / "model.pt")!r},mixed_precision=False); '
              'assert "tensorflow" not in sys.modules; print("torch-only OK")')
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('mcts', [0, 1])
@pytest.mark.parametrize('care', [None, '0110'])
def test_circuit_truth_tables_and_mcts(mcts, care):
    torch.manual_seed(11)
    model = small_circuit()
    original = aig_pair()
    kwargs = {} if care is None else {'care_set_tts': [bitarray.bitarray(care)]}
    result = model.optimize_batch([original], max_inference_seq_length=12,
                                   num_mcts_steps=mcts, num_leaf_parallelization=2,
                                   num_mcts_playouts_per_step=2, **kwargs)[0]
    for expected, actual in zip(compute_tts(original, 2), compute_tts(result, 2)):
        assert expected == actual if care is None else expected[bitarray.bitarray(care)] == actual[bitarray.bitarray(care)]


def test_training_checkpoint_and_inference_after_training(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    aig = write_aiger(aig_pair())
    for i in range(4):
        (data / f'{i}.json').write_text(json.dumps([aig, 1, aig, 1]))
    model = small_circuit()
    before = copy.deepcopy(model._transformer.state_dict())
    model.train(str(data), str(tmp_path / 'checkpoints'), epochs=2, batch_size=2, validation_split=0.25,
                num_workers=0)
    assert (tmp_path / 'checkpoints' / 'model-0002.pt').is_file()
    assert any(not torch.equal(before[k], v) for k, v in model._transformer.state_dict().items())
    assert model._transformer.return_cache
    assert all(np.isfinite(list(row.values())).all() for row in model.training_history)
    restored = small_circuit(ckpt_path=tmp_path / 'checkpoints' / 'model-0002.pt')
    np.testing.assert_array_equal(model._transformer_inference(features())[0],
                                   restored._transformer_inference(features())[0])


def test_masked_metrics_and_schedule():
    labels = torch.tensor([[1, 2, 0]])
    logits = torch.tensor([[[0., 2., 1.], [0., 2., 1.], [5., 0., 0.]]])
    assert masked_accuracy(labels, logits) == 0.5
    expected = -torch.log_softmax(logits, -1)[0, :2, [1, 2]].mean()
    torch.testing.assert_close(masked_loss(labels, logits), expected)
    assert CustomSchedule(16)(0) == 0
    assert CustomSchedule(16)(4000) > CustomSchedule(16)(8000)


def identity_example(value):
    return value


@pytest.mark.skipif(sys.platform != 'linux', reason='Production multiprocessing targets Linux only')
def test_multiprocess_original_epoch_boundary_and_queue_carryover():
    from circuit_transformer.utils import MPDataset
    dataset = MPDataset(list(range(6)), identity_example,
                        num_processes=2, validation_split=0.)
    try:
        assert list(dataset.train_generator()) == [0, 1, 2, 3, 4]
        assert list(dataset.train_generator()) == [5, 0, 1, 2, 3]
    finally:
        dataset.process.terminate()
        dataset.process.join()
    assert not dataset.process.is_alive()


@pytest.mark.skipif(sys.platform != 'linux', reason='Production multiprocessing targets Linux only')
def test_multiprocess_training(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    aig = write_aiger(aig_pair())
    for i in range(4):
        (data / f'{i}.json').write_text(json.dumps([aig, 1, aig, 1]))
    model = small_circuit()
    model.train(str(data), str(tmp_path / 'checkpoints'), epochs=2, batch_size=2,
                validation_split=0.25, num_workers=2)
    assert len(model.training_history) == 2
    assert (tmp_path / 'checkpoints' / 'model-0002.pt').exists()


def test_original_circuit_methods_unchanged():
    import ast
    original = ast.parse(Path(__file__).with_name('reference').joinpath('model.py').read_text())
    migrated = ast.parse(Path(__file__).parents[1].joinpath('circuit_transformer/model.py').read_text())
    def bodies(tree):
        result = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                result[node.name] = node
        return result
    before, after = bodies(original), bodies(migrated)
    for name in ('node_to_int', 'int_to_node', 'encode_aig', 'stack_to_encoding',
                 'deref_node', 'LogicNetworkEnv', 'MCTSNode', 'ucb'):
        assert ast.dump(before[name]) == ast.dump(after[name]), name
    methods_before, methods_after = bodies(before['CircuitTransformer']), bodies(after['CircuitTransformer'])
    for name in ('_copy_env', '_batch_estimate_policy', '_batch_MCTS_policy_with_leaf_parallelization',
                 '_encode_postprocess', 'generate_action_masks', 'load_and_encode',
                 'load_and_encode_formatted', 'optimize', 'optimize_batch'):
        assert ast.dump(methods_before[name]) == ast.dump(methods_after[name]), name
