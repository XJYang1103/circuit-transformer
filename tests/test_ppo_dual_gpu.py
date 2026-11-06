"""Run the dual-GPU algorithm on two CPU/Gloo ranks without requiring GPUs."""
from datetime import timedelta
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from circuit_transformer import CircuitTransformer, Node, NodeWithInv, write_aiger
from circuit_transformer.ppo import update
from circuit_transformer.ppo_dual_gpu import PPOObjective, distributed_update, train_dual_gpu


def model_args():
    return dict(num_inputs=2, embedding_width=64, num_layers=1, num_attention_heads=2,
                intermediate_size=32, max_tree_depth=32, max_seq_length=12,
                mixed_precision=False, device='cpu', verbose=0)


def gradient_check(rank):
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.manual_seed(10)
    model = CircuitTransformer(**model_args())
    reference = CircuitTransformer(**model_args())
    reference._transformer.load_state_dict(model._transformer.state_dict())
    objective = DDP(PPOObjective(model._transformer, 1.0, 0.2))
    optimizer = torch.optim.Adam(model._transformer.parameters(), lr=1e-4)
    ref_optimizer = torch.optim.Adam(reference._transformer.parameters(), lr=1e-4)
    trajectories = []
    for tokens, advantage in (([6, 2, 4], 1.), ([7, 3], -0.7), ([2, 4, 6, 3], 0.5)):
        source = dict(inputs=np.array([[6, 2, 4, 1]]),
                      enc_pos_encoding=np.zeros((1, 4, 64), dtype=np.float32),
                      enc_action_mask=np.ones((1, 4, 8), dtype=bool))
        positions = [np.zeros(64, dtype=np.float32) for _ in tokens]
        masks = [np.ones(8, dtype=bool) for _ in tokens]
        with torch.no_grad():
            logits, _ = reference._transformer(dict(source, targets=np.array([tokens]),
                dec_pos_encoding=np.array([positions]), dec_action_mask=np.array([masks])))
            old = torch.distributions.Categorical(logits=logits).log_prob(torch.tensor([tokens]))[0].tolist()
        trajectories.append(dict(source=source, tokens=tokens, positions=positions,
                                 masks=masks, old_logps=old, advantage=advantage))
    original = model._transformer.enc_embedding_lookup.weight.detach().clone()
    # Verify global weighting for uneven ranks, then a step with an empty rank.
    for global_batch in (trajectories, trajectories[:1]):
        got = distributed_update(objective, global_batch[rank::2], optimizer, 1.0)
        expected = update(reference, global_batch, ref_optimizer, 0.2, 1.0, 1.0)
        for key in ('loss', 'approx_kl', 'clip_fraction', 'grad_norm'):
            assert math.isclose(got[key], expected[key], rel_tol=2e-4, abs_tol=2e-6), (key, got, expected)
        for name, value in model._transformer.state_dict().items():
            torch.testing.assert_close(value, reference._transformer.state_dict()[name], atol=2e-6, rtol=2e-4)
    assert not torch.equal(original, model._transformer.enc_embedding_lookup.weight)


def worker(rank, root):
    import circuit_transformer.ppo_dual_gpu as training
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group('gloo', init_method=(root / 'store').as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        gradient_check(rank)
        cfg = yaml.safe_load((root / 'settings.yaml').read_text())
        real_update = training.distributed_update
        calls = 0
        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError('simulated interruption')
            return real_update(*args, **kwargs)
        training.distributed_update = interrupt
        try:
            train_dual_gpu(cfg, model_args(), device='cpu')
            raise AssertionError('Expected interruption')
        except RuntimeError as exc:
            assert 'simulated interruption' in str(exc), str(exc)
        finally:
            training.distributed_update = real_update
        state_path = Path(cfg['output_dir']) / 'trainer-latest.pt'
        state = torch.load(state_path, weights_only=True)
        assert state['cursor']['processed_circuits'] == 2
        train_dual_gpu(cfg, model_args(), state_path, device='cpu')
        reference = dict(cfg, output_dir=str(root / 'reference'))
        train_dual_gpu(reference, model_args(), device='cpu')
        if rank == 0:
            out, ref = Path(cfg['output_dir']), Path(reference['output_dir'])
            summary = json.loads((out / 'summary.json').read_text())
            assert summary['processed_circuits'] == 3
            assert summary['updates'] == 4
            records = [json.loads(line) for line in (out / 'rollouts.jsonl').read_text().splitlines()]
            assert len(records) == 6
            for i in range(3):
                assert sum(r['file'] == str(root / 'data' / f'{i}.json') for r in records) == 2
            assert {r['rank'] for r in records} == {0, 1}
            for name in ('rollouts.jsonl', 'metrics.json', 'summary.json'):
                assert (out / name).read_bytes() == (ref / name).read_bytes()
            actual = torch.load(out / 'model-ppo.pt', weights_only=True)['state_dict']
            expected = torch.load(ref / 'model-ppo.pt', weights_only=True)['state_dict']
            assert all(torch.equal(actual[k], expected[k]) for k in actual)
            CircuitTransformer(**dict(model_args(), ckpt_path=str(out / 'model-ppo.pt')))
    finally:
        dist.destroy_process_group()


def test_two_rank_ppo_gradients_traversal_and_resume(tmp_path):
    model = CircuitTransformer(**model_args())
    checkpoint = tmp_path / 'initial.pt'
    model.save(checkpoint)
    data = tmp_path / 'data'
    data.mkdir()
    x, y = Node(0, None, None), Node(1, None, None)
    root = Node(None, NodeWithInv(x, False), NodeWithInv(y, False))
    aig = write_aiger([NodeWithInv(root, False)])
    for i in range(3):
        (data / f'{i}.json').write_text(json.dumps([aig, 1, aig, 1]))
    cfg = dict(model_config='unused.yaml', checkpoint=str(checkpoint), data_dir=str(data),
               output_dir=str(tmp_path / 'interrupted'), device='cuda', world_size=2,
               mixed_precision=False, seed=42, num_circuits=None, data_epochs=1,
               rollout_batch_size=2, save_every_batches=1, samples_per_circuit=2,
               max_length=12, failure_penalty=10., temperature=1., ppo_epochs=1,
               minibatch_size=2, learning_rate=1e-5, clip_ratio=0.2, max_grad_norm=1.)
    (tmp_path / 'settings.yaml').write_text(yaml.safe_dump(cfg))
    mp.spawn(worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_dual_gpu_config_dry_run(tmp_path):
    model_config = tmp_path / 'model.yaml'
    model_config.write_text(yaml.safe_dump({'model': model_args()}))
    checkpoint = tmp_path / 'model.pt'
    checkpoint.touch()
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / 'configs/ppo_dual_gpu.yaml').read_text())
    cfg.update(model_config=str(model_config), checkpoint=str(checkpoint),
               data_dir=str(tmp_path), output_dir=str(tmp_path / 'output'), max_length=12)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    command = [sys.executable, str(repo / 'scripts/train_ppo_dual_gpu.py'),
               '--config', str(config), '--dry-run']
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert 'Circuits: ALL' in result.stdout
    assert not (tmp_path / 'output').exists()
    cfg['minibatch_size'] = 3
    config.write_text(yaml.safe_dump(cfg))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'minibatch_size' in result.stderr


def test_dual_gpu_nccl_launcher_and_resume(tmp_path):
    import os
    import pytest
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA GPUs')
    args = model_args()
    checkpoint = tmp_path / 'initial.pt'
    CircuitTransformer(**args).save(checkpoint)
    model_config = tmp_path / 'model.yaml'
    model_config.write_text(yaml.safe_dump({'model': args}))
    data = tmp_path / 'data'
    data.mkdir()
    x, y = Node(0, None, None), Node(1, None, None)
    root = Node(None, NodeWithInv(x, False), NodeWithInv(y, False))
    aig = write_aiger([NodeWithInv(root, False)])
    for i in range(3):
        (data / f'{i}.json').write_text(json.dumps([aig, 1, aig, 1]))
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / 'configs/ppo_dual_gpu.yaml').read_text())
    out = tmp_path / 'dual-gpu-run'
    cfg.update(model_config=str(model_config), checkpoint=str(checkpoint),
               data_dir=str(data), output_dir=str(out), max_length=12,
               rollout_batch_size=2, minibatch_size=2, ppo_epochs=1, save_every_batches=1)
    config = tmp_path / 'dual.yaml'
    config.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='1')
    command = [sys.executable, '-u', str(repo / 'scripts/train_ppo_dual_gpu.py')]
    result = subprocess.run(command + ['--config', str(config)], env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['processed_circuits'] == 3 and summary['updates'] == 4
    assert summary['world_size'] == 2
    final_state = torch.load(out / 'model-ppo.pt', weights_only=True)['state_dict']
    result = subprocess.run(command + ['--resume', str(out / 'trainer-latest.pt')], env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    # Reload through the existing public inference interface.
    CircuitTransformer(**dict(args, ckpt_path=str(out / 'model-ppo.pt')))
    resumed_state = torch.load(out / 'model-ppo.pt', weights_only=True)['state_dict']
    assert all(torch.equal(final_state[k], resumed_state[k]) for k in final_state)
