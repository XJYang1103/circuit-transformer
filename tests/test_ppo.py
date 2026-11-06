import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from circuit_transformer import CircuitTransformer, Node, NodeWithInv, write_aiger
from circuit_transformer.ppo import rollout, assign_advantages


def test_ppo_round_and_sampling_probabilities(tmp_path):
    model_args = dict(num_inputs=2, embedding_width=64, num_layers=1,
                      num_attention_heads=2, intermediate_size=32, max_tree_depth=32,
                      max_seq_length=12, mixed_precision=False, device='cpu', verbose=0)
    model = CircuitTransformer(**model_args)
    ckpt = tmp_path / 'initial.pt'
    model.save(ckpt)
    x, y = Node(0, None, None), Node(1, None, None)
    root = Node(None, NodeWithInv(x, False), NodeWithInv(y, False))
    aig = write_aiger([NodeWithInv(root, False)])
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'one.json').write_text(json.dumps([aig, 1, aig, 1]))
    trajectory = rollout(model, data / 'one.json', 12, 1.0, 10)
    tokens = torch.tensor([trajectory['tokens']])
    inputs = dict(trajectory['source'], targets=tokens,
                  dec_pos_encoding=np.stack(trajectory['positions'])[None],
                  dec_action_mask=np.stack(trajectory['masks'])[None])
    with torch.no_grad():
        logits, _ = model._transformer(inputs)
        logp = torch.distributions.Categorical(logits=logits).log_prob(tokens)[0]
    np.testing.assert_allclose(logp.numpy(), trajectory['old_logps'], atol=2e-5)
    group = [dict(reward=1.), dict(reward=3.)]
    assign_advantages(group)
    assert [x['advantage'] for x in group] == [-2., 2.]
    config = tmp_path / 'model.yaml'
    config.write_text(yaml.safe_dump({'model': model_args}))
    output = tmp_path / 'output'
    ppo_config = tmp_path / 'ppo.yaml'
    ppo_config.write_text(yaml.safe_dump(dict(model_config=str(config), checkpoint=str(ckpt),
        data_dir=str(data), output_dir=str(output), device='cpu', mixed_precision=False,
        seed=42, num_circuits=1, samples_per_circuit=2, max_length=12,
        failure_penalty=10., temperature=1., ppo_epochs=1, minibatch_size=2,
        learning_rate=1e-5, clip_ratio=0.2, max_grad_norm=1.)))
    script = Path(__file__).resolve().parents[1] / 'scripts/train_ppo.py'
    subprocess.run([sys.executable, str(script), '--config', str(ppo_config)], check=True)
    assert (output / 'model-ppo.pt').is_file()
    assert len((output / 'rollouts.jsonl').read_text().splitlines()) == 2
    assert len(json.loads((output / 'metrics.json').read_text())) == 1
    CircuitTransformer(**dict(model_args, ckpt_path=str(output / 'model-ppo.pt')))


def ppo_fixture(tmp_path, name, epochs=1):
    from circuit_transformer.ppo_training import train_ppo
    args = dict(num_inputs=2, embedding_width=64, num_layers=1,
                num_attention_heads=2, intermediate_size=32, max_tree_depth=32,
                max_seq_length=12, mixed_precision=False, device='cpu', verbose=0)
    checkpoint = tmp_path / 'initial.pt'
    if not checkpoint.exists():
        torch.manual_seed(7)
        CircuitTransformer(**args).save(checkpoint)
    data = tmp_path / 'data'
    data.mkdir(exist_ok=True)
    x, y = Node(0, None, None), Node(1, None, None)
    root = Node(None, NodeWithInv(x, False), NodeWithInv(y, False))
    aig = write_aiger([NodeWithInv(root, False)])
    for i in range(3):
        (data / f'{i}.json').write_text(json.dumps([aig, 1, aig, 1]))
    cfg = dict(model_config='unused.yaml', checkpoint=str(checkpoint),
               data_dir=str(data), output_dir=str(tmp_path / name), device='cpu',
               mixed_precision=False, seed=42, num_circuits=None, data_epochs=epochs,
               rollout_batch_size=2, save_every_batches=1, samples_per_circuit=2,
               max_length=12, failure_penalty=10., temperature=1., ppo_epochs=2,
               minibatch_size=3, learning_rate=1e-5, clip_ratio=0.2, max_grad_norm=1.)
    return cfg, args, train_ppo


def test_full_traversal_partial_batch_and_multiple_epochs(tmp_path):
    cfg, args, train = ppo_fixture(tmp_path, 'full', epochs=2)
    train(cfg, args)
    out = Path(cfg['output_dir'])
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['processed_circuits'] == 6
    assert summary['rollout_batch'] == 4
    assert summary['updates'] == 12
    records = [json.loads(line) for line in (out / 'rollouts.jsonl').read_text().splitlines()]
    for epoch in (1, 2):
        paths = [r['file'] for r in records if r['data_epoch'] == epoch]
        assert len(paths) == 6
        assert all(paths.count(str(Path(cfg['data_dir']) / f'{i}.json')) == 2 for i in range(3))
    assert len(json.loads((out / 'metrics.json').read_text())) == 12
    assert len((out / 'data_manifest.jsonl').read_text().splitlines()) == 3
    assert (out / 'model-latest.pt').is_file()
    state = torch.load(out / 'trainer-latest.pt', weights_only=True)
    assert state['cursor']['data_epoch'] == 2


def test_interrupted_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    import pytest
    import circuit_transformer.ppo_training as training
    cfg, args, train = ppo_fixture(tmp_path, 'interrupted')
    real_update = training.update
    calls = 0

    def interrupt(*a, **kw):
        nonlocal calls
        calls += 1
        # First rollout batch has four optimizer updates. Fail after one
        # update in the next batch to exercise log rollback and RNG restore.
        if calls == 6:
            raise RuntimeError('simulated interruption')
        return real_update(*a, **kw)

    monkeypatch.setattr(training, 'update', interrupt)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        train(cfg, args)
    out = Path(cfg['output_dir'])
    state = torch.load(out / 'trainer-latest.pt', map_location='cpu', weights_only=True)
    assert state['cursor']['processed_circuits'] == 2
    assert len(json.loads((out / 'metrics.json').read_text())) == 5
    monkeypatch.setattr(training, 'update', real_update)
    train(state['config'], state['model_config'], state)
    reference_cfg = dict(cfg, output_dir=str(tmp_path / 'uninterrupted'))
    train(reference_cfg, args)
    reference = Path(reference_cfg['output_dir'])
    for name in ('metrics.json', 'rollouts.jsonl', 'summary.json'):
        assert (out / name).read_bytes() == (reference / name).read_bytes()
    resumed = torch.load(out / 'model-ppo.pt', weights_only=True)['state_dict']
    expected = torch.load(reference / 'model-ppo.pt', weights_only=True)['state_dict']
    assert all(torch.equal(resumed[k], expected[k]) for k in expected)
