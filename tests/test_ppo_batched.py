"""Batched decoding must retain masks, cache rows and per-trajectory losses."""
import json

import numpy as np
import torch

from circuit_transformer import CircuitTransformer, Node, NodeWithInv, write_aiger
from circuit_transformer.ppo_batched import rollout_circuits, trajectory_batch
from circuit_transformer.ppo_dual_gpu import PPOObjective


def test_batched_sampling_compaction_and_teacher_forcing(tmp_path):
    torch.manual_seed(3)
    model = CircuitTransformer(num_inputs=2, embedding_width=64, num_layers=1,
        num_attention_heads=2, intermediate_size=32, max_tree_depth=32,
        max_seq_length=16, mixed_precision=False, device='cpu', verbose=0)
    x, y = Node(0, None, None), Node(1, None, None)
    root = Node(None, NodeWithInv(x, False), NodeWithInv(y, False))
    paths = []
    for i, roots in enumerate(([NodeWithInv(x, False)], [NodeWithInv(root, False)],
                               [NodeWithInv(root, False), NodeWithInv(x, True)])):
        aig = write_aiger(roots)
        path = tmp_path / f'{i}.json'
        path.write_text(json.dumps([aig, 0, aig, 0]))
        paths.append(path)
    seen = []
    hook = model._transformer.register_forward_pre_hook(
        lambda module, args: seen.append(args[0]['targets'].shape[0]))
    groups = rollout_circuits(model, paths, 2, 16, 1.0, 10., sampling_batch_size=5)
    hook.remove()
    assert [len(group) for group in groups] == [2, 2, 2]
    assert max(seen) == 5 and min(seen) < 5  # finished rows removed; partial final batch
    trajectories = [t for group in groups for t in group]
    assert len({len(t['tokens']) for t in trajectories}) > 1
    assert all(t['success'] for t in groups[0])
    for t in trajectories:
        assert t['success'] == (t['after'] is not None)
        if not t['success']:
            assert t['reward'] == -10.
        assert all(mask[token] for token, mask in zip(t['tokens'], t['masks']))
        t['advantage'] = 0.5
        inputs = dict(t['source'], targets=np.array([t['tokens']]),
                      dec_pos_encoding=np.array([t['positions']]),
                      dec_action_mask=np.array([t['masks']]))
        with torch.no_grad():
            logits, _ = model._transformer(inputs)
            logp = torch.distributions.Categorical(logits=logits).log_prob(torch.tensor([t['tokens']]))[0]
        np.testing.assert_allclose(logp.numpy(), t['old_logps'], atol=3e-5)
    objective = PPOObjective(model._transformer, 1., .2)
    with torch.no_grad():
        loss, stats = objective(trajectories)
        singles = [objective(t) for t in trajectories]
    torch.testing.assert_close(loss, sum(item[0] for item in singles), atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(stats, sum(item[1] for item in singles), atol=2e-5, rtol=1e-5)
    inputs, old, valid, advantages = trajectory_batch(trajectories, model.device)
    assert inputs['inputs'].shape[1] < model.max_seq_length
    assert valid.sum().item() == sum(len(t['tokens']) for t in trajectories)
    assert (inputs['dec_action_mask'][~valid][:, 0]).all()
