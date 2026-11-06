"""Batched constrained sampling and padded trajectory tensors for dual-GPU PPO."""
from pathlib import Path

import numpy as np
import torch

from circuit_transformer.ppo import source_and_env
from circuit_transformer.utils import compute_tt, count_num_ands


def trajectory_batch(trajectories, device):
    """Pad targets, trim source padding, and preserve equal weight per trajectory."""
    batch, length = len(trajectories), max(len(t['tokens']) for t in trajectories)
    sample = trajectories[0]
    depth = len(sample['positions'][0])
    vocab = len(sample['masks'][0])
    tokens = np.zeros((batch, length), dtype=np.int64)
    positions = np.zeros((batch, length, depth), dtype=np.float32)
    masks = np.zeros((batch, length, vocab), dtype=bool)
    masks[:, :, 0] = True  # padding rows remain valid categorical distributions
    old = np.zeros((batch, length), dtype=np.float32)
    valid = np.zeros((batch, length), dtype=bool)
    for i, t in enumerate(trajectories):
        size = len(t['tokens'])
        tokens[i, :size] = t['tokens']
        positions[i, :size] = t['positions']
        masks[i, :size] = t['masks']
        old[i, :size] = t['old_logps']
        valid[i, :size] = True
    sources = {key: np.concatenate([t['source'][key] for t in trajectories], axis=0)
               for key in sample['source']}
    source_length = max(1, int((sources['inputs'] != 0).sum(axis=1).max()))
    inputs = {key: torch.as_tensor(value[:, :source_length], device=device)
              for key, value in sources.items()}
    inputs.update(targets=torch.as_tensor(tokens, device=device),
                  dec_pos_encoding=torch.as_tensor(positions, device=device),
                  dec_action_mask=torch.as_tensor(masks, device=device))
    return (inputs, torch.as_tensor(old, device=device), torch.as_tensor(valid, device=device),
            torch.tensor([t['advantage'] for t in trajectories], device=device)[:, None])


def select_cache(cache, indices):
    """Compact finished rows while preserving each remaining KV history."""
    return dict(encoder_outputs=cache['encoder_outputs'].index_select(0, indices),
                kv_cache={layer: {name: value.index_select(0, indices)
                                  for name, value in values.items()}
                          for layer, values in cache['kv_cache'].items()})


@torch.no_grad()
def sample_work(model, work, max_length, temperature, failure_penalty):
    """One batched decoder forward per step, independent environments per row."""
    source = {key: np.concatenate([item['source'][key] for item in work], axis=0)
              for key in work[0]['source']}
    source_length = max(1, int((source['inputs'] != 0).sum(axis=1).max()))
    source = {key: torch.as_tensor(value[:, :source_length], device=model.device)
              for key, value in source.items()}
    active = list(range(len(work)))
    previous = torch.zeros((len(active), 1), device=model.device, dtype=torch.long)
    previous_pos = torch.zeros((len(active), 1, model.max_tree_depth * 2), device=model.device)
    cache = None
    model._transformer.eval()
    for _ in range(max_length):
        action_masks = np.stack([work[i]['env'].action_masks[-1] for i in active]).astype(bool)
        if not action_masks.any(axis=1).all():
            raise RuntimeError('No legal action during batched PPO sampling')
        inputs = dict(source, targets=previous, dec_pos_encoding=previous_pos,
                      dec_action_mask=torch.as_tensor(action_masks[:, None], device=model.device), cache=cache)
        logits, cache = model._transformer(inputs, return_kv_cache=True, return_last_token=True)
        distribution = torch.distributions.Categorical(logits=logits.float() / temperature)
        actions = distribution.sample()
        # One device-to-host transfer for the CPU environment transitions.
        values = torch.stack((actions.float(), distribution.log_prob(actions)), dim=1).cpu().numpy()
        keep, positions = [], []
        for row, index in enumerate(active):
            item = work[index]
            token, old_logp = int(values[row, 0]), float(values[row, 1])
            _, done = item['env'].step(token)
            position = item['env'].positional_encodings[-1].copy()
            item['tokens'].append(token)
            item['positions'].append(position)
            item['masks'].append(action_masks[row].copy())
            item['old_logps'].append(old_logp)
            if not done:
                keep.append(row)
                positions.append(position)
        if not keep:
            break
        # Completed circuits stop consuming GPU work; histories of remaining
        # rows keep the same sequence length and decoder cache offset.
        indices = torch.tensor(keep, device=model.device, dtype=torch.long)
        active = [active[row] for row in keep]
        previous = actions.index_select(0, indices)[:, None]
        previous_pos = torch.as_tensor(np.array(positions)[:, None], device=model.device)
        if len(keep) != cache['encoder_outputs'].shape[0]:
            source = {key: value.index_select(0, indices) for key, value in source.items()}
            cache = select_cache(cache, indices)
    results = []
    for item in work:
        env, original = item.pop('env'), item.pop('original')
        success = bool(env.success and len(env.roots) == len(original) and all(
            compute_tt(root, input_tt=model.input_tt) == compute_tt(orig, input_tt=model.input_tt)
            for root, orig in zip(env.roots, original)))
        before = count_num_ands(original)
        after = count_num_ands(env.roots) if success else None
        item.update(success=success, before=before, after=after,
                    reward=float(before - after) if success else float(-failure_penalty))
        results.append(item)
    return results


def rollout_circuits(model, paths, samples_per_circuit, max_length, temperature,
                     failure_penalty, sampling_batch_size):
    """Return one sample group per input circuit, using bounded GPU batches."""
    groups, pending, owners = [[] for _ in paths], [], []

    def flush():
        results = sample_work(model, pending, max_length, temperature, failure_penalty)
        for owner, result in zip(owners, results):
            groups[owner].append(result)
        pending.clear()
        owners.clear()

    for circuit, path in enumerate(paths):
        path = Path(path)
        source, env, original = source_and_env(model, path, max_length)
        for _ in range(samples_per_circuit):
            pending.append(dict(source=source, env=model._copy_env(env), original=original,
                                tokens=[], positions=[], masks=[], old_logps=[], file=str(path)))
            owners.append(circuit)
            if len(pending) == sampling_batch_size:
                flush()
    if pending:
        flush()
    return groups
