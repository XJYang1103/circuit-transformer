"""Small on-policy PPO fine-tuning for constrained circuit generation."""
import json
import random

import numpy as np
import torch

from circuit_transformer.model import LogicNetworkEnv, encode_aig
from circuit_transformer.utils import read_aiger, compute_tt, count_num_ands


def source_and_env(model, path, max_length):
    record = json.loads(path.read_text())
    aig, info = read_aiger(aiger_str=record[0])
    if info[1] > model.num_inputs or info[2] != 0 or info[3] not in (1, 2):
        raise ValueError(f'Unsupported circuit: {path}: {info}')
    aig = aig if isinstance(aig, list) else [aig]
    seq, pos = encode_aig(aig, model.num_inputs)
    if len(seq) + 1 > model.max_seq_length:
        raise ValueError(f'Input circuit exceeds max_seq_length: {path}')
    tts = [compute_tt(root, input_tt=model.input_tt) for root in aig]
    mask = model.generate_action_masks(tts, model.input_tt, None, seq, True, tts)
    seq, pos = model._encode_postprocess(seq, pos)
    source = dict(inputs=seq[None], enc_pos_encoding=pos[None], enc_action_mask=mask[None])
    env = LogicNetworkEnv(tts=tts, num_inputs=model.num_inputs, input_tt=model.input_tt,
                          max_tree_depth=model.max_tree_depth, max_length=max_length,
                          eos_id=model.eos_id, pad_id=model.pad_id)
    return source, env, aig


@torch.no_grad()
def rollout(model, path, max_length, temperature, failure_penalty):
    source, env, original = source_and_env(model, path, max_length)
    net = model._transformer
    net.eval()
    tokens, positions, masks, old_logps = [], [], [], []
    cache = None
    previous = np.zeros((1, 1), dtype=np.int32)
    previous_pos = np.zeros((1, 1, model.max_tree_depth * 2), dtype=np.float32)
    for _ in range(max_length):
        action_mask = np.asarray(env.action_masks[-1], dtype=bool)
        if not action_mask.any():
            break
        inputs = dict(source, targets=previous, dec_pos_encoding=previous_pos,
                      dec_action_mask=action_mask[None, None], cache=cache)
        logits, cache = net(inputs, return_kv_cache=True, return_last_token=True)
        dist = torch.distributions.Categorical(logits=logits.float() / temperature)
        action = dist.sample()
        token = int(action.item())
        masks.append(action_mask.copy())
        old_logps.append(float(dist.log_prob(action).item()))
        _, done = env.step(token)
        tokens.append(token)
        positions.append(env.positional_encodings[-1].copy())
        previous = np.array([[token]], dtype=np.int32)
        previous_pos = positions[-1][None, None, :]
        if done:
            break
    success = bool(env.success)
    # Verify the environment's logical constraint against the complete truth table.
    equivalent = success and all(compute_tt(root, input_tt=model.input_tt) ==
                                 compute_tt(orig, input_tt=model.input_tt)
                                 for root, orig in zip(env.roots, original))
    success = bool(success and equivalent and len(env.roots) == len(original))
    before = count_num_ands(original)
    after = count_num_ands(env.roots) if success else None
    reward = float(before - after) if success else float(-failure_penalty)
    if not tokens:
        raise RuntimeError(f'No legal action for {path}')
    return dict(source=source, tokens=tokens, positions=positions, masks=masks,
                old_logps=old_logps, reward=reward, success=success,
                before=before, after=after, file=str(path))


def update(model, trajectories, optimizer, clip, temperature, max_grad_norm):
    """One optimizer step; trajectory advantages use other samples as baseline."""
    net = model._transformer
    net.eval()  # identical policy semantics during sampling and optimization
    optimizer.zero_grad(set_to_none=True)
    losses, kls, clipped = [], [], []
    for trajectory in trajectories:
        tokens = torch.tensor([trajectory['tokens']], device=model.device, dtype=torch.long)
        inputs = dict(trajectory['source'], targets=tokens,
                      dec_pos_encoding=np.stack(trajectory['positions'])[None],
                      dec_action_mask=np.stack(trajectory['masks'])[None])
        logits, _ = net(inputs)
        dist = torch.distributions.Categorical(logits=logits.float() / temperature)
        logp = dist.log_prob(tokens)[0]
        old = torch.tensor(trajectory['old_logps'], device=model.device)
        log_ratio = logp - old
        ratio = log_ratio.exp()
        advantage = trajectory['advantage']
        objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
        loss = -objective.mean() / len(trajectories)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite PPO loss')
        loss.backward()
        losses.append(float(loss.detach()) * len(trajectories))
        kls.append(float(((ratio - 1) - log_ratio).mean().detach()))
        clipped.append(float(((ratio - 1).abs() > clip).float().mean().detach()))
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(np.mean(losses)), approx_kl=float(np.mean(kls)),
                clip_fraction=float(np.mean(clipped)), grad_norm=float(norm))


def assign_advantages(group):
    if len(group) < 2:
        raise ValueError('At least two samples per circuit are required')
    rewards = [item['reward'] for item in group]
    for item in group:
        item['advantage'] = (item['reward'] - (sum(rewards) - item['reward']) / (len(group) - 1))
