"""双卡 PPO: two processes sample different circuits and share DDP gradients.

The single-GPU training modules remain usable independently. Internal CPU/Gloo
support exists to exercise the same distributed algorithm in regression tests.
"""
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from circuit_transformer import CircuitTransformer
from circuit_transformer.ppo import assign_advantages
from circuit_transformer.ppo_batched import rollout_circuits, trajectory_batch
from circuit_transformer.ppo_training import (
    MetricsWriter, atomic_save, build_manifest, restore_rng, rng_state,
)

STATE_FORMAT = 'circuit-transformer-ppo-dual-gpu-v1'


def broadcast(value, src=0):
    values = [value]
    dist.broadcast_object_list(values, src=src)
    return values[0]


def on_primary(function):
    """Propagate rank-0 filesystem/config errors before other ranks continue."""
    result = None
    if dist.get_rank() == 0:
        try:
            result = dict(value=function())
        except Exception as exc:
            result = dict(error=f'{type(exc).__name__}: {exc}')
    result = broadcast(result)
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result['value']


class PPOObjective(torch.nn.Module):
    def __init__(self, policy, temperature, clip):
        super().__init__()
        self.policy = policy
        self.temperature, self.clip = temperature, clip

    def forward(self, trajectories):
        if trajectories is None or (isinstance(trajectories, list) and not trajectories):
            # Empty ranks still participate in the same DDP reduction.
            zero = sum(parameter.reshape(-1)[0] * 0 for parameter in self.policy.parameters())
            return zero, zero.detach().new_zeros(3)
        if isinstance(trajectories, dict):
            trajectories = [trajectories]
        device = next(self.policy.parameters()).device
        inputs, old, valid, advantages = trajectory_batch(trajectories, device)
        logits, _ = self.policy(inputs)
        distribution = torch.distributions.Categorical(logits=logits.float() / self.temperature)
        logp = distribution.log_prob(inputs['targets'])
        log_ratio = logp - old
        ratio = log_ratio.exp()
        def reduce_tokens(values):
            return ((values * valid).sum(dim=1) / valid.sum(dim=1)).sum()
        loss = -reduce_tokens(torch.minimum(ratio * advantages,
                              ratio.clamp(1 - self.clip, 1 + self.clip) * advantages))
        stats = torch.stack((loss.detach(), reduce_tokens((ratio - 1) - log_ratio).detach(),
                             reduce_tokens(((ratio - 1).abs() > self.clip).float()).detach()))
        return loss, stats


def distributed_update(objective, trajectories, optimizer, max_grad_norm,
                       microbatch_size=None, scaler=None):
    """Batched forwards; exact global trajectory mean with optional accumulation."""
    device = next(objective.parameters()).device
    count = torch.tensor(len(trajectories), device=device, dtype=torch.long)
    dist.all_reduce(count)
    total = int(count.item())
    if total == 0:
        raise ValueError('Cannot update an empty global minibatch')
    optimizer.zero_grad(set_to_none=True)
    stats = torch.zeros(3, device=device)
    size = microbatch_size or max(1, len(trajectories))
    work = [trajectories[start:start + size] for start in range(0, len(trajectories), size)] or [None]
    for index, microbatch in enumerate(work):
        context = objective.no_sync() if index < len(work) - 1 else nullcontext()
        with context:
            loss, values = objective(microbatch)
            weighted = loss * (dist.get_world_size() / total)
            if scaler is not None:
                scaler.scale(weighted).backward()
            else:
                weighted.backward()
        stats += values
    enabled = scaler is not None and scaler.is_enabled()
    if scaler is not None:
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(objective.parameters(), max_grad_norm,
                                         error_if_nonfinite=not enabled)
    if scaler is not None:
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
    else:
        optimizer.step()
        skipped = False
    dist.all_reduce(stats)
    values = (stats / total).tolist()
    norm_value = float(norm)
    return dict(loss=values[0], approx_kl=values[1], clip_fraction=values[2],
                grad_norm=norm_value if math.isfinite(norm_value) else None,
                trajectories=total, optimizer_step_skipped=skipped,
                loss_scale=scaler.get_scale() if enabled else 1.0)


def train_dual_gpu(cfg, model_cfg, resume_path=None, device=None):
    """Run inside an initialized two-rank process group."""
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise ValueError('双卡 PPO requires exactly two processes')
    device = torch.device(device if device is not None else f'cuda:{rank}')
    if cfg['minibatch_size'] < world or cfg['minibatch_size'] % world:
        raise ValueError('minibatch_size is the global size and must be a positive multiple of 2')
    out = Path(cfg['output_dir'])
    resume = None
    if resume_path is not None:
        # Each process reads the same atomic training checkpoint independently.
        resume = torch.load(resume_path, map_location='cpu', weights_only=True)
        if resume.get('format') != STATE_FORMAT or resume['world_size'] != world:
            raise ValueError('Expected a two-GPU PPO training checkpoint')
        if resume['config'] != cfg or resume['model_config'] != model_cfg:
            raise ValueError('Resume configuration does not match the saved run')

    def prepare():
        import yaml
        if resume is None:
            out.mkdir(parents=True, exist_ok=False)
            (out / 'resolved_ppo_dual_gpu.yaml').write_text(
                yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            return build_manifest(cfg, out)
        digest = hashlib.sha256()
        with (out / 'data_manifest.jsonl').open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != resume['manifest_sha256']:
            raise ValueError('Data manifest changed since checkpoint')
        return resume['total_circuits'], resume['manifest_sha256']

    total, digest = on_primary(prepare)
    random.seed(cfg['seed'] + rank)
    np.random.seed((cfg['seed'] + rank) % 2 ** 32)
    torch.manual_seed(cfg['seed'] + rank)
    model_args = dict(model_cfg, ckpt_path=cfg['checkpoint'] if resume is None else None,
                      device=str(device), mixed_precision=cfg['mixed_precision'], verbose=0)
    model = CircuitTransformer(**model_args)
    if cfg['max_length'] > model.max_seq_length:
        raise ValueError('max_length exceeds model max_seq_length')
    if resume is not None:
        model._transformer.load_state_dict(resume['model'])
    model._transformer.eval()
    wrapper = PPOObjective(model._transformer, cfg['temperature'], cfg['clip_ratio'])
    objective = DDP(wrapper, device_ids=[device.index] if device.type == 'cuda' else None,
                    broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.Adam(model._transformer.parameters(), lr=cfg['learning_rate'],
                                 fused=device.type == 'cuda')
    scaler = torch.amp.GradScaler(device.type, enabled=cfg['mixed_precision'] and device.type == 'cuda',
                                  init_scale=1024.0)
    if resume is None:
        cursor = dict(data_epoch=0, manifest_offset=0, circuits_in_epoch=0,
                      processed_circuits=0, rollout_batch=0, updates=0)
    else:
        optimizer.load_state_dict(resume['optimizer'])
        if resume.get('scaler'):
            scaler.load_state_dict(resume['scaler'])
        cursor = dict(resume['cursor'])
        restore_rng(resume['rng_by_rank'][rank])

    # Only rank 0 owns logs and writes model/optimizer checkpoints.
    def open_logs():
        nonlocal metrics, records, performance
        metrics = MetricsWriter(out / 'metrics.json', resume['metrics_size'] if resume else None)
        records = (out / 'rollouts.jsonl').open('r+b' if resume else 'w+b')
        if resume:
            records.truncate(resume['rollouts_size'])
            records.seek(0, 2)
        perf_path = out / 'performance.jsonl'
        performance = perf_path.open('r+b' if perf_path.exists() and resume else 'w+b')
        if resume:
            performance.truncate(resume.get('performance_size', 0))
            performance.seek(0, 2)
    metrics = records = performance = None
    on_primary(open_logs)
    # Release the extra checkpoint tensors after loading the optimizer/model.
    resume = None

    def save(force_log=False):
        states = [None] * world
        dist.all_gather_object(states, rng_state())
        def write():
            records.flush()
            performance.flush()
            atomic_save(dict(format=STATE_FORMAT, world_size=world, config=cfg,
                             model_config=model_cfg, model=model._transformer.state_dict(),
                             optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), cursor=dict(cursor),
                             total_circuits=total, manifest_sha256=digest, rng_by_rank=states,
                             metrics_size=metrics.size(), rollouts_size=records.tell(),
                             performance_size=performance.tell()),
                        out / 'trainer-latest.pt')
            model.save(out / 'model-latest.pt')
            saved_index = cursor['rollout_batch'] // cfg['save_every_batches']
            if force_log or saved_index % cfg.get('log_every_checkpoints', 10) == 0:
                print(f"[双卡] Saved {cursor['processed_circuits']} circuits; "
                      f"{cursor['updates']} updates", flush=True)
        on_primary(write)

    if rank == 0:
        print(f'[双卡 PPO] {total} circuits, {cfg["data_epochs"]} data epoch(s), '
              f'global rollout batch={cfg["rollout_batch_size"]}', flush=True)
    if resume_path is None:
        save(force_log=True)
    manifest = None
    try:
        while cursor['data_epoch'] < cfg['data_epochs']:
            def open_manifest():
                nonlocal manifest
                manifest = (out / 'data_manifest.jsonl').open('rb')
                manifest.seek(cursor['manifest_offset'])
            on_primary(open_manifest)
            while True:
                def read_paths():
                    paths = []
                    for _ in range(cfg['rollout_batch_size']):
                        line = manifest.readline()
                        if not line:
                            break
                        paths.append(json.loads(line))
                    return paths, manifest.tell()
                paths, offset = on_primary(read_paths)
                if not paths:
                    break
                epoch = cursor['data_epoch'] + 1
                rollout_batch = cursor['rollout_batch'] + 1
                log_batch = (rollout_batch == 1 or rollout_batch % cfg.get('log_every_batches', 10) == 0
                             or cursor['circuits_in_epoch'] + len(paths) == total)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                sampling_started = time.perf_counter()
                trajectories, local_records = [], []
                groups = rollout_circuits(model, paths[rank::world], cfg['samples_per_circuit'],
                                          cfg['max_length'], cfg['temperature'], cfg['failure_penalty'],
                                          cfg.get('sampling_batch_size', 32))
                for group in groups:
                    assign_advantages(group)
                    trajectories.extend(group)
                    for item in group:
                        record = {k: item[k] for k in ('file', 'reward', 'advantage', 'success', 'before', 'after')}
                        record.update(rank=rank, data_epoch=epoch, rollout_batch=rollout_batch)
                        local_records.append(record)
                del groups
                sampling_seconds = time.perf_counter() - sampling_started
                gathered = [None] * world
                dist.all_gather_object(gathered, local_records)
                def write_rollouts():
                    for group_records in gathered:
                        for record in group_records:
                            records.write((json.dumps(record) + '\n').encode())
                    records.flush()
                    if log_batch:
                        print(f'[双卡] Rollout batch {rollout_batch}: data epoch {epoch}, '
                              f'{cursor["circuits_in_epoch"] + len(paths)}/{total} circuits', flush=True)
                on_primary(write_rollouts)
                per_rank = cfg['minibatch_size'] // world
                steps = torch.tensor(math.ceil(len(trajectories) / per_rank), device=device)
                dist.all_reduce(steps, op=dist.ReduceOp.MAX)
                update_started = time.perf_counter()
                for ppo_epoch in range(cfg['ppo_epochs']):
                    random.shuffle(trajectories)
                    for index in range(int(steps.item())):
                        batch = trajectories[index * per_rank:(index + 1) * per_rank]
                        m = distributed_update(objective, batch, optimizer, cfg['max_grad_norm'],
                                               cfg.get('train_microbatch_size', 16), scaler)
                        cursor['updates'] += 1
                        m.update(epoch=ppo_epoch + 1, batch=index + 1, data_epoch=epoch,
                                 rollout_batch=rollout_batch, update=cursor['updates'])
                        if rank == 0:
                            metrics.append(m)
                            if cursor['updates'] % cfg.get('log_every_updates', 100) == 0:
                                print(json.dumps(m), flush=True)
                update_seconds = time.perf_counter() - update_started
                local_perf = torch.tensor([sampling_seconds, update_seconds,
                    torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0,
                    torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else 0],
                    dtype=torch.float64, device=device)
                rank_perf = [torch.empty_like(local_perf) for _ in range(world)]
                dist.all_gather(rank_perf, local_perf)
                if rank == 0:
                    values = [value.tolist() for value in rank_perf]
                    sampling_time, update_time = max(v[0] for v in values), max(v[1] for v in values)
                    sampled = len(paths) * cfg['samples_per_circuit']
                    perf = dict(rollout_batch=rollout_batch, data_epoch=epoch,
                                sampling_seconds=sampling_time, update_seconds=update_time,
                                sampling_trajectories_per_second=sampled / max(sampling_time, 1e-9),
                                update_trajectories_per_second=sampled * cfg['ppo_epochs'] / max(update_time, 1e-9),
                                peak_allocated_mib=[v[2] for v in values], peak_reserved_mib=[v[3] for v in values])
                    performance.write((json.dumps(perf) + '\n').encode())
                    performance.flush()
                    if log_batch:
                        print('[双卡性能] ' + json.dumps(perf), flush=True)
                del trajectories, batch, local_records, gathered
                cursor.update(manifest_offset=offset, rollout_batch=rollout_batch,
                              circuits_in_epoch=cursor['circuits_in_epoch'] + len(paths),
                              processed_circuits=cursor['processed_circuits'] + len(paths))
                if rollout_batch % cfg['save_every_batches'] == 0:
                    save()
            if rank == 0:
                manifest.close()
                manifest = None
            if cursor['circuits_in_epoch'] != total:
                raise RuntimeError('Manifest circuit count mismatch')
            cursor.update(data_epoch=cursor['data_epoch'] + 1, manifest_offset=0, circuits_in_epoch=0)
            save(force_log=True)
        def finish():
            model.save(out / 'model-ppo.pt')
            (out / 'summary.json').write_text(json.dumps(
                dict(world_size=world, circuits_per_epoch=total, **cursor, completed=True), indent=2) + '\n')
            print(f'[双卡] Completed: {out / "model-ppo.pt"}', flush=True)
        on_primary(finish)
    finally:
        if manifest is not None:
            manifest.close()
        if metrics is not None:
            metrics.close()
        if records is not None:
            records.close()
        if performance is not None:
            performance.close()
