"""Bounded-memory dataset traversal and resumable PPO training."""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from circuit_transformer import CircuitTransformer
from circuit_transformer.ppo import assign_advantages, rollout, update


STATE_FORMAT = 'circuit-transformer-ppo-v1'


def atomic_save(payload, path):
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_manifest(cfg, out):
    """Freeze traversal order on disk; retain only a small subset when requested."""
    limit = cfg['num_circuits']
    selected = []
    rng = random.Random(cfg['seed'])
    count = 0
    digest = hashlib.sha256()
    manifest = out / 'data_manifest.jsonl'
    with manifest.open('wb') as handle, os.scandir(cfg['data_dir']) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith('.json'):
                continue
            count += 1
            if limit is None:
                line = (json.dumps(entry.path) + '\n').encode()
                handle.write(line)
                digest.update(line)
            elif len(selected) < limit:
                selected.append(entry.path)
            else:
                index = rng.randrange(count)
                if index < limit:
                    selected[index] = entry.path
            if count % 100000 == 0:
                print(f'Manifest: scanned {count} files', flush=True)
        if not count or (limit is not None and count < limit):
            raise ValueError(f'Only {count} JSON files found; requested {limit}')
        if limit is not None:
            for path in selected:
                line = (json.dumps(path) + '\n').encode()
                handle.write(line)
                digest.update(line)
    return count if limit is None else limit, digest.hexdigest()


class MetricsWriter:
    """Append to a valid JSON array without holding the history in memory."""
    def __init__(self, path, size=None):
        if size is None:
            self.handle = path.open('w+b')
            self.handle.write(b'[\n]\n')
        else:
            self.handle = path.open('r+b')
            self.handle.truncate(size)
            self.handle.seek(size - 2)
            self.handle.write(b']\n')
        self.handle.flush()

    def append(self, record):
        self.handle.seek(-2, os.SEEK_END)
        prefix = b'' if self.handle.tell() == 2 else b',\n'
        self.handle.write(prefix + json.dumps(record).encode() + b'\n]\n')
        self.handle.flush()

    def size(self):
        return self.handle.seek(0, os.SEEK_END)

    def close(self):
        self.handle.close()


def rng_state():
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]))


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    value = state['numpy']
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))


def train_ppo(cfg, model_cfg, resume=None):
    out = Path(cfg['output_dir'])
    if resume is None:
        import yaml
        out.mkdir(parents=True, exist_ok=False)
        (out / 'resolved_ppo.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        total, digest = build_manifest(cfg, out)
    else:
        total, digest = resume['total_circuits'], resume['manifest_sha256']
        actual = hashlib.sha256()
        with (out / 'data_manifest.jsonl').open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError('Data manifest changed since checkpoint')
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    model_args = dict(model_cfg, ckpt_path=cfg['checkpoint'] if resume is None else None,
                      device=cfg['device'], mixed_precision=cfg['mixed_precision'], verbose=0)
    model = CircuitTransformer(**model_args)
    if cfg['max_length'] > model.max_seq_length:
        raise ValueError('max_length must not exceed model max_seq_length')
    optimizer = torch.optim.Adam(model._transformer.parameters(), lr=cfg['learning_rate'])
    if resume is None:
        cursor = dict(data_epoch=0, manifest_offset=0, circuits_in_epoch=0,
                      processed_circuits=0, rollout_batch=0, updates=0)
        metrics = MetricsWriter(out / 'metrics.json')
        records = (out / 'rollouts.jsonl').open('w+b')
    else:
        model._transformer.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        cursor = dict(resume['cursor'])
        metrics = MetricsWriter(out / 'metrics.json', resume['metrics_size'])
        records = (out / 'rollouts.jsonl').open('r+b')
        records.truncate(resume['rollouts_size'])
        records.seek(0, os.SEEK_END)
        restore_rng(resume['rng'])

    def save():
        records.flush()
        atomic_save(dict(format=STATE_FORMAT, config=cfg, model_config=model_cfg,
                         model=model._transformer.state_dict(), optimizer=optimizer.state_dict(),
                         cursor=dict(cursor), total_circuits=total, manifest_sha256=digest,
                         metrics_size=metrics.size(), rollouts_size=records.tell(), rng=rng_state()),
                    out / 'trainer-latest.pt')
        model.save(out / 'model-latest.pt')
        print(f"Checkpoint: {cursor['processed_circuits']} circuits, {cursor['updates']} updates", flush=True)

    print(f"Dataset: {total} circuits × {cfg['data_epochs']} pass(es); "
          f"{cfg['rollout_batch_size']} circuits per rollout batch", flush=True)
    # An initial state also permits recovery before the first periodic save.
    if resume is None:
        save()
    try:
        while cursor['data_epoch'] < cfg['data_epochs']:
            with (out / 'data_manifest.jsonl').open('rb') as manifest:
                manifest.seek(cursor['manifest_offset'])
                while True:
                    paths = []
                    for _ in range(cfg['rollout_batch_size']):
                        line = manifest.readline()
                        if not line:
                            break
                        paths.append(Path(json.loads(line)))
                    if not paths:
                        break
                    trajectories = []
                    epoch = cursor['data_epoch'] + 1
                    rollout_batch = cursor['rollout_batch'] + 1
                    for path in paths:
                        group = [rollout(model, path, cfg['max_length'], cfg['temperature'], cfg['failure_penalty'])
                                 for _ in range(cfg['samples_per_circuit'])]
                        assign_advantages(group)
                        trajectories.extend(group)
                        for t in group:
                            record = {k: t[k] for k in ('file', 'reward', 'advantage', 'success', 'before', 'after')}
                            record.update(data_epoch=epoch, rollout_batch=rollout_batch)
                            records.write((json.dumps(record) + '\n').encode())
                    print(f"Rollout batch {rollout_batch}: data epoch {epoch}, "
                          f"{cursor['circuits_in_epoch'] + len(paths)}/{total} circuits, "
                          f"mean reward={np.mean([t['reward'] for t in trajectories]):.3f}", flush=True)
                    for ppo_epoch in range(cfg['ppo_epochs']):
                        random.shuffle(trajectories)
                        for start in range(0, len(trajectories), cfg['minibatch_size']):
                            batch = trajectories[start:start + cfg['minibatch_size']]
                            m = update(model, batch, optimizer, cfg['clip_ratio'], cfg['temperature'], cfg['max_grad_norm'])
                            cursor['updates'] += 1
                            m.update(epoch=ppo_epoch + 1, batch=start // cfg['minibatch_size'] + 1,
                                     data_epoch=epoch, rollout_batch=rollout_batch, update=cursor['updates'])
                            metrics.append(m)
                            print(json.dumps(m), flush=True)
                    # All updates for this rollout batch are complete; old trajectories are discarded.
                    del trajectories, group, batch, t
                    cursor.update(manifest_offset=manifest.tell(), rollout_batch=rollout_batch,
                                  circuits_in_epoch=cursor['circuits_in_epoch'] + len(paths),
                                  processed_circuits=cursor['processed_circuits'] + len(paths))
                    records.flush()
                    if rollout_batch % cfg['save_every_batches'] == 0:
                        save()
            if cursor['circuits_in_epoch'] != total:
                raise RuntimeError('Manifest circuit count mismatch')
            cursor.update(data_epoch=cursor['data_epoch'] + 1, manifest_offset=0, circuits_in_epoch=0)
            save()
        model.save(out / 'model-ppo.pt')
        summary = dict(circuits_per_epoch=total, **cursor, completed=True)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f"Saved: {out / 'model-ppo.pt'}", flush=True)
    finally:
        metrics.close()
        records.close()
