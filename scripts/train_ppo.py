#!/usr/bin/env python3
"""Run one small PPO rollout/update round from a trained checkpoint."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ppo.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    import yaml
    config_path = args.config.expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text())
    required = {'model_config', 'checkpoint', 'data_dir', 'output_dir', 'device', 'mixed_precision',
                'seed', 'num_circuits', 'samples_per_circuit', 'max_length', 'failure_penalty',
                'temperature', 'ppo_epochs', 'minibatch_size', 'learning_rate', 'clip_ratio', 'max_grad_norm'}
    if not isinstance(cfg, dict) or set(cfg) != required:
        parser.error(f'PPO config must contain exactly: {sorted(required)}')
    timestamp = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f')
    for key in ('model_config', 'checkpoint', 'data_dir', 'output_dir'):
        value = Path(os.path.expandvars(str(cfg[key]).replace('{timestamp}', timestamp))).expanduser()
        cfg[key] = str((value if value.is_absolute() else config_path.parent / value).resolve())
    for key in ('model_config', 'checkpoint'):
        if not Path(cfg[key]).is_file():
            parser.error(f'File not found: {cfg[key]}')
    if not Path(cfg['data_dir']).is_dir():
        parser.error(f'Data directory not found: {cfg["data_dir"]}')
    for key in ('num_circuits', 'samples_per_circuit', 'max_length', 'ppo_epochs', 'minibatch_size'):
        if type(cfg[key]) is not int or cfg[key] < (2 if key == 'samples_per_circuit' else 1):
            parser.error(f'Invalid {key}')
    for key in ('temperature', 'learning_rate', 'max_grad_norm', 'failure_penalty'):
        if not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < float('inf'):
            parser.error(f'Invalid {key}')
    if not 0 < cfg['clip_ratio'] < 1 or type(cfg['mixed_precision']) is not bool:
        parser.error('Invalid clip_ratio or mixed_precision')
    print(f"Checkpoint: {cfg['checkpoint']}\nData: {cfg['data_dir']}\nOutput: {cfg['output_dir']}\n"
          f"Rollouts: {cfg['num_circuits']} circuits × {cfg['samples_per_circuit']} samples", flush=True)
    if args.dry_run:
        return
    import numpy as np
    import torch
    from circuit_transformer import CircuitTransformer
    from circuit_transformer.ppo import rollout, update, assign_advantages
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    # Reservoir sampling avoids loading millions of file names into memory.
    selected = []
    rng = random.Random(cfg['seed'])
    count = 0
    with os.scandir(cfg['data_dir']) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith('.json'):
                continue
            count += 1
            if len(selected) < cfg['num_circuits']:
                selected.append(Path(entry.path))
            else:
                index = rng.randrange(count)
                if index < len(selected):
                    selected[index] = Path(entry.path)
    if len(selected) < cfg['num_circuits']:
        raise ValueError(f'Only {len(selected)} JSON files found')
    model_cfg = yaml.safe_load(Path(cfg['model_config']).read_text())['model'].copy()
    model_cfg.update(ckpt_path=cfg['checkpoint'], device=cfg['device'],
                     mixed_precision=cfg['mixed_precision'], verbose=0)
    model = CircuitTransformer(**model_cfg)
    if cfg['max_length'] > model.max_seq_length:
        raise ValueError('max_length must not exceed model max_seq_length')
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'resolved_ppo.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    trajectories = []
    for i, path in enumerate(selected):
        group = [rollout(model, path, cfg['max_length'], cfg['temperature'], cfg['failure_penalty'])
                 for _ in range(cfg['samples_per_circuit'])]
        assign_advantages(group)
        trajectories.extend(group)
        print(f"Rollout {i + 1}/{len(selected)}: rewards={[t['reward'] for t in group]}", flush=True)
    with (out / 'rollouts.jsonl').open('w') as f:
        for t in trajectories:
            f.write(json.dumps({k: t[k] for k in ('file', 'reward', 'advantage', 'success', 'before', 'after')}) + '\n')
    optimizer = torch.optim.Adam(model._transformer.parameters(), lr=cfg['learning_rate'])
    metrics = []
    for epoch in range(cfg['ppo_epochs']):
        random.shuffle(trajectories)
        for start in range(0, len(trajectories), cfg['minibatch_size']):
            batch = trajectories[start:start + cfg['minibatch_size']]
            m = update(model, batch, optimizer, cfg['clip_ratio'], cfg['temperature'], cfg['max_grad_norm'])
            m.update(epoch=epoch + 1, batch=start // cfg['minibatch_size'] + 1)
            metrics.append(m)
            print(json.dumps(m), flush=True)
    model.save(str(out / 'model-ppo.pt'))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(f'Saved: {out / "model-ppo.pt"}', flush=True)


if __name__ == '__main__':
    main()
