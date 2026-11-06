#!/usr/bin/env python3
"""Fine-tune a checkpoint with PPO, streaming through the training dataset."""
import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ppo.yaml')
    parser.add_argument('--resume', type=Path, help='Continue an existing trainer-latest.pt using its saved config')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    import yaml
    resume = None
    if args.resume:
        import torch
        resume = torch.load(args.resume.expanduser().resolve(), map_location='cpu', weights_only=True)
        if resume.get('format') != 'circuit-transformer-ppo-v1':
            parser.error('--resume requires a PPO trainer-latest.pt, not inference weights')
        cfg, model_cfg = resume['config'], resume['model_config']
    else:
        config_path = args.config.expanduser().resolve()
        cfg = yaml.safe_load(config_path.read_text())
        required = {'model_config', 'checkpoint', 'data_dir', 'output_dir', 'device', 'mixed_precision',
                    'seed', 'num_circuits', 'samples_per_circuit', 'max_length', 'failure_penalty',
                    'temperature', 'ppo_epochs', 'minibatch_size', 'learning_rate', 'clip_ratio', 'max_grad_norm'}
        defaults = dict(rollout_batch_size=32, data_epochs=1, save_every_batches=10)
        if not isinstance(cfg, dict) or required - set(cfg) or set(cfg) - required - set(defaults):
            parser.error(f'PPO config requires {sorted(required)}; optional keys: {sorted(defaults)}')
        cfg = dict(defaults, **cfg)
        timestamp = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f')
        for key in ('model_config', 'checkpoint', 'data_dir', 'output_dir'):
            value = Path(os.path.expandvars(str(cfg[key]).replace('{timestamp}', timestamp))).expanduser()
            cfg[key] = str((value if value.is_absolute() else config_path.parent / value).resolve())
        for key in ('model_config', 'checkpoint'):
            if not Path(cfg[key]).is_file():
                parser.error(f'File not found: {cfg[key]}')
        model_cfg = yaml.safe_load(Path(cfg['model_config']).read_text())['model']
    if not Path(cfg['data_dir']).is_dir():
        parser.error(f'Data directory not found: {cfg["data_dir"]}')
    for key in ('samples_per_circuit', 'max_length', 'ppo_epochs', 'minibatch_size',
                'rollout_batch_size', 'data_epochs', 'save_every_batches'):
        if type(cfg[key]) is not int or cfg[key] < (2 if key == 'samples_per_circuit' else 1):
            parser.error(f'Invalid {key}')
    if cfg['num_circuits'] is not None and (type(cfg['num_circuits']) is not int or cfg['num_circuits'] < 1):
        parser.error('num_circuits must be null (all circuits) or a positive integer')
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2 ** 32:
        parser.error('seed must be an integer in [0, 2**32)')
    for key in ('temperature', 'learning_rate', 'max_grad_norm', 'failure_penalty', 'clip_ratio'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            parser.error(f'Invalid {key}')
    if cfg['clip_ratio'] >= 1 or type(cfg['mixed_precision']) is not bool:
        parser.error('Invalid clip_ratio or mixed_precision')
    if cfg['max_length'] > model_cfg.get('max_seq_length', 200):
        parser.error('max_length must not exceed model max_seq_length')
    print(f"Checkpoint: {cfg['checkpoint']}\nData: {cfg['data_dir']}\nOutput: {cfg['output_dir']}\n"
          f"Circuits: {cfg['num_circuits'] if cfg['num_circuits'] is not None else 'ALL'}; "
          f"data epochs: {cfg['data_epochs']}; rollout batch: {cfg['rollout_batch_size']}; "
          f"samples per circuit: {cfg['samples_per_circuit']}", flush=True)
    if args.dry_run:
        return
    from circuit_transformer.ppo_training import train_ppo
    train_ppo(cfg, model_cfg, resume)


if __name__ == '__main__':
    main()
