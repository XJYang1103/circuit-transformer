#!/usr/bin/env python3
"""双卡 PPO 启动入口；原 scripts/train_ppo.py 仍为单卡入口。

python scripts/train_ppo_dual_gpu.py 自动启动两个进程。
也可由 torchrun --standalone --nproc_per_node=2 启动本文件。
"""
import argparse
from datetime import datetime, timedelta
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_settings(config_path, resume_path=None):
    import yaml
    if resume_path is not None:
        import torch
        state = torch.load(resume_path, map_location='cpu', weights_only=True)
        if state.get('format') != 'circuit-transformer-ppo-dual-gpu-v1' or state['world_size'] != 2:
            raise ValueError('--resume 必须指向双卡版 trainer-latest.pt')
        return state['config'], state['model_config']
    config_path = config_path.expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text())
    required = {'model_config', 'checkpoint', 'data_dir', 'output_dir', 'device', 'mixed_precision',
                'seed', 'num_circuits', 'samples_per_circuit', 'max_length', 'failure_penalty',
                'temperature', 'ppo_epochs', 'minibatch_size', 'learning_rate', 'clip_ratio',
                'max_grad_norm', 'rollout_batch_size', 'data_epochs', 'save_every_batches', 'world_size'}
    defaults = dict(sampling_batch_size=32, train_microbatch_size=16, log_every_updates=100,
                    log_every_batches=10, log_every_checkpoints=10)
    if not isinstance(cfg, dict) or required - set(cfg) or set(cfg) - required - set(defaults):
        raise ValueError(f'双卡配置必需字段：{sorted(required)}；可选字段：{sorted(defaults)}')
    cfg = dict(defaults, **cfg)
    timestamp = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f')
    for key in ('model_config', 'checkpoint', 'data_dir', 'output_dir'):
        value = Path(os.path.expandvars(str(cfg[key]).replace('{timestamp}', timestamp))).expanduser()
        cfg[key] = str((value if value.is_absolute() else config_path.parent / value).resolve())
    for key in ('model_config', 'checkpoint'):
        if not Path(cfg[key]).is_file():
            raise ValueError(f'File not found: {cfg[key]}')
    if not Path(cfg['data_dir']).is_dir():
        raise ValueError(f'Data directory not found: {cfg["data_dir"]}')
    for key in ('samples_per_circuit', 'max_length', 'ppo_epochs', 'minibatch_size',
                'rollout_batch_size', 'data_epochs', 'save_every_batches', 'sampling_batch_size',
                'train_microbatch_size', 'log_every_updates', 'log_every_batches', 'log_every_checkpoints'):
        if type(cfg[key]) is not int or cfg[key] < (2 if key == 'samples_per_circuit' else 1):
            raise ValueError(f'Invalid {key}')
    if cfg['num_circuits'] is not None and (type(cfg['num_circuits']) is not int or cfg['num_circuits'] < 1):
        raise ValueError('num_circuits 必须是 null 或正整数')
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2 ** 32:
        raise ValueError('Invalid seed')
    for key in ('temperature', 'learning_rate', 'max_grad_norm', 'failure_penalty', 'clip_ratio'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if type(cfg['world_size']) is not int or cfg['world_size'] != 2 or cfg['device'] != 'cuda':
        raise ValueError('双卡版使用 world_size: 2 和 device: cuda；进程分别绑定两张可见 GPU')
    if type(cfg['mixed_precision']) is not bool:
        raise ValueError('mixed_precision 必须为 true 或 false')
    if cfg['clip_ratio'] >= 1 or cfg['minibatch_size'] % 2 or cfg['minibatch_size'] < 2:
        raise ValueError('clip_ratio 必须小于 1；minibatch_size 为两卡合计轨迹数，必须为正偶数')
    model_cfg = yaml.safe_load(Path(cfg['model_config']).read_text())['model']
    if cfg['max_length'] > model_cfg.get('max_seq_length', 200):
        raise ValueError('max_length exceeds model max_seq_length')
    return cfg, model_cfg


def show_plan(cfg, resume_path):
    print(f"[双卡 PPO] GPU 数量: 2 (DDP)\nCheckpoint: {cfg['checkpoint']}\n"
          f"Data: {cfg['data_dir']}\nOutput: {cfg['output_dir']}\n"
          f"Circuits: {cfg['num_circuits'] if cfg['num_circuits'] is not None else 'ALL'}; "
          f"data epochs: {cfg['data_epochs']}\n"
          f"两卡合计 rollout_batch_size: {cfg['rollout_batch_size']}; "
          f"两卡合计 minibatch_size: {cfg['minibatch_size']}\n"
          f"每卡并行采样轨迹: {cfg.get('sampling_batch_size', 32)}; "
          f"每卡训练 microbatch: {cfg.get('train_microbatch_size', 16)}; "
          f"FP16: {cfg['mixed_precision']}", flush=True)
    if resume_path:
        print(f'Resume: {resume_path} (使用保存的双卡配置)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ppo_dual_gpu.yaml')
    parser.add_argument('--resume', type=Path, help='双卡训练状态 trainer-latest.pt')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.resume:
        args.resume = args.resume.expanduser().resolve()
    launched = 'RANK' in os.environ
    if args.dry_run or not launched:
        try:
            cfg, _ = load_settings(args.config, args.resume)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
        show_plan(cfg, args.resume)
        if args.dry_run:
            print('配置预览完成；未检查 GPU、未扫描全部数据、未启动训练。', flush=True)
            return
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            parser.error('双卡训练需要至少两张可见 CUDA GPU；请检查解释器和 CUDA_VISIBLE_DEVICES')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   str(Path(__file__).resolve()), '--config', str(args.config.expanduser().resolve())]
        if args.resume:
            command.extend(['--resume', str(args.resume)])
        raise SystemExit(subprocess.run(command).returncode)

    import torch
    import torch.distributed as dist
    if int(os.environ.get('WORLD_SIZE', 0)) != 2:
        parser.error('请使用 --nproc_per_node=2 启动双卡版')
    local_rank = int(os.environ['LOCAL_RANK'])
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        parser.error('双卡训练需要两张可见 CUDA GPU')
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=120))
    try:
        from circuit_transformer.ppo_dual_gpu import on_primary, train_dual_gpu
        cfg, model_cfg = on_primary(lambda: load_settings(args.config, args.resume))
        if dist.get_rank() == 0:
            show_plan(cfg, args.resume)
        # Verify that the collective returns the expected value before training.
        probe = torch.tensor(float(dist.get_rank() + 1), device=f'cuda:{local_rank}')
        dist.all_reduce(probe)
        if probe.item() != 3.0:
            raise RuntimeError('双卡通信检查失败：all_reduce 结果不正确')
        train_dual_gpu(cfg, model_cfg, args.resume, device=f'cuda:{local_rank}')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
