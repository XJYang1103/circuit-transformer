#!/usr/bin/env python3
"""在 Linux 上使用 YAML 配置训练 PyTorch 版 Circuit Transformer。

本命令行工具会展开并检查路径，为每次运行创建独立目录，
并将配置中的模型参数和训练参数传给 CircuitTransformer。
模型实现和训练循环位于 circuit_transformer 包中。

输入和输出的目录结构（项目约定详见 readme.md)

    ~/data/circuit-transformer/train/<数据集>/     .json 训练数据文件
    ~/runs/circuit-transformer/<时间戳>-<名称>/
        ckpt/                  模型权重:model-<轮次>.pt
        tensorboard/           train.profile 为 true 时生成的 PyTorch 性能分析文件
        resolved_config.yaml   本次运行实际使用的配置，包含命令行覆盖的参数
        train.log              本次运行的标准输出和标准错误日志

使用方法：

    # 使用指定配置训练
    python scripts/train.py --config configs/train_deepsyn.yaml

    # 仅检查并显示运行配置，不执行训练
    python scripts/train.py --config configs/train_deepsyn.yaml --dry-run

    # 将训练总轮数覆盖为 1
    python scripts/train.py --config configs/train_deepsyn.yaml --epochs 1
"""

import argparse
import datetime
import inspect
import os
import sys
from pathlib import Path


# Running `python scripts/train.py` sets sys.path to scripts/, not the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def resolve(path):
    """Expand ~ and $VARS, then make the result absolute."""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def load_config(path):
    try:
        import yaml
    except ImportError:
        sys.exit("This script needs PyYAML: pip install pyyaml")

    if not os.path.isfile(path):
        sys.exit("Config file not found: %s" % path)

    with open(path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        sys.exit("%s: expected a YAML mapping at the top level" % path)

    return config


def effective_options(call, supplied):
    """Fill in callable defaults without changing the supplied arguments."""
    bound = inspect.signature(call).bind_partial(**supplied)
    bound.apply_defaults()
    return dict(bound.arguments)


def check_data_dir(path):
    """Validate the directory the way `train()` consumes it.

    `train()` lists every directory entry and treats it as
    a training file -- it does not filter by extension. A nested subdirectory
    therefore passes `listdir` and only blows up much later, inside the data
    pipeline, where the error is hard to read. Catching it here is worth the
    few lines.
    """
    if not os.path.isdir(path):
        sys.exit(
            "Training data directory not found:\n"
            "  %s\n\n"
            "Download and extract the dataset described in readme.md,\n"
            "then point paths.data_dir at the directory that directly holds\n"
            "the .json files." % path)

    entries = sorted(os.listdir(path))
    if not entries:
        sys.exit("Training data directory is empty:\n  %s" % path)

    subdirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    if subdirs:
        # The published archive ships a single top-level directory, so
        # unzipping without stripping it leaves you exactly one level too high.
        sys.exit(
            "Training data directory contains %d subdirector%s (e.g. %r).\n"
            "train() treats every directory entry as a training file, so\n"
            "point paths.data_dir one level deeper:\n"
            "  %s"
            % (len(subdirs), "y" if len(subdirs) == 1 else "ies",
               subdirs[0], os.path.join(path, subdirs[0])))

    return len(entries)


class _Tee:
    """Mirror writes to several streams, so each run keeps its own log."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a Circuit Transformer from scratch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every flag below overrides the matching value in the config file.")
    parser.add_argument('--config', default='configs/train_deepsyn.yaml',
                        help='YAML config (default: %(default)s)')
    parser.add_argument('--data-dir', help='override paths.data_dir')
    parser.add_argument('--output-root', help='override paths.output_root')
    parser.add_argument('--run-name', help='override run.name')
    parser.add_argument('--epochs', type=int, help='override train.epochs')
    parser.add_argument('--batch-size', type=int, help='override train.batch_size')
    parser.add_argument('--device', help='override model.device (e.g. cuda or cpu)')
    parser.add_argument('--num-workers', type=int,
                        help='override train.num_workers (default 8; preserves original loader)')
    parser.add_argument('--resume', metavar='WEIGHTS_PT',
                        help='initialize model weights from a .pt file; optimizer and '
                             'learning-rate step are reset as in the original train()')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the resolved plan and exit; imports no '
                             'PyTorch, so it works before the env is set up')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    paths = config.get('paths') or {}
    run = config.get('run') or {}
    model_cfg = dict(config.get('model') or {})
    train_cfg = dict(config.get('train') or {})

    data_dir = args.data_dir or paths.get('data_dir')
    if not data_dir:
        sys.exit("%s: paths.data_dir is not set" % args.config)
    data_dir = resolve(data_dir)

    output_root = resolve(args.output_root or paths.get('output_root')
                          or '~/runs/circuit-transformer')
    run_name = args.run_name or run.get('name') or 'run'

    if args.epochs is not None:
        train_cfg['epochs'] = args.epochs
    if args.batch_size is not None:
        train_cfg['batch_size'] = args.batch_size
    if args.device is not None:
        model_cfg['device'] = args.device
    if args.num_workers is not None:
        train_cfg['num_workers'] = args.num_workers
    if args.resume is not None:
        model_cfg['ckpt_path'] = args.resume
    model_cfg['ckpt_path'] = (resolve(model_cfg['ckpt_path'])
                              if model_cfg.get('ckpt_path') else None)

    num_files = check_data_dir(data_dir)

    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(output_root, '%s-%s' % (stamp, run_name))
    ckpt_dir = os.path.join(run_dir, 'ckpt')
    log_dir = os.path.join(run_dir, 'tensorboard')

    print("Config       : %s" % os.path.abspath(args.config))
    print("Training data: %s (%d files)" % (data_dir, num_files))
    print("Run directory: %s" % run_dir)
    print("Checkpoints  : %s" % ckpt_dir)
    print("Epochs/batch : %s / %s" % (train_cfg.get('epochs'), train_cfg.get('batch_size')))
    print("Device/workers: %s / %s" % (model_cfg.get('device', 'auto'),
                                       train_cfg.get('num_workers', 8)))
    print("Initial weights: %s" % (model_cfg['ckpt_path'] or 'random initialization'))

    if args.dry_run:
        print("\n--dry-run: plan resolved, nothing was trained.")
        return 0

    # Keep the run directory ready for both checkpoints and optional traces.
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Keep optional PyTorch profiler traces with this run.
    train_cfg['log_dir'] = log_dir

    log_handle = open(os.path.join(run_dir, 'train.log'), 'a',
                      encoding='utf-8', buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_handle)
    sys.stderr = _Tee(sys.__stderr__, log_handle)

    try:
        import circuit_transformer as ct
    except ImportError as exc:
        sys.exit("Cannot import circuit_transformer: %s\n"
                 "Install the package and its dependencies first:\n"
                 "  pip install -e ." % exc)

    model = ct.CircuitTransformer(**model_cfg)

    # Record the exact options used, including defaults omitted from the YAML.
    model_options = effective_options(ct.CircuitTransformer, model_cfg)
    model_options['device'] = str(model.device)
    train_options = effective_options(model.train, dict(
        train_data_dir=data_dir, ckpt_save_path=ckpt_dir, **train_cfg))
    train_options.pop('train_data_dir')
    train_options.pop('ckpt_save_path')
    resolved_config = {
        'run': {'name': run_name},
        'paths': {'data_dir': data_dir, 'output_root': output_root},
        'model': model_options,
        'train': train_options,
    }
    import yaml
    resolved_config_path = os.path.join(run_dir, 'resolved_config.yaml')
    with open(resolved_config_path, 'w', encoding='utf-8') as config_handle:
        yaml.safe_dump(resolved_config, config_handle, sort_keys=False,
                       allow_unicode=True)
    print("Resolved config: %s" % resolved_config_path)

    model.train(train_data_dir=data_dir,
                ckpt_save_path=ckpt_dir,
                **train_cfg)

    print("\nDone. Checkpoints in %s" % ckpt_dir)
    print("Load one with: CircuitTransformer(ckpt_path='%s/model-XXXX.pt')"
          % ckpt_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
