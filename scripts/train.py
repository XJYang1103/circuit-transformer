#!/usr/bin/env python3
"""Train the PyTorch Circuit Transformer with a YAML config on Linux.

This CLI expands and validates paths, creates one directory per run, and
passes the configured model and training options to CircuitTransformer.
The model and its training loop remain in the circuit_transformer package.

Layout produced and consumed (see readme.md for the project conventions):

    ~/data/circuit-transformer/train/<dataset>/     .json training files
    ~/runs/circuit-transformer/<timestamp>-<name>/
        ckpt/           model-<epoch>.pt
        tensorboard/    PyTorch profiler traces when `train.profile` is true
        train.log       stdout/stderr of this run

Usage:

    python scripts/train.py --config configs/train_deepsyn.yaml
    python scripts/train.py --config configs/train_deepsyn.yaml --dry-run
    python scripts/train.py --config configs/train_deepsyn.yaml --epochs 1
"""

import argparse
import datetime
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
            "Fetch the dataset first:\n"
            "  scripts/download_data.sh\n"
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

    if args.dry_run:
        print("\n--dry-run: plan resolved, nothing was trained.")
        return 0

    # Keep the run directory ready for both checkpoints and optional traces.
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

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

    if args.resume:
        model_cfg['ckpt_path'] = resolve(args.resume)

    model = ct.CircuitTransformer(**model_cfg)

    # Keep optional PyTorch profiler traces with this run.
    train_cfg['log_dir'] = log_dir

    model.train(train_data_dir=data_dir,
                ckpt_save_path=ckpt_dir,
                **train_cfg)

    print("\nDone. Checkpoints in %s" % ckpt_dir)
    print("Load one with: CircuitTransformer(ckpt_path='%s/model-XXXX.pt')"
          % ckpt_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
