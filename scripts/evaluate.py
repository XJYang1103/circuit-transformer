#!/usr/bin/env python3
"""Evaluate JSON circuit datasets using a trained native checkpoint."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-config', type=Path, default=PROJECT_ROOT / 'configs/evaluate.yaml',
                        help='Evaluation YAML (default: configs/evaluate.yaml)')
    parser.add_argument('--config', type=Path, help='Training resolved_config.yaml')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--limit', type=int, help='Evaluate only the first N files (smoke test)')
    parser.add_argument('--mcts-steps', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    import yaml
    args.eval_config = args.eval_config.expanduser().resolve()
    if not args.eval_config.is_file():
        parser.error(f'Evaluation config not found: {args.eval_config}')
    settings = yaml.safe_load(args.eval_config.read_text())
    allowed = {'config', 'checkpoint', 'data_dir', 'output_dir', 'device',
               'batch_size', 'limit', 'mcts_steps'}
    if not isinstance(settings, dict) or set(settings) - allowed:
        parser.error(f'Evaluation YAML must be a mapping with keys: {sorted(allowed)}')
    defaults = dict(device='cuda', batch_size=128, limit=None, mcts_steps=0)
    timestamp = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f')
    for key in allowed:
        cli_value = getattr(args, key)
        value = cli_value if cli_value is not None else settings.get(key, defaults.get(key))
        if key in ('config', 'checkpoint', 'data_dir', 'output_dir'):
            if not isinstance(value, (str, Path)) or not str(value).strip():
                parser.error(f'{key} must specify a path')
            value = Path(os.path.expandvars(str(value)).replace('{timestamp}', timestamp)).expanduser()
            # YAML paths are relative to the YAML file; CLI paths retain cwd semantics.
            if not value.is_absolute() and cli_value is None:
                value = args.eval_config.parent / value
            value = value.resolve()
        setattr(args, key, value)
    for key in ('batch_size', 'mcts_steps', 'limit'):
        value = getattr(args, key)
        if key == 'limit' and value is None:
            continue
        if type(value) is not int:
            parser.error(f'{key} must be an integer')
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1) or args.mcts_steps < 0:
        parser.error('batch-size/limit must be positive; mcts-steps must be nonnegative')
    for path in (args.config, args.checkpoint):
        if not path.is_file():
            parser.error(f'File not found: {path}')
    files = sorted(args.data_dir.rglob('*.json'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error(f'No JSON files in {args.data_dir}')
    config = yaml.safe_load(args.config.read_text())
    model_args = dict(config['model'])
    model_args.update(ckpt_path=str(args.checkpoint.resolve()), device=args.device,
                      inference_batch_size=args.batch_size, verbose=0)
    print(f'Checkpoint: {args.checkpoint}\nSamples: {len(files)}\nOutput: {args.output_dir}', flush=True)
    if args.dry_run:
        return
    # Refuse to overwrite previous evaluation artifacts.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import circuit_transformer as ct
    model = ct.CircuitTransformer(**model_args)
    started = time.perf_counter()
    groups = {}
    with (args.output_dir / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['file', 'subset', 'original_ands',
                                'reference_ands', 'optimized_ands', 'equivalent'])
        writer.writeheader()
        for start in range(0, len(files), args.batch_size):
            batch = files[start:start + args.batch_size]
            records = [json.loads(path.read_text()) for path in batch]
            parsed = [ct.read_aiger(aiger_str=record[0]) for record in records]
            for path, (_, info) in zip(batch, parsed):
                if info[1] > model.num_inputs or info[2] != 0 or info[3] not in (1, 2):
                    raise ValueError(f'Unsupported input/latch/output counts: {path}: {info}')
            aigs = [aig for aig, _ in parsed]
            outputs = model.optimize(aigs, num_mcts_steps=args.mcts_steps)
            if len(outputs) != len(batch):
                raise RuntimeError('Output count differs from input count')
            for path, record, original, optimized in zip(batch, records, aigs, outputs):
                subset = str(path.parent.relative_to(args.data_dir))
                before, after = ct.count_num_ands(original), ct.count_num_ands(optimized)
                equivalent = ct.compute_tts(original, num_inputs=model.num_inputs) == ct.compute_tts(optimized, num_inputs=model.num_inputs)
                writer.writerow(dict(file=str(path.relative_to(args.data_dir)), subset=subset,
                                     original_ands=before, reference_ands=record[3],
                                     optimized_ands=after, equivalent=equivalent))
                stats = groups.setdefault(subset, dict(samples=0, equivalent=0, original_ands=0,
                                                      optimized_ands=0, reference_ands=0))
                for key, value in dict(samples=1, equivalent=int(equivalent), original_ands=before,
                                       optimized_ands=after, reference_ands=record[3]).items():
                    stats[key] += value
            handle.flush()
            print(f'{min(start + args.batch_size, len(files))}/{len(files)} evaluated', flush=True)
    for stats in groups.values():
        stats['equivalence_rate'] = stats['equivalent'] / stats['samples']
        stats['and_reduction_rate'] = (1 - stats['optimized_ands'] / stats['original_ands']) if stats['original_ands'] else None
    summary = dict(checkpoint=str(args.checkpoint.resolve()), config=str(args.config.resolve()),
                   device=args.device, batch_size=args.batch_size, mcts_steps=args.mcts_steps,
                   limit=args.limit, elapsed_seconds=time.perf_counter() - started, subsets=groups)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if any(stats['equivalent'] != stats['samples'] for stats in groups.values()):
        raise SystemExit('Equivalence check failed; inspect results.csv')


if __name__ == '__main__':
    main()
