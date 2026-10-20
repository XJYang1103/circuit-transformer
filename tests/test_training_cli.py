"""Exercise the merged YAML training entry point with a small real circuit."""
import json
from pathlib import Path
import subprocess
import sys

import yaml

from circuit_transformer import Node, NodeWithInv, write_aiger


def test_yaml_cli_dry_run_and_train(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    x, y = Node(0, None, None), Node(1, None, None)
    root = Node(None, NodeWithInv(x, True), NodeWithInv(y, False))
    aig = write_aiger([NodeWithInv(root, False), NodeWithInv(x, False)])
    for index in range(4):
        (data / f'{index}.json').write_text(json.dumps([aig, 1, aig, 1]))

    output = tmp_path / 'runs'
    config = tmp_path / 'train.yaml'
    config.write_text(yaml.safe_dump({
        'run': {'name': 'cli-smoke'},
        'paths': {'data_dir': str(data), 'output_root': str(output)},
        'model': {'num_inputs': 2, 'embedding_width': 64, 'num_layers': 2,
                  'num_attention_heads': 2, 'intermediate_size': 24,
                  'max_tree_depth': 32, 'max_seq_length': 12,
                  'mixed_precision': False, 'device': 'cuda'},
        'train': {'epochs': 1, 'batch_size': 2, 'validation_split': 0.25,
                  'num_workers': 8},
    }))
    script = Path(__file__).parents[1] / 'scripts' / 'train.py'
    command = [sys.executable, str(script), '--config', str(config),
               '--device', 'cpu', '--num-workers', '0']
    planned = subprocess.run(command + ['--dry-run'], capture_output=True,
                             text=True, check=True)
    assert 'Device/workers: cpu / 0' in planned.stdout
    assert not output.exists()

    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    run_dir, = output.glob('*-cli-smoke')
    checkpoint = run_dir / 'ckpt' / 'model-0001.pt'
    assert checkpoint.is_file()
    assert 'Epoch 1/1:' in (run_dir / 'train.log').read_text()
    assert 'model-XXXX.pt' in completed.stdout
