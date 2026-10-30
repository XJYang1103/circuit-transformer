import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from circuit_transformer import CircuitTransformer, Node, NodeWithInv, write_aiger
from circuit_transformer.ppo import rollout, assign_advantages


def test_ppo_round_and_sampling_probabilities(tmp_path):
    model_args = dict(num_inputs=2, embedding_width=64, num_layers=1,
                      num_attention_heads=2, intermediate_size=32, max_tree_depth=32,
                      max_seq_length=12, mixed_precision=False, device='cpu', verbose=0)
    model = CircuitTransformer(**model_args)
    ckpt = tmp_path / 'initial.pt'
    model.save(ckpt)
    x, y = Node(0, None, None), Node(1, None, None)
    root = Node(None, NodeWithInv(x, False), NodeWithInv(y, False))
    aig = write_aiger([NodeWithInv(root, False)])
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'one.json').write_text(json.dumps([aig, 1, aig, 1]))
    trajectory = rollout(model, data / 'one.json', 12, 1.0, 10)
    tokens = torch.tensor([trajectory['tokens']])
    inputs = dict(trajectory['source'], targets=tokens,
                  dec_pos_encoding=np.stack(trajectory['positions'])[None],
                  dec_action_mask=np.stack(trajectory['masks'])[None])
    with torch.no_grad():
        logits, _ = model._transformer(inputs)
        logp = torch.distributions.Categorical(logits=logits).log_prob(tokens)[0]
    np.testing.assert_allclose(logp.numpy(), trajectory['old_logps'], atol=2e-5)
    group = [dict(reward=1.), dict(reward=3.)]
    assign_advantages(group)
    assert [x['advantage'] for x in group] == [-2., 2.]
    config = tmp_path / 'model.yaml'
    config.write_text(yaml.safe_dump({'model': model_args}))
    output = tmp_path / 'output'
    ppo_config = tmp_path / 'ppo.yaml'
    ppo_config.write_text(yaml.safe_dump(dict(model_config=str(config), checkpoint=str(ckpt),
        data_dir=str(data), output_dir=str(output), device='cpu', mixed_precision=False,
        seed=42, num_circuits=1, samples_per_circuit=2, max_length=12,
        failure_penalty=10., temperature=1., ppo_epochs=1, minibatch_size=2,
        learning_rate=1e-5, clip_ratio=0.2, max_grad_norm=1.)))
    script = Path(__file__).resolve().parents[1] / 'scripts/train_ppo.py'
    subprocess.run([sys.executable, str(script), '--config', str(ppo_config)], check=True)
    assert (output / 'model-ppo.pt').is_file()
    assert len((output / 'rollouts.jsonl').read_text().splitlines()) == 2
    assert len(json.loads((output / 'metrics.json').read_text())) == 1
    CircuitTransformer(**dict(model_args, ckpt_path=str(output / 'model-ppo.pt')))
