"""Strict, name-based conversion of TF object checkpoints to PyTorch weights.

TensorFlow is imported only when a legacy checkpoint is explicitly loaded.
Conversion never guesses a parameter by shape or silently skips a model weight.
"""
import os
from pathlib import Path

import numpy as np
import torch


FORMAT = 'circuit-transformer-pytorch-v1'


def parameter_mapping(model):
    """Yield (PyTorch name, TF object path, TF-to-Torch array transform)."""
    identity = lambda a: a
    transpose = lambda a: a.T
    qkv = lambda a: a.reshape(a.shape[0], -1).T
    out = lambda a: a.reshape(-1, a.shape[-1]).T
    for side in ('enc', 'dec'):
        yield f'{side}_embedding_lookup.weight', f'{side}_embedding_lookup/embeddings', identity
    if model.tree_position_embedding is not None:
        yield 'tree_position_embedding.p', 'tree_position_embedding/p', identity
    if model.add_action_mask_to_inputs:
        for side in ('enc', 'dec'):
            yield f'{side}_candidate_ratio', f'{side}_candidate_ratio', identity
    for side, stack in (('encoder', model.encoder_layer), ('decoder', model.decoder_layer)):
        root = f'{side}_layer'
        for i in range(stack.num_layers):
            pt = f'{root}.layers.{i}'
            tf = f'{root}/{side}_layers/{i}'
            attentions = [('self_attention', '_attention_layer' if side == 'encoder' else 'self_attention')]
            if side == 'decoder':
                attentions.append(('cross_attention', 'encdec_attention'))
            for pt_att, tf_att in attentions:
                for projection in ('query', 'key', 'value', 'output'):
                    yield (f'{pt}.{pt_att}.{projection}.weight',
                           f'{tf}/{tf_att}/_{projection}_dense/kernel', out if projection == 'output' else qkv)
            for pt_name, tf_name in (
                    ('intermediate', '_intermediate_dense' if side == 'encoder' else 'intermediate_dense'),
                    ('output', '_output_dense' if side == 'encoder' else 'output_dense')):
                yield f'{pt}.{pt_name}.weight', f'{tf}/{tf_name}/kernel', transpose
                yield f'{pt}.{pt_name}.bias', f'{tf}/{tf_name}/bias', identity
            norms = [('self_norm', '_attention_layer_norm' if side == 'encoder' else 'self_attention_layer_norm'),
                     ('output_norm', '_output_layer_norm' if side == 'encoder' else 'output_layer_norm')]
            if side == 'decoder':
                norms.append(('cross_norm', 'encdec_attention_layer_norm'))
            for pt_name, tf_name in norms:
                for pt_par, tf_par in (('weight', 'gamma'), ('bias', 'beta')):
                    yield f'{pt}.{pt_name}.{pt_par}', f'{tf}/{tf_name}/{tf_par}', identity
        for pt_par, tf_par in (('weight', 'gamma'), ('bias', 'beta')):
            yield f'{root}.output_normalization.{pt_par}', f'{root}/output_normalization/{tf_par}', identity


def save_checkpoint(model, path):
    path = Path(path)
    payload = dict(format=FORMAT, config=model.config,
                   state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()})
    # Same-directory replacement avoids leaving a partial checkpoint on failure.
    import tempfile
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(model, path):
    path = os.fspath(path)
    if path.endswith('.index'):
        path = path[:-6]
    if os.path.isfile(path + '.index'):
        return load_tensorflow_checkpoint(model, path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Expected a Circuit Transformer PyTorch checkpoint saved with save().')
    if payload['config'] != model.config:
        raise ValueError(f'Checkpoint configuration mismatch: {payload["config"]} != {model.config}')
    state = payload['state_dict']
    validate_state(model, state)
    model.load_state_dict(state, strict=True)


def validate_state(model, state):
    expected = model.state_dict()
    if expected.keys() != state.keys():
        raise ValueError(f'Checkpoint parameter mismatch; missing={expected.keys() - state.keys()}, '
                         f'unexpected={state.keys() - expected.keys()}')
    for name, value in state.items():
        if value.shape != expected[name].shape:
            raise ValueError(f'Checkpoint shape mismatch for {name}: {value.shape} != {expected[name].shape}')
        if not torch.isfinite(value).all():
            raise ValueError(f'Non-finite checkpoint parameter: {name}')


def load_tensorflow_checkpoint(model, prefix):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError('Legacy checkpoint conversion requires TensorFlow once. Install '
                          'tensorflow in a conversion environment, load(prefix), then save("model.pt"). '
                          'Native .pt inference and training do not require TensorFlow.') from exc
    reader = tf.train.load_checkpoint(prefix)
    variables = reader.get_variable_to_shape_map()
    state, used = {}, set()
    for pt_name, tf_path, transform in parameter_mapping(model):
        suffix = tf_path + '/.ATTRIBUTES/VARIABLE_VALUE'
        matches = [name for name in variables if name == suffix or name.endswith('/' + suffix)]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one TF variable for {pt_name}: {suffix}; found {matches}')
        name = matches[0]
        state[pt_name] = torch.from_numpy(np.array(transform(reader.get_tensor(name)), copy=True))
        used.add(name)
    unconverted = [name for name in variables if name not in used and name != '_CHECKPOINTABLE_OBJECT_GRAPH'
                   and not name.startswith(('optimizer/', 'save_counter/'))]
    if unconverted:
        raise ValueError(f'Unconverted TensorFlow variables: {unconverted}')
    validate_state(model, state)
    model.load_state_dict(state, strict=True)
