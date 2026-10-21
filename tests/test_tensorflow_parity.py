"""Real cross-framework comparisons, optional only when TF is not installed."""
import numpy as np
import random
import subprocess
import sys
import pytest
import torch

tf = pytest.importorskip('tensorflow')
tf_keras = pytest.importorskip('tf_keras')
pytest.importorskip('official')
from reference.loader import load_reference, load_reference_circuit
from test_pytorch_migration import features, small_circuit, aig_pair, SearchOptions
from circuit_transformer.checkpoints import parameter_mapping, load_tensorflow_checkpoint
from circuit_transformer.pytorch_transformer import masked_loss, masked_accuracy, CustomSchedule, KerasAdam


def pair(tmp_path, candidate=False, mixed=False):
    tf_keras.mixed_precision.set_global_policy('mixed_float16' if mixed else 'float32')
    # tf_keras 2.18's global seed generator passes float 1e9 to randint,
    # which Python 3.12 rejects. Seed the RNGs directly instead.
    random.seed(17)
    np.random.seed(17)
    tf.random.set_seed(17)
    reference = load_reference()
    from official.nlp.modeling.models.seq2seq_transformer import TransformerEncoder, TransformerDecoder
    old = reference.Seq2SeqTransformer(enc_vocab_size=8, dec_vocab_size=8,
                                       embedding_width=64, max_tree_depth=32, add_action_mask_to_inputs=candidate,
                                       encoder_layer=TransformerEncoder(num_layers=2, num_attention_heads=2, intermediate_size=24),
                                       decoder_layer=TransformerDecoder(num_layers=2, num_attention_heads=2, intermediate_size=24))
    old(features(), training=False)
    prefix = str(tmp_path / 'reference')
    old.save_weights(prefix)
    new = small_circuit(add_action_mask_to_input=candidate)._transformer
    new.mixed_precision = mixed
    load_tensorflow_checkpoint(new, prefix)
    old.return_cache = new.return_cache = True
    new.eval()
    return old, new, prefix, reference


def test_checkpoint_conversion_cli(tmp_path):
    _, original, prefix, _ = pair(tmp_path)
    destination = tmp_path / 'converted.pt'
    result = subprocess.run([
        sys.executable, '-m', 'circuit_transformer.convert_checkpoint', prefix + '.index',
        str(destination), '--num-inputs', '2', '--embedding-width', '64',
        '--num-layers', '2', '--num-attention-heads', '2', '--intermediate-size', '24',
    ], check=True, capture_output=True, text=True, timeout=30)
    assert 'Converted all' in result.stdout
    restored = small_circuit(ckpt_path=destination)._transformer
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize('candidate', [False, True])
@pytest.mark.parametrize('mixed', [False, True])
def test_logits_loss_cache_and_gradients(tmp_path, candidate, mixed):
    old, new, prefix, reference = pair(tmp_path, candidate, mixed)
    x = features()
    tolerance = 0.035 if mixed else 2e-5
    expected, tc = old(x, training=False)
    actual, pc = new(x)
    legal = x['dec_action_mask']
    error = np.abs(expected.numpy()[legal] - actual.detach().numpy()[legal]).max()
    print(f'candidate={candidate} mixed={mixed} max_logit_abs_error={error:.9g}')
    np.testing.assert_allclose(actual.detach().numpy()[legal], expected.numpy()[legal], atol=tolerance, rtol=tolerance)
    np.testing.assert_array_equal(actual.argmax(-1).detach(), tf.argmax(expected, -1).numpy())
    np.testing.assert_allclose(pc['encoder_outputs'].detach(), tc['encoder_outputs'].numpy(), atol=tolerance, rtol=tolerance)
    assert torch.equal(actual[~torch.from_numpy(legal)], torch.full_like(actual[~torch.from_numpy(legal)], torch.finfo(torch.float32).min))
    # Incremental decoding must match teacher forcing with shifted tokens/positions.
    tc = pc = None
    for step in range(x['targets'].shape[1]):
        item = dict(x)
        item['targets'] = np.zeros((2, 1), np.int32) if step == 0 else x['targets'][:, step-1:step]
        item['dec_pos_encoding'] = np.zeros((2, 1, 64), np.float32) if step == 0 else x['dec_pos_encoding'][:, step-1:step]
        item['dec_action_mask'] = x['dec_action_mask'][:, step:step+1]
        item['cache'] = tc
        tl, tc = old(item, training=False, return_kv_cache=True, return_last_token=True)
        item['cache'] = pc
        pl, pc = new(item, return_kv_cache=True, return_last_token=True)
        np.testing.assert_allclose(pl.detach(), tl.numpy(), atol=tolerance, rtol=tolerance)
        np.testing.assert_allclose(pl.detach(), actual[:, step].detach(), atol=tolerance, rtol=tolerance)
        for layer in pc['kv_cache']:
            for name in ('key', 'value'):
                np.testing.assert_allclose(pc['kv_cache'][layer][name].detach(),
                                           tc['kv_cache'][layer][name].numpy(), atol=tolerance, rtol=tolerance)
    old.return_cache = new.return_cache = False
    with tf.GradientTape() as tape:
        expected = old(x, training=True)
        tf_loss = reference.masked_loss(x['targets'], expected)
    tf_grads = tape.gradient(tf_loss, old.trainable_variables)
    actual = new(x)
    loss = masked_loss(x['targets'], actual)
    np.testing.assert_allclose(loss.detach(), tf_loss.numpy(), atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(masked_accuracy(x['targets'], actual).detach(), reference.masked_accuracy(x['targets'], expected).numpy())
    loss.backward()
    # Resolve TF object paths on the built model, independent of checkpoint reader.
    for pt_name, path, transform in parameter_mapping(new):
        obj = old
        for component in path.split('/'):
            obj = obj[int(component)] if component.isdigit() else getattr(obj, component)
        index = next(i for i, variable in enumerate(old.trainable_variables) if variable is obj)
        gradient = tf.convert_to_tensor(tf_grads[index]).numpy()
        actual_gradient = dict(new.named_parameters())[pt_name].grad
        assert actual_gradient is not None
        np.testing.assert_allclose(actual_gradient, transform(gradient),
                                   atol=tolerance, rtol=tolerance, err_msg=pt_name)


def test_adam_and_schedule_match_tensorflow():
    reference = load_reference()
    x = tf.Variable([0.2, -0.5, 1.2], dtype=tf.float32)
    y = torch.nn.Parameter(torch.tensor([0.2, -0.5, 1.2]))
    old = tf_keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    new = KerasAdam([y], lr=0.01)
    for grad in ([1e-10, 0.2, -0.3], [0., 0., 0.], [0.2, -0.1, 0.4]):
        old.apply_gradients([(tf.constant(grad), x)])
        y.grad = torch.tensor(grad)
        new.step()
        np.testing.assert_allclose(y.detach(), x.numpy(), atol=2e-7, rtol=2e-6)
    for step in (0, 1, 100, 3999, 4000, 8000):
        np.testing.assert_allclose(CustomSchedule(512)(step), reference.CustomSchedule(512)(step).numpy(), rtol=2e-7)


@pytest.mark.parametrize('candidate', [False, True])
def test_three_training_updates_same_batch(tmp_path, candidate):
    old, new, _, reference = pair(tmp_path, candidate)
    old.return_cache = new.return_cache = False
    tf_optimizer = tf_keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    pt_optimizer = KerasAdam(new.parameters(), lr=0.001)
    x = features()
    for step in range(3):
        with tf.GradientTape() as tape:
            loss = reference.masked_loss(x['targets'], old(x, training=True))
        gradients = tape.gradient(loss, old.trainable_variables)
        tf_optimizer.apply_gradients(zip(gradients, old.trainable_variables))
        pt_optimizer.zero_grad()
        pt_loss = masked_loss(x['targets'], new(x))
        pt_loss.backward()
        pt_optimizer.step()
        np.testing.assert_allclose(pt_loss.detach(), loss.numpy(), atol=2e-5, rtol=2e-5)
        for name, path, transform in parameter_mapping(new):
            value = old
            for component in path.split('/'):
                value = value[int(component)] if component.isdigit() else getattr(value, component)
            np.testing.assert_allclose(dict(new.named_parameters())[name].detach(),
                                       transform(value.numpy()), atol=2e-5, rtol=3e-5,
                                       err_msg=f'step={step} {name}')


def test_loss_scaling_overflow_iteration_semantics():
    variable = tf.Variable([1., 2.])
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    old = tf_keras.mixed_precision.LossScaleOptimizer(
        tf_keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9))
    new = KerasAdam([parameter], lr=0.001)
    scaler = torch.amp.GradScaler('cpu', init_scale=32768, growth_interval=2000)
    for gradient in ([float('inf'), 1.], [0.3, 0.5], [float('inf'), 1.], [0.1, -0.2]):
        old.apply_gradients([(tf.constant(gradient), variable)])
        new.zero_grad()
        scaler.scale((parameter * torch.tensor(gradient)).sum()).backward()
        old_scale = scaler.get_scale()
        scaler.step(new)
        scaler.update()
        if scaler.get_scale() < old_scale:
            new.skipped_step()
        np.testing.assert_allclose(parameter.detach(), variable.numpy(), atol=2e-7)
        assert new.param_groups[0]['iterations'] == int(old.iterations.numpy())
        assert scaler.get_scale() == float(old.loss_scale.numpy())


def test_low_level_beam_search(tmp_path):
    old, new, _, _ = pair(tmp_path)
    old._decode_max_length = new.decode_max_length = 7
    x = features()
    del x['targets']
    expected = old(x, training=False)
    actual = new(x)
    np.testing.assert_array_equal(actual['outputs'], expected['outputs'].numpy())
    np.testing.assert_allclose(actual['scores'], expected['scores'].numpy(), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('mcts', [0, 1])
def test_end_to_end_same_decode_and_rewards(tmp_path, mcts):
    _, _, prefix, _ = pair(tmp_path)
    reference = load_reference_circuit()
    old = reference.CircuitTransformer(num_inputs=2, embedding_width=64, num_layers=2,
                                       num_attention_heads=2, intermediate_size=24, max_tree_depth=32, max_seq_length=12,
                                       mixed_precision=False, ckpt_path=prefix)
    new = small_circuit(ckpt_path=prefix)
    options = SearchOptions(max_inference_seq_length=12, num_mcts_steps=mcts,
                            num_leaf_parallelization=2, num_mcts_playouts_per_step=2, return_envs=True)
    aigs = [aig_pair(), aig_pair()]
    before = old.optimize_batch(aigs, **options)
    after = new.optimize_batch(aigs, **options)
    for a, b in zip(before, after):
        assert a.tokens == b.tokens
        assert a.rewards == b.rewards
        assert a.success == b.success
        for ma, mb in zip(a.action_masks, b.action_masks):
            np.testing.assert_array_equal(ma, mb)
