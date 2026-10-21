"""Regressions for public Transformer calls outside the circuit search wrapper."""
import pytest
import torch

from circuit_transformer.pytorch_transformer import (
    Float32LayerNorm, KerasAdam, Seq2SeqTransformer,
)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('length', [1, 3])
def test_action_masks_preserve_batch_and_sequence_dimensions(cached, length):
    torch.manual_seed(7)
    model = Seq2SeqTransformer(
        enc_vocab_size=8, dec_vocab_size=8, embedding_width=8, num_layers=1,
        num_attention_heads=2, intermediate_size=16, max_tree_depth=None,
    ).eval()
    mask = torch.ones(2, length, 8, dtype=torch.bool)
    # Distinct masks for each example and position catch accidental broadcasting.
    for batch in range(2):
        for step in range(length):
            mask[batch, step, batch + step + 2] = False
    inputs = {
        'inputs': torch.tensor([[2, 3], [4, 5]]),
        'targets': torch.tensor([[2, 3, 1], [4, 5, 1]])[:, :length],
        'dec_action_mask': mask,
    }
    with torch.no_grad():
        full = model(inputs, return_kv_cache=cached)
        last = model(inputs, return_kv_cache=cached, return_last_token=True)
    assert full.shape == (2, length, 8)
    assert last.shape == (2, 8)
    torch.testing.assert_close(last, full[:, -1])
    assert torch.all(full[~mask] == torch.finfo(torch.float32).min)
    assert torch.all(full[mask] > torch.finfo(torch.float32).min)


@pytest.mark.parametrize('affine,bias', [(True, True), (True, False), (False, False)])
def test_float32_layer_norm_accepts_base_class_options(affine, bias):
    layer = Float32LayerNorm(4, elementwise_affine=affine, bias=bias)
    inputs = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float16)
    result = layer(input=inputs)
    expected = torch.nn.functional.layer_norm(
        inputs.float(), (4,), layer.weight, layer.bias, layer.eps,
    )
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('tensor_loss', [False, True])
def test_keras_adam_closure_returns_loss_and_updates_parameter(tensor_loss):
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = KerasAdam([parameter], lr=0.1)
    losses = []

    def closure():
        assert torch.is_grad_enabled()
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        losses.append(loss if tensor_loss else loss.item())
        return losses[-1]

    result = optimizer.step(closure)
    assert len(losses) == 1
    assert result is losses[0]
    assert parameter.item() < 1.
    assert optimizer.step() is None
