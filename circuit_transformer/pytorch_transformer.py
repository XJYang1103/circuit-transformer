# Port of the Seq2Seq Transformer wrapper and TF Models layer computations.
# Original wrapper: Copyright 2022 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch implementation of the repository's TF Models Transformer.

This deliberately uses explicit attention rather than nn.Transformer: the
original uses pre-norm, biased feed-forward layers, bias-free attention, an
extra final normalization, and tied decoder embedding/output weights.
"""
import math
from collections.abc import Callable
from typing import TypeVar, overload

import torch
from torch import nn
from torch.nn import functional as F


_Loss = TypeVar('_Loss', float, torch.Tensor)


class TreePositionEncodingPostProcessing(nn.Module):
    def __init__(self, max_tree_depth, num_repeat):
        super().__init__()
        self._max_tree_depth = max_tree_depth
        self._num_repeat = num_repeat
        self.p = nn.Parameter(torch.full((num_repeat,), 2.0))

    def forward(self, inputs):
        p = self.p.to(inputs.dtype).tanh()
        powers = torch.arange(self._max_tree_depth, device=p.device)
        vector = p[:, None].pow(powers) * (1 - p.square()).sqrt()[:, None]
        vector = vector.flatten().repeat_interleave(2)[None, None, :]
        return inputs.repeat(1, 1, self._num_repeat) * vector * math.sqrt(
            self._max_tree_depth * self._num_repeat)


class Float32LayerNorm(nn.LayerNorm):
    def forward(self, input):
        # TF Models explicitly sets layer normalization dtype to float32.
        return F.layer_norm(input.float(), self.normalized_shape,
                            None if self.weight is None else self.weight.float(),
                            None if self.bias is None else self.bias.float(), self.eps)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        def projection():
            layer = nn.Linear(width, width, bias=False)
            nn.init.uniform_(layer.weight, -math.sqrt(3 / width), math.sqrt(3 / width))
            return layer

        self.query = projection()
        self.key = projection()
        self.value = projection()
        self.output = projection()

    def forward(self, query, memory, mask, cache=None, dtype=torch.float32):
        def project(layer, x):
            return F.linear(x.to(dtype), layer.weight.to(dtype)).reshape(
                x.shape[0], x.shape[1], self.heads, self.head_dim)
        q, k, v = (project(self.query, query), project(self.key, memory),
                   project(self.value, memory))
        if cache is not None:
            k = torch.cat((cache['key'].to(dtype), k), dim=1)
            v = torch.cat((cache['value'].to(dtype), v), dim=1)
            cache['key'], cache['value'] = k, v
        scores = torch.einsum('bshd,bthd->bhts', k, q * (self.head_dim ** -0.5))
        # Keras Softmax adds a finite mask bias (including all-masked rows).
        bias = -30000.0 if dtype == torch.float16 else -1e9
        scores = scores + (~mask[:, None].bool()).to(dtype) * bias
        probs = scores.softmax(dim=-1)
        result = torch.einsum('bhts,bshd->bthd', probs, v).flatten(2)
        return F.linear(result, self.output.weight.to(dtype))


class TransformerBlock(nn.Module):
    def __init__(self, width, heads, intermediate_size, decoder=False):
        super().__init__()
        self.self_attention = Attention(width, heads)
        self.self_norm = Float32LayerNorm(width, eps=1e-6)
        self.output_norm = Float32LayerNorm(width, eps=1e-6)
        self.intermediate = nn.Linear(width, intermediate_size)
        self.output = nn.Linear(intermediate_size, width)
        for layer in (self.intermediate, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        if decoder:
            self.cross_attention = Attention(width, heads)
            self.cross_norm = Float32LayerNorm(width, eps=1e-6)

    def forward(self, x, mask, memory=None, cross_mask=None, cache=None,
                dtype=torch.float32):
        x = x.to(dtype)
        normalized = self.self_norm(x)
        x = x + self.self_attention(normalized, normalized, mask, cache, dtype)
        if memory is not None:
            x = x + self.cross_attention(self.cross_norm(x), memory, cross_mask,
                                         dtype=dtype)
        y = F.linear(self.output_norm(x).to(dtype), self.intermediate.weight.to(dtype),
                     self.intermediate.bias.to(dtype)).relu()
        return x + F.linear(y, self.output.weight.to(dtype), self.output.bias.to(dtype))


class TransformerStack(nn.Module):
    def __init__(self, width, num_layers, num_attention_heads, intermediate_size,
                 decoder=False):
        super().__init__()
        self.num_layers, self.num_attention_heads = num_layers, num_attention_heads
        self.layers = nn.ModuleList([
            TransformerBlock(width, num_attention_heads, intermediate_size, decoder)
            for _ in range(num_layers)])
        self.output_normalization = Float32LayerNorm(width, eps=1e-6)

    def forward(self, x, mask, memory=None, cross_mask=None, cache=None,
                dtype=torch.float32):
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, memory, cross_mask,
                      None if cache is None else cache[str(i)], dtype)
        return self.output_normalization(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, enc_vocab_size=20, dec_vocab_size=20, embedding_width=512,
                 num_layers=12, num_attention_heads=8, intermediate_size=2048,
                 dropout_rate=0.0, max_tree_depth: int | None = 32,
                 add_action_mask_to_inputs=False, eos_id=1, mixed_precision=False,
                 padded_decode=False, decode_max_length=None, extra_decode_length=0,
                 beam_size=4, alpha=0.6,
                 **kwargs):
        super().__init__()
        if kwargs:
            raise TypeError('Unsupported Transformer options: ' + ', '.join(kwargs))
        for name, value in (('embedding_width', embedding_width), ('num_layers', num_layers),
                            ('num_attention_heads', num_attention_heads),
                            ('intermediate_size', intermediate_size), ('enc_vocab_size', enc_vocab_size),
                            ('dec_vocab_size', dec_vocab_size), ('beam_size', beam_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not 0 <= eos_id < dec_vocab_size:
            raise ValueError('eos_id must be in the decoder vocabulary')
        if max_tree_depth is not None and max_tree_depth < 1:
            raise ValueError('max_tree_depth must be positive or None')
        if not 0 <= dropout_rate <= 1:
            raise ValueError('dropout_rate must be in [0, 1]')
        if decode_max_length is not None and decode_max_length < 1:
            raise ValueError('decode_max_length must be positive')
        if extra_decode_length < 0:
            raise ValueError('extra_decode_length must be nonnegative')
        if embedding_width % num_attention_heads:
            raise ValueError('embedding_width must be divisible by num_attention_heads')
        if max_tree_depth is not None and embedding_width % (max_tree_depth * 2):
            raise ValueError('embedding_width must be divisible by 2 * max_tree_depth')
        self.config = dict(enc_vocab_size=enc_vocab_size, dec_vocab_size=dec_vocab_size,
                           embedding_width=embedding_width, num_layers=num_layers,
                           num_attention_heads=num_attention_heads,
                           intermediate_size=intermediate_size, dropout_rate=dropout_rate,
                           max_tree_depth=max_tree_depth,
                           add_action_mask_to_inputs=add_action_mask_to_inputs, eos_id=eos_id)
        self.width, self.heads = embedding_width, num_attention_heads
        self.max_tree_depth = max_tree_depth
        self.add_action_mask_to_inputs = add_action_mask_to_inputs
        self.mixed_precision = mixed_precision
        self.return_cache = False
        self.enc_embedding_lookup = nn.Embedding(enc_vocab_size, embedding_width)
        self.dec_embedding_lookup = nn.Embedding(dec_vocab_size, embedding_width)
        for layer in (self.enc_embedding_lookup, self.dec_embedding_lookup):
            nn.init.normal_(layer.weight, std=embedding_width ** -0.5)
        self.encoder_layer = TransformerStack(embedding_width, num_layers,
                                              num_attention_heads, intermediate_size)
        self.decoder_layer = TransformerStack(embedding_width, num_layers,
                                              num_attention_heads, intermediate_size, True)
        self.tree_position_embedding = (TreePositionEncodingPostProcessing(
            max_tree_depth, embedding_width // (2 * max_tree_depth))
            if max_tree_depth is not None else None)
        self.dropout_rate = dropout_rate
        self.eos_id, self.beam_size, self.alpha = eos_id, beam_size, alpha
        self.padded_decode = padded_decode
        self.decode_max_length, self.extra_decode_length = decode_max_length, extra_decode_length
        if add_action_mask_to_inputs:
            self.enc_candidate_ratio = nn.Parameter(torch.tensor(0.1))
            self.dec_candidate_ratio = nn.Parameter(torch.tensor(0.1))

    @property
    def compute_dtype(self):
        return torch.float16 if self.mixed_precision else torch.float32

    def _tensor(self, value, dtype=None):
        return torch.as_tensor(value, device=self.enc_embedding_lookup.weight.device,
                               dtype=dtype)

    def _position(self, value, length):
        if value is not None:
            result = self._tensor(value, self.compute_dtype)
            return (self.tree_position_embedding(result)
                    if self.tree_position_embedding is not None else result)
        # TF RelativePositionEmbedding concatenates sin and cos halves.
        n = (self.width + 1) // 2
        inv = torch.exp(torch.arange(n, device=self.enc_embedding_lookup.weight.device,
                                     dtype=torch.float32) * -(math.log(10000) / max(n - 1, 1)))
        angles = torch.arange(length, device=inv.device)[:, None] * inv[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)[None, :, :self.width].to(self.compute_dtype)

    def _embed(self, tokens, layer):
        return F.embedding(tokens.long(), layer.weight.to(self.compute_dtype)) * math.sqrt(self.width)

    def _candidates(self, mask, layer, ratio):
        values = self._tensor(mask, self.compute_dtype)
        return ratio.to(self.compute_dtype) * (
            values @ layer.weight.to(self.compute_dtype) / values.sum(-1, keepdim=True).clamp_min(1))

    def forward(self, inputs, training=None, mask=None, return_kv_cache=False,
                return_last_token=False):
        training = self.training if training is None else training
        targets = inputs.get('targets')
        if targets is not None:
            targets = self._tensor(targets, torch.long)
        sources = inputs.get('inputs')
        if sources is not None:
            sources = self._tensor(sources, torch.long)
            boolean_mask = sources != 0
            embedded = self._embed(sources, self.enc_embedding_lookup)
        else:
            embedded = self._tensor(inputs['embedded_inputs'], self.compute_dtype)
            boolean_mask = self._tensor(inputs['input_masks'], torch.bool)
        cache = inputs.get('cache')
        if not cache:
            encoder_inputs = embedded * boolean_mask[..., None]
            encoder_inputs = encoder_inputs + self._position(inputs.get('enc_pos_encoding'), embedded.shape[1])
            if self.add_action_mask_to_inputs and inputs.get('enc_action_mask') is not None:
                encoder_inputs = encoder_inputs + self._candidates(
                    inputs['enc_action_mask'], self.enc_embedding_lookup, self.enc_candidate_ratio)
            encoder_inputs = F.dropout(encoder_inputs, self.dropout_rate, training)
            encoder_outputs = self.encoder_layer(encoder_inputs, boolean_mask[:, None, :],
                                                 dtype=self.compute_dtype)
            kv_cache = None
        else:
            encoder_outputs = self._tensor(cache['encoder_outputs'])
            kv_cache = cache['kv_cache']
        if targets is None:
            return self._beam_search(encoder_outputs, boolean_mask)
        if not return_kv_cache:
            kv_cache = None
        offset = kv_cache['0']['key'].shape[1] if kv_cache is not None else 0
        if not return_kv_cache:
            targets = F.pad(targets, (1, 0))[:, :-1]
        decoder_inputs = self._embed(targets, self.dec_embedding_lookup)
        if self.add_action_mask_to_inputs:
            decoder_inputs = decoder_inputs + self._candidates(
                inputs['dec_action_mask'], self.dec_embedding_lookup, self.dec_candidate_ratio)
        explicit_positions = inputs.get('dec_pos_encoding')
        pos = self._position(explicit_positions, targets.shape[1] + offset)
        if explicit_positions is None:
            pos = pos[:, offset:]
        if not return_kv_cache and inputs.get('dec_pos_encoding') is not None:
            pos = F.pad(pos, (0, 0, 1, 0))[:, :-1]
        decoder_inputs = F.dropout(decoder_inputs + pos, self.dropout_rate, training)
        b, length = targets.shape
        if return_kv_cache and kv_cache is None:
            shape = (b, 0, self.heads, self.width // self.heads)
            kv_cache = {str(i): {name: decoder_inputs.new_zeros(shape) for name in ('key', 'value')}
                        for i in range(self.decoder_layer.num_layers)}
        offset = kv_cache['0']['key'].shape[1] if kv_cache is not None else 0
        causal = torch.ones((length + offset, length + offset), device=targets.device,
                            dtype=torch.bool).tril()[None, offset:, :]
        outputs = self.decoder_layer(decoder_inputs, causal, encoder_outputs,
                                     boolean_mask[:, None, :], kv_cache, self.compute_dtype)
        logits = F.linear(outputs.to(self.compute_dtype),
                          self.dec_embedding_lookup.weight.to(self.compute_dtype)).float()
        action_mask = self._tensor(inputs['dec_action_mask'], torch.bool)
        if return_last_token:
            logits = logits[:, -1]
            action_mask = action_mask[:, -1]
        logits = torch.where(action_mask, logits, torch.finfo(torch.float32).min)
        if self.return_cache:
            return logits, dict(encoder_outputs=encoder_outputs, outputs=outputs, kv_cache=kv_cache)
        return logits

    @torch.no_grad()
    def _beam_search(self, memory, boolean_mask):
        """TF Models' alive/finished beam search, including its stopping bound.

        CircuitTransformer.optimize uses constrained decoding instead. This
        preserves the original low-level, unconstrained Seq2Seq entry point.
        """
        b, source_length = memory.shape[:2]
        beam, vocab = self.beam_size, self.dec_embedding_lookup.num_embeddings
        max_length = self.decode_max_length or (source_length + self.extra_decode_length)
        dtype, device = self.compute_dtype, memory.device
        infinity = 65504.0 if dtype == torch.float16 else 1e7
        memory = memory.to(dtype).repeat_interleave(beam, 0)
        cross_mask = boolean_mask.repeat_interleave(beam, 0)[:, None, :]
        sequence = torch.zeros((b, beam, 1), dtype=torch.long, device=device)
        alive = torch.full((b, beam), -float('inf'), dtype=dtype, device=device)
        alive[:, 0] = 0
        finished_seq = sequence.clone()
        finished_scores = torch.full_like(alive, -infinity)
        finished = torch.zeros_like(alive, dtype=torch.bool)
        cache = {str(i): {key: memory.new_zeros((b * beam, 0, self.heads, self.width // self.heads))
                          for key in ('key', 'value')} for i in range(self.decoder_layer.num_layers)}
        timing = self._position(None, max_length + 1)
        batch_indices = torch.arange(b, device=device)[:, None]

        def top(values, count):
            # TF top_k resolves equal scores by smaller index.
            indices = values.argsort(dim=-1, descending=True, stable=True)[:, :count]
            return values.gather(1, indices), indices

        for step in range(max_length):
            worst = (finished_scores * finished).amin(1)
            worst = worst + (~finished.any(1)).to(dtype) * -infinity
            bound = alive[:, 0] / (((5 + max_length) / 6) ** self.alpha)
            if bool((worst > bound).all()):
                break
            x = self._embed(sequence[:, :, -1].reshape(b * beam, 1), self.dec_embedding_lookup)
            x = x + timing[:, step:step + 1]
            causal = torch.ones((1, 1, step + 1), dtype=torch.bool, device=device)
            output = self.decoder_layer(x, causal, memory, cross_mask, cache, dtype)
            logits = F.linear(output.to(dtype), self.dec_embedding_lookup.weight.to(dtype))[:, 0]
            probabilities = logits.log_softmax(-1).reshape(b, beam, vocab) + alive[:, :, None]
            scores, indices = top(probabilities.flatten(1), beam * 2)
            parents, tokens = indices // vocab, indices % vocab
            candidates = torch.cat((sequence[batch_indices, parents], tokens[:, :, None]), dim=-1)
            flags = tokens == self.eos_id
            # Both survivor caches and sequences originate from the selected parents.
            alive, survivors = top(scores + flags.to(dtype) * -infinity, beam)
            sequence = candidates[batch_indices, survivors]
            selected_parents = parents.gather(1, survivors)
            flat_parents = (selected_parents + batch_indices * beam).flatten()
            cache = {layer: {name: value[flat_parents] for name, value in entries.items()}
                     for layer, entries in cache.items()}
            finished_seq = F.pad(finished_seq, (0, 1))
            new_scores = scores / (((5 + step + 1) / 6) ** self.alpha)
            new_scores = new_scores + (~flags).to(dtype) * -infinity
            all_scores = torch.cat((finished_scores, new_scores), dim=1)
            all_seq = torch.cat((finished_seq, candidates), dim=1)
            all_flags = torch.cat((finished, flags), dim=1)
            finished_scores, selected = top(all_scores, beam)
            finished_seq = all_seq[batch_indices, selected]
            finished = all_flags[batch_indices, selected]
        has_finished = finished.any(1)
        output = torch.where(has_finished[:, None], finished_seq[:, 0], sequence[:, 0])[:, 1:]
        scores = torch.where(has_finished, finished_scores[:, 0], alive[:, 0])
        if self.padded_decode:
            output = F.pad(output, (0, max_length - output.shape[1]))
        return dict(outputs=output.to(torch.int32), scores=scores)


def masked_loss(label, pred):
    label = torch.as_tensor(label, device=pred.device, dtype=torch.long)
    loss = F.cross_entropy(pred.flatten(0, 1), label.flatten(), reduction='none').reshape_as(label)
    mask = label != 0
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def masked_accuracy(label, pred):
    label = torch.as_tensor(label, device=pred.device, dtype=torch.long)
    mask = label != 0
    return ((pred.argmax(dim=2) == label) & mask).float().sum() / mask.sum().clamp_min(1)


class CustomSchedule:
    def __init__(self, d_model, warmup_steps=4000):
        if d_model <= 0 or warmup_steps <= 0:
            raise ValueError('d_model and warmup_steps must be positive')
        self.d_model, self.warmup_steps = d_model, warmup_steps

    def __call__(self, step):
        step = torch.as_tensor(step, dtype=torch.float32)
        return self.d_model ** -0.5 * torch.minimum(step.rsqrt(), step * self.warmup_steps ** -1.5)


class KerasAdam(torch.optim.Optimizer):
    """Keras Adam update, including epsilon-hat and dense embedding moments."""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.98), eps=1e-9):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, iterations=0))

    def skipped_step(self):
        """Keras LossScaleOptimizer advances iterations even after overflow."""
        for group in self.param_groups:
            group['iterations'] += 1

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], _Loss]) -> _Loss: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], _Loss] | None = None) -> _Loss | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group['iterations'] += 1
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.to_dense() if p.grad.is_sparse else p.grad
                state = self.state[p]
                if not state:
                    state.update(exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                t = torch.tensor(group['iterations'], device=p.device, dtype=p.dtype)
                b1, b2 = p.new_tensor(beta1), p.new_tensor(beta2)
                alpha = group['lr'] * (1 - b2.pow(t)).sqrt() / (1 - b1.pow(t))
                m, v = state['exp_avg'], state['exp_avg_sq']
                m.add_((grad - m) * (1 - beta1))
                v.add_((grad.square() - v) * (1 - beta2))
                p.sub_((m * alpha) / (v.sqrt() + group['eps']))
        return loss
