import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.modeling_layers import GenericForSequenceClassification, GenericForTokenClassification, GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from .configuration_hydrogen import HydrogenAudioConfig, HydrogenConfig, HydrogenVisionConfig

# Conv-state slots inside one layer's cache entry (`HydrogenConfig.number_of_conv_states`)
CANON_A, MIXER_CONV, CANON_C, CANON_D, DECODER_KV_CONV = 0, 1, 2, 3, 4
CANON_KERNEL, CONV_KERNEL, SINKHORN_ITERS = 4, 3, 4
# Session learning takes one step per this many input tokens: 64 kept 91% of the accuracy gain of 32 at 1.7x the speed
SESSION_CHUNK = 64
EVICT_EVERY = 64  # cached keys are tested for eviction this often (in tokens), so the test costs nothing per token
EVICT_SINKS = 4  # the first tokens are never evicted: attention parks probability mass on them


# --------------------------------------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------------------------------------


@use_kernel_forward_from_hub("RMSNorm")
class HydrogenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def causal_conv(x: torch.Tensor, weight: torch.Tensor, cache: Cache | None = None, layer_idx: int = 0, slot: int = 0):
    """
    Depthwise causal convolution over time, (batch, seq, channels) -> same shape, with `weight` (channels, 1, kernel).
    With a cache the last `kernel - 1` inputs live in the layer's conv state `slot`, so a decoding step sees exactly the
    window it saw during prefill.
    """
    kernel = weight.shape[-1]
    seq_len = x.shape[1]
    if cache is not None:
        x = cache.update_conv_state(x.transpose(1, 2), layer_idx, state_idx=slot, conv_kernel_size=kernel - 1).transpose(1, 2)
    if x.shape[1] < seq_len + kernel - 1:  # no (or too short a) cached window: the past is zeros
        x = F.pad(x, (0, 0, seq_len + kernel - 1 - x.shape[1], 0))
    x = x[:, -(seq_len + kernel - 1) :]
    # A handful of taps: shifted multiply-adds beat a depthwise conv call, whose setup dwarfs the arithmetic
    taps = weight.to(x.dtype).squeeze(1).t()  # (kernel, channels)
    if seq_len == 1:  # decoding: one dot product over the cached window
        return (x * taps).sum(1, keepdim=True)
    out = x[:, :seq_len] * taps[0]
    for k in range(1, kernel):
        out = out + x[:, k : k + seq_len] * taps[k]
    return out


class CanonConv(nn.Module):
    """
    Canon layer ("Physics of Language Models, Part 4.1"): a causal depthwise convolution added residually. It gives every
    position a cheap, learned summary of its immediate neighbours, which small models otherwise spend attention heads on.
    Zero-initialized, so it starts as the identity.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, 1, CANON_KERNEL))

    def forward(self, x, cache=None, layer_idx=0, slot=0):
        return x + causal_conv(x, self.weight, cache, layer_idx, slot)


def mask_padding(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Zero padded positions before they enter a convolution."""
    return x if mask is None else x * mask[:, :, None].to(x.dtype)


class HydrogenCache(DynamicCache):
    """
    `DynamicCache` plus Hydrogen's side states (the fp32 cumulative forget gates of forgetting attention, per layer, and
    the session-learning state), which beam search reorders together with the KV and conv states.
    """

    def side(self) -> dict:
        return self.__dict__.setdefault("hydrogen_side", {})

    def reorder_cache(self, beam_idx: torch.LongTensor):
        super().reorder_cache(beam_idx)
        side = self.side()
        side.pop("novelty_steps", None)  # recorded steps belong to the pre-reorder beams
        for key, value in side.items():
            if isinstance(value, torch.Tensor):
                side[key] = value.index_select(0, beam_idx.to(value.device))

    def reset(self):
        super().reset()
        self.side().clear()


def detach_cache(cache: Cache) -> None:
    """Cut every cached tensor from the autograd graph (session learning runs the model with gradients on)."""
    for layer in cache.layers:
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor) and value.grad_fn is not None:
                setattr(layer, name, value.detach())
        for states in (getattr(layer, "conv_states", None), getattr(layer, "recurrent_states", None)):
            for key, value in (states or {}).items():
                if isinstance(value, torch.Tensor) and value.grad_fn is not None:
                    states[key] = value.detach()
    side = side_states(cache)
    for key, value in side.items():
        if isinstance(value, torch.Tensor) and value.grad_fn is not None:  # leaves (session corrections) stay leaves
            side[key] = value.detach()


def side_states(cache: Cache | None) -> dict | None:
    """The side-state dict of a cache, upgrading a plain `DynamicCache` (e.g. one built by `generate`) in place."""
    if cache is None:
        return None
    if type(cache) is DynamicCache:
        cache.__class__ = HydrogenCache
    return cache.side() if isinstance(cache, HydrogenCache) else cache.__dict__.setdefault("hydrogen_side", {})


class HydrogenRotaryEmbedding(nn.Module):
    def __init__(self, config: HydrogenConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config: HydrogenConfig, device=None, **kwargs) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), 1.0

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = position_ids[:, :, None].float() * self.inv_freq.float()
            freqs = freqs.repeat_interleave(2, dim=-1)  # interleaved layout: every frequency covers two adjacent channels
            return (freqs.cos() * self.attention_scaling).to(x.dtype), (freqs.sin() * self.attention_scaling).to(x.dtype)


def rotate_half(x):
    """Rotates adjacent channel pairs (interleaved RoPE layout)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies interleaved RoPE to the first `cos.shape[-1]` channels of `x` (batch, heads, seq_len, head_dim)."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    return torch.cat([(x_rot * cos) + (rotate_half(x_rot) * sin), x_pass], dim=-1)


# --------------------------------------------------------------------------------------------------------------------
# Token mixers
# --------------------------------------------------------------------------------------------------------------------


class HydrogenAttention(nn.Module):
    """
    Forgetting attention: grouped-query attention whose logits are down-weighted by the product of learned per-head,
    per-token forget gates between query and key (`logit_ij += F_i - F_j`, F the cumulative log gate), with QK-norm and a
    learned temperature, partial interleaved RoPE, a Canon convolution on q/k/v and a query-dependent output gate.

    Encoder layers read q/k/v from their own residual (each from its own hyper-connection blend). Decoder layers read
    q from their own residual but k, v and the forget gates from the encoder output, so no decoder layer needs another
    token's decoder state - which is what lets prefill skip most of the decoder.
    """

    forget_block_entries = 1 << 24  # bias entries per query block (bounds peak memory on long sequences)

    def __init__(self, config: HydrogenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.is_decoder = layer_idx >= config.encoder_layers

        q_size = config.num_attention_heads * self.head_dim
        kv_size = config.num_key_value_heads * self.head_dim
        self.split = (q_size, kv_size, kv_size)
        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, q_size, bias=True)
        # Heads start spread from fast (sigmoid(1) = 0.73 per token) to slow (sigmoid(6)) forgetting (see _init_weights)
        self.forget_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=True)
        self.q_norm = HydrogenRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = HydrogenRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # normalized q.k is bounded, so the temperature must be learnable or attention could not sharpen
        self.log_temperature = nn.Parameter(torch.zeros(()))
        if self.is_decoder:
            self.canon = CanonConv(q_size)
            self.canon_kv = CanonConv(2 * kv_size)
        else:
            self.canon = CanonConv(q_size + 2 * kv_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        key_value_states: tuple[torch.Tensor, torch.Tensor] | None = None,
        encoder_states: torch.Tensor | None = None,
        encoder_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        encoder_padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        `key_value_states` feeds an encoder layer's k/v from other inputs than its query. A decoder layer takes k/v from
        `encoder_states`, which cover more (earlier) positions than `hidden_states` while a prompt's end is replayed.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings
        if self.is_decoder:
            query = self.canon(mask_padding(self.q_proj(hidden_states), padding_mask), past_key_values, self.layer_idx, MIXER_CONV)
            kv = mask_padding(torch.cat([self.k_proj(encoder_states), self.v_proj(encoder_states)], -1), encoder_padding_mask)
            key, value = self.canon_kv(kv, past_key_values, self.layer_idx, DECODER_KV_CONV).chunk(2, -1)
            key_cos, key_sin = encoder_position_embeddings
            key_shape = (*encoder_states.shape[:-1], -1, self.head_dim)
            gate_source = encoder_states
        else:
            key_input, value_input = key_value_states if key_value_states is not None else (hidden_states, hidden_states)
            mixed = torch.cat([self.q_proj(hidden_states), self.k_proj(key_input), self.v_proj(value_input)], -1)
            mixed = mask_padding(mixed, padding_mask)
            query, key, value = self.canon(mixed, past_key_values, self.layer_idx, MIXER_CONV).split(self.split, dim=-1)
            key_cos, key_sin, key_shape = cos, sin, hidden_shape
            gate_source = hidden_states

        key_states = apply_rotary(self.k_norm(key.view(key_shape)).transpose(1, 2), key_cos, key_sin)
        value_states = value.view(key_shape).transpose(1, 2)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        query_states = apply_rotary(self.q_norm(query.view(hidden_shape)).transpose(1, 2), cos, sin)
        query_states = query_states * self.log_temperature.exp().to(query_states.dtype)

        attn_output = self._attend(gate_source, query_states, key_states, value_states, attention_mask, past_key_values)
        attn_output = attn_output.reshape(*input_shape, -1) * torch.sigmoid(self.gate_proj(hidden_states))
        return self.o_proj(attn_output)

    def _evict(self, past_key_values, keys_forget, queries):
        """
        Drop keys the forget gates have made irrelevant. Attention weights here are bounded: queries and keys are
        RMS-normalized, so a logit cannot exceed `exp(log_temperature) * sqrt(head_dim)` in magnitude, and key j is
        down-weighted by `exp(F_i - F_j)` for every later query i. Once

            2 * logit_bound + (F_i - F_j) < log(eviction epsilon)

        holds for every head, key j can never again take more than epsilon of any future attention weight (F_i only
        falls as i grows), so dropping it changes nothing beyond that tolerance. The first `EVICT_SINKS` keys and the
        current query block are always kept. Returns the kept-column index, or None if nothing is dropped.

        Invariant: eviction only removes whole positions from this layer's KV cache and its forget sums, which stay
        aligned; it is skipped while an attention mask is in play, because that mask is indexed by position.
        """
        layer = past_key_values.layers[self.layer_idx]
        length = keys_forget.shape[-1]
        if length < EVICT_SINKS + EVICT_EVERY or (length - queries) % EVICT_EVERY:
            return None
        bound = 2 * self.log_temperature.exp().to(keys_forget.dtype) * self.head_dim**0.5
        margin = keys_forget[..., -1:] - keys_forget  # F_i - F_j at the newest query, per head (<= 0)
        keep = (margin.amax(dim=1)[0] + bound > math.log(self.config.kv_evict_epsilon)).clone()
        keep[:EVICT_SINKS] = True
        keep[-queries:] = True
        if bool(keep.all()):
            return None
        index = keep.nonzero(as_tuple=True)[0]
        layer.keys = layer.keys.index_select(2, index)
        layer.values = layer.values.index_select(2, index)
        return index

    def _attend(self, gate_source, query, key, value, attention_mask, past_key_values):
        # Cumulative log forget gates; the key-side sums stay in fp32 next to the cache, since they grow with the context
        cumulative = F.logsigmoid(self.forget_proj(gate_source).float()).transpose(1, 2).cumsum(-1)  # (batch, heads, seq)
        keys_forget = cumulative
        if past_key_values is not None:
            store = side_states(past_key_values)
            previous = store.get(("forget", self.layer_idx))
            if previous is not None:
                cumulative = cumulative + previous[..., -1:]
                keys_forget = torch.cat([previous, cumulative], -1)
            else:
                keys_forget = cumulative
            if self.config.kv_evict_epsilon and attention_mask is None and keys_forget.shape[0] == 1:
                index = self._evict(past_key_values, keys_forget, query.shape[-2])
                if index is not None:
                    keys_forget = keys_forget.index_select(-1, index)
                    key, value = past_key_values.layers[self.layer_idx].keys, past_key_values.layers[self.layer_idx].values
            store[("forget", self.layer_idx)] = keys_forget
        queries, k_len = query.shape[-2], key.shape[-2]
        query_forget = cumulative[..., -queries:]
        keys_forget = keys_forget[..., -k_len:]
        offset = k_len - queries  # query i sits at key position offset + i
        outputs = []
        # Queries are processed in blocks so the (heads, q, k) bias stays bounded; at scale a fused
        # forgetting-attention kernel computes the same thing without materializing it.
        block = max(1, self.forget_block_entries // max(1, k_len * query.shape[1] * query.shape[0]))
        for start in range(0, queries, block):
            stop = min(queries, start + block)
            bias = query_forget[..., start:stop].unsqueeze(-1) - keys_forget.unsqueeze(-2)
            if attention_mask is None:
                rows = torch.arange(start, stop, device=bias.device)[:, None] + offset
                bias = bias.masked_fill(torch.arange(k_len, device=bias.device)[None, :] > rows, float("-inf"))
            elif attention_mask.dtype == torch.bool:
                bias = bias.masked_fill(~attention_mask[..., start:stop, :], float("-inf"))
            else:
                bias = bias + attention_mask[..., start:stop, :].float()
            outputs.append(
                F.scaled_dot_product_attention(
                    query[..., start:stop, :], key, value, attn_mask=bias.to(query.dtype), scale=self.scaling, enable_gqa=True
                )
            )
        return torch.cat(outputs, dim=-2).transpose(1, 2)


class HydrogenShortConv(nn.Module):
    """
    Gated short convolution: `out(C * conv(B * x))`. A convolution over a few tokens with input- and output-dependent
    gates handles the local structure small Transformers otherwise spend attention layers on, in linear time and with a
    constant-size cache.
    """

    def __init__(self, config: HydrogenConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d = config.hidden_size
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(d, 1, CONV_KERNEL))
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, hidden_states, past_key_values=None, padding_mask=None, **kwargs):
        b, c, x = self.in_proj(mask_padding(hidden_states, padding_mask)).chunk(3, dim=-1)
        return self.out_proj(c * causal_conv(b * x, self.conv_weight, past_key_values, self.layer_idx, MIXER_CONV))


# --------------------------------------------------------------------------------------------------------------------
# Channel mixers
# --------------------------------------------------------------------------------------------------------------------


class HydrogenMLP(nn.Module):
    """Gated MLP with a Canon convolution on its hidden state."""

    def __init__(self, config: HydrogenConfig, layer_idx: int, intermediate_size: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.canon = CanonConv(2 * intermediate_size)
        self.record = None  # novelty decoding: the last token's hidden activation

    def forward(self, hidden_states, past_key_values=None, padding_mask=None):
        up = mask_padding(self.gate_up_proj(hidden_states), padding_mask)
        gate, up = self.canon(up, past_key_values, self.layer_idx, CANON_D).chunk(2, dim=-1)
        hidden = up * self.activation_fn(gate)
        if self.record is not None:
            self.record = hidden[:, -1].float()
        out = self.down_proj(hidden)
        delta = None if past_key_values is None else side_states(past_key_values).get(("session", self.layer_idx))
        if delta is not None:  # what this session has learned so far: a per-sequence correction of down_proj
            out = out + torch.einsum("bti,boi->bto", hidden, delta.to(hidden.dtype))
        return out


class HydrogenMoE(HydrogenMLP):
    """
    Fine-grained mixture of experts: one always-on shared expert (the inherited MLP) plus `num_experts` small routed
    experts, top-k by sigmoid affinity. Load balance comes from a per-expert bias that only affects *which* experts are
    chosen, nudged after each training step towards equal load (auxiliary-loss-free), so no balancing loss competes with
    language modeling.
    """

    def __init__(self, config: HydrogenConfig, layer_idx: int):
        super().__init__(config, layer_idx, intermediate_size=config.moe_intermediate_size)
        E, I, d = config.num_experts, config.moe_intermediate_size, config.hidden_size
        self.num_experts, self.top_k = E, config.num_experts_per_tok
        self.bias_rate = config.router_bias_update_rate
        self.router = nn.Linear(d, E, bias=False)
        self.expert_bias = nn.Buffer(torch.zeros(E))
        self.experts_gate_up = nn.Parameter(torch.empty(E, 2 * I, d))
        self.experts_down = nn.Parameter(torch.empty(E, d, I))
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, past_key_values=None, padding_mask=None):
        shared = super().forward(hidden_states, past_key_values, padding_mask)
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        scores = torch.sigmoid(self.router(x).float())
        chosen = (scores + self.expert_bias).topk(self.top_k, dim=-1).indices
        weights = scores.gather(-1, chosen)
        weights = (weights / weights.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(x)
        for expert in chosen.unique().tolist():
            token, slot = (chosen == expert).nonzero(as_tuple=True)
            gate, up = F.linear(x[token], self.experts_gate_up[expert]).chunk(2, dim=-1)
            y = F.linear(self.activation(gate) * up, self.experts_down[expert])
            out.index_add_(0, token, y * weights[token, slot, None])
        if self.training:
            with torch.no_grad():
                load = torch.bincount(chosen.flatten(), minlength=self.num_experts).float()
                self.expert_bias.add_(self.bias_rate * torch.sign(load.mean() - load))
        return shared + out.view_as(hidden_states)


# --------------------------------------------------------------------------------------------------------------------
# Residual streams
# --------------------------------------------------------------------------------------------------------------------


class HydrogenHyperConnection(nn.Module):
    """
    Manifold-constrained hyper-connection. The residual is `n` streams wide; before a sublayer, token-dependent weights
    collapse them into its input (`pre`), and afterwards the output is written back (`post`) while the streams are remixed
    by a doubly stochastic matrix (`comb`, Sinkhorn-projected, so repeated mixing neither blows up nor vanishes with
    depth). With `outputs=3` an attention layer reads separate blends for its query, key and value.
    """

    def __init__(self, config: HydrogenConfig, outputs: int = 1):
        super().__init__()
        n, d = config.hc_streams, config.hidden_size
        self.n, self.outputs = n, outputs
        self.hc_proj = nn.Linear(n * d, (outputs + 1 + n) * n, bias=True)
        self.hc_scale = nn.Parameter(torch.ones(3))
        self.eps = config.rms_norm_eps

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.n
        flat = streams.flatten(-2).float()
        flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.eps)
        logits = F.linear(flat, self.hc_proj.weight.float(), self.hc_proj.bias.float())
        pre, post, comb = logits.split([self.outputs * n, n, n * n], dim=-1)
        pre = torch.sigmoid(pre * self.hc_scale[0]).unflatten(-1, (self.outputs, n)) + 1e-6
        post = 2 * torch.sigmoid(post * self.hc_scale[1])
        comb = torch.softmax((comb * self.hc_scale[2]).unflatten(-1, (n, n)), dim=-1) + 1e-6
        for _ in range(SINKHORN_ITERS - 1):  # 4 iterations matched 10 on the bench
            comb = comb / comb.sum(-2, keepdim=True)
            comb = comb / comb.sum(-1, keepdim=True)
        collapsed = (pre @ streams.float()).to(streams.dtype)  # (batch, seq, outputs, hidden)
        return collapsed, post.to(streams.dtype), comb.to(streams.dtype)

    @staticmethod
    def write(streams, output, post, comb):
        return post.unsqueeze(-1) * output.unsqueeze(-2) + comb.transpose(-1, -2) @ streams


# --------------------------------------------------------------------------------------------------------------------
# Decoder layer
# --------------------------------------------------------------------------------------------------------------------


class HydrogenDecoderLayer(GradientCheckpointingLayer):
    """
    Streams -> [collapse] -> norm -> Canon A -> mixer -> [write back] -> [collapse] -> norm -> Canon C -> MLP (Canon D)
    -> [write back]. Hidden states are (batch, seq, streams, hidden) throughout.
    """

    def __init__(self, config: HydrogenConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.mixer_type = config.mixers[layer_idx]
        if self.mixer_type == "attention":
            self.self_attn = HydrogenAttention(config, layer_idx)
        else:
            self.conv = HydrogenShortConv(config, layer_idx)
        self.mlp = HydrogenMoE(config, layer_idx) if config.is_moe_layer(layer_idx) else HydrogenMLP(config, layer_idx)
        self.input_layernorm = HydrogenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HydrogenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.canon_a = CanonConv(config.hidden_size)
        self.canon_c = CanonConv(config.hidden_size)
        split = self.mixer_type == "attention" and layer_idx < config.encoder_layers
        self.hc_mix = HydrogenHyperConnection(config, outputs=3 if split else 1)
        self.hc_mlp = HydrogenHyperConnection(config)

    def _mix(self, collapsed, past_key_values=None, padding_mask=None, **kwargs):
        if collapsed.shape[-2] == 3:
            # query, key and value read different stream blends: normalize and convolve them in one stacked call
            stacked = self.input_layernorm(collapsed).flatten(-2)
            weight = self.canon_a.weight.repeat(3, 1, 1)
            stacked = stacked + causal_conv(mask_padding(stacked, padding_mask), weight, past_key_values, self.layer_idx, CANON_A)
            query_input, key_input, value_input = stacked.chunk(3, dim=-1)
            return self.self_attn(
                query_input, past_key_values=past_key_values, key_value_states=(key_input, value_input), padding_mask=padding_mask, **kwargs
            )
        x = self.canon_a(mask_padding(self.input_layernorm(collapsed[..., 0, :]), padding_mask), past_key_values, self.layer_idx, CANON_A)
        if self.mixer_type == "attention":
            return self.self_attn(x, past_key_values=past_key_values, padding_mask=padding_mask, **kwargs)
        return self.conv(x, past_key_values=past_key_values, padding_mask=padding_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        collapsed, post, comb = self.hc_mix(hidden_states)
        mixed = self._mix(
            collapsed,
            past_key_values=past_key_values,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.hc_mix.write(hidden_states, mixed, post, comb)
        collapsed, post, comb = self.hc_mlp(hidden_states)
        if self.mlp.record is not None:
            self.record_post = post[:, -1].float().mean(-1)  # how strongly the MLP output reaches the mean of the streams
        normed = self.canon_c(mask_padding(self.post_attention_layernorm(collapsed[..., 0, :]), padding_mask), past_key_values, self.layer_idx, CANON_C)
        return self.hc_mlp.write(hidden_states, self.mlp(normed, past_key_values, padding_mask), post, comb)


# --------------------------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------------------------


class HydrogenPreTrainedModel(PreTrainedModel):
    config: HydrogenConfig
    config_class = HydrogenConfig  # so `from_pretrained` matches the checkpoint's model_type ("hydrogen")
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HydrogenDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _is_stateful = True  # conv states cannot be rolled back token by token
    _can_compile_fullgraph = False  # the prefill replay is data-dependent
    _can_record_outputs = {"hidden_states": HydrogenDecoderLayer}

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, CanonConv):
            init.zeros_(module.weight)
        elif isinstance(module, HydrogenShortConv):
            init.normal_(module.conv_weight, std=1.0 / CONV_KERNEL)
        elif isinstance(module, HydrogenAttention):
            init.zeros_(module.log_temperature)
            init.zeros_(module.forget_proj.weight)
            heads = module.forget_proj.bias.numel()
            spread = torch.linspace(1.0, 6.0, heads) if heads > 1 else torch.full((1,), 3.0)
            init.copy_(module.forget_proj.bias, spread)
        elif isinstance(module, HydrogenHyperConnection):
            init.ones_(module.hc_scale)
        if isinstance(module, HydrogenMoE):
            init.normal_(module.experts_gate_up, std=std)
            init.normal_(module.experts_down, std=std)
            init.zeros_(module.expert_bias)


class HydrogenModel(HydrogenPreTrainedModel):
    def __init__(self, config: HydrogenConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # No padding_idx: with tied embeddings the pad id is often also EOS, whose embedding must keep learning
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HydrogenDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.encoder_norm = HydrogenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = HydrogenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HydrogenRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # How many trailing prompt tokens the decoder must recompute so that its Canon convolutions (the only decoder
        # parts that look at other tokens' decoder states) end in exactly the state a full prefill would leave
        decoder_layers = config.num_hidden_layers - config.encoder_layers
        self.replay_tokens = decoder_layers * 4 * (CANON_KERNEL - 1) + 1
        self.record_final = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        decoder_tokens: int | None = None,
        session: "SessionLearner | None" = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        decoder_tokens (`int`, *optional*):
            With a cache, only the last `decoder_tokens` positions are needed from the decoder (e.g. the one token
            `generate` samples from): the decoder then runs on just enough trailing tokens to leave its caches exact, and
            `last_hidden_state` covers only those. Set automatically by `HydrogenForCausalLM` from `logits_to_keep`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = HydrogenCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        mask_kwargs = dict(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids,
        )
        causal_mask = create_causal_mask(**mask_kwargs, layer_idx=self.config.mixers.index("attention"))
        padding_mask = create_recurrent_attention_mask(self.config, inputs_embeds, attention_mask, past_key_values)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)
        common = dict(past_key_values=past_key_values, position_embeddings=position_embeddings, padding_mask=padding_mask)

        hidden_states = inputs_embeds.unsqueeze(-2).expand(-1, -1, self.config.hc_streams, -1).contiguous()
        replay = None
        for layer_idx, layer in enumerate(self.layers):
            layer_mask = causal_mask if layer.mixer_type == "attention" else None
            if layer_mask is not None and replay is not None:
                layer_mask = layer_mask[..., -replay:, :]
            hidden_states = layer(hidden_states, attention_mask=layer_mask, **common, **kwargs)
            if layer_idx == self.config.encoder_layers - 1:
                # The encoder output becomes every decoder layer's key/value/forget-gate source
                kwargs["encoder_states"] = self.encoder_norm(hidden_states.mean(-2))
                kwargs["encoder_position_embeddings"] = position_embeddings
                kwargs["encoder_padding_mask"] = padding_mask
                if session is not None:
                    return self._decode_while_learning(hidden_states, causal_mask, common, kwargs, session)
                needed = max(decoder_tokens or 0, self.replay_tokens)
                if decoder_tokens and not self.training and past_key_values is not None and hidden_states.shape[1] > needed:
                    # Prefill: the decoder only has to see the end of the prompt (its keys come from the encoder)
                    replay = needed
                    hidden_states = hidden_states[:, -replay:]
                    cos, sin = position_embeddings
                    common["position_embeddings"] = (cos[:, -replay:], sin[:, -replay:])
                    if padding_mask is not None:
                        common["padding_mask"] = padding_mask[:, -replay:]

        final = hidden_states.mean(-2)
        if self.record_final:
            self.recorded_final = final[:, -1].float()
        return BaseModelOutputWithPast(last_hidden_state=self.norm(final), past_key_values=past_key_values)

    def _decode_while_learning(self, hidden_states, causal_mask, common, kwargs, session):
        """
        Session learning: the encoder has already read the whole input in one pass (it holds no session state); the
        decoder now reads it chunk by chunk - predict a chunk with what the session knows, learn from it, move on.
        """
        encoder_states = kwargs["encoder_states"]
        cos, sin = common["position_embeddings"]
        padding_mask = common["padding_mask"]
        past = common["past_key_values"]
        length = hidden_states.shape[1]
        key_offset = causal_mask.shape[-1] - length if causal_mask is not None else 0
        finals = []
        for start in range(0, length, session.chunk):
            stop = min(length, start + session.chunk)
            span = slice(start, stop)
            chunk_kwargs = dict(
                kwargs,
                encoder_states=encoder_states[:, span],
                encoder_position_embeddings=(cos[:, span], sin[:, span]),
                encoder_padding_mask=None if padding_mask is None else padding_mask[:, span],
            )
            chunk_common = dict(
                common,
                position_embeddings=(cos[:, span], sin[:, span]),
                padding_mask=None if padding_mask is None else padding_mask[:, span],
            )
            mask = None if causal_mask is None else causal_mask[..., span, : key_offset + stop]
            session.prepare()
            with torch.enable_grad():
                h = hidden_states[:, span]
                for layer in self.layers[self.config.encoder_layers :]:
                    h = layer(h, attention_mask=mask, **chunk_common, **chunk_kwargs)
                final = self.norm(h.mean(-2))
                session.learn(final, span)
            detach_cache(past)
            finals.append(final.detach())
        return BaseModelOutputWithPast(last_hidden_state=torch.cat(finals, dim=1), past_key_values=past)


def decoder_mlps(model) -> list:
    return [layer.mlp for layer in model.model.layers[model.config.encoder_layers :]]


def session_update(model, store, mlp, delta, step) -> None:
    """
    Apply one session step to a layer's correction and store it again.

    `step` is the raw gradient direction. The correction is then bounded to `session_drift_cap` times the RMS of the
    weight it corrects, so however long a session runs it stays a correction rather than a different model, and it is
    kept in bf16, which halves the per-session state and measured the same on the bench. (Momentum and per-layer
    normalized steps were measured too: both were worse, so neither is here.)
    """
    delta = delta.detach().float() + step
    limit = model.session_drift_cap * mlp.down_proj.weight.float().pow(2).mean().sqrt()
    scale = (limit / (delta.pow(2).mean((-2, -1), keepdim=True).sqrt() + 1e-12)).clamp(max=1.0)
    store[("session", mlp.layer_idx)] = (delta * scale).to(model.session_dtype).detach()


def session_deltas(mlps, store, batch) -> list[torch.Tensor]:
    """This session's corrections to the decoder MLP output projections, as fresh leaves that gradients can reach."""
    deltas = []
    for mlp in mlps:
        delta = store.get(("session", mlp.layer_idx))
        if delta is None:
            delta = mlp.down_proj.weight.new_zeros(batch, *mlp.down_proj.weight.shape)
        if not (delta.is_leaf and delta.requires_grad):  # reuse the leaf while it is unchanged (one answer = one leaf)
            delta = delta.detach().requires_grad_(True)
        store[("session", mlp.layer_idx)] = delta
        deltas.append(delta)
    return deltas


def flush_novelty(model, store) -> None:
    """
    Turn the generated tokens recorded so far into novelty gradients: for every decoder layer, the direct-path
    gradient of -log p(own token) with respect to its session correction, summed over tokens. The direct path is the
    route every MLP output takes into the residual streams and on through the final norm to the output head; it needs
    no backward pass - only what the forward already computed and one product with the output embedding per token.
    """
    steps = [step for step in store.pop("novelty_steps", []) if step["token"] is not None]
    if not steps:
        return
    embedding = model.lm_head.weight.float()
    norm = model.model.norm
    token = torch.cat([step["token"] for step in steps], dim=1)  # (batch, T)
    g_h = torch.stack([step["expected"] for step in steps], dim=1) - embedding[token]  # E^T (p - onehot)
    x = torch.stack([step["final"] for step in steps], dim=1)
    w = norm.weight.float()
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    wg = w * g_h
    g_x = wg * r - x * (wg * x).mean(-1, keepdim=True) * r.pow(3)  # back through the RMSNorm
    for i, mlp in enumerate(decoder_mlps(model)):
        hidden = torch.stack([step["hidden"][i] for step in steps], dim=1)
        reach = torch.stack([step["post"][i] for step in steps], dim=1)[..., None]
        grad = torch.einsum("bto,bti->boi", g_x * reach, hidden)
        key = ("novelty", mlp.layer_idx)
        store[key] = grad if key not in store else store[key] + grad
    store["novelty_count"] = store.get("novelty_count", 0) + len(steps)


def apply_novelty(model, store) -> None:
    """
    Novelty learning, applied between answers: the gradient that makes the tokens the model just generated more
    likely was summed while it generated them; now the session corrections take a step *against* it, so the next answer
    is pushed away from the previous one. Nothing changes in the middle of an answer, which stays self-consistent.
    """
    flush_novelty(model, store)
    count = store.pop("novelty_count", 0)
    for mlp in decoder_mlps(model):
        grad = store.pop(("novelty", mlp.layer_idx), None)
        if grad is None or not count:
            continue
        delta = store.get(("session", mlp.layer_idx))
        delta = torch.zeros_like(grad) if delta is None else delta
        session_update(model, store, mlp, delta, model.config.novelty_rate * grad / count)


class SessionLearner:
    """
    Learns while reading: holds this session's corrections to the decoder MLP output projections (stored in the cache
    side states), and after each chunk has been predicted takes one SGD step on that chunk's next-token loss.
    """

    def __init__(self, model: "HydrogenForCausalLM", input_ids, attention_mask, past_key_values):
        self.model, self.ids, self.cache = model, input_ids, past_key_values
        self.store = side_states(past_key_values)
        self.rate = model.config.session_learning_rate
        self.chunk = model.session_chunk
        past = past_key_values.get_seq_length()
        self.mask = None if attention_mask is None else attention_mask[:, past:]
        self.mlps = decoder_mlps(model)
        self.deltas = []
        self.logits = []
        apply_novelty(model, self.store)

    def prepare(self):
        self.deltas = session_deltas(self.mlps, self.store, self.ids.shape[0])

    def learn(self, final_hidden, span):
        logits = self.model.lm_head(final_hidden)
        self.logits.append(logits.detach())
        ids = self.ids[:, span]
        grads = [None] * len(self.deltas)
        if ids.shape[1] > 1:
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten(), reduction="none")
            loss = loss.view(ids.shape[0], -1)
            weight = torch.ones_like(loss) if self.mask is None else self.mask[:, span][:, 1:].to(loss.dtype)
            # every sequence learns from its own mean loss, so sessions in a batch stay independent
            loss = ((loss * weight).sum(-1) / weight.sum(-1).clamp_min(1)).sum()
            grads = torch.autograd.grad(loss, self.deltas, allow_unused=True)
        for mlp, delta, grad in zip(self.mlps, self.deltas, grads):
            if grad is None:
                self.store[("session", mlp.layer_idx)] = delta.detach()
            else:
                session_update(self.model, self.store, mlp, delta, -self.rate * grad)


class HydrogenForCausalLM(HydrogenPreTrainedModel, GenerationMixin):
    session_chunk = SESSION_CHUNK
    session_drift_cap = 0.1  # a correction stays this small relative to the weight it corrects
    session_dtype = torch.bfloat16
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = HydrogenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _session_learning(self, input_ids, attention_mask, position_ids, past_key_values, logits_to_keep):
        """
        Learn while reading. The encoder reads the input in one pass; the decoder then reads it in chunks of
        `session_chunk` tokens, and each chunk is first predicted with what the session has learned so far (its logits
        are honest) and then learned from with one SGD step on the session's corrections to every decoder MLP output
        projection. The corrections live in the cache - a new cache is a new session, nothing touches the weights - and
        only input the model is *given* (prompts, tool results, user turns) is learned from, never its own generations.
        """
        learner = SessionLearner(self, input_ids, attention_mask, past_key_values)
        with torch.no_grad():
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                session=learner,
            )
        logits = torch.cat(learner.logits, dim=1)
        return logits[:, -logits_to_keep:] if isinstance(logits_to_keep, int) and logits_to_keep > 0 else logits

    def _decode_with_novelty(self, input_ids, attention_mask, position_ids, past_key_values):
        """
        A decoding step that also records what novelty learning needs about it (see `flush_novelty`): the token
        arriving now is the one sampled from the previous step. Nothing changes until the next input arrives, so an
        answer stays self-consistent; then `apply_novelty` steps the session away from it.
        """
        store = side_states(past_key_values)
        steps = store.setdefault("novelty_steps", [])
        if steps and steps[-1]["token"] is None:
            steps[-1]["token"] = input_ids[:, -1:]
        if len(steps) >= 256:  # fold long answers in as they go, to bound the recorded state
            flush_novelty(self, store)
            steps = store.setdefault("novelty_steps", [])
        mlps = decoder_mlps(self)
        layers = self.model.layers[self.config.encoder_layers :]
        for mlp in mlps:
            mlp.record = True
        self.model.record_final = True
        try:
            hidden = self.model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=True,
            ).last_hidden_state
            logits = self.lm_head(hidden[:, -1:])
            probs = logits[:, -1].float().softmax(-1)
            steps.append(
                dict(
                    token=None,
                    expected=probs @ self.lm_head.weight.float(),  # E^T p
                    final=self.model.recorded_final,
                    hidden=[mlp.record for mlp in mlps],
                    post=[layer.record_post for layer in layers],
                )
            )
        finally:
            for mlp in mlps:
                mlp.record = None
            self.model.record_final = False
        return logits

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from Hydrogen_LM import HydrogenForCausalLM
        >>> from transformers import AutoTokenizer

        >>> model = HydrogenForCausalLM.from_pretrained("checkpoints/hydrogen-135m")
        >>> tokenizer = AutoTokenizer.from_pretrained("checkpoints/hydrogen-135m")
        >>> inputs = tokenizer("def fibonacci(n):", return_tensors="pt")
        >>> tokenizer.decode(model.generate(**inputs, max_new_tokens=30)[0])
        ```
        """
        if use_cache and past_key_values is None:
            past_key_values = HydrogenCache(config=self.config)  # a new cache is a new session
        if (
            self.config.session_learning_rate
            and not self.training
            and labels is None
            and input_ids is not None
            and input_ids.shape[1] > 1
            and past_key_values is not None
            and not torch.is_inference_mode_enabled()
        ):
            logits = self._session_learning(input_ids, attention_mask, position_ids, past_key_values, logits_to_keep)
            return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
        if (
            self.config.novelty_rate
            and not self.training
            and labels is None
            and input_ids is not None
            and input_ids.shape[1] == 1
            and past_key_values is not None
            and not torch.is_inference_mode_enabled()
        ):
            logits = self._decode_with_novelty(input_ids, attention_mask, position_ids, past_key_values)
            return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
        if self.config.novelty_rate and past_key_values is not None and not self.training:
            apply_novelty(self, side_states(past_key_values))  # new input: step away from the last answer
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            decoder_tokens=logits_to_keep if isinstance(logits_to_keep, int) and logits_to_keep > 0 else None,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states
        )


class HydrogenForSequenceClassification(GenericForSequenceClassification, HydrogenPreTrainedModel):
    pass


class HydrogenForTokenClassification(GenericForTokenClassification, HydrogenPreTrainedModel):
    pass


def register_hydrogen_auto_classes() -> None:
    """
    Registers Hydrogen with the `Auto*` classes, so `AutoConfig` and `AutoModelForCausalLM` work on Hydrogen checkpoints
    (whose `model_type` is `"hydrogen"`). Safe to call repeatedly.
    """
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )

    try:
        AutoConfig.register("hydrogen", HydrogenConfig)
    except ValueError:
        return  # already registered
    AutoModel.register(HydrogenConfig, HydrogenModel)
    AutoModelForCausalLM.register(HydrogenConfig, HydrogenForCausalLM)
    AutoModelForSequenceClassification.register(HydrogenConfig, HydrogenForSequenceClassification)
    AutoModelForTokenClassification.register(HydrogenConfig, HydrogenForTokenClassification)


# --------------------------------------------------------------------------------------------------------------------
# Perception: vision and audio encoders
#
# Both answer the same question - how does the model perceive more without spending context? A naive patch encoder costs
# one token per patch (576 for a 384px image at 16px patches), and a 10ms-hop spectrogram costs 100 tokens per second.
# Both encoders therefore end in a resampler, or in strided convolutions, so that what reaches the language model is a
# small, predictable number of tokens. They share the primitives below because perception is not causal in either case.
# --------------------------------------------------------------------------------------------------------------------


class EncoderRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon)).to(input_dtype)


def sinusoidal_positions(length: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Positions with no parameters and no length limit, so one checkpoint serves any duration or resolution."""
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / width))
    angles = position * scale
    encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles.cos()[:, : encoding[:, 1::2].shape[-1]]
    return encoding.to(dtype)[None]


class EncoderBlock(nn.Module):
    """Pre-norm bidirectional transformer block: perception has no causal structure, so attention is unmasked."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.attention_norm = EncoderRMSNorm(hidden_size, eps)
        self.mlp_norm = EncoderRMSNorm(hidden_size, eps)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.attention_norm(hidden_states)
        hidden_states = hidden_states + self.attention(normed, normed, normed, need_weights=False)[0]
        normed = self.mlp_norm(hidden_states)
        gate, up = self.gate_up_proj(normed).chunk(2, dim=-1)
        return hidden_states + self.down_proj(up * self.activation(gate))


class Resampler(nn.Module):
    """
    Learned queries cross-attend to the encoded input, fixing how many tokens it occupies.

    This is what keeps perception off the context budget: patches grow with resolution squared and audio frames grow
    with duration, but latents do not. Each latent carries its own embedding, so they can specialize instead of each
    summarizing an arbitrary slice.
    """

    def __init__(self, hidden_size: int, num_latents: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_size) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.latent_norm = EncoderRMSNorm(hidden_size, eps)
        self.input_norm = EncoderRMSNorm(hidden_size, eps)
        self.out_norm = EncoderRMSNorm(hidden_size, eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        latents = self.latent_norm(self.latents).expand(hidden_states.shape[0], -1, -1)
        encoded = self.input_norm(hidden_states)
        return self.out_norm(latents + self.attention(latents, encoded, encoded, need_weights=False)[0])


def encoder_block_flops(tokens: int, width: int, intermediate_size: int) -> float:
    """Matmul FLOPs for one `EncoderBlock`, counting a multiply-add as 2."""
    attention = 2 * 4 * tokens * width * width + 2 * 2 * tokens * tokens * width
    return attention + 2 * 3 * tokens * width * intermediate_size


class HydrogenVisionEncoder(nn.Module):
    """
    Pixels -> a fixed number of language-model tokens.

    `hidden_size` is the *language model's* width; the encoder works in its own `config.hidden_size` and projects once
    at the end, so a small encoder can feed a larger model (or the reverse) without changing either.
    """

    def __init__(self, config: HydrogenVisionConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(3, config.hidden_size, config.patch_size, stride=config.patch_size)
        self.position_embed = nn.Parameter(torch.randn(1, config.num_patches, config.hidden_size) * 0.02)
        self.blocks = nn.ModuleList(
            EncoderBlock(config.hidden_size, config.num_heads, config.intermediate_size, config.rms_norm_eps)
            for _ in range(config.num_layers)
        )
        self.resampler = Resampler(config.hidden_size, config.latent_tokens, config.resampler_heads, config.rms_norm_eps)
        self.projection = nn.Linear(config.hidden_size, hidden_size, bias=False)

    @property
    def tokens_per_image(self) -> int:
        return self.config.latent_tokens

    def flops_per_image(self, patches_per_side: int | None = None) -> float:
        """Matmul FLOPs for one image. Encoder compute is what buys the context saving, so both are reported."""
        config = self.config
        side = patches_per_side or config.patches_per_side
        patches, width = side * side, config.hidden_size
        patch_embed = 2 * patches * width * 3 * config.patch_size**2
        resampler = 2 * 4 * config.latent_tokens * width * width + 2 * 2 * config.latent_tokens * patches * width
        return patch_embed + config.num_layers * encoder_block_flops(patches, width, config.intermediate_size) + resampler

    def interpolate_positions(self, patches_per_side: int) -> torch.Tensor:
        """Position embeddings are resized, so one checkpoint serves several resolutions."""
        if patches_per_side == self.config.patches_per_side:
            return self.position_embed
        grid = self.position_embed.reshape(1, self.config.patches_per_side, self.config.patches_per_side, -1)
        grid = nn.functional.interpolate(
            grid.permute(0, 3, 1, 2), size=(patches_per_side, patches_per_side), mode="bicubic", align_corners=False
        )
        return grid.permute(0, 2, 3, 1).reshape(1, patches_per_side**2, -1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """`pixel_values` is (batch, 3, height, width) in [-1, 1]; returns (batch, latent_tokens, lm_hidden_size)."""
        patches = self.patch_embed(pixel_values)
        patches_per_side = patches.shape[-1]
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches + self.interpolate_positions(patches_per_side).to(patches.dtype)
        for block in self.blocks:
            patches = block(patches)
        return self.projection(self.resampler(patches))


def mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> torch.Tensor:
    """Triangular mel filters, (n_mels, n_fft // 2 + 1). Written in torch so inference needs no audio library."""

    def hz_to_mel(hz):
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    bins = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    edges = mel_to_hz(torch.linspace(hz_to_mel(torch.tensor(0.0)), hz_to_mel(torch.tensor(sample_rate / 2.0)), n_mels + 2))
    lower = (bins[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    upper = (edges[2:, None] - bins[None, :]) / (edges[2:, None] - edges[1:-1, None])
    return torch.clamp(torch.minimum(lower, upper), min=0.0)


class HydrogenAudioEncoder(nn.Module):
    """
    Waveform -> language-model tokens at `config.tokens_per_second` (or a fixed count with `latent_tokens`).

    The mel filterbank is built with plain torch, so a deployed model needs no torchaudio or librosa.
    """

    def __init__(self, config: HydrogenAudioConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.window = nn.Buffer(torch.hann_window(config.n_fft), persistent=False)
        self.filters = nn.Buffer(mel_filterbank(config.n_mels, config.n_fft, config.sample_rate), persistent=False)

        channels, convolutions = config.n_mels, []
        for stride in config.conv_strides:
            convolutions += [nn.Conv1d(channels, config.hidden_size, kernel_size=3, stride=stride, padding=1), nn.SiLU()]
            channels = config.hidden_size
        self.convolutions = nn.Sequential(*convolutions)

        self.blocks = nn.ModuleList(
            EncoderBlock(config.hidden_size, config.num_heads, config.intermediate_size, config.rms_norm_eps)
            for _ in range(config.num_layers)
        )
        self.resampler = (
            Resampler(config.hidden_size, config.latent_tokens, config.resampler_heads, config.rms_norm_eps)
            if config.latent_tokens
            else None
        )
        self.projection = nn.Linear(config.hidden_size, hidden_size, bias=False)

    def tokens_for(self, seconds: float) -> int:
        """How much context a clip of this length will occupy."""
        if self.resampler is not None:
            return self.config.latent_tokens
        frames = int(seconds * self.config.sample_rate) // self.config.hop_length + 1
        return math.ceil(frames / self.config.downsample)

    def flops_per_second(self) -> float:
        """Matmul FLOPs for one second of audio (convolutions, blocks and one resampler pass)."""
        config = self.config
        tokens, width = max(1, int(config.tokens_per_second)), config.hidden_size
        convolutions, channels, rate = 0.0, config.n_mels, config.sample_rate / config.hop_length
        for stride in config.conv_strides:
            rate = rate / stride
            convolutions += 2 * rate * width * channels * 3
            channels = width
        resampler = 0.0
        if self.resampler is not None:
            resampler = 2 * 4 * config.latent_tokens * width * width + 2 * 2 * config.latent_tokens * tokens * width
        return convolutions + config.num_layers * encoder_block_flops(tokens, width, config.intermediate_size) + resampler

    def resample(self, waveform: torch.Tensor, sample_rate: int | None) -> torch.Tensor:
        """
        Bring audio to the rate this encoder was configured for.

        The hop is measured in samples, so a 48kHz file would otherwise produce 300 spectrogram frames per second
        instead of 100 and silently cost three times the context. Linear interpolation keeps the model free of any
        audio library; resample offline with a better filter if fidelity matters.
        """
        if not sample_rate or sample_rate == self.config.sample_rate:
            return waveform
        target = round(waveform.shape[-1] * self.config.sample_rate / sample_rate)
        return nn.functional.interpolate(waveform[:, None], size=target, mode="linear", align_corners=False)[:, 0]

    def log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """(batch, samples) -> (batch, n_mels, frames), normalized the way Whisper does for stable scales."""
        spectrogram = torch.stft(
            waveform,
            self.config.n_fft,
            self.config.hop_length,
            window=self.window.to(waveform.dtype),
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        mel = self.filters.to(spectrogram.real.dtype) @ spectrogram.abs().pow(2)
        log_mel = torch.log10(mel.clamp_min(1e-10))
        log_mel = torch.maximum(log_mel, log_mel.amax(dim=(-2, -1), keepdim=True) - 8.0)
        return (log_mel + 4.0) / 4.0

    def forward(self, waveform: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        """
        `waveform` is (batch, samples); returns (batch, tokens, lm_hidden_size).

        Pass the file's `sample_rate` whenever it may differ from `config.sample_rate` - the audio is resampled so that
        a clip costs the same context no matter how it was recorded.
        """
        features = self.convolutions(self.log_mel(self.resample(waveform, sample_rate))).transpose(1, 2)
        features = features + sinusoidal_positions(features.shape[1], features.shape[-1], features.device, features.dtype)
        for block in self.blocks:
            features = block(features)
        if self.resampler is not None:
            features = self.resampler(features)
        return self.projection(features)



__all__ = [
    "HydrogenPreTrainedModel",
    "HydrogenModel",
    "HydrogenForCausalLM",
    "HydrogenForSequenceClassification",
    "HydrogenForTokenClassification",
    "HydrogenVisionEncoder",
    "HydrogenAudioEncoder",
    "EncoderBlock",
    "Resampler",
    "mel_filterbank",
    "register_hydrogen_auto_classes",
]
