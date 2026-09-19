import math
from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


# Deep-and-thin shapes with tied embeddings (49152-token vocabulary). On the research bench (test/RESEARCH.md) the
# Hydrogen stack matched a plain Transformer with ~4x its parameters at a short training budget and ~1.3x after 9M
# tokens, while extrapolating to 4x its training context with almost no loss increase.
HYDROGEN_PRESETS = {
    "hydrogen-25m": dict(hidden_size=256, intermediate_size=768, num_hidden_layers=12, num_attention_heads=4, num_key_value_heads=2, head_dim=64),
    "hydrogen-50m": dict(hidden_size=384, intermediate_size=1024, num_hidden_layers=16, num_attention_heads=6, num_key_value_heads=2, head_dim=64),
    "hydrogen-135m": dict(hidden_size=576, intermediate_size=1536, num_hidden_layers=27, num_attention_heads=9, num_key_value_heads=3, head_dim=64),
    "hydrogen-350m": dict(hidden_size=960, intermediate_size=2560, num_hidden_layers=27, num_attention_heads=15, num_key_value_heads=5, head_dim=64),
    # Scale-out presets: fine-grained mixture of experts with a shared expert, for multi-accelerator training.
    "hydrogen-3b-a0.7b": dict(
        hidden_size=1536, intermediate_size=4096, num_hidden_layers=30, num_attention_heads=12, num_key_value_heads=4,
        head_dim=128, num_experts=40, num_experts_per_tok=4, moe_intermediate_size=512, first_dense_layers=2,
    ),
    "hydrogen-20b-a3b": dict(
        hidden_size=2560, intermediate_size=7168, num_hidden_layers=42, num_attention_heads=20, num_key_value_heads=4,
        head_dim=128, num_experts=80, num_experts_per_tok=6, moe_intermediate_size=768, first_dense_layers=3,
        max_position_embeddings=262144,
    ),
}


@strict
class HydrogenConfig(PreTrainedConfig):
    r"""
    Configuration for Hydrogen models. The defaults describe `hydrogen-135m`; use [`HydrogenConfig.from_preset`] for other
    sizes. The architecture is fixed - every part of it won its place on the research bench - so the fields are sizes,
    not switches:

    - A causal encoder-decoder: the lower half of the layers encodes; every upper-half layer is attention whose keys,
      values and forget gates come from the encoder output. Prefill therefore runs the decoder half on only the last few
      prompt tokens (exactly), roughly halving its cost.
    - The encoder alternates two gated short-convolution layers with one attention layer ("CCA"), so only a third of the
      encoder keeps a KV cache.
    - Forgetting attention (a learned per-head forget gate on the logits, heads initialized from fast to slow forgetting),
      QK-norm with a learned temperature, and a query-dependent output gate.
    - Canon layers: causal depthwise convolutions before the mixer, on the attention inputs, before the MLP and inside it.
    - Hyper-connections: `hc_streams` residual streams mixed by Sinkhorn-projected matrices, with separate stream reads for
      the attention query, key and value.

    Args:
        hc_streams (`int`): Width of the hyper-connection residual.
        num_experts (`int`): Routed experts per MoE layer (0 = dense MLPs). Sigmoid top-k routing with an
            auxiliary-loss-free balancing bias, plus one always-on shared expert of `moe_intermediate_size`.
        num_experts_per_tok (`int`), moe_intermediate_size (`int`), first_dense_layers (`int`),
        router_bias_update_rate (`float`): MoE shape and balancing-bias step.
        session_learning_rate (`float`): Step size of session learning (0 disables it). With a cache, every input the
            model reads (prompt, tool result, user turn) is predicted in 64-token chunks and then learned from with one
            gradient step on the session's own corrections to the decoder MLP output projections, kept in the cache and
            dropped with it. On the bench: -0.06 nats on 1K-token documents (more on a less trained model), growing
            the longer a session runs, for ~3x the compute on *input* tokens; decoding speed is unchanged. Set 0 for
            throughput-bound serving. Tuned at the bench scale (0.3-0.5 best, 1.0 diverges); re-tune with
            test/session_eval.py for new sizes.

    Example:

    ```python
    >>> from Hydrogen_LM import HydrogenConfig, HydrogenForCausalLM
    >>> model = HydrogenForCausalLM(HydrogenConfig.from_preset("hydrogen-25m"))
    ```"""

    model_type = "hydrogen"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_up_proj": "colwise_gather_output",
        "layers.*.mlp.down_proj": "rowwise_split_input",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 27
    num_attention_heads: int = 9
    num_key_value_heads: int | None = 3
    head_dim: int | None = 64
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = 0
    bos_token_id: int | None = 0

    hc_streams: int = 4
    num_experts: int = 0
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    first_dense_layers: int = 1
    router_bias_update_rate: float | int = 1e-3
    session_learning_rate: float | int = 0.5

    layer_types: list[str] | None = None  # derived: the cache layout transformers builds for generation

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        if self.num_experts and not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if self.hc_streams < 1:
            raise ValueError("hc_streams must be >= 1")
        if self.num_hidden_layers < 2:
            raise ValueError("Hydrogen needs at least one encoder and one decoder layer")
        # Every layer keeps conv states (Canon); attention layers also keep a KV cache ("hybrid").
        self.layer_types = ["hybrid" if m == "attention" else "conv" for m in self.mixers]
        self.number_of_conv_states = 5  # canon A, mixer / canon B, canon C, canon D, a decoder's encoder-k/v canon
        super().__post_init__(**kwargs)

    @property
    def encoder_layers(self) -> int:
        return self.num_hidden_layers // 2

    @property
    def mixers(self) -> list[str]:
        """Per layer, `conv` or `attention`: the encoder repeats conv, conv, attention; the decoder is all attention."""
        encoder = ["attention" if i % 3 == 2 else "conv" for i in range(self.encoder_layers)]
        return encoder + ["attention"] * (self.num_hidden_layers - self.encoder_layers)

    def is_moe_layer(self, layer_idx: int) -> bool:
        return bool(self.num_experts) and layer_idx >= self.first_dense_layers

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "HydrogenConfig":
        if name not in HYDROGEN_PRESETS:
            raise ValueError(f"Unknown preset {name!r}; choose from {sorted(HYDROGEN_PRESETS)}")
        return cls(**{**HYDROGEN_PRESETS[name], **overrides})


@dataclass
class HydrogenVisionConfig:
    """
    Vision encoder shape. The field that matters for deployment is `latent_tokens`: an image costs that many tokens of
    the language model's context whatever its resolution, so raising the resolution buys detail with encoder FLOPs
    instead of with context.
    """

    image_size: int = 224
    patch_size: int = 16
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    intermediate_size: int = 1024
    latent_tokens: int = 64
    resampler_heads: int = 6
    rms_norm_eps: float = 1e-6

    @property
    def patches_per_side(self) -> int:
        if self.image_size % self.patch_size:
            raise ValueError(f"image_size {self.image_size} is not divisible by patch_size {self.patch_size}")
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patches_per_side**2


@dataclass
class HydrogenAudioConfig:
    """
    Audio encoder shape. A 10ms hop gives 100 spectrogram frames per second; the strided convolutions reduce that to
    `tokens_per_second` (12.5 by default), and `latent_tokens > 0` adds a resampler for a fixed budget per clip.
    """

    sample_rate: int = 16000
    n_fft: int = 400
    """25ms analysis window at 16kHz."""
    hop_length: int = 160
    """10ms hop, so the spectrogram has 100 frames per second before downsampling."""
    n_mels: int = 80
    conv_strides: tuple[int, ...] = (2, 2, 2)
    """Each convolution halves the frame rate: three of them turn 100 frames/s into 12.5 tokens/s."""
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    intermediate_size: int = 1024
    latent_tokens: int = 0
    """0 keeps one token per downsampled frame (length grows with duration); >0 fixes the budget per clip."""
    resampler_heads: int = 6
    rms_norm_eps: float = 1e-6

    @property
    def downsample(self) -> int:
        return math.prod(self.conv_strides)

    @property
    def tokens_per_second(self) -> float:
        return self.sample_rate / self.hop_length / self.downsample


__all__ = ["HydrogenConfig", "HYDROGEN_PRESETS", "HydrogenVisionConfig", "HydrogenAudioConfig"]
