import math

import torch
from torch import nn


def newton_schulz(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately orthogonalizes a matrix, or a batch of matrices (quintic Newton-Schulz iteration from Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.float()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.mT if tall else x).to(grad.dtype)


class Muon(torch.optim.Optimizer):
    """
    MomentUm Orthogonalized by Newton-Schulz, for hidden weight matrices. Each group may set `splits` to orthogonalize a
    matrix as that many equal row blocks: fused projections (`gate_up_proj`: 2) and per-head Q/K projections (one
    block per head, so each head gets its own preconditioner). 3D parameters (stacked experts) are orthogonalized per
    expert.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, splits=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay, splits=splits)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buffer = state["momentum_buffer"]
                buffer.lerp_(p.grad, 1 - momentum)
                grad = p.grad.lerp(buffer, momentum) if group["nesterov"] else buffer
                matrices = grad.reshape(-1, grad.size(-2), grad.size(-1)) if grad.ndim == 3 else grad.reshape(1, grad.size(0), -1)
                blocks = matrices.chunk(group["splits"], dim=-2)
                update = torch.cat([newton_schulz(block, group["ns_steps"]) for block in blocks], dim=-2).view_as(p)
                rows, cols = blocks[0].shape[-2:]
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * max(1.0, rows / cols) ** 0.5)
        return loss


NO_DECAY_NAMES = ("A_log", "dt_bias", "log_temperature", "gain", "expert_bias")


def build_optimizers(
    model: nn.Module,
    lr: float = 3e-3,
    weight_decay: float = 0.1,
    optimizer: str = "adamw",
    muon_lr: float = 0.02,
    betas: tuple[float, float] = (0.9, 0.95),
) -> list[torch.optim.Optimizer]:
    """
    Returns the optimizers for a Hydrogen model. With `optimizer="muon"`, weight matrices inside the decoder layers use
    Muon (head-wise for Q/K, per expert for MoE) and everything else - embeddings, norms, biases, convolution kernels,
    decay rates and learned scales - uses AdamW. Every group records its `base_lr` so schedules can be applied with
    [`set_lr_factor`].
    """
    config = getattr(model, "config", None)
    q_heads = getattr(config, "num_attention_heads", 1)
    kv_heads = getattr(config, "num_key_value_heads", 1) or q_heads
    decay, no_decay = [], []
    muon_groups: dict[int, list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_layers = ".layers." in f".{name}"
        leaf = name.rsplit(".", 1)[-1]
        matrix = p.ndim == 2 and min(p.shape) > 1 or name.endswith(("experts_gate_up", "experts_down"))
        if optimizer == "muon" and in_layers and matrix and not any(k in name for k in ("router", "hc_proj", "forget_proj")):
            if name.endswith(("gate_up_proj.weight", "experts_gate_up")):
                splits = 2
            elif name.endswith("self_attn.q_proj.weight"):
                splits = q_heads
            elif name.endswith("self_attn.k_proj.weight"):
                splits = kv_heads
            else:
                splits = 1
            muon_groups.setdefault(splits, []).append(p)
        elif p.ndim >= 2 and leaf not in NO_DECAY_NAMES:
            decay.append(p)  # includes convolution kernels: decay pulls a Canon layer back towards the identity
        else:
            no_decay.append(p)

    groups = [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]
    optimizers = [torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, betas=betas, eps=1e-8)]
    if muon_groups:
        optimizers.append(
            Muon([dict(params=ps, splits=s) for s, ps in muon_groups.items()], lr=muon_lr, weight_decay=weight_decay)
        )
    for opt in optimizers:
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]
    return optimizers


def lr_factor(step: int, total_steps: int, warmup_steps: int, schedule: str = "cosine", min_factor: float = 0.1) -> float:
    """Linear warmup, then `cosine` decay or `wsd` (warmup-stable-decay: flat, then linear decay over the last 20%)."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    if schedule == "wsd":
        decay_start = 0.8
        return 1.0 if progress < decay_start else 1.0 - (1.0 - min_factor) * (progress - decay_start) / (1 - decay_start)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr_factor(optimizers: list[torch.optim.Optimizer], factor: float) -> None:
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * factor


__all__ = ["Muon", "build_optimizers", "lr_factor", "set_lr_factor", "newton_schulz"]
