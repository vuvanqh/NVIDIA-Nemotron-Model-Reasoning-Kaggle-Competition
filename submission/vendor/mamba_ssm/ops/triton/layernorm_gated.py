"""Small PyTorch fallback for the mamba_ssm gated RMSNorm symbol.

The downloaded Nemotron H Transformers code unconditionally imports
`mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn`, even when its CUDA fast path
is unavailable. This fallback supports the CPU/slow-path call shape used by the
local scaffold. It is not a replacement for the full mamba-ssm package.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _rms_norm_grouped(x: torch.Tensor, eps: float, group_size: int | None) -> torch.Tensor:
    input_dtype = x.dtype
    values = x.to(torch.float32)

    if group_size is not None and group_size > 0 and values.shape[-1] % group_size == 0:
        original_shape = values.shape
        values = values.reshape(*values.shape[:-1], values.shape[-1] // group_size, group_size)
        variance = values.pow(2).mean(dim=-1, keepdim=True)
        values = values * torch.rsqrt(variance + eps)
        values = values.reshape(original_shape)
    else:
        variance = values.pow(2).mean(dim=-1, keepdim=True)
        values = values * torch.rsqrt(variance + eps)

    return values.to(input_dtype)


def rmsnorm_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    **_kwargs: object,
) -> torch.Tensor:
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)

    output = _rms_norm_grouped(x, eps=eps, group_size=group_size)
    output = output * weight.to(dtype=output.dtype, device=output.device)

    if bias is not None:
        output = output + bias.to(dtype=output.dtype, device=output.device)

    if z is not None and norm_before_gate:
        output = output * F.silu(z)

    return output

