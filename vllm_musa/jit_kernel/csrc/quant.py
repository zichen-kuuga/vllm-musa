from __future__ import annotations

import torch

from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit
from vllm_musa.jit_kernel.utils import cache_once


@cache_once
def _quant_v2_module():
    return load_musa_jit(
        "vllm_musa_quant_v2",
        ("quant/per_token_group_quant_8bit_v2.mu",),
    )


def per_token_group_quant_8bit_v2(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: torch.Tensor | None = None,
) -> None:
    if masked_m is None:
        masked_m = torch.empty((1,), device=input.device, dtype=torch.int32)
        has_masked_m = False
    else:
        has_masked_m = True
    _quant_v2_module().sgl_per_token_group_quant_8bit_v2(
        input,
        output_q,
        output_s,
        int(group_size),
        float(eps),
        float(min_8bit),
        float(max_8bit),
        bool(scale_ue8m0),
        bool(fuse_silu_and_mul),
        masked_m,
        bool(has_masked_m),
    )


def _per_token_group_quant_8bit_custom(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    has_masked_m: bool = False,
) -> None:
    _quant_v2_module().sgl_per_token_group_quant_8bit_v2(
        input,
        output_q,
        output_s,
        int(group_size),
        float(eps),
        float(min_8bit),
        float(max_8bit),
        bool(scale_ue8m0),
        bool(fuse_silu_and_mul),
        masked_m,
        bool(has_masked_m),
    )


def _per_token_group_quant_8bit_custom_fake(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    has_masked_m: bool = False,
) -> None:
    return


direct_register_custom_op(
    op_name="musa_per_token_group_quant_8bit",
    op_func=_per_token_group_quant_8bit_custom,
    mutates_args=["output_q", "output_s"],
    fake_impl=_per_token_group_quant_8bit_custom_fake,
)


def per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: torch.Tensor | None = None,
    enable_v2: bool | None = None,
) -> None:
    if enable_v2 is False:
        raise ValueError("MUSA csrc quant only supports the v2 kernel path.")
    if masked_m is None:
        masked_m = torch.empty((1,), device=input.device, dtype=torch.int32)
        has_masked_m = False
    else:
        has_masked_m = True
    torch.ops.vllm.musa_per_token_group_quant_8bit(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        min_8bit,
        max_8bit,
        masked_m,
        scale_ue8m0,
        fuse_silu_and_mul,
        has_masked_m,
    )
