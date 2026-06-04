# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from vllm.model_executor.layers.layernorm import RMSNorm

try:
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
except ImportError:
    GemmaRMSNorm = None

try:
    from vllm.model_executor.layers.layernorm import fused_add_rms_norm
except ImportError:
    fused_add_rms_norm = None

from vllm_musa import _custom_ops as musa_ops
from vllm_musa.utils.environ import envs


def _can_use_musa_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    hidden_size = x.shape[-1]
    return (
        envs.VLLM_MUSA_FUSED_ADD_RMSNORM.get()
        and x.device.type == "musa"
        and residual.device.type == "musa"
        and weight.device.type == "musa"
        and x.dim() == 2
        and residual.dim() == 2
        and weight.dim() == 1
        and x.shape == residual.shape
        and hidden_size == weight.numel()
        and hidden_size % 8 == 0
        and hidden_size <= 16384
        and x.dtype in (torch.float16, torch.bfloat16)
        and residual.dtype == x.dtype
        and weight.dtype == x.dtype
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and hasattr(torch.ops, "_C_musa_ops")
        and hasattr(torch.ops._C_musa_ops, "musa_fused_add_rms_norm")
    )


def _can_use_musa_jit_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    hidden_size = x.shape[-1]
    return (
        x.device.type == "musa"
        and weight.device.type == "musa"
        and x.dim() == 2
        and weight.dim() == 1
        and hidden_size > 0
        and hidden_size == weight.numel()
        and hidden_size <= 32768
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
        and x.is_contiguous()
        and weight.is_contiguous()
    )


def _can_use_musa_jit_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    return (
        _can_use_musa_jit_rmsnorm(x, weight)
        and residual.device.type == "musa"
        and residual.dim() == 2
        and residual.shape == x.shape
        and residual.dtype == x.dtype
        and residual.is_contiguous()
    )


def _maybe_import_musa_jit_norm():
    try:
        from vllm_musa.jit_kernel.csrc import norm as musa_jit_norm
    except (ImportError, ModuleNotFoundError):
        return None
    return musa_jit_norm


@RMSNorm.register_oot
class MusaRMSNorm(RMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get():
            return self.forward_native(x, residual)

        # ==================== MUSA ADAPTATION ====================
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        add_residual = residual is not None
        if add_residual:
            if (
                envs.VLLM_MUSA_ENABLE_JIT_RMSNORM.get()
                and _can_use_musa_jit_fused_add_rmsnorm(
                    x, residual, self.weight.data
                )
            ):
                musa_jit_norm = _maybe_import_musa_jit_norm()
                if musa_jit_norm is not None:
                    musa_jit_norm.fused_add_rmsnorm(
                        x, residual, self.weight.data, self.variance_epsilon
                    )
                    return x, residual

            if _can_use_musa_fused_add_rms_norm(x, residual, self.weight.data):
                musa_ops.musa_fused_add_rms_norm(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            if fused_add_rms_norm is not None:
                return fused_add_rms_norm(
                    x, residual, self.weight.data, self.variance_epsilon
                )
            return self.forward_native(x, residual)
        else:
            if (
                envs.VLLM_MUSA_ENABLE_JIT_RMSNORM.get()
                and _can_use_musa_jit_rmsnorm(x, self.weight.data)
            ):
                musa_jit_norm = _maybe_import_musa_jit_norm()
                if musa_jit_norm is not None:
                    return musa_jit_norm.rmsnorm(
                        x, self.weight.data, self.variance_epsilon
                    )

            out = nn.functional.rms_norm(
                x, (self.hidden_size,), self.weight.data, self.variance_epsilon
            )
            return out
        # ========================== END ==========================


if GemmaRMSNorm is not None:

    @GemmaRMSNorm.register_oot
    class MusaGemmaRMSNorm(GemmaRMSNorm):
        def forward_oot(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            if (
                envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get()
                or getattr(self, "variance_size_override", None) is not None
            ):
                return self.forward_native(x, residual)

            if residual is not None:
                if (
                    envs.VLLM_MUSA_ENABLE_JIT_GEMMA_RMSNORM.get()
                    and _can_use_musa_jit_fused_add_rmsnorm(
                        x, residual, self.weight.data
                    )
                ):
                    musa_jit_norm = _maybe_import_musa_jit_norm()
                    if musa_jit_norm is not None:
                        musa_jit_norm.gemma_fused_add_rmsnorm(
                            x, residual, self.weight.data, self.variance_epsilon
                        )
                        return x, residual
                return self.forward_native(x, residual)

            if (
                envs.VLLM_MUSA_ENABLE_JIT_GEMMA_RMSNORM.get()
                and _can_use_musa_jit_rmsnorm(x, self.weight.data)
            ):
                musa_jit_norm = _maybe_import_musa_jit_norm()
                if musa_jit_norm is not None:
                    return musa_jit_norm.gemma_rmsnorm(
                        x, self.weight.data, self.variance_epsilon
                    )

            return self.forward_native(x, residual)
