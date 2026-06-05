"""Tests for MUSA csrc JIT kernels ported from SGLang."""

import os

import pytest
import torch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("VLLM_MUSA_JIT_CACHE_DIR", "/tmp/vllm_musa_pytest_jit_cache")
os.environ.setdefault("VLLM_MUSA_ARCH_LIST", "31")


def _require_musa() -> None:
    pytest.importorskip("torch_musa")
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.musa.set_device(0)


@pytest.fixture(scope="module", autouse=True)
def _musa_device():
    _require_musa()


def _sync() -> None:
    torch.musa.synchronize()


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> None:
    _sync()
    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol), (
        (actual.float() - expected.float()).abs().max().item()
    )


def _rmsnorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    gemma: bool = False,
) -> torch.Tensor:
    scale_weight = weight.float() + (1.0 if gemma else 0.0)
    normalized = x.float() * torch.rsqrt(
        x.float().pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * scale_weight).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_csrc_rmsnorm_kernels_match_reference(dtype: torch.dtype) -> None:
    from vllm_musa.jit_kernel.csrc.norm import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )

    torch.manual_seed(123)
    device = torch.device("musa")
    eps = 1e-6
    x = torch.randn((5, 128), device=device, dtype=dtype)
    weight = torch.randn((128,), device=device, dtype=dtype)

    _assert_close(rmsnorm(x, weight, eps), _rmsnorm_ref(x, weight, eps))
    _assert_close(
        gemma_rmsnorm(x, weight, eps),
        _rmsnorm_ref(x, weight, eps, gemma=True),
    )

    x0 = torch.randn((5, 128), device=device, dtype=dtype)
    residual0 = torch.randn((5, 128), device=device, dtype=dtype)
    residual_ref = (x0.float() + residual0.float()).to(dtype)

    x_jit = x0.clone()
    residual_jit = residual0.clone()
    fused_add_rmsnorm(x_jit, residual_jit, weight, eps)
    _assert_close(x_jit, _rmsnorm_ref(residual_ref, weight, eps))
    _assert_close(residual_jit, residual_ref)

    x_jit = x0.clone()
    residual_jit = residual0.clone()
    gemma_fused_add_rmsnorm(x_jit, residual_jit, weight, eps)
    _assert_close(x_jit, _rmsnorm_ref(residual_ref, weight, eps, gemma=True))
    _assert_close(residual_jit, residual_ref)


def test_csrc_per_token_group_quant_fp8_matches_native_op() -> None:
    import vllm_musa._custom_ops  # noqa: F401
    from vllm_musa.jit_kernel.csrc.quant import per_token_group_quant_8bit

    torch.manual_seed(456)
    device = torch.device("musa")
    group_size = 128
    eps = 1e-10
    fp8_min = -448.0
    fp8_max = 448.0
    x = torch.randn((6, 256), device=device, dtype=torch.bfloat16)

    q_jit = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s_jit = torch.empty(
        (x.shape[0], x.shape[1] // group_size),
        device=device,
        dtype=torch.float32,
    )
    per_token_group_quant_8bit(x, q_jit, s_jit, group_size, eps, fp8_min, fp8_max)

    if not hasattr(torch.ops, "_C_musa_ops") or not hasattr(
        torch.ops._C_musa_ops, "per_token_group_fp8_quant"
    ):
        pytest.skip("native MUSA per-token-group FP8 quant op is not registered")

    q_native = torch.empty_like(q_jit)
    s_native = torch.empty_like(s_jit)
    torch.ops._C_musa_ops.per_token_group_fp8_quant(
        x,
        q_native,
        s_native,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        False,
        False,
        False,
    )
    _sync()

    _assert_close(s_jit, s_native, atol=1e-6, rtol=1e-5)
    assert torch.equal(q_jit.view(torch.uint8).cpu(), q_native.view(torch.uint8).cpu())


def _topk_softmax_ref(
    gating: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = gating.float()
    if correction_bias is not None:
        logits = logits + correction_bias.float().unsqueeze(0)
    scores = torch.softmax(logits, dim=-1)
    _, ids = torch.topk(logits, topk, dim=-1)
    values = scores.gather(1, ids)
    if renormalize:
        values = values / values.sum(dim=-1, keepdim=True)
    return values.float(), ids.int()


def _topk_sigmoid_ref(
    gating: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(gating.float())
    choice = scores
    if correction_bias is not None:
        choice = choice + correction_bias.float().unsqueeze(0)
    _, ids = torch.topk(choice, topk, dim=-1)
    values = scores.gather(1, ids)
    if renormalize:
        values = values / values.sum(dim=-1, keepdim=True)
    return values.float(), ids.int()


def _assert_ids_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    _sync()
    assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("renormalize", [False, True])
def test_csrc_topk_kernels_match_reference(
    dtype: torch.dtype,
    renormalize: bool,
) -> None:
    from vllm_musa.jit_kernel.csrc.topk import topk_sigmoid, topk_softmax

    torch.manual_seed(789)
    device = torch.device("musa")
    rows = 7
    experts = 128
    topk = 8
    values = torch.linspace(-4.0, 4.0, experts, device=device, dtype=torch.float32)
    perm = torch.randperm(experts, device=device)
    base = torch.empty((experts,), device=device, dtype=torch.float32)
    base[perm] = values
    row_offsets = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    gating = (base.unsqueeze(0) + row_offsets * 0.001).to(dtype)
    bias = torch.randn((experts,), device=device, dtype=torch.float32) * 0.01

    weights = torch.empty((rows, topk), device=device, dtype=torch.float32)
    ids = torch.empty((rows, topk), device=device, dtype=torch.int32)
    topk_softmax(weights, ids, gating, renormalize)
    ref_weights, ref_ids = _topk_softmax_ref(gating, topk, renormalize)
    _assert_ids_equal(ids, ref_ids)
    _assert_close(weights, ref_weights, atol=3e-3, rtol=3e-3)

    weights = torch.empty((rows, topk), device=device, dtype=torch.float32)
    ids = torch.empty((rows, topk), device=device, dtype=torch.int32)
    topk_softmax(weights, ids, gating, renormalize, correction_bias=bias)
    ref_weights, ref_ids = _topk_softmax_ref(gating, topk, renormalize, bias)
    _assert_ids_equal(ids, ref_ids)
    _assert_close(weights, ref_weights, atol=3e-3, rtol=3e-3)

    weights = torch.empty((rows, topk), device=device, dtype=torch.float32)
    ids = torch.empty((rows, topk), device=device, dtype=torch.int32)
    topk_sigmoid(weights, ids, gating, renormalize, correction_bias=bias)
    ref_weights, ref_ids = _topk_sigmoid_ref(gating, topk, renormalize, bias)
    _assert_ids_equal(ids, ref_ids)
    _assert_close(weights, ref_weights, atol=3e-3, rtol=3e-3)


def test_csrc_jit_integration_imports() -> None:
    import vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router as topk_router
    import vllm_musa.model_executor.layers.layernorm as layernorm
    import vllm_musa.model_executor.layers.quantization.utils.fp8_utils as fp8_utils

    assert hasattr(layernorm, "MusaRMSNorm")
    assert hasattr(fp8_utils, "_can_use_musa_jit_per_token_group_quant_fp8")
    assert hasattr(topk_router, "_musa_jit_fused_topk")
