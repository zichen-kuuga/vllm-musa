"""Microbenchmark MUSA csrc JIT kernels against native/torch baselines.

Run on a MUSA host from the vllm-musa repository root:

    python tests/jit_kernel/benchmark_csrc_jit_kernels.py

The numbers are microbenchmarks for the direct kernel wrappers. They are useful
for checking whether the opt-in csrc JIT path is faster for representative
tensor shapes; model-level throughput still needs an end-to-end benchmark.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

# torchada redirects torch.cuda symbols to MUSA. Keep this import first.
import torchada  # noqa: F401

import vllm_musa  # noqa: F401
import vllm_musa._custom_ops  # noqa: F401
from vllm_musa.jit_kernel.csrc.norm import (
    fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
)
from vllm_musa.jit_kernel.csrc.quant import per_token_group_quant_8bit
from vllm_musa.jit_kernel.csrc.topk import topk_sigmoid, topk_softmax

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("VLLM_MUSA_JIT_CACHE_DIR", "/tmp/vllm_musa_bench_jit_cache")
os.environ.setdefault("VLLM_MUSA_ARCH_LIST", "31")


@dataclass(frozen=True)
class BenchCase:
    kernel: str
    label: str
    baseline: str
    iters: int
    baseline_event_us: float
    jit_event_us: float
    baseline_wall_us: float
    jit_wall_us: float

    @property
    def event_speedup(self) -> float:
        if self.jit_event_us == 0:
            return math.nan
        return self.baseline_event_us / self.jit_event_us

    @property
    def wall_speedup(self) -> float:
        if self.jit_wall_us == 0:
            return math.nan
        return self.baseline_wall_us / self.jit_wall_us


def _sync() -> None:
    torch.cuda.synchronize()


def _measure(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    _sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    wall_end = time.perf_counter()
    return (
        start.elapsed_time(end) * 1000.0 / iters,
        (wall_end - wall_start) * 1_000_000.0 / iters,
    )


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> None:
    _sync()
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        max_abs = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"max_abs={max_abs}")


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


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _bench_rmsnorm(
    *,
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> list[BenchCase]:
    torch.manual_seed(123)
    device = torch.device("musa")
    eps = 1e-6
    x = torch.randn((rows, hidden), device=device, dtype=dtype)
    residual = torch.randn((rows, hidden), device=device, dtype=dtype)
    weight = torch.randn((hidden,), device=device, dtype=dtype)

    _assert_close(rmsnorm(x, weight, eps), _rmsnorm_ref(x, weight, eps))
    _assert_close(
        gemma_rmsnorm(x, weight, eps),
        _rmsnorm_ref(x, weight, eps, gemma=True),
    )

    x_jit = x.clone()
    residual_jit = residual.clone()
    fused_add_rmsnorm(x_jit, residual_jit, weight, eps)
    residual_ref = (x.float() + residual.float()).to(dtype)
    _assert_close(x_jit, _rmsnorm_ref(residual_ref, weight, eps))
    _assert_close(residual_jit, residual_ref)

    cases: list[BenchCase] = []
    label = f"rows={rows},hidden={hidden},dtype={dtype}"

    def torch_rms() -> None:
        torch.nn.functional.rms_norm(x, (hidden,), weight, eps)

    def jit_rms() -> None:
        rmsnorm(x, weight, eps)

    baseline_event_us, baseline_wall_us = _measure(
        torch_rms, warmup=warmup, iters=iters
    )
    jit_event_us, jit_wall_us = _measure(jit_rms, warmup=warmup, iters=iters)
    cases.append(
        BenchCase(
            "rmsnorm",
            label,
            "torch.nn.functional.rms_norm",
            iters,
            baseline_event_us,
            jit_event_us,
            baseline_wall_us,
            jit_wall_us,
        )
    )

    def torch_gemma() -> None:
        _rmsnorm_ref(x, weight, eps, gemma=True)

    def jit_gemma() -> None:
        gemma_rmsnorm(x, weight, eps)

    baseline_event_us, baseline_wall_us = _measure(
        torch_gemma, warmup=warmup, iters=iters
    )
    jit_event_us, jit_wall_us = _measure(jit_gemma, warmup=warmup, iters=iters)
    cases.append(
        BenchCase(
            "gemma_rmsnorm",
            label,
            "torch_reference",
            iters,
            baseline_event_us,
            jit_event_us,
            baseline_wall_us,
            jit_wall_us,
        )
    )

    def native_fused_add() -> None:
        x_buf = x.clone()
        residual_buf = residual.clone()
        torch.ops._C_musa_ops.musa_fused_add_rms_norm(
            x_buf,
            residual_buf,
            weight,
            eps,
        )

    def jit_fused_add() -> None:
        x_buf = x.clone()
        residual_buf = residual.clone()
        fused_add_rmsnorm(x_buf, residual_buf, weight, eps)

    if hasattr(torch.ops, "_C_musa_ops") and hasattr(
        torch.ops._C_musa_ops, "musa_fused_add_rms_norm"
    ):
        baseline_event_us, baseline_wall_us = _measure(
            native_fused_add, warmup=warmup, iters=iters
        )
        jit_event_us, jit_wall_us = _measure(jit_fused_add, warmup=warmup, iters=iters)
        cases.append(
            BenchCase(
                "fused_add_rmsnorm",
                label,
                "_C_musa_ops.musa_fused_add_rms_norm",
                iters,
                baseline_event_us,
                jit_event_us,
                baseline_wall_us,
                jit_wall_us,
            )
        )

    return cases


def _bench_quant(
    *,
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> list[BenchCase]:
    if not hasattr(torch.ops, "_C_musa_ops") or not hasattr(
        torch.ops._C_musa_ops, "per_token_group_fp8_quant"
    ):
        return []

    torch.manual_seed(456)
    device = torch.device("musa")
    group_size = 128
    eps = 1e-10
    fp8_min = -448.0
    fp8_max = 448.0
    x = torch.randn((rows, hidden), device=device, dtype=dtype)
    q_jit = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s_jit = torch.empty(
        (rows, hidden // group_size), device=device, dtype=torch.float32
    )
    q_native = torch.empty_like(q_jit)
    s_native = torch.empty_like(s_jit)

    per_token_group_quant_8bit(x, q_jit, s_jit, group_size, eps, fp8_min, fp8_max)
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
    _assert_close(s_jit, s_native, atol=1e-6, rtol=1e-5)
    if not torch.equal(q_jit.view(torch.uint8).cpu(), q_native.view(torch.uint8).cpu()):
        raise AssertionError("quant FP8 bytes differ")

    def native_quant() -> None:
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

    def jit_quant() -> None:
        per_token_group_quant_8bit(
            x,
            q_jit,
            s_jit,
            group_size,
            eps,
            fp8_min,
            fp8_max,
        )

    baseline_event_us, baseline_wall_us = _measure(
        native_quant, warmup=warmup, iters=iters
    )
    jit_event_us, jit_wall_us = _measure(jit_quant, warmup=warmup, iters=iters)
    return [
        BenchCase(
            "per_token_group_quant_fp8",
            f"rows={rows},hidden={hidden},group=128,dtype={dtype}",
            "_C_musa_ops.per_token_group_fp8_quant",
            iters,
            baseline_event_us,
            jit_event_us,
            baseline_wall_us,
            jit_wall_us,
        )
    ]


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


def _topk_inputs(
    rows: int,
    experts: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("musa")
    values = torch.linspace(-4.0, 4.0, experts, device=device, dtype=torch.float32)
    perm = torch.randperm(experts, device=device)
    base = torch.empty((experts,), device=device, dtype=torch.float32)
    base[perm] = values
    row_offsets = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    gating = (base.unsqueeze(0) + row_offsets * 0.001).to(dtype)
    bias = torch.randn((experts,), device=device, dtype=torch.float32) * 0.01
    return gating, bias


def _bench_topk(
    *,
    rows: int,
    experts: int,
    topk: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> list[BenchCase]:
    torch.manual_seed(789)
    device = torch.device("musa")
    gating, bias = _topk_inputs(rows, experts, dtype)
    weights = torch.empty((rows, topk), device=device, dtype=torch.float32)
    ids = torch.empty((rows, topk), device=device, dtype=torch.int32)
    label = f"rows={rows},experts={experts},topk={topk},dtype={dtype}"

    topk_softmax(weights, ids, gating, renormalize=True)
    ref_weights, ref_ids = _topk_softmax_ref(gating, topk, True)
    _assert_close(weights, ref_weights, atol=3e-3, rtol=3e-3)
    if not torch.equal(ids.cpu(), ref_ids.cpu()):
        raise AssertionError("topk_softmax ids differ")

    def torch_softmax() -> None:
        _topk_softmax_ref(gating, topk, True)

    def jit_softmax() -> None:
        topk_softmax(weights, ids, gating, renormalize=True)

    baseline_event_us, baseline_wall_us = _measure(
        torch_softmax, warmup=warmup, iters=iters
    )
    jit_event_us, jit_wall_us = _measure(jit_softmax, warmup=warmup, iters=iters)
    cases = [
        BenchCase(
            "topk_softmax",
            label,
            "torch.softmax+topk",
            iters,
            baseline_event_us,
            jit_event_us,
            baseline_wall_us,
            jit_wall_us,
        )
    ]

    topk_sigmoid(weights, ids, gating, renormalize=True, correction_bias=bias)
    ref_weights, ref_ids = _topk_sigmoid_ref(gating, topk, True, bias)
    _assert_close(weights, ref_weights, atol=3e-3, rtol=3e-3)
    if not torch.equal(ids.cpu(), ref_ids.cpu()):
        raise AssertionError("topk_sigmoid ids differ")

    def torch_sigmoid_bias() -> None:
        _topk_sigmoid_ref(gating, topk, True, bias)

    def jit_sigmoid_bias() -> None:
        topk_sigmoid(weights, ids, gating, renormalize=True, correction_bias=bias)

    baseline_event_us, baseline_wall_us = _measure(
        torch_sigmoid_bias, warmup=warmup, iters=iters
    )
    jit_event_us, jit_wall_us = _measure(jit_sigmoid_bias, warmup=warmup, iters=iters)
    cases.append(
        BenchCase(
            "topk_sigmoid_bias",
            label,
            "torch.sigmoid+topk",
            iters,
            baseline_event_us,
            jit_event_us,
            baseline_wall_us,
            jit_wall_us,
        )
    )
    return cases


def _print_results(results: list[BenchCase]) -> None:
    print(
        "kernel                      label                                      "
        "baseline                         iters  baseline_event_us  jit_event_us  "
        "event_speedup  baseline_wall_us  jit_wall_us  wall_speedup"
    )
    for case in results:
        print(
            f"{case.kernel:<27} {case.label:<42} {case.baseline:<32} "
            f"{case.iters:>5} {case.baseline_event_us:>18.3f} "
            f"{case.jit_event_us:>12.3f} {case.event_speedup:>14.2f} "
            f"{case.baseline_wall_us:>17.3f} {case.jit_wall_us:>11.3f} "
            f"{case.wall_speedup:>12.2f}"
        )
        print(
            "BENCH "
            f"kernel={case.kernel} label={case.label.replace(' ', '_')} "
            f"baseline={case.baseline.replace(' ', '_')} iters={case.iters} "
            f"baseline_event_us={case.baseline_event_us:.3f} "
            f"jit_event_us={case.jit_event_us:.3f} "
            f"event_speedup={case.event_speedup:.3f} "
            f"baseline_wall_us={case.baseline_wall_us:.3f} "
            f"jit_wall_us={case.jit_wall_us:.3f} "
            f"wall_speedup={case.wall_speedup:.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark MUSA csrc JIT kernels against native/torch baselines."
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        print("FAIL no MUSA device available")
        return 1
    torch.cuda.set_device("cuda:0")
    dtype = _dtype_from_name(args.dtype)

    results: list[BenchCase] = []
    results.extend(
        _bench_rmsnorm(
            rows=args.rows,
            hidden=args.hidden,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
    )
    results.extend(
        _bench_quant(
            rows=args.rows,
            hidden=args.hidden,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
    )
    results.extend(
        _bench_topk(
            rows=args.rows,
            experts=args.experts,
            topk=args.topk,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
    )
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
