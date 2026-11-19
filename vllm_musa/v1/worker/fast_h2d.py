# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse pinned staging and destination buffers for per-step metadata uploads.

vLLM's ``async_tensor_h2d`` re-stages every numpy input into a fresh pinned
buffer and, for the ``device=`` form, allocates a new device tensor on every
call.  Measured on MTT S5000 that costs 27-33 us of CPU time per call for a
256 B / 516 B metadata upload (117 us/step for ``idx_mapping`` under load),
while a reused pinned buffer plus a reused destination is ~10 us.  The copy
itself is required: a pageable ``copy_(non_blocking=True)`` blocks the CPU for
323 us while the device is busy.

This module installs a shim that keeps the pinned path and removes the
re-staging cost:

  * one pinned staging tensor per (numel, dtype), reused across calls;
  * one device destination per (numel, dtype, device) for the ``device=`` form,
    so no device allocation happens per step;
  * when the caller owns the destination (``out=``), the write is skipped if the
    staged bytes equal the last bytes written to that destination - uniform
    decode re-uploads an identical ``query_start_loc`` every step.

Kill switches (environment):

  VLLM_MUSA_FAST_H2D=0             restore the upstream code path
  VLLM_MUSA_FAST_H2D_REUSE_DEST=0  keep the pinned cache but allocate a fresh
                                   device tensor per call (conservative mode)
"""

import os

import numpy as np
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_INSTALL_ATTR = "_musa_fast_h2d_installed"


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


class FastH2DUploader:
    """Pinned/destination-reusing wrapper around ``async_tensor_h2d``."""

    def __init__(self, fallback, reuse_dest: bool = True):
        self._fallback = fallback
        self.reuse_dest = reuse_dest
        self._enabled = None
        self._pinned: dict = {}
        self._dest: dict = {}
        self._written: dict = {}
        self.pinned_uploads = 0
        self.dest_reuses = 0
        self.skipped_uploads = 0

    def _pinned_for(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        key = (numel, dtype)
        buf = self._pinned.get(key)
        if buf is None:
            buf = torch.empty(numel, dtype=dtype, pin_memory=True)
            self._pinned[key] = buf
        return buf

    def _dest_for(self, numel: int, dtype: torch.dtype, device) -> torch.Tensor:
        key = (numel, dtype, str(device))
        buf = self._dest.get(key)
        if buf is None:
            buf = torch.empty(numel, dtype=dtype, device=device)
            self._dest[key] = buf
        else:
            self.dest_reuses += 1
        return buf

    def stats(self) -> dict:
        return {
            "pinned_uploads": self.pinned_uploads,
            "dest_reuses": self.dest_reuses,
            "skipped_uploads": self.skipped_uploads,
            "pinned_buffers": len(self._pinned),
            "dest_buffers": len(self._dest),
        }

    def _on_musa(self) -> bool:
        # Resolved lazily: the object patch is installed while the platform
        # plugin is still being activated, before `current_platform` settles.
        if self._enabled is None:
            try:
                from vllm.platforms import current_platform

                self._enabled = bool(current_platform.is_musa())
            except Exception:
                self._enabled = False
        return self._enabled

    def __call__(self, data, device=None, dtype=None, out=None):
        fallback = self._fallback
        if not self._on_musa():
            return fallback(data, device=device, dtype=dtype, out=out)
        if dtype is None and out is not None:
            dtype = out.dtype

        array = None
        if isinstance(data, np.ndarray):
            if not data.flags.c_contiguous:
                return fallback(data, device=device, dtype=dtype, out=out)
            array = data
            src = torch.from_numpy(data)
        elif (
            isinstance(data, torch.Tensor)
            and data.is_cpu
            and data.is_contiguous()
            and not data.is_pinned()
        ):
            src = data
        else:
            return fallback(data, device=device, dtype=dtype, out=out)

        numel = src.numel()
        target_dtype = dtype or src.dtype
        if numel == 0 or src.dtype != target_dtype:
            return fallback(data, device=device, dtype=dtype, out=out)
        if out is not None and (out.numel() != numel or out.dtype != target_dtype):
            return fallback(data, device=device, dtype=dtype, out=out)

        pinned = self._pinned_for(numel, target_dtype)
        try:
            pinned.copy_(src)
        except RuntimeError:
            return fallback(data, device=device, dtype=dtype, out=out)
        self.pinned_uploads += 1

        if out is not None:
            dest = out
            # Only a caller-owned destination can already hold the payload.
            payload = array.tobytes() if array is not None else None
            key = (dest.data_ptr(), numel, target_dtype)
            if payload is not None and self._written.get(key) == payload:
                self.skipped_uploads += 1
                return dest
            dest.copy_(pinned, non_blocking=True)
            if payload is not None:
                self._written[key] = payload
            return dest

        if self.reuse_dest and device is not None:
            dest = self._dest_for(numel, target_dtype, device)
            dest.copy_(pinned, non_blocking=True)
            return dest

        return pinned.to(device=device, dtype=target_dtype, non_blocking=True)


def install(torch_utils_module) -> FastH2DUploader | None:
    """Wrap ``vllm.utils.torch_utils.async_tensor_h2d`` once per process."""
    if getattr(torch_utils_module, _INSTALL_ATTR, None) is not None:
        return getattr(torch_utils_module, _INSTALL_ATTR)

    original = getattr(torch_utils_module, "async_tensor_h2d", None)
    if original is None or isinstance(original, FastH2DUploader):
        return None

    uploader = FastH2DUploader(
        original, reuse_dest=_env_flag("VLLM_MUSA_FAST_H2D_REUSE_DEST", True)
    )
    torch_utils_module.async_tensor_h2d = uploader
    setattr(torch_utils_module, _INSTALL_ATTR, uploader)

    # Modules that already did ``from ... import async_tensor_h2d`` keep the old
    # binding; rebind them so the shim applies regardless of import order.
    import sys

    rebound = 0
    for module in list(sys.modules.values()):
        if module is None or module is torch_utils_module:
            continue
        if getattr(module, "async_tensor_h2d", None) is original:
            try:
                setattr(module, "async_tensor_h2d", uploader)
                rebound += 1
            except Exception:  # pragma: no cover - exotic module types
                pass

    logger.info_once(
        "MUSA fast H2D metadata uploader active "
        f"(reuse_dest={uploader.reuse_dest}, rebound_modules={rebound}).",
        scope="global",
    )
    return uploader
