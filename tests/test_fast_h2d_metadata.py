# SPDX-License-Identifier: Apache-2.0
"""CPU-side contract tests for the MUSA fast H2D metadata uploader."""

import numpy as np
import pytest
import torch

from vllm_musa.v1.worker.fast_h2d import FastH2DUploader


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, data, device=None, dtype=None, out=None):
        self.calls.append((np.asarray(data).copy() if hasattr(data, "__array__") else data,
                           device, dtype, out))
        if out is not None:
            if out.numel() == np.asarray(data).size:
                out.copy_(torch.from_numpy(np.asarray(data)))
            return out
        src = torch.from_numpy(np.asarray(data))
        t = src.to(device=device, dtype=dtype or src.dtype)
        return t


def _uploader(reuse_dest=True):
    rec = _Recorder()
    return FastH2DUploader(rec, reuse_dest=reuse_dest), rec


def test_pinned_staging_is_reused():
    up, rec = _uploader()
    src = np.arange(129, dtype=np.int32)
    first = up(src, device="cpu")
    for value in range(1, 5):
        src[1] = value
        up(src, device="cpu")
    assert up.stats()["pinned_buffers"] == 1
    assert up.stats()["pinned_uploads"] == 5
    assert up.stats()["dest_buffers"] == 1
    assert not rec.calls, "fast path must not consult the upstream implementation"
    assert first is up(src, device="cpu"), "device destination must be reused"


def test_destination_carries_latest_payload():
    up, _ = _uploader()
    src = np.arange(257, dtype=np.int32)
    out = up(src, device="cpu")
    np.testing.assert_array_equal(out.numpy(), src)
    src[:] = 7
    out = up(src, device="cpu")
    np.testing.assert_array_equal(out.numpy(), np.full(257, 7, dtype=np.int32))


def test_out_upload_is_skipped_when_unchanged():
    up, rec = _uploader()
    src = np.zeros(129, dtype=np.int32)
    dst = torch.zeros(129, dtype=torch.int32)
    up(src, out=dst)
    assert up.stats()["skipped_uploads"] == 0
    src[3] = 5
    up(src, out=dst)
    assert dst[3].item() == 5
    for _ in range(4):
        up(src, out=dst)
    assert up.stats()["skipped_uploads"] == 4
    src[3] = 6
    up(src, out=dst)
    assert dst[3].item() == 6
    assert up.stats()["skipped_uploads"] == 4


def test_reuse_dest_disabled_allocates_fresh_destination():
    # The destination tensor is only reused on the ``device=`` fast path; with
    # reuse disabled the call must fall through to ``pinned.to(device)``.
    up, _ = _uploader(reuse_dest=False)
    src = np.arange(32, dtype=np.int64)
    up(src, device="cpu")
    up(src, device="cpu")
    assert up.stats()["dest_buffers"] == 0
    assert up.stats()["dest_reuses"] == 0
    assert up.stats()["pinned_buffers"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        np.arange(32, dtype=np.float32),          # dtype conversion requested below
        np.arange(64, dtype=np.int32)[::2],       # non-contiguous view
    ],
)
def test_unsupported_sources_fall_back(payload):
    up, rec = _uploader()
    if payload.dtype == np.float32:
        up(payload, device="cpu", dtype=torch.int32)
    else:
        up(payload, device="cpu")
    assert len(rec.calls) == 1


def test_out_size_mismatch_falls_back():
    up, rec = _uploader()
    src = np.arange(16, dtype=np.int32)
    dst = torch.zeros(17, dtype=torch.int32)
    up(src, out=dst)
    assert len(rec.calls) == 1


def test_list_input_falls_back():
    up, rec = _uploader()
    up([1, 2, 3], device="cpu")
    assert len(rec.calls) == 1
