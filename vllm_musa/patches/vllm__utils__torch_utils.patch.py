# SPDX-License-Identifier: Apache-2.0
"""MUSA fast H2D metadata uploads.

Installs :class:`vllm_musa.v1.worker.fast_h2d.FastH2DUploader` over
``vllm.utils.torch_utils.async_tensor_h2d`` so the per-step
``query_start_loc`` / ``idx_mapping`` uploads reuse one pinned staging buffer
and one device destination instead of allocating both on every decode step.
"""

PATCHES: list = []
PATCH_PHASE = "runtime"
PATCH_REQUIRED = False


def apply() -> None:
    import os

    from vllm.logger import init_logger

    logger = init_logger(__name__)

    if os.environ.get("VLLM_MUSA_FAST_H2D", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
        "",
    ):
        logger.info_once("MUSA fast H2D uploader disabled by VLLM_MUSA_FAST_H2D.")
        return

    import vllm.utils.torch_utils as torch_utils

    from vllm_musa.v1.worker.fast_h2d import install

    install(torch_utils)
