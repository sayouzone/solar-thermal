"""GPU 가속 백엔드 — 런타임 capability 감지 + CPU fallback.

이 모듈은 사용 가능한 GPU 라이브러리를 import 단계에서 감지하여
런타임에 안전하게 분기할 수 있도록 plain 플래그/얇은 wrapper 만 제공한다.

핵심 원칙
---------
1. 어떤 라이브러리든 import 실패해도 절대 모듈 import 가 죽지 않게.
2. 호출부는 capability 플래그(``HAS_CUPY`` 등)만 보고 분기.
3. GPU 사용 후 메모리는 명시적으로 해제 (장기 실행 시 누수 방지).

환경변수
--------
``GEOREF_DISABLE_GPU=1`` 로 모든 GPU 경로를 강제 비활성화 (벤치마크/디버깅용).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CuPy
# ---------------------------------------------------------------------------
HAS_CUPY: bool = False
cp = None  # type: ignore
try:
    import cupy as _cp  # type: ignore

    # 실제 디바이스 한 번 찔러서 동작 검증 (드라이버/런타임 불일치 잡기).
    _cp.cuda.runtime.getDeviceCount()
    cp = _cp
    HAS_CUPY = True
    logger.info("[GPU] CuPy 활성화 (CUDA device count=%d)",
                _cp.cuda.runtime.getDeviceCount())
except Exception as e:  # ImportError, CUDARuntimeError, etc.
    logger.info("[GPU] CuPy 비활성 (%s) — BA/triangulation/warp CPU 사용",
                type(e).__name__)


# ---------------------------------------------------------------------------
# OpenCV CUDA
# ---------------------------------------------------------------------------
HAS_CV_CUDA: bool = False
HAS_CV_CUDA_SIFT: bool = False
HAS_CV_CUDA_ORB: bool = False
HAS_CV_CUDA_REMAP: bool = False
try:
    import cv2 as _cv2

    if _cv2.cuda.getCudaEnabledDeviceCount() > 0:
        HAS_CV_CUDA = True
        # SIFT_CUDA 는 OpenCV 4.5.1+ + contrib 가 필요.
        HAS_CV_CUDA_SIFT = hasattr(_cv2.cuda, "SIFT_CUDA") or hasattr(
            _cv2.cuda, "SIFT_create"
        )
        HAS_CV_CUDA_ORB = hasattr(_cv2.cuda, "ORB_create")
        HAS_CV_CUDA_REMAP = hasattr(_cv2.cuda, "remap")
        logger.info(
            "[GPU] OpenCV CUDA 활성화 (SIFT=%s, ORB=%s, remap=%s)",
            HAS_CV_CUDA_SIFT, HAS_CV_CUDA_ORB, HAS_CV_CUDA_REMAP,
        )
    else:
        logger.info("[GPU] OpenCV CUDA 비활성 (device 0개)")
except Exception as e:
    logger.info("[GPU] OpenCV CUDA 비활성 (%s)", type(e).__name__)


# ---------------------------------------------------------------------------
# PyTorch (LightGlue 등 학습 기반 매칭 옵션)
# ---------------------------------------------------------------------------
HAS_TORCH: bool = False
HAS_LIGHTGLUE: bool = False
torch = None  # type: ignore
try:
    import torch as _torch  # type: ignore

    if _torch.cuda.is_available():
        torch = _torch
        HAS_TORCH = True
        try:
            from lightglue import LightGlue, SuperPoint  # type: ignore  # noqa: F401

            HAS_LIGHTGLUE = True
            logger.info("[GPU] PyTorch + LightGlue 활성화")
        except Exception:
            logger.info("[GPU] PyTorch CUDA 활성, LightGlue 미설치")
    else:
        logger.info("[GPU] PyTorch CUDA 비활성")
except Exception as e:
    logger.info("[GPU] PyTorch 비활성 (%s)", type(e).__name__)


# ---------------------------------------------------------------------------
# 환경변수로 강제 비활성
# ---------------------------------------------------------------------------
if os.environ.get("GEOREF_DISABLE_GPU", "").lower() in ("1", "true", "yes"):
    logger.warning("[GPU] GEOREF_DISABLE_GPU 설정 — 모든 GPU 경로 비활성화")
    HAS_CUPY = False
    HAS_CV_CUDA = False
    HAS_CV_CUDA_SIFT = False
    HAS_CV_CUDA_ORB = False
    HAS_CV_CUDA_REMAP = False
    HAS_TORCH = False
    HAS_LIGHTGLUE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def free_gpu_memory() -> None:
    """루프 사이에 호출해서 캐시된 GPU 메모리 풀을 비운다."""
    if HAS_CUPY:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    if HAS_TORCH:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def gpu_summary() -> str:
    """현재 활성화된 가속 경로 요약 (시작 로그용)."""
    parts: list[str] = []
    if HAS_CUPY:
        parts.append("CuPy")
    if HAS_CV_CUDA:
        bits = []
        if HAS_CV_CUDA_SIFT:
            bits.append("SIFT")
        if HAS_CV_CUDA_ORB:
            bits.append("ORB")
        if HAS_CV_CUDA_REMAP:
            bits.append("remap")
        parts.append(f"OpenCV-CUDA[{','.join(bits) or 'none'}]")
    if HAS_LIGHTGLUE:
        parts.append("LightGlue")
    elif HAS_TORCH:
        parts.append("PyTorch")
    if not parts:
        return "CPU only"
    return " + ".join(parts)


__all__ = [
    "HAS_CUPY", "cp",
    "HAS_CV_CUDA", "HAS_CV_CUDA_SIFT", "HAS_CV_CUDA_ORB", "HAS_CV_CUDA_REMAP",
    "HAS_TORCH", "HAS_LIGHTGLUE", "torch",
    "free_gpu_memory", "gpu_summary",
]