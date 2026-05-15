"""특징점 추출 — CUDA SIFT / CUDA ORB / CPU SIFT(ProcessPool) 자동 선택.

가속 우선순위
-------------
1. **CUDA SIFT** (OpenCV 4.5.1+ contrib 빌드 필요) — 원본 SIFT 알고리즘,
   거의 동일한 결과.
2. **CUDA ORB** — SIFT 대비 5~10배 빠름, descriptor 가 binary 라
   매칭은 Hamming distance 사용. 정확도는 약간 낮지만 RTK 가 cm 급
   prior 를 제공하므로 충분.
3. **CPU SIFT + ProcessPool** — GPU 모두 불가용일 때 ``n_workers`` 개
   프로세스로 병렬 추출. 워커 프로세스에서 ``cv2.KeyPoint`` 직접 반환은
   pickle 불가이므로 numpy 배열로 직렬화 후 메인 프로세스에서 복원.

호출자는 단순히 ``extract_all_features(paths)`` 만 부르면 되고, 반환 dict
에 ``__desc_type__`` 키로 descriptor 종류("SIFT" or "ORB") 가 들어가서
매칭 단계가 거리 함수를 자동 선택할 수 있다.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..gpu_backend import HAS_CV_CUDA_ORB, HAS_CV_CUDA_SIFT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-image extractors
# ---------------------------------------------------------------------------
def _extract_cuda_sift(image_path: Path, max_features: int):
    """OpenCV CUDA SIFT (contrib 모듈 필요)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    # API 위치가 OpenCV 버전마다 다름.
    if hasattr(cv2.cuda, "SIFT_CUDA"):
        sift = cv2.cuda.SIFT_CUDA.create(nfeatures=max_features)
    else:
        sift = cv2.cuda.SIFT_create(nfeatures=max_features)
    kp_gpu, desc_gpu = sift.detectAndComputeAsync(gpu_img, None)
    kp = sift.convert(kp_gpu)
    desc = desc_gpu.download() if desc_gpu is not None else None
    return kp, desc, img.shape


def _extract_cuda_orb(image_path: Path, max_features: int):
    """OpenCV CUDA ORB — SIFT 대비 훨씬 빠름, descriptor 는 binary."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    orb = cv2.cuda.ORB_create(nfeatures=max_features)
    kp_gpu, desc_gpu = orb.detectAndComputeAsync(gpu_img, None)
    kp = orb.convert(kp_gpu)
    desc = desc_gpu.download() if desc_gpu is not None else None
    return kp, desc, img.shape


def _extract_cpu_sift_pickleable(image_path, max_features: int):
    """CPU SIFT — ProcessPool 워커에서 호출 (top-level 함수).

    ``cv2.KeyPoint`` 객체는 pickle 불가능하므로 keypoint 속성만 ``(N, 7)``
    numpy 배열로 직렬화. 메인 프로세스에서 ``_unpack_kp_array`` 로 복원.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp, desc = sift.detectAndCompute(img, None)
    kp_arr = (
        np.array(
            [(k.pt[0], k.pt[1], k.size, k.angle,
              k.response, k.octave, k.class_id) for k in kp],
            dtype=np.float32,
        )
        if kp else np.zeros((0, 7), dtype=np.float32)
    )
    return kp_arr, desc, img.shape


def _unpack_kp_array(kp_arr: np.ndarray) -> list:
    """``(N, 7)`` 직렬화 배열 → ``cv2.KeyPoint`` 리스트 복원."""
    return [
        cv2.KeyPoint(
            x=float(r[0]), y=float(r[1]), size=float(r[2]),
            angle=float(r[3]), response=float(r[4]),
            octave=int(r[5]), class_id=int(r[6]),
        )
        for r in kp_arr
    ]


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------
def extract_all_features(image_paths: list[Path],
                         max_features: int = 8000,
                         n_workers: int | None = None) -> dict[Any, Any]:
    """이미지 경로 리스트 → ``{idx: (keypoints, descriptors, shape)}`` dict.

    반환 dict 에는 추가로 ``"__desc_type__"`` 키로 descriptor 종류
    (``"SIFT"`` 또는 ``"ORB"``) 가 들어간다. 매칭 단계에서 거리 함수
    (L2 vs Hamming) 를 자동 선택하기 위함.

    Parameters
    ----------
    image_paths : 추출할 이미지 경로 리스트.
    max_features : 한 장당 추출 상한 (기본 8000).
    n_workers : CPU SIFT fallback 시 프로세스 수. ``None`` 이면
        ``min(8, os.cpu_count())``. GPU 경로에서는 무시.
    """
    features: dict[Any, Any] = {}

    if HAS_CV_CUDA_SIFT:
        logger.info("특징점 추출: CUDA SIFT (이미지 %d장)", len(image_paths))
        for i, path in enumerate(image_paths):
            features[i] = _extract_cuda_sift(path, max_features)
        features["__desc_type__"] = "SIFT"
        return features

    if HAS_CV_CUDA_ORB:
        logger.info("특징점 추출: CUDA ORB (이미지 %d장)", len(image_paths))
        for i, path in enumerate(image_paths):
            features[i] = _extract_cuda_orb(path, max_features)
        features["__desc_type__"] = "ORB"
        return features

    if n_workers is None:
        n_workers = min(8, (os.cpu_count() or 4))
    logger.info("특징점 추출: CPU SIFT × %d workers (이미지 %d장)",
                n_workers, len(image_paths))

    if n_workers <= 1:
        for i, path in enumerate(image_paths):
            kp_arr, desc, shape = _extract_cpu_sift_pickleable(path, max_features)
            features[i] = (_unpack_kp_array(kp_arr), desc, shape)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {
                pool.submit(_extract_cpu_sift_pickleable, p, max_features): i
                for i, p in enumerate(image_paths)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                kp_arr, desc, shape = fut.result()
                features[i] = (_unpack_kp_array(kp_arr), desc, shape)
    features["__desc_type__"] = "SIFT"
    return features


__all__ = ["extract_all_features"]