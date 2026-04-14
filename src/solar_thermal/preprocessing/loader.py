"""RGB / IR 이미지 로더.

지원 형식
---------
* RGB: JPEG/PNG (OpenCV)
* IR radiometric TIFF: 16-bit raw (tifffile)
* IR pseudo-color JPEG: FLIR/DJI pseudo-color 이미지 (OpenCV)
* IR gray16 PNG: 16-bit 그레이스케일

클라우드 URI (gs://, s3://, http(s)://) 는 `cloud.storage` 를 통해 로컬로 다운로드 후 처리.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tifffile
from loguru import logger

from ..cloud.storage import ensure_local


@dataclass
class ImagePair:
    """정합 전 RGB + IR 원본 이미지 쌍."""

    rgb: np.ndarray           # (H, W, 3) BGR uint8
    ir_raw: np.ndarray        # (H, W) float32 (°C) 또는 (H, W, 3) BGR
    ir_format: str            # "radiometric_tiff" | "pseudo_color" | "gray16"
    rgb_path: str
    ir_path: str


def load_rgb(uri: str) -> np.ndarray:
    """RGB 이미지를 OpenCV BGR uint8 배열로 로드."""

    local_path = ensure_local(uri)
    img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read RGB image: {uri}")
    return img


def load_thermal(
    uri: str,
    ir_format: str = "radiometric_tiff",
    temp_range: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """열화상을 로드.

    Returns
    -------
    np.ndarray
        * ir_format == "radiometric_tiff" or "gray16": (H, W) float32, 단위 °C
        * ir_format == "pseudo_color": (H, W, 3) BGR uint8

    Notes
    -----
    pseudo_color 는 색상-온도 팔레트 역변환이 camera-specific 이므로
    정확한 온도가 필요하면 radiometric_tiff 또는 FLIR SDK 파싱이 필요합니다.
    """

    local_path = ensure_local(uri)
    suffix = Path(local_path).suffix.lower()

    if ir_format == "radiometric_tiff":
        arr = tifffile.imread(str(local_path))
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D radiometric array, got shape {arr.shape}")
        # FLIR radiometric TIFF 는 centi-Kelvin (1/100 K) 단위인 경우가 많음
        # heuristic: 값 범위가 크면 centi-Kelvin 로 간주
        arr = arr.astype(np.float32)
        if arr.max() > 1000:  # centi-Kelvin
            arr = arr / 100.0 - 273.15
        logger.debug(f"Loaded radiometric TIFF: range [{arr.min():.1f}, {arr.max():.1f}] °C")
        return arr

    if ir_format == "gray16":
        arr = cv2.imread(str(local_path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Failed to read IR gray16 image: {uri}")
        if arr.dtype != np.uint16:
            logger.warning(f"Expected uint16, got {arr.dtype}")
        # temp_range 를 이용한 선형 디코딩: value -> temp
        if temp_range is None:
            temp_range = (-20.0, 120.0)
        t_min, t_max = temp_range
        arr_f = arr.astype(np.float32)
        arr_norm = (arr_f - arr_f.min()) / max(1.0, (arr_f.max() - arr_f.min()))
        return (arr_norm * (t_max - t_min) + t_min).astype(np.float32)

    if ir_format == "pseudo_color":
        img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read pseudo-color IR: {uri}")
        return img

    raise ValueError(f"Unsupported ir_format: {ir_format} (suffix={suffix})")
