"""열화상 변환 유틸리티.

* raw_to_temperature: 카메라 raw 값 → 온도 (°C)
* pseudo_color_to_temp: 의사컬러 이미지 → 근사 온도 (heuristic)
* normalize_thermal: 시각화용 0..255 정규화
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def raw_to_temperature(
    raw: np.ndarray,
    emissivity: float = 0.95,
    reflected_temp_c: float = 20.0,
    atmospheric_temp_c: float = 20.0,
    distance_m: float = 5.0,
    humidity: float = 0.5,
) -> np.ndarray:
    """FLIR Planck 근사 역변환.

    정확한 값을 위해서는 카메라별 R1, R2, B, F, O 상수가 필요하지만,
    여기서는 raw 가 이미 centi-Kelvin (1/100 K) 단위라고 가정하고 °C 로 변환.
    """

    raw = raw.astype(np.float32)
    temp_k = raw / 100.0
    temp_c = temp_k - 273.15
    # 방사율 보정 (간소화)
    if emissivity < 1.0:
        temp_c = temp_c + (1.0 - emissivity) * (reflected_temp_c - temp_c) * 0.5
    return temp_c


# FLIR Iron / Rainbow 팔레트 근사 LUT (BGR → normalized 0..1)
# 실제 운용 시 camera SDK 에서 추출한 정확한 LUT 사용 권장
_IRON_LUT_BGR = np.array(
    [
        [0, 0, 0],
        [64, 0, 64],
        [128, 0, 128],
        [160, 0, 192],
        [0, 0, 255],
        [0, 128, 255],
        [0, 255, 255],
        [0, 255, 128],
        [0, 255, 0],
        [128, 255, 0],
        [255, 255, 0],
        [255, 192, 0],
        [255, 128, 0],
        [255, 64, 0],
        [255, 0, 0],
        [255, 255, 255],
    ],
    dtype=np.float32,
)


def pseudo_color_to_temp(
    bgr: np.ndarray,
    temp_min_c: float,
    temp_max_c: float,
    lut_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pseudo-color IR 이미지 → 근사 온도 맵.

    주의: 이는 근사치이며, 정확한 라디오메트릭 분석에는 적합하지 않음.
    camera display min/max (temp_min_c, temp_max_c) 가 EXIF/메타데이터에
    존재할 때 가장 정확합니다.
    """

    if lut_bgr is None:
        lut_bgr = _IRON_LUT_BGR

    h, w = bgr.shape[:2]
    flat = bgr.reshape(-1, 3).astype(np.float32)  # (N, 3)
    # 각 픽셀에 대해 LUT 와의 최소거리 인덱스 탐색
    # (N, K) distance
    dists = np.linalg.norm(flat[:, None, :] - lut_bgr[None, :, :], axis=2)
    idx = np.argmin(dists, axis=1).astype(np.float32)
    norm = idx / max(1.0, (len(lut_bgr) - 1))
    temp = temp_min_c + norm * (temp_max_c - temp_min_c)
    return temp.reshape(h, w)


def normalize_thermal(
    temp: np.ndarray,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
) -> np.ndarray:
    """시각화 용도의 0..255 uint8 정규화 (선형)."""

    if t_lo is None:
        t_lo = float(np.percentile(temp, 1))
    if t_hi is None:
        t_hi = float(np.percentile(temp, 99))
    clipped = np.clip(temp, t_lo, t_hi)
    norm = (clipped - t_lo) / max(1e-6, (t_hi - t_lo))
    return (norm * 255.0).astype(np.uint8)


def to_heatmap_bgr(temp: np.ndarray) -> np.ndarray:
    """열화상 온도맵 → JET colormap BGR 이미지."""

    gray = normalize_thermal(temp)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
