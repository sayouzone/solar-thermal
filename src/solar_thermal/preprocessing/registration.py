"""RGB ↔ IR 이미지 정합(registration).

드론 기반 dual-sensor 카메라는 RGB 와 IR 가 서로 다른 해상도/FOV 를 가집니다.
- method="homography": ORB + RANSAC homography (기본)
- method="affine":     ORB + affine
- method="identity":   단순 resize (카메라가 이미 하드웨어 정합된 경우)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger


@dataclass
class AlignedPair:
    rgb: np.ndarray           # (H, W, 3) BGR
    ir_temp: np.ndarray       # (H, W) float32 °C, RGB 해상도와 동일
    warp_matrix: np.ndarray   # IR→RGB 변환 행렬 (3x3 or 2x3)
    method: str


def align_ir_to_rgb(
    rgb: np.ndarray,
    ir_temp: np.ndarray,
    method: str = "homography",
    orb_features: int = 2000,
    ransac_reproj_threshold: float = 5.0,
) -> AlignedPair:
    """IR 온도 맵을 RGB 좌표계로 워프.

    Parameters
    ----------
    rgb : (H, W, 3) BGR uint8
    ir_temp : (h, w) float32 단위 °C
    """

    h_rgb, w_rgb = rgb.shape[:2]

    if method == "identity":
        ir_resized = cv2.resize(ir_temp, (w_rgb, h_rgb), interpolation=cv2.INTER_LINEAR)
        return AlignedPair(rgb=rgb, ir_temp=ir_resized, warp_matrix=np.eye(3), method="identity")

    # feature 기반 정합을 위해 IR 온도 맵을 8-bit 로 변환
    ir_8u = cv2.normalize(ir_temp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=orb_features)
    kp1, des1 = orb.detectAndCompute(ir_8u, None)
    kp2, des2 = orb.detectAndCompute(rgb_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        logger.warning("ORB features insufficient; falling back to identity warp")
        return align_ir_to_rgb(rgb, ir_temp, method="identity")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)[: max(50, int(0.2 * len(matches)))]

    if len(matches) < 10:
        logger.warning("Not enough matches; falling back to identity warp")
        return align_ir_to_rgb(rgb, ir_temp, method="identity")

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if method == "homography":
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj_threshold)
        if M is None:
            logger.warning("Homography failed; falling back to identity")
            return align_ir_to_rgb(rgb, ir_temp, method="identity")
        warped = cv2.warpPerspective(
            ir_temp, M, (w_rgb, h_rgb), flags=cv2.INTER_LINEAR, borderValue=float("nan")
        )
        return AlignedPair(rgb=rgb, ir_temp=warped, warp_matrix=M, method="homography")

    if method == "affine":
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        if M is None:
            return align_ir_to_rgb(rgb, ir_temp, method="identity")
        warped = cv2.warpAffine(
            ir_temp, M, (w_rgb, h_rgb), flags=cv2.INTER_LINEAR, borderValue=float("nan")
        )
        M3 = np.vstack([M, [0, 0, 1]])
        return AlignedPair(rgb=rgb, ir_temp=warped, warp_matrix=M3, method="affine")

    raise ValueError(f"Unknown registration method: {method}")
