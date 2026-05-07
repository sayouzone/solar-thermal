"""
RGB-IR Co-registration for DJI ZH20T Dual Camera
=================================================

DJI Zenmuse H20T로 동시 촬영된 RGB(Zoom) 이미지와 IR(Thermal) 이미지를
픽셀 단위로 정합하여 다음을 가능하게 한다:

  1. 패널 분할 마스크(RGB 검출 결과)를 IR 좌표계로 투영
  2. IR 결함 위치를 RGB 고해상도 이미지로 검증
  3. RGB-IR 듀얼 모달 학습용 정합 데이터셋 구축

[정합 방식 - 3가지 단계적 접근]
  1) Initial: 카메라 캘리브레이션 + 짐벌/LRF 메타데이터 → 호모그래피 초기 추정
  2) Refine:  ECC (Enhanced Correlation Coefficient) 기반 미세 조정
  3) Verify:  ORB/SIFT feature matching (가능 시 fallback)

[참고]
  - DJI ZH20T 사양:
    Wide   (RGB):  4056×3040 / 24mm equiv / DFOV 82.9°
    Zoom   (RGB):  5184×3888 / 31-129mm equiv / DFOV 4-66.6° (가변 줌)
    Thermal (IR):  640×512   / 13.5mm / 35mm equiv 58mm / DFOV 40.6° / HFOV 35.4°

[작성자] sayouzone / SeongJung Kim
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ============================================================
# DJI ZH20T 카메라 사양 (제조사 공식 + 메타데이터 검증값)
# ============================================================

@dataclass(frozen=True)
class CameraSpec:
    """카메라 내부 파라미터 (FOV는 도(degree), focal_35mm는 35mm 환산)."""
    name: str
    width: int
    height: int
    focal_35mm: float       # mm, 35mm 환산
    hfov_deg: float
    vfov_deg: float

    @property
    def fx_normalized(self) -> float:
        """이미지 폭 대비 정규화 초점거리 (= width / (2·tan(HFOV/2)))"""
        return self.width / (2.0 * np.tan(np.radians(self.hfov_deg) / 2.0))

    @property
    def fy_normalized(self) -> float:
        return self.height / (2.0 * np.tan(np.radians(self.vfov_deg) / 2.0))

    @property
    def cx(self) -> float: return self.width / 2.0

    @property
    def cy(self) -> float: return self.height / 2.0

    def K(self) -> np.ndarray:
        """카메라 내부 행렬 (대략값 — 정합 초기화용)."""
        return np.array([[self.fx_normalized, 0, self.cx],
                         [0, self.fy_normalized, self.cy],
                         [0, 0, 1]], dtype=np.float64)


# ZH20T 사양 (DJI 공식 매뉴얼 + 업로드 EXIF 검증)
ZH20T_THERMAL = CameraSpec(
    name="ZH20T_Thermal",
    width=640, height=512,
    focal_35mm=58.0,        # XMP의 35mm 환산값
    hfov_deg=40.6,          # 공식 사양 DFOV에서 환산
    vfov_deg=33.0,          # = 2·atan(tan(DFOV/2) / sqrt(1+(W/H)²))
)

ZH20T_ZOOM_WIDE = CameraSpec(
    name="ZH20T_Zoom_Wide",
    width=5184, height=3888,
    focal_35mm=47.0,        # XMP의 35mm 환산값 (촬영 당시 줌)
    hfov_deg=43.0,          # 47mm 환산 기준
    vfov_deg=33.0,
)


# ============================================================
# 정합 결과 컨테이너
# ============================================================

@dataclass
class RegistrationResult:
    homography: np.ndarray           # 3×3 RGB → IR 좌표계 변환
    inverse_homography: np.ndarray   # 3×3 IR → RGB
    method: str                      # 'metadata' / 'ecc' / 'feature'
    confidence: float                # 0~1
    rgb_size: Tuple[int, int]        # (W, H)
    ir_size: Tuple[int, int]
    inliers: int = 0
    rmse_px: float = float("nan")

    def warp_rgb_to_ir(self, rgb: np.ndarray, interp: int = cv2.INTER_AREA) -> np.ndarray:
        """RGB 이미지를 IR 해상도/좌표계로 변환."""
        return cv2.warpPerspective(rgb, self.homography, self.ir_size, flags=interp)

    def warp_mask_to_ir(self, mask: np.ndarray) -> np.ndarray:
        """이진 마스크는 NEAREST 보간으로 변환."""
        return cv2.warpPerspective(
            mask.astype(np.uint8), self.homography, self.ir_size,
            flags=cv2.INTER_NEAREST,
        ).astype(bool)

    def warp_points_rgb_to_ir(self, pts_rgb: np.ndarray) -> np.ndarray:
        """포인트 좌표 변환. pts_rgb shape: (N,2) → (N,2)"""
        if pts_rgb.size == 0:
            return pts_rgb
        pts = pts_rgb.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "confidence": self.confidence,
            "rgb_size": self.rgb_size,
            "ir_size": self.ir_size,
            "inliers": self.inliers,
            "rmse_px": self.rmse_px,
            "homography": self.homography.tolist(),
        }


# ============================================================
# 단계 1: 메타데이터 기반 호모그래피 초기 추정
# ============================================================

def estimate_homography_from_metadata(
    rgb_spec: CameraSpec,
    ir_spec: CameraSpec,
    distance_m: float,
    baseline_m: float = 0.07,    # ZH20T 듀얼 광학계 간 약 7cm 베이스라인
) -> np.ndarray:
    """
    카메라 사양과 피사체 거리만으로 RGB → IR 호모그래피 초기 추정.

    원리: 두 카메라의 내부행렬과 광학축 정렬을 가정하고, 피사체가 평면에 있으면
    유효 스케일 = (ir_focal / rgb_focal) × (rgb_size / ir_size)

    distance_m: LRF 측정값 또는 비행 고도 사용.
    """
    K_rgb = rgb_spec.K()
    K_ir = ir_spec.K()

    # 단순 모델: 두 카메라가 동축이고 피사체가 충분히 멀다고 가정하면
    # 동일 3D 점이 두 이미지에 투영되는 변환은
    #   x_ir = K_ir · K_rgb⁻¹ · x_rgb   (회전 없음, 평행이동 무시)
    # 베이스라인 보정은 시차(parallax)가 픽셀 미만일 때 무시 가능.
    H_init = K_ir @ np.linalg.inv(K_rgb)

    # 베이스라인으로 인한 시차 보정 (소형 평행이동 항)
    # 시차 [px] ≈ baseline × ir_focal_px / distance
    parallax_px = baseline_m * ir_spec.fx_normalized / max(distance_m, 1e-3)
    H_init[0, 2] -= parallax_px / 2.0   # 좌측 보정

    return H_init


# ============================================================
# 단계 2: ECC 기반 정합 미세 조정
# ============================================================

def refine_homography_ecc(
    rgb_gray: np.ndarray,
    ir_gray: np.ndarray,
    H_init: np.ndarray,
    n_iter: int = 200,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    ECC (Enhanced Correlation Coefficient) 기반 호모그래피 정밀화.

    Returns: (H_refined, correlation_coeff)
    """
    # IR 크기에 맞춰 RGB를 H_init으로 워핑한 뒤 ECC 적용
    h_ir, w_ir = ir_gray.shape
    rgb_warped = cv2.warpPerspective(rgb_gray, H_init, (w_ir, h_ir))

    # 8-bit 정규화 (ECC 권장)
    def _norm(img):
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.ptp() + 1e-9)
        return (img * 255).astype(np.uint8)

    rgb_n = _norm(rgb_warped)
    ir_n  = _norm(ir_gray)

    warp = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, eps)

    try:
        cc, warp = cv2.findTransformECC(
            templateImage=ir_n, inputImage=rgb_n, warpMatrix=warp,
            motionType=cv2.MOTION_HOMOGRAPHY,
            criteria=criteria, inputMask=None, gaussFiltSize=5,
        )
    except cv2.error as e:
        # ECC 발산 시 초기값 그대로 반환
        return H_init.astype(np.float32), 0.0

    # 최종 = ECC 보정 × 초기값
    H_refined = warp.astype(np.float64) @ H_init
    return H_refined, float(cc)


# ============================================================
# 단계 3: ORB feature 기반 정합 (검증/fallback)
# ============================================================

def estimate_homography_features(
    rgb_gray: np.ndarray,
    ir_gray: np.ndarray,
    H_prior: Optional[np.ndarray] = None,
    min_matches: int = 12,
) -> Tuple[Optional[np.ndarray], int, float]:
    """
    ORB descriptor 매칭 + RANSAC 호모그래피.
    RGB-IR은 모달리티가 다르므로 신뢰도가 낮을 수 있음 → ECC 결과 검증용.

    Returns: (H, num_inliers, mean_reprojection_error_px)
    """
    # CLAHE로 IR 콘트라스트 강화 (RGB와 텍스처 매칭 가능성 ↑)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ir_eq = clahe.apply(ir_gray)
    rgb_eq = clahe.apply(rgb_gray)

    orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8)
    kp_rgb, des_rgb = orb.detectAndCompute(rgb_eq, None)
    kp_ir,  des_ir  = orb.detectAndCompute(ir_eq, None)

    if des_rgb is None or des_ir is None or len(kp_rgb) < min_matches or len(kp_ir) < min_matches:
        return None, 0, float("nan")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des_rgb, des_ir, k=2)

    # Lowe ratio test
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    if len(good) < min_matches:
        return None, 0, float("nan")

    pts_rgb = np.float32([kp_rgb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_ir  = np.float32([kp_ir[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_rgb, pts_ir, cv2.RANSAC,
                                  ransacReprojThreshold=3.0, maxIters=2000)
    if H is None:
        return None, 0, float("nan")

    inliers = int(mask.sum())
    # 재투영 오차
    pts_rgb_inl = pts_rgb[mask.ravel() == 1]
    pts_ir_inl  = pts_ir[mask.ravel() == 1]
    proj = cv2.perspectiveTransform(pts_rgb_inl, H)
    rmse = float(np.sqrt(((proj - pts_ir_inl) ** 2).sum(axis=2).mean()))

    return H, inliers, rmse


# ============================================================
# 메인 정합 파이프라인
# ============================================================

def coregister_rgb_ir(
    rgb_path: str | Path,
    ir_visual_path: str | Path,
    distance_m: float,
    rgb_spec: CameraSpec = ZH20T_ZOOM_WIDE,
    ir_spec: CameraSpec = ZH20T_THERMAL,
    use_ecc: bool = True,
    use_features: bool = True,
) -> RegistrationResult:
    """
    RGB와 IR(시각화 JPEG) 이미지 정합.

    ir_visual_path: R-JPEG의 시각화 레이어 (640×512 컬러 JPEG).
                    온도 행렬을 [0,255]로 정규화한 그레이스케일도 가능.
    """
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    ir  = cv2.imread(str(ir_visual_path), cv2.IMREAD_COLOR)
    if rgb is None or ir is None:
        raise FileNotFoundError(f"이미지 로드 실패: {rgb_path} / {ir_visual_path}")

    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    ir_gray  = cv2.cvtColor(ir,  cv2.COLOR_BGR2GRAY)

    # 1) 메타데이터 기반 초기값
    H_init = estimate_homography_from_metadata(rgb_spec, ir_spec, distance_m)
    method, confidence, inliers, rmse = "metadata", 0.5, 0, float("nan")
    H_final = H_init

    # 2) ECC 정밀화
    if use_ecc:
        H_ecc, cc = refine_homography_ecc(rgb_gray, ir_gray, H_init)
        if cc > 0.3:   # 상관계수 양호
            H_final = H_ecc
            method = "ecc"
            confidence = float(cc)

    # 3) Feature matching 검증 (있으면 가장 신뢰)
    if use_features:
        H_feat, n_inl, rmse_feat = estimate_homography_features(rgb_gray, ir_gray)
        if H_feat is not None and n_inl >= 20 and rmse_feat < 5.0:
            H_final = H_feat
            method = "feature"
            confidence = min(1.0, n_inl / 100.0)
            inliers, rmse = n_inl, rmse_feat

    H_inv = np.linalg.inv(H_final)

    return RegistrationResult(
        homography=H_final,
        inverse_homography=H_inv,
        method=method,
        confidence=confidence,
        rgb_size=(rgb.shape[1], rgb.shape[0]),
        ir_size=(ir.shape[1], ir.shape[0]),
        inliers=inliers,
        rmse_px=rmse,
    )


# ============================================================
# 시각화
# ============================================================

def visualize_registration(
    rgb_path: str | Path,
    ir_visual_path: str | Path,
    result: RegistrationResult,
    out_path: str | Path,
    alpha: float = 0.5,
) -> None:
    """RGB 워핑 + IR 오버레이 + 차이맵 4-pane 비교."""
    import matplotlib.pyplot as plt

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)[:, :, ::-1]
    ir  = cv2.imread(str(ir_visual_path), cv2.IMREAD_COLOR)[:, :, ::-1]
    rgb_warp = result.warp_rgb_to_ir(rgb, interp=cv2.INTER_AREA)

    overlay = (alpha * ir + (1 - alpha) * rgb_warp).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"RGB 원본 ({result.rgb_size[0]}×{result.rgb_size[1]})")
    axes[0, 0].set_axis_off()

    axes[0, 1].imshow(ir)
    axes[0, 1].set_title(f"IR ({result.ir_size[0]}×{result.ir_size[1]})")
    axes[0, 1].set_axis_off()

    axes[1, 0].imshow(rgb_warp)
    axes[1, 0].set_title(f"RGB → IR 워핑\n방법: {result.method}, 신뢰도: {result.confidence:.2f}")
    axes[1, 0].set_axis_off()

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f"IR + RGB 오버레이 (α={alpha})")
    axes[1, 1].set_axis_off()

    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DJI ZH20T RGB-IR 정합")
    parser.add_argument("--rgb", required=True, help="RGB JPG (예: ..._Z.JPG)")
    parser.add_argument("--ir",  required=True, help="IR R-JPEG (예: ..._T.JPG)")
    parser.add_argument("--distance", type=float, required=True,
                        help="LRF 거리 [m] (XMP의 LRFTargetDistance)")
    parser.add_argument("--out-vis",  default="registration.png")
    parser.add_argument("--out-json", default="registration.json")
    parser.add_argument("--no-ecc",      action="store_true")
    parser.add_argument("--no-features", action="store_true")
    args = parser.parse_args()

    result = coregister_rgb_ir(
        rgb_path=args.rgb, ir_visual_path=args.ir,
        distance_m=args.distance,
        use_ecc=not args.no_ecc, use_features=not args.no_features,
    )
    print(f"[정합 완료] 방법={result.method}, 신뢰도={result.confidence:.3f}, "
          f"inliers={result.inliers}, RMSE={result.rmse_px:.2f}px")

    visualize_registration(args.rgb, args.ir, result, args.out_vis)
    Path(args.out_json).write_text(json.dumps(result.to_dict(), indent=2))
    print(f"  → {args.out_vis}")
    print(f"  → {args.out_json}")
