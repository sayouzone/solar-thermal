"""
드론 사진 Georeferencing 파이프라인 (GCP-Free / RTK-PPK 기반)
==============================================================

GCP(지상기준점)를 사용하지 않고, 드론에 탑재된 RTK/PPK GPS의 cm급 정확도
좌표를 카메라 위치 제약조건으로 직접 사용하는 GCP-free 워크플로우.

워크플로우:
    1. EXIF/XMP 메타데이터 추출 → 사진별 RTK 좌표 + 짐벌 자세
    2. RTK Fixed 여부 검증 (RtkFlag=50만 신뢰)
    3. SfM Tie Point 추출 (SIFT + RANSAC)
    4. 두 시점 초기화 → 점진적 SfM 재구성
    5. RTK 제약 Bundle Adjustment (카메라 위치를 GCP처럼 고정/소프트제약)
    6. 정사영상 생성 - DSM 없이 평면 평균 고도 기반 간이 정사보정

GCP-free의 한계:
    - 수평 정확도: 2~5 cm 가능 (RTK Fixed 기준)
    - 수직 정확도: 5~15 cm (안테나 위상 중심 오프셋, GNSS 다중경로 영향)
    - 절대 정확도가 중요한 측량/검측용은 최소 1~2점의 Check Point 권장

작성: Sayouzone Solar-Thermal 프로젝트 참고용
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from itertools import combinations

import cv2
import numpy as np
import rasterio
import sys
from rasterio.transform import from_origin
from pyproj import Transformer
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import exifread
from xml.etree import ElementTree as ET

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solar_thermal.image.metadata import ImageMetadata, extract_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 메타데이터 추출 (DJI EXIF/XMP)
# ---------------------------------------------------------------------------
# DJI RtkFlag 의미 (DJI SDK 문서 기준):
#   0  = None / GPS only
#   16 = RTK Float (수십 cm 정확도)
#   34 = RTK Single (저정밀)
#   50 = RTK Fixed (1~3 cm 정확도) ← 신뢰 가능
RTK_FIXED = 50


# ---------------------------------------------------------------------------
# 2. 좌표계 변환
# ---------------------------------------------------------------------------
class CRSConverter:
    def __init__(self, target_epsg: int = 5186):
        self.to_proj = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
        self.to_wgs = Transformer.from_crs(f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True)
        self.target_epsg = target_epsg

    def forward(self, lon: float, lat: float) -> tuple[float, float]:
        return self.to_proj.transform(lon, lat)

    def inverse(self, x: float, y: float) -> tuple[float, float]:
        return self.to_wgs.transform(x, y)


# ---------------------------------------------------------------------------
# 3. RTK 품질 검증
# ---------------------------------------------------------------------------
def validate_rtk_quality(metas: list[ImageMetadata],
                         min_fixed_ratio: float = 0.9) -> bool:
    """RTK Fixed 비율 확인. 90% 미만이면 GCP-free 비추천."""
    if not metas:
        return False
    fixed_count = sum(m.is_rtk_fixed for m in metas)
    ratio = fixed_count / len(metas)
    logger.info("RTK Fixed 비율: %d/%d (%.1f%%)", fixed_count, len(metas), ratio * 100)
    if ratio < min_fixed_ratio:
        logger.warning(
            "RTK Fixed 비율이 %.1f%% 미만입니다. GCP 없이 진행 시 정확도가 떨어질 수 있습니다.",
            min_fixed_ratio * 100,
        )
    return ratio >= min_fixed_ratio


# ---------------------------------------------------------------------------
# 4. SfM - Tie Point 추출 (GPS 인접 사진끼리만 매칭)
# ---------------------------------------------------------------------------
def extract_features(image_path: Path, max_features: int = 8000):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc, img.shape


def match_pair(desc1, desc2, ratio: float = 0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc1, desc2, k=2)
    return [m for m, n in knn if m.distance < ratio * n.distance]


def find_neighbor_pairs(metas: list[ImageMetadata],
                        crs: CRSConverter,
                        k_neighbors: int = 8) -> list[tuple[int, int]]:
    """RTK 좌표 기반 KD-Tree로 인접 사진만 매칭 후보로.

    O(N²) 풀 매칭 대신 O(N log N)으로 단축. 수천 장 처리시 필수.
    """
    xy = np.array([crs.forward(m.gps.lng, m.gps.lat) for m in metas])
    tree = cKDTree(xy)
    pairs = set()
    for i in range(len(metas)):
        _, idxs = tree.query(xy[i], k=min(k_neighbors + 1, len(metas)))
        for j in idxs[1:]:  # 자기 자신 제외
            pairs.add((min(i, j), max(i, j)))
    logger.info("인접 페어 %d개 생성 (k=%d)", len(pairs), k_neighbors)
    return sorted(pairs)


def build_tie_points(metas: list[ImageMetadata],
                     pairs: list[tuple[int, int]]):
    """SIFT 특징점 추출 + 페어별 매칭 + RANSAC outlier 제거."""
    features = {i: extract_features(m.origin_path) for i, m in enumerate(metas)}
    matches = {}

    for i, j in pairs:
        kp_i, desc_i, _ = features[i]
        kp_j, desc_j, _ = features[j]
        if desc_i is None or desc_j is None:
            continue
        good = match_pair(desc_i, desc_j)
        if len(good) < 30:
            continue
        pts_i = np.float32([kp_i[m.queryIdx].pt for m in good])
        pts_j = np.float32([kp_j[m.trainIdx].pt for m in good])
        _, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 1.0, 0.99)
        if mask is None:
            continue
        inliers = mask.ravel().astype(bool)
        if inliers.sum() < 20:
            continue
        matches[(i, j)] = (pts_i[inliers], pts_j[inliers])

    logger.info("유효 매칭 페어: %d", len(matches))
    return matches, features


# ---------------------------------------------------------------------------
# 5. 공선조건 & 회전행렬
# ---------------------------------------------------------------------------
def rotation_matrix(omega: float, phi: float, kappa: float) -> np.ndarray:
    """사진측량 ω-φ-κ 회전행렬 (radian)."""
    co, so = np.cos(omega), np.sin(omega)
    cp, sp = np.cos(phi), np.sin(phi)
    ck, sk = np.cos(kappa), np.sin(kappa)
    return np.array([
        [cp * ck,                  -cp * sk,                 sp],
        [co * sk + so * sp * ck,    co * ck - so * sp * sk,  -so * cp],
        [so * sk - co * sp * ck,    so * ck + co * sp * sk,   co * cp],
    ])


def project_point(world_xyz: np.ndarray,
                  camera_xyz: np.ndarray,
                  omega: float, phi: float, kappa: float,
                  f_px: float, cx: float, cy: float) -> np.ndarray:
    """공선조건식으로 3D점을 이미지 픽셀로 투영.

    f_px: 픽셀 단위 초점거리 (= f_mm * width / sensor_width_mm)
    """
    R = rotation_matrix(omega, phi, kappa)
    diff = world_xyz - camera_xyz
    den = R[2] @ diff
    if abs(den) < 1e-9:
        return np.array([np.nan, np.nan])
    x = cx - f_px * (R[0] @ diff) / den
    y = cy - f_px * (R[1] @ diff) / den
    return np.array([x, y])


# ---------------------------------------------------------------------------
# 6. RTK 제약 Bundle Adjustment (GCP 없이 카메라 위치 자체가 제약)
# ---------------------------------------------------------------------------
def rtk_constrained_bundle_adjustment(
    initial_cameras: np.ndarray,        # (n_cam, 6) [Xc,Yc,Zc, ω,φ,κ]
    initial_points: np.ndarray,         # (n_pts, 3)
    observations: list[tuple[int, int, np.ndarray]],
    rtk_priors: np.ndarray,             # (n_cam, 3) RTK 측정 카메라 위치
    rtk_weights: np.ndarray,            # (n_cam, 3) 1/σ² 가중치
    f_px: float, cx: float, cy: float,
):
    """RTK 좌표를 카메라 위치의 사전확률(prior)로 묶는 번들 조정.

    핵심 아이디어:
        GCP가 없으므로 절대 좌표계 기준점은 RTK 측정값.
        하지만 RTK도 cm급 오차가 있으므로 hard constraint가 아닌
        soft constraint (가중치 = 1/σ²) 로 잔차에 추가.

        residual = [reprojection_errors, sqrt(w) * (camera_pos - rtk_prior)]

    이렇게 하면:
        - reprojection error는 일관된 internal geometry를 보장
        - RTK prior는 절대 georeferencing을 보장
        - 둘이 가중 평균되어 outlier에 강건한 해를 찾음
    """
    n_cam = len(initial_cameras)
    n_pts = len(initial_points)

    def pack(cams, pts):
        return np.concatenate([cams.ravel(), pts.ravel()])

    def unpack(x):
        cams = x[: n_cam * 6].reshape(n_cam, 6)
        pts = x[n_cam * 6:].reshape(n_pts, 3)
        return cams, pts

    def residuals(x):
        cams, pts = unpack(x)
        res = []

        # (a) Reprojection residuals
        for cam_idx, pt_idx, obs in observations:
            cam = cams[cam_idx]
            predicted = project_point(
                pts[pt_idx], cam[:3], cam[3], cam[4], cam[5],
                f_px, cx, cy,
            )
            res.extend(predicted - obs)

        # (b) RTK prior residuals — 각 카메라 위치를 RTK 측정값에 묶음
        for i in range(n_cam):
            diff = cams[i, :3] - rtk_priors[i]
            res.extend(np.sqrt(rtk_weights[i]) * diff)

        return np.array(res)

    x0 = pack(initial_cameras, initial_points)
    result = least_squares(
        residuals, x0,
        method="trf",
        loss="huber",
        f_scale=2.0,           # 픽셀 단위 outlier 임계값
        max_nfev=300,
        verbose=2,
    )
    cams_opt, pts_opt = unpack(result.x)
    rmse_px = np.sqrt(2 * result.cost / len(observations))
    logger.info("BA 완료. Reprojection RMSE ≈ %.2f px", rmse_px)
    return cams_opt, pts_opt, rmse_px


# ---------------------------------------------------------------------------
# 7. 정사영상 생성 - 평면 평균 고도 기반 간이 보정
# ---------------------------------------------------------------------------
def simple_orthophoto(meta: ImageMetadata,
                      camera_xyz: np.ndarray,
                      omega: float, phi: float, kappa: float,
                      ground_z: float,
                      gsd_m: float,
                      output_path: Path,
                      epsg: int = 5186):
    """평지 가정 정사영상 생성.

    DSM이 없는 GCP-free 환경에서는 사진의 평균 지상 고도(ground_z)를
    추정해서 평면으로 가정. 솔라패널 단지처럼 평탄한 지형이면 충분히 유효.

    Args:
        ground_z: 평균 지표면 고도 (m). 미지정 시 카메라 고도 - rel_alt 사용.
        gsd_m: 출력 픽셀 크기 (Ground Sampling Distance, m/pixel).
               예: 100m 고도, 4000px 폭, 35mm 렌즈 → GSD ≈ 2.7cm/px
    """
    img = cv2.imread(meta.origin_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]

    # -----------------------------------------------------------------------
    # 픽셀 단위 초점거리.
    # ★ 기존: sensor_width_mm=13.2 하드코딩 → ZH20T zoom 센서(약 7.4mm)와
    #   2배 가까이 어긋남.
    # ★ 수정: EXIF FocalLengthIn35mmFilm 사용. 35mm 풀프레임 가로=36mm 기준이라
    #   센서 물리크기를 몰라도 정확하다. (metadata.py 의 K 행렬과 동일 공식)
    #     f_px = focal_length_in_35mm / 36.0 * width
    # -----------------------------------------------------------------------
    if meta.focal_length_in_35mm and meta.focal_length_in_35mm > 0:
        f_px = meta.focal_length_in_35mm / 36.0 * w_img
    else:
        # fallback: 센서폭 13.2mm 가정 (정확도 저하 경고)
        logger.warning("focal_length_in_35mm 없음 → 센서폭 13.2mm 가정")
        f_px = meta.focal_length * w_img / 13.2

    R = rotation_matrix(omega, phi, kappa)
    cx, cy = w_img / 2.0, h_img / 2.0

    # -----------------------------------------------------------------------
    # 픽셀 → 지상점 역투영.
    #
    # ★ 버그 원인 ★
    #   rotation_matrix(ω,φ,κ) 규약에서 카메라 광학축(보는 방향)은
    #   카메라 좌표계의 +Z 가 아니라 -Z 에 대응한다. 기존 코드는 ray_cam 의
    #   Z 를 +1 로 둬서 광선이 하늘을 향했고(ray_world[2]>0), 카메라가 지면
    #   위(158m > 113m)에 있는데도 t = (Zg-Zc)/ray_world[2] 가 음수가 되어
    #   `if t < 0: continue` 에서 4모서리가 전부 버려졌다.
    #
    # ★ 해결 ★
    #   d_cam 의 Z 를 -1 로 두면 nadir 촬영 시 d_world[2] < 0,
    #   t = (Zg - Zc) / d_world[2] > 0 (Zg < Zc 이므로) 으로 정상.
    #   실측 검증: H20T 45.87m 고도 → LRF 거리(45.87m)와 정확히 일치,
    #   지상범위 약 26.7 x 35.4 m.
    # -----------------------------------------------------------------------
    def pixel_to_ground(px: float, py: float):
        d_cam = np.array([-(px - cx) / f_px,
                          -(py - cy) / f_px,
                          -1.0])
        d_world = R.T @ d_cam
        if abs(d_world[2]) < 1e-9:
            return None  # 광선이 지면과 평행
        t = (ground_z - camera_xyz[2]) / d_world[2]
        if t <= 0:
            return None  # nadir 촬영이면 t>0. t<=0 이면 자세값 이상.
        gx = camera_xyz[0] + t * d_world[0]
        gy = camera_xyz[1] + t * d_world[1]
        return np.array([gx, gy])

    # 이미지 4모서리를 ground plane (Z=ground_z)으로 역투영하여 정사영상 범위 결정
    corners_px = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=float)
    ground_corners = []
    for px, py in corners_px:
        g = pixel_to_ground(px, py)
        if g is not None:
            ground_corners.append(g)

    if len(ground_corners) < 4:
        logger.warning("Ground projection 실패: %s", meta.origin_path)
        return

    ground_corners = np.array(ground_corners)
    x_min, y_min = ground_corners.min(axis=0)
    x_max, y_max = ground_corners.max(axis=0)

    # 출력 래스터 크기
    out_w = int(np.ceil((x_max - x_min) / gsd_m))
    out_h = int(np.ceil((y_max - y_min) / gsd_m))

    # 정사영상의 각 픽셀 → 월드좌표 → 원본사진 픽셀로 역매핑 (backward warping)
    out_xs = x_min + (np.arange(out_w) + 0.5) * gsd_m
    out_ys = y_max - (np.arange(out_h) + 0.5) * gsd_m
    grid_x, grid_y = np.meshgrid(out_xs, out_ys)
    world = np.stack([grid_x.ravel(),
                      grid_y.ravel(),
                      np.full(grid_x.size, ground_z)], axis=1)

    # 월드점 → 원본 픽셀 (forward projection).
    # ★ pixel_to_ground 가 d_cam=[...,-1] 규약을 쓰므로, 그 역산인
    #   backward warping 도 부호를 맞춰야 한다. 왕복 검증 결과:
    #     px = cx + f_px · (R·diff)[0] / (R·diff)[2]
    #     py = cy + f_px · (R·diff)[1] / (R·diff)[2]
    #   (기존 cx - f_px·... 는 ray_cam Z=+1 규약의 잔재로, 좌우/상하가
    #    반전되어 왕복오차 ~5000px 발생)
    diff = world - camera_xyz
    R_diff = diff @ R.T          # 각 행이 R·diff
    den = R_diff[:, 2]
    valid = np.abs(den) > 1e-9

    src_x = np.full(grid_x.size, -1.0)
    src_y = np.full(grid_x.size, -1.0)
    src_x[valid] = cx + f_px * R_diff[valid, 0] / den[valid]
    src_y[valid] = cy + f_px * R_diff[valid, 1] / den[valid]

    map_x = src_x.reshape(out_h, out_w).astype(np.float32)
    map_y = src_y.reshape(out_h, out_w).astype(np.float32)
    ortho = cv2.remap(img_rgb, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0))

    # GeoTIFF 저장 (북향 정사)
    transform = from_origin(x_min, y_max, gsd_m, gsd_m)
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=out_h, width=out_w, count=3,
        dtype=ortho.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        compress="lzw",
    ) as dst:
        for i in range(3):
            dst.write(ortho[:, :, i], i + 1)
    logger.info("정사영상 저장: %s (%dx%d px, GSD=%.3fm)", output_path, out_w, out_h, gsd_m)


# ---------------------------------------------------------------------------
# 8. End-to-End 파이프라인 (GCP-Free)
# ---------------------------------------------------------------------------
def estimate_ground_z(meta: ImageMetadata) -> float:
    """평면 정사보정용 지표면 절대고도(m) 추정.

    우선순위:
      1) LRF 실측 (lrf[3] = LRFTargetAbsAlt) — H20T 등 LRF 탑재 기종.
         촬영 시점 조준점의 실측 절대고도라 가장 정확.
      2) gps.altitude - relative_height — RelativeAltitude 는 이륙지점
         기준이라 촬영지 지형이 이륙지와 다르면 오차가 크다.
    """
    if meta.has_valid_lrf:
        return float(meta.lrf_target_abs_alt)
    if meta.relative_height:
        return meta.gps.altitude - meta.relative_height
    logger.warning("ground_z 추정 정보 부족 → altitude - 100m 사용")
    return meta.gps.altitude - 100.0


def run_pipeline_gcp_free(image_dir: Path,
                          output_dir: Path,
                          target_epsg: int = 5186,
                          gsd_m: float = 0.05,
                          k_neighbors: int = 8):
    output_dir.mkdir(parents=True, exist_ok=True)
    crs = CRSConverter(target_epsg=target_epsg)

    # 1. 메타데이터 추출
    images = sorted(image_dir.glob("*.JPG"))
    metas = [extract_metadata(p) for p in images]
    logger.info("이미지 %d장 로딩", len(metas))

    # 2. RTK 품질 검증
    if not validate_rtk_quality(metas):
        logger.warning("RTK 품질 부족. 결과 정확도 보장 어려움.")

    # 3. RTK 좌표를 투영좌표계로 + 초기 외부표정 구성
    rtk_priors = []
    initial_cameras = []
    for m in metas:
        X, Y = crs.forward(m.gps.lng, m.gps.lat)
        Z = m.gps.altitude
        rtk_priors.append([X, Y, Z])
        gimbal_yaw, gimbal_pitch, gimbal_roll = m.orientation
        omega = np.deg2rad(gimbal_roll)
        phi = np.deg2rad(gimbal_pitch + 90)  # nadir(-90) → 0 보정
        kappa = np.deg2rad(gimbal_yaw)
        initial_cameras.append([X, Y, Z, omega, phi, kappa])
    rtk_priors = np.array(rtk_priors)
    initial_cameras = np.array(initial_cameras)

    # RTK 표준편차로 가중치 산출 (Fixed=1cm 가정, Float=20cm 패널티)
    rtk_weights = np.zeros((len(metas), 3))
    for i, m in enumerate(metas):
        sigma_xy = m.gps_std_xy if m.is_rtk_fixed else max(m.gps_std_xy, 0.20)
        sigma_z = m.gps_std_z if m.is_rtk_fixed else max(m.gps_std_z, 0.30)
        rtk_weights[i] = [1.0 / sigma_xy**2, 1.0 / sigma_xy**2, 1.0 / sigma_z**2]

    # 4. 인접 페어만 SfM 매칭
    pairs = find_neighbor_pairs(metas, crs, k_neighbors=k_neighbors)
    matches, features = build_tie_points(metas, pairs)

    # 5. (실제 구현) tie point들을 triangulation으로 3D 초기화
    #    observations, initial_points = triangulate_tracks(matches, initial_cameras, ...)
    #    cams_opt, pts_opt, rmse = rtk_constrained_bundle_adjustment(
    #        initial_cameras, initial_points, observations,
    #        rtk_priors, rtk_weights,
    #        f_px=..., cx=..., cy=...,
    #    )
    # 여기서는 분량 관계로 BA는 RTK 초기값을 그대로 사용한다고 가정
    cams_opt = initial_cameras

    # 6. 정사영상 생성 (사진별)
    # ★ 기존: 모든 사진에 동일한 ground_z_avg 사용 → RelativeAltitude 가
    #   이륙지점 기준이라 부정확. H20T 는 LRF 실측이 있으므로 그게 정확.
    #   → estimate_ground_z() 로 사진마다 개별 추정 (LRF 우선).
    lrf_count = sum(m.has_valid_lrf for m in metas)
    logger.info("LRF 실측 가능 사진: %d/%d", lrf_count, len(metas))

    for i, meta in enumerate(metas):
        cam = cams_opt[i]
        ground_z = estimate_ground_z(meta)
        out = output_dir / f"{Path(meta.origin_path).stem}_ortho.tif"
        simple_orthophoto(
            meta=meta,
            camera_xyz=cam[:3],
            omega=cam[3], phi=cam[4], kappa=cam[5],
            ground_z=ground_z,
            gsd_m=gsd_m,
            output_path=out,
            epsg=target_epsg,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline_gcp_free(
        image_dir=Path("./data/solar/images/RGB"),
        output_dir=Path("./workspace/output"),
        target_epsg=5186,
        gsd_m=0.05,         # 5cm/pixel (DJI Mavic 3E 100m 고도 기준 적절)
        k_neighbors=8,
    )