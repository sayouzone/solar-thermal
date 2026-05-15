"""평면 가정 정사영상 생성 — backward warping.

DSM 이 없는 GCP-free 환경에서는 사진의 평균 지상 고도(``ground_z``)를
추정해서 평면으로 가정한다. 솔라패널 단지처럼 평탄한 지형이면 충분히 유효.

부호 규약 (중요)
----------------
``geometry.rotation_matrix(ω,φ,κ)`` 규약에서 카메라 광학축(보는 방향)은
카메라 좌표계 **-Z** 에 대응한다. 따라서:

* **픽셀 → 지상점 (forward 화면범위 계산)** 의 ray 는 ``d_cam = [-x/f, -y/f, -1]``
  로 둬야 nadir 촬영에서 ``d_world[2] < 0`` 이 나와 ``t = (Zg-Zc)/d_world[2] > 0``
  (지상이 카메라 아래이므로 ``Zg < Zc``) 으로 정상.
* **지상점 → 픽셀 (backward warping)** 도 같은 부호 규약을 유지해야 왕복 일치::

    px = cx + f_px · (R·diff)[0] / (R·diff)[2]
    py = cy + f_px · (R·diff)[1] / (R·diff)[2]

  (기존 ``cx - f_px·...`` 는 ray_cam Z=+1 규약의 잔재로, 좌우/상하 반전되어
  왕복오차 ~5000 px 발생.)

가속 전략
---------
* **좌표 생성 (가장 큰 연산)**: CuPy meshgrid + einsum-like 벡터 연산.
  ``out_w × out_h`` 픽셀 (수백만~수천만) 의 ray 를 한 번에 계산.
* **remap**: ``cv2.cuda.remap`` 사용 시 GpuMat 업로드/다운로드 시간이 있어
  단일 호출 효과는 출력 해상도가 1M 픽셀 이상일 때만 의미가 있다.
* 입력 이미지 크기가 작으면 CPU 가 더 빠르므로 임계치 가드를 둔다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

from solar_thermal.image.metadata import ImageMetadata

from ..geometry import compute_focal_px, rotation_matrix
from ..gpu_backend import HAS_CUPY, HAS_CV_CUDA_REMAP, cp

logger = logging.getLogger(__name__)

# 출력 픽셀 수 임계치 — 이 이하에서는 GPU 전송 오버헤드가 SIMT 이득보다 크다.
_GPU_MIN_PIXELS = 1_000_000


# ---------------------------------------------------------------------------
# Backward warping 좌표 생성
# ---------------------------------------------------------------------------
def _build_warp_maps_gpu(out_w: int, out_h: int,
                         x_min: float, y_max: float, gsd_m: float,
                         ground_z: float,
                         R: np.ndarray, camera_xyz: np.ndarray,
                         f_px: float, cx: float, cy: float):
    """CuPy 로 (map_x, map_y) 생성. 반환은 numpy float32."""
    out_xs = x_min + (cp.arange(out_w, dtype=cp.float64) + 0.5) * gsd_m
    out_ys = y_max - (cp.arange(out_h, dtype=cp.float64) + 0.5) * gsd_m
    grid_x, grid_y = cp.meshgrid(out_xs, out_ys)
    world = cp.stack([grid_x.ravel(),
                      grid_y.ravel(),
                      cp.full(grid_x.size, ground_z, dtype=cp.float64)],
                     axis=1)
    R_gpu = cp.asarray(R)
    cam_gpu = cp.asarray(camera_xyz)
    diff = world - cam_gpu
    R_diff = diff @ R_gpu.T
    den = R_diff[:, 2]
    valid = cp.abs(den) > 1e-9
    safe_den = cp.where(valid, den, 1)
    src_x = cp.where(valid, cx + f_px * R_diff[:, 0] / safe_den, -1.0)
    src_y = cp.where(valid, cy + f_px * R_diff[:, 1] / safe_den, -1.0)
    map_x = cp.asnumpy(src_x.reshape(out_h, out_w).astype(cp.float32))
    map_y = cp.asnumpy(src_y.reshape(out_h, out_w).astype(cp.float32))
    return map_x, map_y


def _build_warp_maps_np(out_w: int, out_h: int,
                        x_min: float, y_max: float, gsd_m: float,
                        ground_z: float,
                        R: np.ndarray, camera_xyz: np.ndarray,
                        f_px: float, cx: float, cy: float):
    out_xs = x_min + (np.arange(out_w) + 0.5) * gsd_m
    out_ys = y_max - (np.arange(out_h) + 0.5) * gsd_m
    grid_x, grid_y = np.meshgrid(out_xs, out_ys)
    world = np.stack([grid_x.ravel(),
                      grid_y.ravel(),
                      np.full(grid_x.size, ground_z)], axis=1)
    diff = world - camera_xyz
    R_diff = diff @ R.T
    den = R_diff[:, 2]
    valid = np.abs(den) > 1e-9
    src_x = np.full(grid_x.size, -1.0)
    src_y = np.full(grid_x.size, -1.0)
    src_x[valid] = cx + f_px * R_diff[valid, 0] / den[valid]
    src_y[valid] = cy + f_px * R_diff[valid, 1] / den[valid]
    return (src_x.reshape(out_h, out_w).astype(np.float32),
            src_y.reshape(out_h, out_w).astype(np.float32))


# ---------------------------------------------------------------------------
# Pixel → ground (4모서리 화면범위 계산용)
# ---------------------------------------------------------------------------
def _pixel_to_ground(px: float, py: float,
                     R: np.ndarray, camera_xyz: np.ndarray,
                     ground_z: float,
                     f_px: float, cx: float, cy: float) -> np.ndarray | None:
    """단일 픽셀 → ground plane 교점. 광선이 평행이거나 위로 향하면 None."""
    d_cam = np.array([-(px - cx) / f_px, -(py - cy) / f_px, -1.0])
    d_world = R.T @ d_cam
    if abs(d_world[2]) < 1e-9:
        return None  # 광선이 지면과 평행.
    t = (ground_z - camera_xyz[2]) / d_world[2]
    if t <= 0:
        return None  # nadir 촬영이면 t>0. t<=0 이면 자세값 이상.
    gx = camera_xyz[0] + t * d_world[0]
    gy = camera_xyz[1] + t * d_world[1]
    return np.array([gx, gy])


# ---------------------------------------------------------------------------
# remap (cv2.cuda.remap 또는 CPU)
# ---------------------------------------------------------------------------
def _remap_gpu(img_rgb: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    gpu_src = cv2.cuda_GpuMat()
    gpu_src.upload(img_rgb)
    gpu_mx = cv2.cuda_GpuMat()
    gpu_mx.upload(map_x)
    gpu_my = cv2.cuda_GpuMat()
    gpu_my.upload(map_y)
    gpu_dst = cv2.cuda.remap(
        gpu_src, gpu_mx, gpu_my,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return gpu_dst.download()


def _remap_cpu(img_rgb: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        img_rgb, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def simple_orthophoto(meta: ImageMetadata,
                      camera_xyz: np.ndarray,
                      omega: float, phi: float, kappa: float,
                      ground_z: float,
                      gsd_m: float,
                      output_path: Path,
                      epsg: int = 5186) -> None:
    """평지 가정 정사영상 생성 → GeoTIFF 저장.

    Parameters
    ----------
    meta : 메타데이터 (원본 이미지 경로, 내부 파라미터 등).
    camera_xyz, omega/phi/kappa : BA 후 카메라 외부표정 (또는 RTK 초기값).
    ground_z : 평균 지표면 절대고도 (m). ``rtk.estimate_ground_z`` 권장.
    gsd_m : 출력 픽셀 크기 (Ground Sampling Distance, m/pixel).
        예: 100m 고도, 4000px 폭, 35mm 렌즈 → GSD ≈ 2.7cm/px.
    output_path : GeoTIFF 저장 경로.
    epsg : 출력 좌표계 EPSG (기본 5186).
    """
    img = cv2.imread(meta.origin_path, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("이미지 로드 실패: %s", meta.origin_path)
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]

    f_px = compute_focal_px(meta.focal_length_in_35mm,
                            meta.focal_length, w_img)
    R = rotation_matrix(omega, phi, kappa)
    cx, cy = w_img / 2.0, h_img / 2.0

    # 1) 4 모서리 → ground plane 으로 출력 범위 계산.
    corners_px = np.array([[0, 0], [w_img, 0],
                           [w_img, h_img], [0, h_img]], dtype=float)
    ground_corners = []
    for px, py in corners_px:
        g = _pixel_to_ground(px, py, R, camera_xyz, ground_z, f_px, cx, cy)
        if g is not None:
            ground_corners.append(g)

    if len(ground_corners) < 4:
        logger.warning("Ground projection 실패: %s", meta.origin_path)
        return

    ground_corners = np.array(ground_corners)
    x_min, y_min = ground_corners.min(axis=0)
    x_max, y_max = ground_corners.max(axis=0)
    out_w = int(np.ceil((x_max - x_min) / gsd_m))
    out_h = int(np.ceil((y_max - y_min) / gsd_m))
    n_pixels = out_w * out_h

    # 2) backward warping maps 생성 (GPU/CPU 자동 분기).
    if HAS_CUPY and n_pixels >= _GPU_MIN_PIXELS:
        map_x, map_y = _build_warp_maps_gpu(
            out_w, out_h, x_min, y_max, gsd_m,
            ground_z, R, camera_xyz, f_px, cx, cy,
        )
    else:
        map_x, map_y = _build_warp_maps_np(
            out_w, out_h, x_min, y_max, gsd_m,
            ground_z, R, camera_xyz, f_px, cx, cy,
        )

    # 3) remap (GPU 가용 + 큰 출력일 때만 GPU).
    if HAS_CV_CUDA_REMAP and n_pixels >= _GPU_MIN_PIXELS:
        try:
            ortho = _remap_gpu(img_rgb, map_x, map_y)
        except Exception as e:
            logger.warning("cuda.remap 실패 (%s) → CPU remap", e)
            ortho = _remap_cpu(img_rgb, map_x, map_y)
    else:
        ortho = _remap_cpu(img_rgb, map_x, map_y)

    # 4) GeoTIFF 저장 (북향 정사).
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
    logger.info("정사영상 저장: %s (%dx%d px, GSD=%.3fm)",
                output_path, out_w, out_h, gsd_m)


__all__ = ["simple_orthophoto"]