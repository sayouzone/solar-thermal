"""RTK 제약 Bundle Adjustment.

핵심 아이디어
-------------
GCP 가 없으므로 절대 좌표계 기준점은 RTK 측정값. 하지만 RTK 도 cm 급 오차가
있으므로 hard constraint 가 아닌 **soft constraint (가중치 = 1/σ²)** 로 잔차에
추가::

    residual = [reprojection_errors, sqrt(w) * (camera_pos - rtk_prior)]

* reprojection error 가 일관된 internal geometry 를 보장
* RTK prior 가 절대 georeferencing 을 보장
* 둘이 가중 평균되어 outlier 에 강건한 해를 찾음

가속 전략
---------
1. **관측 평탄화**: 모든 관측 ``(cam_idx, pt_idx, uv)`` 를 (M, 2) 배열로 펴서
   ``residuals(x)`` 호출당 단일 ``einsum`` 으로 처리. 원본은 Python 루프로
   M 번 ``project_point()`` 를 불러 호출 오버헤드가 압도적이었다.
2. **희소 jacobian**: 각 reprojection 잔차는 *자기 카메라 6개 + 자기 점 3개*
   파라미터에만 의존. 각 RTK 잔차는 *자기 카메라 위치 3개* 에만 의존.
   이 sparsity 패턴을 ``scipy.optimize.least_squares`` 에 넘기면 finite-diff
   jacobian 계산 비용이 수십 배 감소. **GPU 없이도 가장 큰 가속 효과.**
3. **CuPy backend**: 관측이 충분히 많을 때 (M ≥ 5000) ``residuals`` 본체를
   GPU 로 옮긴다. 적은 관측은 PCIe 전송 오버헤드가 SIMT 이득보다 커서
   numpy 가 더 빠르다.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, lil_matrix

from ..geometry import rotation_matrices_batch_cp, rotation_matrices_batch_np
from ..gpu_backend import HAS_CUPY, cp, free_gpu_memory

logger = logging.getLogger(__name__)

# GPU 전환 임계치 (관측 수). 이 이하에서는 numpy 벡터화가 빠르다.
_GPU_MIN_OBS = 5000


# ---------------------------------------------------------------------------
# Residual 콜백 빌더
# ---------------------------------------------------------------------------
def _build_residuals_gpu(n_cam: int, n_pts: int,
                         obs_cam_idx: np.ndarray, obs_pt_idx: np.ndarray,
                         obs_uv: np.ndarray,
                         rtk_priors: np.ndarray, rtk_w_sqrt: np.ndarray,
                         f_px: float, cx: float, cy: float):
    """CuPy 벡터화 residual. 입력은 numpy, 내부에서 GPU 로 이전."""
    obs_cam_idx_g = cp.asarray(obs_cam_idx)
    obs_pt_idx_g = cp.asarray(obs_pt_idx)
    obs_uv_g = cp.asarray(obs_uv)
    rtk_priors_g = cp.asarray(rtk_priors)
    rtk_w_sqrt_g = cp.asarray(rtk_w_sqrt)

    def residuals(x):
        x_g = cp.asarray(x)
        cams = x_g[: n_cam * 6].reshape(n_cam, 6)
        pts = x_g[n_cam * 6:].reshape(n_pts, 3)

        R = rotation_matrices_batch_cp(cams[:, 3:6], cp)  # (n_cam, 3, 3)
        C = cams[:, :3]                                    # (n_cam, 3)

        R_o = R[obs_cam_idx_g]               # (M, 3, 3)
        C_o = C[obs_cam_idx_g]               # (M, 3)
        P_o = pts[obs_pt_idx_g]              # (M, 3)
        diff = P_o - C_o                     # (M, 3)
        Rdiff = cp.einsum("mij,mj->mi", R_o, diff)  # (M, 3)
        den = Rdiff[:, 2]
        den_safe = cp.where(cp.abs(den) < 1e-9, 1e-9, den)
        x_pred = cx - f_px * Rdiff[:, 0] / den_safe
        y_pred = cy - f_px * Rdiff[:, 1] / den_safe
        reproj = cp.stack([x_pred - obs_uv_g[:, 0],
                           y_pred - obs_uv_g[:, 1]], axis=1)  # (M, 2)

        rtk_res = rtk_w_sqrt_g * (C - rtk_priors_g)  # (n_cam, 3)

        return cp.asnumpy(cp.concatenate([reproj.ravel(), rtk_res.ravel()]))

    return residuals


def _build_residuals_np(n_cam: int, n_pts: int,
                        obs_cam_idx: np.ndarray, obs_pt_idx: np.ndarray,
                        obs_uv: np.ndarray,
                        rtk_priors: np.ndarray, rtk_w_sqrt: np.ndarray,
                        f_px: float, cx: float, cy: float):
    """numpy 벡터화 residual (CPU). 원본 Python 루프 대비 수십~수백 배 빠름."""

    def residuals(x):
        cams = x[: n_cam * 6].reshape(n_cam, 6)
        pts = x[n_cam * 6:].reshape(n_pts, 3)

        R = rotation_matrices_batch_np(cams[:, 3:6])
        C = cams[:, :3]

        R_o = R[obs_cam_idx]
        C_o = C[obs_cam_idx]
        P_o = pts[obs_pt_idx]
        diff = P_o - C_o
        Rdiff = np.einsum("mij,mj->mi", R_o, diff)
        den = Rdiff[:, 2]
        den_safe = np.where(np.abs(den) < 1e-9, 1e-9, den)
        x_pred = cx - f_px * Rdiff[:, 0] / den_safe
        y_pred = cy - f_px * Rdiff[:, 1] / den_safe
        reproj = np.stack([x_pred - obs_uv[:, 0],
                           y_pred - obs_uv[:, 1]], axis=1)

        rtk_res = rtk_w_sqrt * (C - rtk_priors)

        return np.concatenate([reproj.ravel(), rtk_res.ravel()])

    return residuals


# ---------------------------------------------------------------------------
# Jacobian sparsity
# ---------------------------------------------------------------------------
def _build_jacobian_sparsity(n_cam: int, n_pts: int,
                             obs_cam_idx: np.ndarray,
                             obs_pt_idx: np.ndarray) -> csr_matrix:
    """trf 알고리즘에 sparsity 패턴 전달용 binary 행렬.

    각 reprojection residual (2개) 은 자기 카메라 6개 + 자기 3D 점 3개에만 의존.
    각 RTK residual (3개) 은 자기 카메라 위치 3개에만 의존. → 극도로 희소.
    """
    M = obs_cam_idx.shape[0]
    n_params = n_cam * 6 + n_pts * 3
    n_res = 2 * M + 3 * n_cam
    J = lil_matrix((n_res, n_params), dtype=np.uint8)

    rows = np.arange(M) * 2
    for k in range(6):
        J[rows, obs_cam_idx * 6 + k] = 1
        J[rows + 1, obs_cam_idx * 6 + k] = 1
    pt_offset = n_cam * 6
    for k in range(3):
        J[rows, pt_offset + obs_pt_idx * 3 + k] = 1
        J[rows + 1, pt_offset + obs_pt_idx * 3 + k] = 1

    rtk_row_start = 2 * M
    for i in range(n_cam):
        for k in range(3):
            J[rtk_row_start + 3 * i + k, i * 6 + k] = 1
    return J.tocsr()


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------
def rtk_constrained_bundle_adjustment(
    initial_cameras: np.ndarray,
    initial_points: np.ndarray,
    observations: list[tuple[int, int, np.ndarray]],
    rtk_priors: np.ndarray,
    rtk_weights: np.ndarray,
    f_px: float, cx: float, cy: float,
):
    """RTK prior 가중을 포함한 번들 조정.

    Parameters
    ----------
    initial_cameras : (n_cam, 6) ``[Xc, Yc, Zc, ω, φ, κ]``
    initial_points : (n_pts, 3)
    observations : ``[(cam_idx, point_idx, uv), ...]``
    rtk_priors : (n_cam, 3) RTK 측정 카메라 위치
    rtk_weights : (n_cam, 3) 1/σ² 가중치
    f_px, cx, cy : 대표 내부 파라미터 (모든 카메라 공통 가정)

    Returns
    -------
    cams_opt : (n_cam, 6)
    pts_opt : (n_pts, 3)
    rmse_px : 픽셀 단위 reprojection RMSE
    """
    n_cam = len(initial_cameras)
    n_pts = len(initial_points)
    M = len(observations)

    # 관측을 배열로 평탄화.
    obs_cam_idx = np.empty(M, dtype=np.int64)
    obs_pt_idx = np.empty(M, dtype=np.int64)
    obs_uv = np.empty((M, 2), dtype=np.float64)
    for k, (ci, pi, uv) in enumerate(observations):
        obs_cam_idx[k] = ci
        obs_pt_idx[k] = pi
        obs_uv[k] = uv

    rtk_w_sqrt = np.sqrt(rtk_weights)

    use_gpu = HAS_CUPY and M >= _GPU_MIN_OBS
    backend = "GPU(CuPy)" if use_gpu else "CPU(numpy 벡터화)"
    logger.info("BA backend: %s (관측 %d개, 카메라 %d, 점 %d)",
                backend, M, n_cam, n_pts)

    if use_gpu:
        residuals = _build_residuals_gpu(
            n_cam, n_pts, obs_cam_idx, obs_pt_idx, obs_uv,
            rtk_priors, rtk_w_sqrt, f_px, cx, cy,
        )
    else:
        residuals = _build_residuals_np(
            n_cam, n_pts, obs_cam_idx, obs_pt_idx, obs_uv,
            rtk_priors, rtk_w_sqrt, f_px, cx, cy,
        )

    jac_sparsity = _build_jacobian_sparsity(n_cam, n_pts, obs_cam_idx, obs_pt_idx)

    x0 = np.concatenate([initial_cameras.ravel(), initial_points.ravel()])
    t0 = time.perf_counter()
    result = least_squares(
        residuals, x0,
        method="trf",
        loss="huber",
        f_scale=2.0,            # 픽셀 단위 outlier 임계값
        max_nfev=300,
        jac_sparsity=jac_sparsity,
        verbose=2,
    )
    logger.info("BA wallclock: %.2fs", time.perf_counter() - t0)

    cams_opt = result.x[: n_cam * 6].reshape(n_cam, 6)
    pts_opt = result.x[n_cam * 6:].reshape(n_pts, 3)
    rmse_px = np.sqrt(2 * result.cost / M)
    logger.info("BA 완료. Reprojection RMSE ≈ %.2f px", rmse_px)
    free_gpu_memory()
    return cams_opt, pts_opt, rmse_px


__all__ = ["rtk_constrained_bundle_adjustment"]