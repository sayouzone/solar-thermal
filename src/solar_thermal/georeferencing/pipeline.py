"""End-to-End GCP-Free Georeferencing 파이프라인.

워크플로우
----------
1. EXIF/XMP 메타데이터 추출 → 사진별 RTK 좌표 + 짐벌 자세
2. RTK Fixed 여부 검증 (RtkFlag=50 만 신뢰)
3. SfM Tie Point 추출 (SIFT/ORB + RANSAC) — GPU 가속
4. 두 시점 초기화 → 점진적 SfM 재구성
5. RTK 제약 Bundle Adjustment (카메라 위치를 GCP 처럼 soft constraint)
6. 정사영상 생성 — DSM 없이 평면 평균 고도 기반 간이 정사보정

GCP-free 한계
-------------
* 수평 정확도: 2~5 cm 가능 (RTK Fixed 기준)
* 수직 정확도: 5~15 cm (안테나 위상 중심 오프셋, GNSS 다중경로 영향)
* 절대 정확도가 중요한 측량/검측용은 최소 1~2 점의 Check Point 권장.
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from solar_thermal.image.metadata import extract_metadata

from .crs import CRSConverter
from .features import build_tie_points, find_neighbor_pairs
from .geometry import compute_focal_px
from .gpu_backend import gpu_summary
from .ortho import simple_orthophoto
from .rtk import (
    compute_rtk_prior_weights,
    estimate_ground_z,
    validate_rtk_quality,
)
from .sfm import (
    build_tracks,
    rtk_constrained_bundle_adjustment,
    triangulate_tracks,
)

logger = logging.getLogger(__name__)


def _build_initial_state(metas, crs: CRSConverter):
    """RTK + 짐벌 자세 → 초기 외부표정 + RTK prior 배열.

    ω = roll, φ = pitch + 90° (nadir 보정), κ = yaw  — 모두 radian.
    """
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
    return np.array(rtk_priors), np.array(initial_cameras)


def run_pipeline_gcp_free(image_dir: Path,
                          output_dir: Path,
                          target_epsg: int = 5186,
                          gsd_m: float = 0.05,
                          k_neighbors: int = 8) -> None:
    """전체 파이프라인 실행.

    Parameters
    ----------
    image_dir : DJI JPG 디렉토리.
    output_dir : 정사영상 GeoTIFF 출력 디렉토리.
    target_epsg : 출력 좌표계 EPSG (한국 측량 표준 5186 권장).
    gsd_m : 출력 픽셀 크기 (Ground Sampling Distance, m/pixel).
        예: DJI Mavic 3E 100m 고도 → 0.05 가 적절.
    k_neighbors : KD-Tree 인접 페어 수.
    """
    logger.info("=" * 70)
    logger.info("Georeferencing Pipeline (GCP-Free) — 가속 백엔드: %s",
                gpu_summary())
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    crs = CRSConverter(target_epsg=target_epsg)

    # 1. 메타데이터 추출.
    images = sorted(image_dir.glob("*.JPG"))
    metas = [extract_metadata(p) for p in images]
    logger.info("이미지 %d장 로딩", len(metas))

    # 2. RTK 품질 검증.
    if not validate_rtk_quality(metas):
        logger.warning("RTK 품질 부족. 결과 정확도 보장 어려움.")

    # 3. RTK → 투영좌표 + 초기 외부표정 + prior 가중치.
    rtk_priors, initial_cameras = _build_initial_state(metas, crs)
    rtk_weights = compute_rtk_prior_weights(metas)

    # 4. 인접 페어 SfM 매칭.
    t0 = time.perf_counter()
    pairs = find_neighbor_pairs(metas, crs, k_neighbors=k_neighbors)
    matches, features = build_tie_points(metas, pairs)
    logger.info("[stage] SfM 매칭: %.1fs", time.perf_counter() - t0)

    intrinsics = [
        (compute_focal_px(m.focal_length_in_35mm, m.focal_length, m.width),
         m.width / 2.0, m.height / 2.0)
        for m in metas
    ]

    # 5. Tracks → Triangulation → BA.
    t0 = time.perf_counter()
    tracks = build_tracks(matches, features, min_track_len=2, max_track_len=30)
    logger.info("[stage] track 빌드: %.1fs", time.perf_counter() - t0)

    if len(tracks) < 10:
        logger.warning("track 이 너무 적음(%d). BA 생략, RTK 초기값 사용.",
                       len(tracks))
        cams_opt = initial_cameras
    else:
        t0 = time.perf_counter()
        observations, initial_points, _ = triangulate_tracks(
            tracks, initial_cameras, intrinsics,
            max_reproj_err_px=3.0,
            min_triangulation_angle_deg=2.0,
        )
        logger.info("[stage] triangulation: %.1fs", time.perf_counter() - t0)

        if len(initial_points) < 10 or len(observations) < 30:
            logger.warning("삼각측량 결과 부족. BA 생략, RTK 초기값 사용.")
            cams_opt = initial_cameras
        else:
            # rtk_constrained_bundle_adjustment 는 단일 (f_px, cx, cy) 를 받으므로
            # 대표값(중앙값) 사용. 전 사진 동일 카메라/줌이면 정확히 일치.
            f_px_rep = float(np.median([k[0] for k in intrinsics]))
            cx_rep = float(np.median([k[1] for k in intrinsics]))
            cy_rep = float(np.median([k[2] for k in intrinsics]))
            t0 = time.perf_counter()
            cams_opt, _pts_opt, _rmse = rtk_constrained_bundle_adjustment(
                initial_cameras, initial_points, observations,
                rtk_priors, rtk_weights,
                f_px=f_px_rep, cx=cx_rep, cy=cy_rep,
            )
            logger.info("[stage] bundle adjustment: %.1fs",
                        time.perf_counter() - t0)
            logger.info(
                "BA 후 카메라 위치 평균 이동량: %.3f m",
                float(np.linalg.norm(
                    cams_opt[:, :3] - initial_cameras[:, :3], axis=1
                ).mean()),
            )

    # 6. 정사영상 생성 (사진별).
    lrf_count = sum(m.has_valid_lrf for m in metas)
    logger.info("LRF 실측 가능 사진: %d/%d", lrf_count, len(metas))

    t0 = time.perf_counter()
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
    logger.info("[stage] 정사영상 %d장: %.1fs",
                len(metas), time.perf_counter() - t0)


def main():
    """CLI 진입점. 데모용 기본 경로 사용."""
    start = time.perf_counter()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_pipeline_gcp_free(
        image_dir=Path("./data/solar/images/RGB"),
        output_dir=Path("./workspace/output"),
        target_epsg=5186,
        gsd_m=0.05,
        k_neighbors=8,
    )
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {timedelta(seconds=int(elapsed))}")


if __name__ == "__main__":
    main()


__all__ = ["run_pipeline_gcp_free", "main"]