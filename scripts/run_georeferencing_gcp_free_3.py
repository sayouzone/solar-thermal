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
    if len(metas) < 2:
        logger.warning("이미지가 2장 미만 → 페어 매칭 불가")
        return []
    tree = cKDTree(xy)
    pairs = set()
    for i in range(len(metas)):
        _, idxs = tree.query(xy[i], k=min(k_neighbors + 1, len(metas)))
        idxs = np.atleast_1d(idxs)  # k=1 이면 스칼라가 반환되므로 배열로 강제
        for j in idxs:
            j = int(j)
            if j != i:  # 자기 자신 제외
                pairs.add((min(i, j), max(i, j)))
    logger.info("인접 페어 %d개 생성 (k=%d)", len(pairs), k_neighbors)
    return sorted(pairs)


def build_tie_points(metas: list[ImageMetadata],
                     pairs: list[tuple[int, int]]):
    """SIFT 특징점 추출 + 페어별 매칭 + RANSAC outlier 제거.

    Returns
    -------
    matches : dict[(i, j) -> (pts_i, pts_j, idx_i, idx_j)]
        pts_i, pts_j : (N, 2) 매칭된 픽셀 좌표
        idx_i, idx_j : (N,)   각 매칭의 keypoint 인덱스
                       (track 생성 시 어느 keypoint끼리 연결됐는지 추적용)
    features : dict[i -> (keypoints, descriptors, shape)]
    """
    if not pairs:
        logger.warning("매칭할 페어가 없음 → tie point 생성 생략")
        return {}, {}

    # 페어에 등장하는 이미지만 SIFT 추출 (불필요한 추출/메모리 절약)
    needed = sorted({i for pair in pairs for i in pair})
    features = {i: extract_features(metas[i].origin_path) for i in needed}
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
        idx_i = np.array([m.queryIdx for m in good], dtype=np.int64)
        idx_j = np.array([m.trainIdx for m in good], dtype=np.int64)
        _, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 1.0, 0.99)
        if mask is None:
            continue
        inliers = mask.ravel().astype(bool)
        if inliers.sum() < 20:
            continue
        matches[(i, j)] = (pts_i[inliers], pts_j[inliers],
                           idx_i[inliers], idx_j[inliers])

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


def matrix_to_opk(R: np.ndarray) -> tuple[float, float, float]:
    """회전행렬 → ω-φ-κ (rotation_matrix 의 역변환).

    rotation_matrix 정의에서:
        R[0,2] =  sin(φ)
        R[1,2] = -sin(ω)·cos(φ)
        R[2,2] =  cos(ω)·cos(φ)
        R[0,1] = -cos(φ)·sin(κ)
        R[0,0] =  cos(φ)·cos(κ)
    RGB↔IR rig 변환을 ω-φ-κ 파라미터로 다시 표현할 때 필요.
    """
    phi = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    cp = np.cos(phi)
    if abs(cp) < 1e-8:
        # gimbal lock (φ ≈ ±90°) — 드물지만 안전 처리
        omega = 0.0
        kappa = np.arctan2(R[1, 0], R[1, 1])
    else:
        omega = np.arctan2(-R[1, 2], R[2, 2])
        kappa = np.arctan2(-R[0, 1], R[0, 0])
    return float(omega), float(phi), float(kappa)


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


def camera_projection_matrix(camera_xyz: np.ndarray,
                             omega: float, phi: float, kappa: float,
                             f_px: float, cx: float, cy: float) -> np.ndarray:
    """project_point()와 동일한 투영을 표현하는 3x4 카메라 행렬 P.

    project_point 의 공선조건식:
        x = cx - f_px·(R[0]·diff)/(R[2]·diff),  diff = world - camera
    은 다음 P 로 표현된다 (검증 완료):
        K_neg = [[-f_px, 0,     cx],
                 [0,     -f_px, cy],
                 [0,     0,     1 ]]
        P = K_neg · [R | -R·C]
    그러면  [u·w, v·w, w]ᵀ = P · [X, Y, Z, 1]ᵀ,  (u, v) = (u·w/w, v·w/w).

    DLT 삼각측량은 이 P 행렬을 그대로 사용한다.
    """
    R = rotation_matrix(omega, phi, kappa)
    K_neg = np.array([
        [-f_px, 0.0,   cx],
        [0.0,   -f_px, cy],
        [0.0,   0.0,   1.0],
    ])
    Rt = np.hstack([R, (-R @ camera_xyz).reshape(3, 1)])  # [R | -R·C]
    return K_neg @ Rt


# ---------------------------------------------------------------------------
# 5b. Tie point tracks → 3D 점 (Triangulation 초기화)
# ---------------------------------------------------------------------------
class _UnionFind:
    """경로 압축 + 랭크 기반 Union-Find.

    (image_idx, keypoint_idx) 노드들을 연결해서, 여러 사진에 걸쳐 같은
    지상점을 보고 있는 특징점들을 하나의 집합(track)으로 묶는다.
    """

    def __init__(self):
        self.parent: dict = {}
        self.rank: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        self.rank.setdefault(x, 0)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # 경로 압축
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_tracks(matches: dict,
                 features: dict,
                 min_track_len: int = 2,
                 max_track_len: int = 30) -> list[list[tuple[int, int, float, float]]]:
    """페어별 매칭들을 연결해 track 리스트를 만든다.

    track = 동일 지상점에 대응하는 관측들의 집합
          = [(image_idx, keypoint_idx, px, py), ...]

    Parameters
    ----------
    matches : build_tie_points 의 출력
    features : build_tie_points 의 출력 (keypoint 픽셀좌표 조회용)
    min_track_len : track 으로 인정할 최소 관측 수 (2 = 최소 삼각측량 가능)
    max_track_len : 비정상적으로 긴 track 제거 (보통 매칭 오류).
                    하나의 점이 30장 이상에 나타나면 의심스러움.

    Returns
    -------
    tracks : list of track
    """
    uf = _UnionFind()

    # 1) 모든 매칭을 union — (i, kp_i) 와 (j, kp_j) 를 같은 집합으로
    for (i, j), (_, _, idx_i, idx_j) in matches.items():
        for ki, kj in zip(idx_i, idx_j):
            uf.union((i, int(ki)), (j, int(kj)))

    # 2) 루트별로 관측을 모음
    groups: dict = {}
    for (i, j), (pts_i, pts_j, idx_i, idx_j) in matches.items():
        for n in range(len(idx_i)):
            ki, kj = int(idx_i[n]), int(idx_j[n])
            root = uf.find((i, ki))
            obs = groups.setdefault(root, {})
            # 같은 (image, keypoint) 는 dict 키로 중복 자동 제거
            obs[(i, ki)] = (i, ki, float(pts_i[n][0]), float(pts_i[n][1]))
            obs[(j, kj)] = (j, kj, float(pts_j[n][0]), float(pts_j[n][1]))

    # 3) track 필터링
    tracks = []
    dropped_short = dropped_long = dropped_conflict = 0
    for root, obs in groups.items():
        # 한 이미지에서 두 개 이상의 keypoint 가 한 track 에 들어가면
        # 매칭 충돌 → track 전체를 버린다 (오염 방지).
        images_seen = [img for (img, _kp) in obs.keys()]
        if len(images_seen) != len(set(images_seen)):
            dropped_conflict += 1
            continue
        track = list(obs.values())
        if len(track) < min_track_len:
            dropped_short += 1
            continue
        if len(track) > max_track_len:
            dropped_long += 1
            continue
        tracks.append(track)

    logger.info(
        "track 생성: %d개 (제거: 짧음 %d, 김 %d, 충돌 %d)",
        len(tracks), dropped_short, dropped_long, dropped_conflict,
    )
    if tracks:
        lengths = np.array([len(t) for t in tracks])
        logger.info("  track 길이 분포: 평균 %.1f, 최대 %d, 2장짜리 %d개",
                    lengths.mean(), lengths.max(), (lengths == 2).sum())
    return tracks


def triangulate_dlt(proj_mats: list[np.ndarray],
                    points_2d: list[np.ndarray]) -> np.ndarray:
    """DLT(Direct Linear Transform) 다중시점 삼각측량.

    각 관측 (u, v) 와 카메라 P 에 대해, [u·w, v·w, w]ᵀ = P·[X,Y,Z,1]ᵀ 에서
    u, v 를 소거하면 X=[X,Y,Z,1]ᵀ 에 대한 2개의 동차 선형식을 얻는다:
        u·(P[2]·X) - (P[0]·X) = 0
        v·(P[2]·X) - (P[1]·X) = 0
    N개 시점이면 2N x 4 행렬 A 가 되고, A·X = 0 의 최소제곱해는
    AᵀA 의 최소 특이값에 대응하는 우특이벡터 (SVD 마지막 행).

    Parameters
    ----------
    proj_mats : 각 관측에 대응하는 3x4 카메라 행렬 P 리스트
    points_2d : 각 관측의 (u, v) 픽셀좌표 리스트

    Returns
    -------
    point_3d : (3,) 복원된 지상점 (X, Y, Z)
    """
    A = []
    for P, (u, v) in zip(proj_mats, points_2d):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)                       # (2N, 4)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]                              # 최소 특이값의 우특이벡터
    if abs(X[3]) < 1e-12:
        return np.array([np.nan, np.nan, np.nan])
    return X[:3] / X[3]                     # 동차 → 3D


def triangulate_tracks(tracks: list,
                       cameras: np.ndarray,
                       intrinsics: list[tuple[float, float, float]],
                       max_reproj_err_px: float = 3.0,
                       min_triangulation_angle_deg: float = 2.0):
    """track 들을 삼각측량해서 BA 입력(observations, initial_points) 생성.

    Parameters
    ----------
    tracks : build_tracks 의 출력
    cameras : (n_cam, 6) 초기 외부표정 [Xc, Yc, Zc, ω, φ, κ]
              (RTK 좌표 + 짐벌 자세에서 만든 초기값)
    intrinsics : 카메라별 (f_px, cx, cy) 리스트
    max_reproj_err_px : 삼각측량 후 재투영 오차가 이 값을 넘는 track 은 제거
    min_triangulation_angle_deg : 시선각(parallax)이 너무 작으면 깊이가
        불안정하므로 제거. nadir 드론 사진은 베이스라인이 짧아 각이
        작은 편이라 2도 정도로 완화.

    Returns
    -------
    observations : list[(cam_idx, point_idx, np.array([u, v]))]
    initial_points : (n_pts, 3) 삼각측량된 3D 점들
    track_point_map : list[int] — initial_points[k] 가 몇 번째 track 인지
    """
    # 카메라별 P 행렬 미리 계산
    proj_cache = {}
    centers = {}
    for cam_idx in range(len(cameras)):
        Xc, Yc, Zc, om, ph, ka = cameras[cam_idx]
        f_px, cx, cy = intrinsics[cam_idx]
        C = np.array([Xc, Yc, Zc])
        proj_cache[cam_idx] = camera_projection_matrix(C, om, ph, ka, f_px, cx, cy)
        centers[cam_idx] = C

    observations = []
    initial_points = []
    track_point_map = []
    dropped_angle = dropped_reproj = dropped_degenerate = 0

    for t_idx, track in enumerate(tracks):
        cam_indices = [obs[0] for obs in track]
        pts2d = [np.array([obs[2], obs[3]]) for obs in track]
        Ps = [proj_cache[ci] for ci in cam_indices]

        # --- DLT 삼각측량 ---
        X = triangulate_dlt(Ps, pts2d)
        if not np.all(np.isfinite(X)):
            dropped_degenerate += 1
            continue

        # --- 시선각(parallax) 체크 ---
        # 가장 멀리 떨어진 두 카메라에서 점을 바라본 방향벡터 사이 각도.
        rays = [(X - centers[ci]) for ci in cam_indices]
        rays = [r / (np.linalg.norm(r) + 1e-12) for r in rays]
        max_angle = 0.0
        for a in range(len(rays)):
            for b in range(a + 1, len(rays)):
                cos_ang = np.clip(rays[a] @ rays[b], -1.0, 1.0)
                max_angle = max(max_angle, np.degrees(np.arccos(cos_ang)))
        if max_angle < min_triangulation_angle_deg:
            dropped_angle += 1
            continue

        # --- 재투영 오차 체크 ---
        Xh = np.append(X, 1.0)
        reproj_errs = []
        for P, uv in zip(Ps, pts2d):
            uvw = P @ Xh
            if abs(uvw[2]) < 1e-12:
                reproj_errs.append(1e9)
                continue
            uv_pred = uvw[:2] / uvw[2]
            reproj_errs.append(np.linalg.norm(uv_pred - uv))
        if np.mean(reproj_errs) > max_reproj_err_px:
            dropped_reproj += 1
            continue

        # --- 통과: observations 에 추가 ---
        point_idx = len(initial_points)
        initial_points.append(X)
        track_point_map.append(t_idx)
        for ci, uv in zip(cam_indices, pts2d):
            observations.append((ci, point_idx, uv))

    initial_points = (np.array(initial_points)
                      if initial_points else np.zeros((0, 3)))
    logger.info(
        "삼각측량: %d개 3D점 복원, %d개 관측 (제거: 시선각 %d, 재투영 %d, 퇴화 %d)",
        len(initial_points), len(observations),
        dropped_angle, dropped_reproj, dropped_degenerate,
    )
    if len(initial_points) > 0:
        zs = initial_points[:, 2]
        logger.info("  복원점 Z범위: %.1f ~ %.1f m (중앙값 %.1f)",
                    zs.min(), zs.max(), np.median(zs))
    return observations, initial_points, track_point_map


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
# 6b. RGB-IR 동시 Georeferencing (H20T 멀티센서 rig BA)
# ---------------------------------------------------------------------------
#
# H20T 는 광각 RGB(Wide/Zoom)와 협시야 열화상 IR 을 같은 짐벌에 탑재한다.
# 두 센서는 물리적으로 다른 렌즈/해상도/초점거리를 가지지만, 강체(rigid)로
# 고정되어 있어 "rig 변환" 하나로 묶인다.
#
# 핵심 설계:
#   - RGB 카메라를 주(primary) 카메라로 두고, 그 외부표정 [X,Y,Z,ω,φ,κ] 만
#     BA 변수로 최적화한다.
#   - IR 카메라 포즈는 독립 변수가 아니라 RGB 포즈 + 고정 rig 변환으로 유도:
#         R_ir = R_rig · R_rgb
#         C_ir = C_rgb + R_rgbᵀ · t_rig
#     (검증 완료 — 두 경로로 월드점을 IR 좌표계에 넣으면 일치)
#   - RGB 관측과 IR 관측의 재투영 오차를 같은 잔차 벡터에 넣되, 각자의
#     intrinsic(f_px, cx, cy)을 쓴다. IR 은 픽셀 스케일이 작으므로(640px급)
#     픽셀 오차를 가중치로 정규화해 RGB(4000px급)와 균형을 맞춘다.
#   - rig 변환은 기본적으로 고정(공장 캘리브레이션 값)이지만, optimize_rig=True
#     이면 6-DOF rig 파라미터도 약한 prior 와 함께 추정한다.
#
# 결과: RGB 정사영상과 IR 정사영상이 BA 단계에서 강제로 정합된다.
# ---------------------------------------------------------------------------
@dataclass
class RigTransform:
    """RGB 카메라 좌표계 → IR 카메라 좌표계 강체 변환.

    t_rig : (3,) RGB 카메라 좌표계에서 본 IR 카메라 위치 (meter).
            H20T 는 렌즈 간격이 수 cm 수준이라 작은 값.
    rig_opk : (3,) RGB→IR 미세 회전 ω-φ-κ (radian). 거의 0 에 가까움.

    공장값을 모르면 t_rig=[0,0,0], rig_opk=[0,0,0] 으로 두고
    optimize_rig=True 로 BA 가 추정하게 할 수 있다 (단 충분한 공통
    track 이 있어야 관측 가능).
    """
    t_rig: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rig_opk: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.t_rig, self.rig_opk])

    @staticmethod
    def from_vector(v: np.ndarray) -> "RigTransform":
        return RigTransform(t_rig=np.asarray(v[:3], float),
                            rig_opk=np.asarray(v[3:6], float))


def derive_ir_pose(rgb_cam: np.ndarray, rig: RigTransform) -> np.ndarray:
    """RGB 외부표정 + rig → IR 외부표정 [X,Y,Z,ω,φ,κ].

    rgb_cam : (6,) [Xc,Yc,Zc, ω,φ,κ] RGB 카메라
    반환    : (6,) IR 카메라 외부표정
    """
    C_rgb = rgb_cam[:3]
    R_rgb = rotation_matrix(rgb_cam[3], rgb_cam[4], rgb_cam[5])
    R_rig = rotation_matrix(rig.rig_opk[0], rig.rig_opk[1], rig.rig_opk[2])

    R_ir = R_rig @ R_rgb
    C_ir = C_rgb + R_rgb.T @ rig.t_rig
    om, ph, ka = matrix_to_opk(R_ir)
    return np.array([C_ir[0], C_ir[1], C_ir[2], om, ph, ka])


def rgb_ir_joint_bundle_adjustment(
    initial_rgb_cameras: np.ndarray,    # (n_cam, 6) RGB 외부표정
    initial_points: np.ndarray,         # (n_pts, 3) 삼각측량된 3D 점
    rgb_observations: list,             # [(cam_idx, pt_idx, np.array([u,v])), ...] RGB
    ir_observations: list,              # [(cam_idx, pt_idx, np.array([u,v])), ...] IR
    rtk_priors: np.ndarray,             # (n_cam, 3) RGB 카메라 RTK 위치
    rtk_weights: np.ndarray,            # (n_cam, 3) 1/σ²
    rgb_intrinsics: tuple,              # (f_px, cx, cy) RGB 대표값
    ir_intrinsics: tuple,               # (f_px, cx, cy) IR 대표값
    initial_rig: RigTransform | None = None,
    optimize_rig: bool = False,
    rig_t_sigma: float = 0.05,          # rig 평행이동 prior σ (5cm)
    rig_r_sigma_deg: float = 1.0,       # rig 회전 prior σ (1도)
):
    """RGB-IR 공동 Bundle Adjustment.

    잔차 벡터 구성:
        [ RGB 재투영오차,
          IR 재투영오차 (RGB↔IR 픽셀스케일 차이 보정),
          RTK prior 오차,
          (optimize_rig 시) rig prior 오차 ]

    cam_idx 는 RGB/IR 공통 — i 번째 RGB 사진과 i 번째 IR 사진은 같은
    셔터에 찍힌 한 쌍으로 가정한다 (H20T 는 동시 촬영).

    ★ optimize_rig 사용 시 주의 ★
        nadir 드론 사진에서 수 cm 크기의 rig baseline 은 관측성이 매우 약하다.
        카메라-지면 거리(~45m)에 비해 rig 가 너무 작아, rig 평행이동이 만드는
        픽셀 변화가 노이즈에 묻힌다. 합성 검증 결과:
          - rig 회전:    0.1° 이내로 잘 추정됨
          - rig 평행이동: 수 cm 오차 (부정확)
        따라서 실무에서는 **공장 캘리브레이션 값을 initial_rig 로 주고
        optimize_rig=False (고정)** 하는 것을 강력 권장한다. rig 값을 모르면
        지상에 명확한 특징물이 있는 별도 캘리브레이션 비행으로 구해야 한다.

    Returns
    -------
    rgb_cams_opt : (n_cam, 6) 최적화된 RGB 외부표정
    ir_cams_opt  : (n_cam, 6) 유도된 IR 외부표정
    pts_opt      : (n_pts, 3) 최적화된 3D 점
    rig_opt      : RigTransform
    info         : dict (rmse 등)
    """
    if optimize_rig:
        logger.warning(
            "optimize_rig=True: nadir 사진에서 rig 평행이동은 관측성이 약합니다. "
            "가능하면 공장 캘리브레이션 값으로 고정(optimize_rig=False)하세요."
        )
    n_cam = len(initial_rgb_cameras)
    n_pts = len(initial_points)
    rig0 = initial_rig if initial_rig is not None else RigTransform()

    f_rgb, cx_rgb, cy_rgb = rgb_intrinsics
    f_ir, cx_ir, cy_ir = ir_intrinsics

    # IR 재투영오차를 RGB 와 같은 스케일로: IR 은 초점거리가 작아 같은
    # 각도오차라도 픽셀오차가 작게 나온다. f_rgb/f_ir 비율로 가중.
    ir_pixel_weight = f_rgb / f_ir

    rig_dim = 6 if optimize_rig else 0

    def pack(rgb_cams, pts, rig):
        parts = [rgb_cams.ravel(), pts.ravel()]
        if optimize_rig:
            parts.append(rig.as_vector())
        return np.concatenate(parts)

    def unpack(x):
        rgb_cams = x[: n_cam * 6].reshape(n_cam, 6)
        pts = x[n_cam * 6: n_cam * 6 + n_pts * 3].reshape(n_pts, 3)
        if optimize_rig:
            rig = RigTransform.from_vector(x[n_cam * 6 + n_pts * 3:])
        else:
            rig = rig0
        return rgb_cams, pts, rig

    def residuals(x):
        rgb_cams, pts, rig = unpack(x)
        res = []

        # (a) RGB 재투영 오차
        for cam_idx, pt_idx, obs in rgb_observations:
            cam = rgb_cams[cam_idx]
            pred = project_point(pts[pt_idx], cam[:3], cam[3], cam[4], cam[5],
                                 f_rgb, cx_rgb, cy_rgb)
            res.extend(pred - obs)

        # (b) IR 재투영 오차 — IR 포즈는 RGB 포즈 + rig 로 유도
        for cam_idx, pt_idx, obs in ir_observations:
            ir_cam = derive_ir_pose(rgb_cams[cam_idx], rig)
            pred = project_point(pts[pt_idx], ir_cam[:3],
                                 ir_cam[3], ir_cam[4], ir_cam[5],
                                 f_ir, cx_ir, cy_ir)
            res.extend((pred - obs) * ir_pixel_weight)

        # (c) RTK prior — RGB 카메라 위치를 RTK 측정값에 묶음
        for i in range(n_cam):
            diff = rgb_cams[i, :3] - rtk_priors[i]
            res.extend(np.sqrt(rtk_weights[i]) * diff)

        # (d) rig prior — optimize_rig 시 공장값/0 에서 너무 벗어나지 않게
        if optimize_rig:
            w_t = 1.0 / rig_t_sigma
            w_r = 1.0 / np.deg2rad(rig_r_sigma_deg)
            res.extend(w_t * (rig.t_rig - rig0.t_rig))
            res.extend(w_r * (rig.rig_opk - rig0.rig_opk))

        return np.array(res)

    x0 = pack(initial_rgb_cameras, initial_points, rig0)
    result = least_squares(
        residuals, x0,
        method="trf",
        loss="huber",
        f_scale=2.0,
        max_nfev=400,
        verbose=2,
    )
    rgb_cams_opt, pts_opt, rig_opt = unpack(result.x)
    ir_cams_opt = np.array([derive_ir_pose(rgb_cams_opt[i], rig_opt)
                            for i in range(n_cam)])

    n_obs = len(rgb_observations) + len(ir_observations)
    rmse_px = np.sqrt(2 * result.cost / max(n_obs, 1))

    # RGB/IR 각각의 재투영 RMSE 도 따로 계산 (정합 품질 진단용)
    def _reproj_rmse(obs_list, cams, intr):
        f, cx, cy = intr
        errs = []
        for cam_idx, pt_idx, obs in obs_list:
            cam = cams[cam_idx]
            pred = project_point(pts_opt[pt_idx], cam[:3], cam[3], cam[4], cam[5],
                                 f, cx, cy)
            if np.all(np.isfinite(pred)):
                errs.append(np.linalg.norm(pred - obs))
        return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")

    rgb_rmse = _reproj_rmse(rgb_observations, rgb_cams_opt, rgb_intrinsics)
    ir_rmse = _reproj_rmse(ir_observations, ir_cams_opt, ir_intrinsics)

    info = {
        "total_rmse_px": rmse_px,
        "rgb_rmse_px": rgb_rmse,
        "ir_rmse_px": ir_rmse,
        "rig": rig_opt,
        "rig_baseline_cm": float(np.linalg.norm(rig_opt.t_rig) * 100),
    }
    logger.info(
        "RGB-IR 공동 BA 완료. RGB RMSE=%.2fpx, IR RMSE=%.2fpx, rig baseline=%.2fcm",
        rgb_rmse, ir_rmse, info["rig_baseline_cm"],
    )
    return rgb_cams_opt, ir_cams_opt, pts_opt, rig_opt, info


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

    # 픽셀 단위 초점거리 — FocalLengthIn35mmFilm 기반 (compute_focal_px 참조)
    f_px = compute_focal_px(meta)

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


def compute_focal_px(meta: ImageMetadata) -> float:
    """픽셀 단위 초점거리. FocalLengthIn35mmFilm 우선 (센서크기 불필요)."""
    if meta.focal_length_in_35mm and meta.focal_length_in_35mm > 0:
        return meta.focal_length_in_35mm / 36.0 * meta.width
    logger.warning("focal_length_in_35mm 없음 → 센서폭 13.2mm 가정")
    return meta.focal_length * meta.width / 13.2


def build_rgb_ir_observations(
    ir_metas: list[ImageMetadata],
    initial_points: np.ndarray,
    rgb_cameras: np.ndarray,
    initial_rig: RigTransform,
    ir_intrinsics: tuple,
    search_radius_px: float = 8.0,
    max_features_ir: int = 4000,
):
    """RGB BA 로 복원한 3D 점을 IR 이미지와 매칭해 IR observations 생성.

    ★ 왜 직접 SIFT 매칭이 안 되나 ★
        RGB(가시광)와 IR(열복사)은 외관이 근본적으로 다르다. 같은 패널이라도
        RGB 는 표면 무늬·색, IR 은 온도 분포를 본다. SIFT 디스크립터가
        cross-modal 로는 거의 매칭되지 않는다.

    ★ 해결: 3D 점을 매개로 한 간접(guided) 매칭 ★
        1) RGB BA 가 이미 3D 점(initial_points)과 RGB 포즈를 복원해 둠.
        2) rig 초기값으로 각 3D 점을 IR 이미지에 투영 → 예측 픽셀 위치.
        3) IR 이미지에서 그 예측 위치 근처(search_radius_px)에 실제 IR
           특징점(코너 등)이 있으면 대응으로 채택.
        이렇게 하면 RGB↔IR 직접 매칭 없이도 공동 BA 입력을 만들 수 있다.

    Parameters
    ----------
    ir_metas : IR 이미지 메타데이터 (RGB 와 인덱스 1:1 대응)
    initial_points : (n_pts, 3) RGB BA 로 복원된 3D 점
    rgb_cameras : (n_cam, 6) RGB 외부표정
    initial_rig : 공장 캘리브레이션 rig (예측 투영용)
    ir_intrinsics : (f_px, cx, cy) IR 카메라
    search_radius_px : 예측 위치 주변 탐색 반경 (IR 픽셀 기준)

    Returns
    -------
    ir_observations : [(cam_idx, pt_idx, np.array([u, v])), ...]
    """
    f_ir, cx_ir, cy_ir = ir_intrinsics
    ir_observations = []

    # IR 이미지별 특징점(코너) 미리 추출
    ir_corners = {}
    for ci, meta in enumerate(ir_metas):
        img = cv2.imread(meta.origin_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # 열화상은 대비가 낮으므로 CLAHE 로 향상 후 코너 검출
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        corners = cv2.goodFeaturesToTrack(
            enhanced, maxCorners=max_features_ir,
            qualityLevel=0.01, minDistance=5,
        )
        if corners is not None:
            ir_corners[ci] = corners.reshape(-1, 2)  # (N, 2)

    matched = 0
    for ci, meta in enumerate(ir_metas):
        if ci not in ir_corners:
            continue
        corners = ir_corners[ci]
        ir_tree = cKDTree(corners)

        # 이 IR 사진의 포즈 = RGB 포즈 + rig
        ir_cam = derive_ir_pose(rgb_cameras[ci], initial_rig)

        for pt_idx, pt3d in enumerate(initial_points):
            # 3D 점을 IR 이미지에 투영
            uv_pred = project_point(pt3d, ir_cam[:3],
                                    ir_cam[3], ir_cam[4], ir_cam[5],
                                    f_ir, cx_ir, cy_ir)
            if not np.all(np.isfinite(uv_pred)):
                continue
            if not (0 <= uv_pred[0] < meta.width and 0 <= uv_pred[1] < meta.height):
                continue
            # 예측 위치 근처에 IR 코너가 있으면 대응으로 채택
            dist, idx = ir_tree.query(uv_pred)
            if dist <= search_radius_px:
                ir_observations.append((ci, pt_idx, corners[idx].astype(float)))
                matched += 1

    logger.info("RGB-IR cross-modal 관측: %d개 (IR 사진 %d장)",
                matched, len(ir_corners))
    return ir_observations


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

    # 5. Tie point tracks → 3D 점 (Triangulation 초기화) → RTK 제약 BA
    intrinsics = [
        (compute_focal_px(m), m.width / 2.0, m.height / 2.0)
        for m in metas
    ]

    # 5a. 페어 매칭들을 track 으로 연결
    tracks = build_tracks(matches, features,
                          min_track_len=2, max_track_len=30)

    if len(tracks) < 10:
        logger.warning("track 이 너무 적음(%d). BA 생략, RTK 초기값 사용.",
                       len(tracks))
        cams_opt = initial_cameras
    else:
        # 5b. track 들을 DLT 삼각측량 → observations + initial_points
        observations, initial_points, _ = triangulate_tracks(
            tracks, initial_cameras, intrinsics,
            max_reproj_err_px=3.0,
            min_triangulation_angle_deg=2.0,
        )

        if len(initial_points) < 10 or len(observations) < 30:
            logger.warning("삼각측량 결과 부족. BA 생략, RTK 초기값 사용.")
            cams_opt = initial_cameras
        else:
            # 5c. RTK 제약 Bundle Adjustment
            # 카메라마다 f_px 가 다를 수 있으나, rtk_constrained_bundle_adjustment
            # 는 단일 (f_px, cx, cy) 를 받으므로 대표값(중앙값) 사용.
            # (전 사진 동일 카메라/줌이면 정확히 일치)
            f_px_rep = float(np.median([k[0] for k in intrinsics]))
            cx_rep = float(np.median([k[1] for k in intrinsics]))
            cy_rep = float(np.median([k[2] for k in intrinsics]))
            cams_opt, pts_opt, rmse = rtk_constrained_bundle_adjustment(
                initial_cameras, initial_points, observations,
                rtk_priors, rtk_weights,
                f_px=f_px_rep, cx=cx_rep, cy=cy_rep,
            )
            logger.info("BA 후 카메라 위치 평균 이동량: %.3f m",
                        float(np.linalg.norm(
                            cams_opt[:, :3] - initial_cameras[:, :3], axis=1
                        ).mean()))

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


# ---------------------------------------------------------------------------
# 9. RGB-IR 동시 Georeferencing 파이프라인 (H20T)
# ---------------------------------------------------------------------------
def run_pipeline_rgb_ir(rgb_dir: Path,
                        ir_dir: Path,
                        output_dir: Path,
                        rig: RigTransform | None = None,
                        target_epsg: int = 5186,
                        gsd_m: float = 0.05,
                        k_neighbors: int = 8,
                        optimize_rig: bool = False):
    """H20T RGB-IR 동시 georeferencing.

    워크플로우:
        1. RGB/IR 메타데이터 추출 (파일명 기준 페어링)
        2. RGB 만으로 SfM: 매칭 → track → 삼각측량
        3. RGB observations 로 1차 삼각측량 (3D 점 생성)
        4. 3D 점을 IR 에 guided-projection → IR observations 생성
        5. RGB-IR 공동 BA: RGB 포즈 최적화 + IR 은 rig 로 유도
        6. RGB/IR 정사영상 각각 생성 (동일 좌표계로 정합됨)

    Parameters
    ----------
    rgb_dir, ir_dir : RGB / IR 이미지 폴더. 같은 셔터의 사진은
        파일명 stem 일부가 대응한다고 가정 (정렬 순서로 페어링).
    rig : RGB→IR 강체 변환. None 이면 zero rig 로 시작.
          ★ 공장 캘리브레이션 값을 넣는 것을 강력 권장 (관측성 한계 참고).
    optimize_rig : True 면 rig 도 BA 변수 (비권장 — docstring 참고).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rgb").mkdir(exist_ok=True)
    (output_dir / "ir").mkdir(exist_ok=True)
    crs = CRSConverter(target_epsg=target_epsg)

    # 1. 메타데이터 + 페어링
    rgb_images = sorted(rgb_dir.glob("*.JPG")) + sorted(rgb_dir.glob("*.jpg"))
    ir_images = sorted(ir_dir.glob("*.JPG")) + sorted(ir_dir.glob("*.jpg"))
    if len(rgb_images) != len(ir_images):
        logger.warning("RGB(%d장)와 IR(%d장) 수가 다름. 정렬 순서로 페어링.",
                       len(rgb_images), len(ir_images))
    n_pair = min(len(rgb_images), len(ir_images))
    rgb_images, ir_images = rgb_images[:n_pair], ir_images[:n_pair]

    rgb_metas = [extract_metadata(p) for p in rgb_images]
    ir_metas = [extract_metadata(p) for p in ir_images]
    logger.info("RGB-IR 페어 %d쌍 로딩", n_pair)

    if rig is None:
        rig = RigTransform()
        logger.warning("rig 미지정 → zero rig 사용. 공장 캘리브레이션 값 권장.")

    # 2. RGB 초기 외부표정 + RTK prior
    rtk_priors, initial_rgb_cameras = [], []
    for m in rgb_metas:
        X, Y = crs.forward(m.gps.lng, m.gps.lat)
        Z = m.gps.altitude
        rtk_priors.append([X, Y, Z])
        gy, gp, gr = m.orientation
        initial_rgb_cameras.append([X, Y, Z,
                                    np.deg2rad(gr),
                                    np.deg2rad(gp + 90),
                                    np.deg2rad(gy)])
    rtk_priors = np.array(rtk_priors)
    initial_rgb_cameras = np.array(initial_rgb_cameras)

    rtk_weights = np.zeros((n_pair, 3))
    for i, m in enumerate(rgb_metas):
        sxy = m.gps_std_xy if m.is_rtk_fixed else max(m.gps_std_xy, 0.20)
        sz = m.gps_std_z if m.is_rtk_fixed else max(m.gps_std_z, 0.30)
        rtk_weights[i] = [1.0 / sxy**2, 1.0 / sxy**2, 1.0 / sz**2]

    # 3. RGB SfM: 매칭 → track → 삼각측량
    rgb_intr_list = [(compute_focal_px(m), m.width / 2.0, m.height / 2.0)
                     for m in rgb_metas]
    pairs = find_neighbor_pairs(rgb_metas, crs, k_neighbors=k_neighbors)
    matches, features = build_tie_points(rgb_metas, pairs)
    tracks = build_tracks(matches, features, min_track_len=2, max_track_len=30)

    if len(tracks) < 10:
        logger.warning("RGB track 부족(%d). RGB-IR 공동 BA 생략, RTK 초기값 사용.",
                       len(tracks))
        rgb_cams_opt = initial_rgb_cameras
        ir_cams_opt = np.array([derive_ir_pose(c, rig)
                                for c in initial_rgb_cameras])
    else:
        rgb_observations, initial_points, _ = triangulate_tracks(
            tracks, initial_rgb_cameras, rgb_intr_list,
            max_reproj_err_px=3.0, min_triangulation_angle_deg=2.0,
        )

        if len(initial_points) < 10:
            logger.warning("삼각측량 결과 부족. 공동 BA 생략.")
            rgb_cams_opt = initial_rgb_cameras
            ir_cams_opt = np.array([derive_ir_pose(c, rig)
                                    for c in initial_rgb_cameras])
        else:
            # 대표 intrinsic
            rgb_intr = (float(np.median([k[0] for k in rgb_intr_list])),
                        float(np.median([k[1] for k in rgb_intr_list])),
                        float(np.median([k[2] for k in rgb_intr_list])))
            ir_intr_list = [(compute_focal_px(m), m.width / 2.0, m.height / 2.0)
                            for m in ir_metas]
            ir_intr = (float(np.median([k[0] for k in ir_intr_list])),
                       float(np.median([k[1] for k in ir_intr_list])),
                       float(np.median([k[2] for k in ir_intr_list])))

            # 4. 3D 점 → IR guided projection → IR observations
            ir_observations = build_rgb_ir_observations(
                ir_metas, initial_points, initial_rgb_cameras,
                rig, ir_intr, search_radius_px=8.0,
            )

            # 5. RGB-IR 공동 BA
            if len(ir_observations) < 20:
                logger.warning("IR 관측 부족(%d). RGB 단독 BA 로 대체.",
                               len(ir_observations))
                rgb_cams_opt, _, _ = rtk_constrained_bundle_adjustment(
                    initial_rgb_cameras, initial_points, rgb_observations,
                    rtk_priors, rtk_weights, *rgb_intr,
                )
                ir_cams_opt = np.array([derive_ir_pose(c, rig)
                                        for c in rgb_cams_opt])
            else:
                rgb_cams_opt, ir_cams_opt, _, rig_opt, info = \
                    rgb_ir_joint_bundle_adjustment(
                        initial_rgb_cameras, initial_points,
                        rgb_observations, ir_observations,
                        rtk_priors, rtk_weights,
                        rgb_intr, ir_intr,
                        initial_rig=rig, optimize_rig=optimize_rig,
                    )
                rig = rig_opt

    # 6. RGB / IR 정사영상 각각 생성 (동일 좌표계 → 정합됨)
    logger.info("RGB 정사영상 생성 중...")
    for i, meta in enumerate(rgb_metas):
        cam = rgb_cams_opt[i]
        gz = estimate_ground_z(meta)
        out = output_dir / "rgb" / f"{Path(meta.origin_path).stem}_ortho.tif"
        simple_orthophoto(meta, cam[:3], cam[3], cam[4], cam[5],
                          gz, gsd_m, out, target_epsg)

    logger.info("IR 정사영상 생성 중...")
    for i, meta in enumerate(ir_metas):
        cam = ir_cams_opt[i]
        # IR 의 ground_z 는 대응 RGB 의 LRF 값을 공유 (IR 엔 LRF 없음)
        gz = estimate_ground_z(rgb_metas[i])
        out = output_dir / "ir" / f"{Path(meta.origin_path).stem}_ortho.tif"
        simple_orthophoto(meta, cam[:3], cam[3], cam[4], cam[5],
                          gz, gsd_m, out, target_epsg)

    logger.info("RGB-IR 동시 georeferencing 완료. 출력: %s", output_dir)
    return rgb_cams_opt, ir_cams_opt, rig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # --- RGB 단독 파이프라인 ---
    # run_pipeline_gcp_free(
    #     image_dir=Path("./data/solar/images/RGB"),
    #     output_dir=Path("./workspace/output"),
    #     target_epsg=5186, gsd_m=0.05, k_neighbors=8,
    # )

    # --- RGB-IR 동시 파이프라인 (H20T) ---
    # H20T 공장 캘리브레이션 rig 값을 알면 여기 넣으세요.
    # 모르면 RigTransform() (zero) 로 두되, 정합 정확도가 떨어질 수 있습니다.
    h20t_rig = RigTransform(
        t_rig=np.array([0.0, 0.0, 0.0]),       # RGB→IR 평행이동 (m)
        rig_opk=np.deg2rad(np.array([0.0, 0.0, 0.0])),  # RGB→IR 회전
    )
    run_pipeline_rgb_ir(
        rgb_dir=Path("./data/solar/images/RGB"),
        ir_dir=Path("./data/solar/images/IR"),
        output_dir=Path("./workspace/output_rgb_ir"),
        rig=h20t_rig,
        target_epsg=5186,
        gsd_m=0.05,
        k_neighbors=8,
        optimize_rig=False,   # 관측성 한계 — 공장값 고정 권장
    )