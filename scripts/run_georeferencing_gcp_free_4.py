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
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from itertools import combinations

import cv2
import numpy as np
import rasterio
import sys
from rasterio.transform import from_origin
from pyproj import Transformer
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
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
# 좌표계 정의 (CRS): EPSG:4326 또는 EPSG:32652
# EPSG:4326 → EPSG:5186
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


# ---------------------------------------------------------------------------
# 4-1. RANSAC 안전 wrapper — cv2.findFundamentalMat 견고화
# ---------------------------------------------------------------------------
# OpenCV 4.13 의 findFundamentalMat 는 다음 케이스에서 내부 SVD 분해 단계에서
# matrix.cpp:764 의 row range assertion 으로 크래시한다 (cv2.error 로 raise):
#   - 입력 점이 8 점 미만 (7-point algorithm 최소)
#   - 점들이 동일선상에 있거나 모두 동일점 (degenerate, 분산 0)
#   - dtype 이 float32/float64 가 아니거나 shape 가 (N, 2) 가 아닌 경우
# 솔라패널 단지처럼 반복 텍스처가 많은 환경에서는 잘못된 매칭이 degenerate
# 배치로 모이는 경우가 발생 → 사전 검사 + try/except 로 방어한다.
_MIN_POINTS_FOR_F_MATRIX = 20  # 8-point 최소보다 여유 있게


def _safe_find_fundamental(pts_i: np.ndarray, pts_j: np.ndarray):
    """``cv2.findFundamentalMat`` 호출을 사전 검사 + 예외 방어로 감싸기.

    Returns
    -------
    inlier mask : (N,) bool array 또는 None (방어 케이스).
    """
    # 1) 사전 조건.
    if pts_i is None or pts_j is None:
        return None
    if pts_i.shape != pts_j.shape:
        return None
    if pts_i.ndim != 2 or pts_i.shape[1] != 2:
        return None
    if pts_i.shape[0] < _MIN_POINTS_FOR_F_MATRIX:
        return None

    # 2) 동일선상 (degenerate) 사전 차단 — SVD 분해가 죽는 가장 흔한 원인.
    if np.std(pts_i, axis=0).min() < 1e-6:
        return None
    if np.std(pts_j, axis=0).min() < 1e-6:
        return None

    # 3) dtype/메모리 보장.
    pts_i = np.ascontiguousarray(pts_i, dtype=np.float32)
    pts_j = np.ascontiguousarray(pts_j, dtype=np.float32)

    # 4) 호출 + 예외 방어.
    try:
        F, mask = cv2.findFundamentalMat(
            pts_i, pts_j, cv2.FM_RANSAC, 1.0, 0.99,
        )
    except cv2.error as e:
        # 원인 진단에 필요한 정보를 함께 로깅 (debug 레벨로).
        logger.debug(
            "findFundamentalMat 실패: N=%d, dtype=%s, "
            "pts_i_range=[%.1f, %.1f]x[%.1f, %.1f], err=%s",
            pts_i.shape[0], pts_i.dtype,
            float(pts_i[:, 0].min()), float(pts_i[:, 0].max()),
            float(pts_i[:, 1].min()), float(pts_i[:, 1].max()),
            str(e).split("\n")[0],
        )
        return None
    if mask is None or F is None or F.size == 0:
        return None
    return mask.ravel().astype(bool)


def build_tie_points(metas: list[ImageMetadata],
                     pairs: list[tuple[int, int]]):
    """SIFT 특징점 추출 + 페어별 매칭 + RANSAC outlier 제거.

    견고성
    ------
    * 특징점이 너무 적은 이미지는 사전 제외 (전체 페어에서 빠짐).
    * ``cv2.findFundamentalMat`` 의 OpenCV assertion 을 ``_safe_find_fundamental``
      이 흡수 — 동일선상/저점수/dtype 문제로 죽지 않는다.
    * 각 페어가 어떤 이유로 제외됐는지 통계 로그 출력.

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

    # 진단: 추출 실패 / 특징점 부족 이미지.
    empty_imgs = [
        i for i, (kp, desc, _shape) in features.items()
        if desc is None or len(kp) < _MIN_POINTS_FOR_F_MATRIX
    ]
    if empty_imgs:
        logger.warning(
            "특징점이 부족한 이미지 %d장 (전체 %d장): 인덱스 예시 %s ...",
            len(empty_imgs), len(features), empty_imgs[:5],
        )

    matches = {}
    dropped_no_desc = 0
    dropped_few_matches = 0
    dropped_ransac = 0
    dropped_few_inliers = 0

    for i, j in pairs:
        kp_i, desc_i, _ = features[i]
        kp_j, desc_j, _ = features[j]

        # 빈 descriptor 차단.
        if desc_i is None or desc_j is None:
            dropped_no_desc += 1
            continue
        if (len(kp_i) < _MIN_POINTS_FOR_F_MATRIX
                or len(kp_j) < _MIN_POINTS_FOR_F_MATRIX):
            dropped_no_desc += 1
            continue

        # 매칭 (BFMatcher 예외 흡수).
        try:
            good = match_pair(desc_i, desc_j)
        except cv2.error as e:
            logger.debug("매칭 실패 (pair %d-%d): %s", i, j, e)
            dropped_ransac += 1
            continue

        if len(good) < 30:
            dropped_few_matches += 1
            continue

        try:
            pts_i = np.float32([kp_i[m.queryIdx].pt for m in good])
            pts_j = np.float32([kp_j[m.trainIdx].pt for m in good])
        except (IndexError, AttributeError) as e:
            logger.warning("매칭 인덱스 오류 (pair %d-%d): %s", i, j, e)
            dropped_ransac += 1
            continue

        idx_i = np.array([m.queryIdx for m in good], dtype=np.int64)
        idx_j = np.array([m.trainIdx for m in good], dtype=np.int64)

        # RANSAC (방어된 wrapper 사용).
        inliers = _safe_find_fundamental(pts_i, pts_j)
        if inliers is None:
            dropped_ransac += 1
            continue
        if inliers.sum() < 20:
            dropped_few_inliers += 1
            continue

        matches[(i, j)] = (pts_i[inliers], pts_j[inliers],
                           idx_i[inliers], idx_j[inliers])

    logger.info(
        "유효 매칭 페어: %d/%d (제거: desc없음 %d, 매칭부족 %d, "
        "RANSAC실패 %d, inlier부족 %d)",
        len(matches), len(pairs),
        dropped_no_desc, dropped_few_matches,
        dropped_ransac, dropped_few_inliers,
    )
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
    """경로 압축 + 랭크 기반 Union-Find (호환성 유지용 stub).

    .. deprecated::
        ``build_tracks`` 는 이제 ``scipy.sparse.csgraph.connected_components``
        기반 C 구현을 사용해서 본 클래스를 호출하지 않는다. 외부 코드가
        직접 import 하는 경우를 대비해 stub 으로 남겨둠.
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

    구현 (vectorized)
    -----------------
    원본 dict 기반 Union-Find 는 Python 인터프리터 오버헤드 + 튜플 hash 비용
    + 캐시 미스로 매우 느렸다. 특히 GCE g2 같은 vCPU 환경 (Cascade Lake,
    낮은 클럭 + 작은 L3) 에서는 M4 Pro 대비 3~5 배 느림.

    재구현 핵심:

    1) ``(image_idx, keypoint_idx)`` 튜플을 flat 정수 ID 로 변환::

           node_id = image_idx * MAX_KP + keypoint_idx

       이렇게 하면 dict hash + 튜플 비용이 제거되고 모든 연산이 정수 배열 위에서
       이뤄진다.

    2) 모든 매칭을 ``(N, 2)`` int 엣지 배열 한 번에 모은 뒤
       ``scipy.sparse.csgraph.connected_components`` (C 구현) 로 한 호출에
       Union-Find 처리.

    3) 그룹별 관측 모으기는 ``np.argsort(labels)`` + ``np.split`` 로 벡터화.

    4) 충돌 검사 (한 이미지에서 두 keypoint 가 같은 track 에 들어감) 도
       그룹별로 numpy 로 처리.

    Parameters
    ----------
    matches : ``build_tie_points`` 의 출력
    features : ``build_tie_points`` 의 출력 (keypoint 픽셀좌표 조회용 — 본
        함수는 이미 ``matches`` 안의 ``pts_*`` 에 좌표가 있어서 직접 사용은
        하지 않지만, 기존 API 호환성을 위해 받는다)
    min_track_len : track 으로 인정할 최소 관측 수 (2 = 최소 삼각측량 가능)
    max_track_len : 비정상적으로 긴 track 제거 (보통 매칭 오류).
                    하나의 점이 30 장 이상에 나타나면 의심스러움.

    Returns
    -------
    tracks : list of track. 각 track 은
        ``[(image_idx, keypoint_idx, px, py), ...]`` 형식 — 원본 API 동일.
    """
    if not matches:
        logger.info("track 생성: 0개 (매칭이 비어있음)")
        return []

    t0 = time.perf_counter()

    # -----------------------------------------------------------------------
    # 1) 모든 관측을 평탄화 — 단일 numpy 배열로
    # -----------------------------------------------------------------------
    # 각 페어 (i, j) 의 매칭 K 개에 대해 i-쪽 관측 K 개 + j-쪽 관측 K 개.
    # 전체 관측 수 = sum(2 * K_pair) 이고 보통 수십만 ~ 수백만.
    n_obs_per_pair = [int(len(m[2])) for m in matches.values()]  # idx_i 길이
    total_obs = 2 * sum(n_obs_per_pair)

    if total_obs == 0:
        logger.info("track 생성: 0개 (매칭은 있으나 관측 없음)")
        return []

    # node_id 인코딩에 필요한 최대 keypoint 인덱스 산출.
    # 각 페어의 idx_i, idx_j 의 최대값 + 1 (안전 마진).
    max_kp = 1
    max_img = 0
    for (i, j), (_, _, idx_i, idx_j) in matches.items():
        if len(idx_i) > 0:
            max_kp = max(max_kp,
                         int(idx_i.max()) + 1,
                         int(idx_j.max()) + 1)
        max_img = max(max_img, i, j)
    n_images = max_img + 1

    # flat node id = image_idx * max_kp + keypoint_idx.
    # int64 로 안전한 범위: max_kp * n_images < 2^63. 보통 5000장 * 8000kp =
    # 4천만 → 한참 여유.
    obs_img = np.empty(total_obs, dtype=np.int32)
    obs_kp = np.empty(total_obs, dtype=np.int64)
    obs_px = np.empty(total_obs, dtype=np.float64)
    obs_py = np.empty(total_obs, dtype=np.float64)

    # 엣지 배열 — 그래프 입력. 페어당 K 개 엣지.
    edges_a = np.empty(total_obs // 2, dtype=np.int64)
    edges_b = np.empty(total_obs // 2, dtype=np.int64)

    off_obs = 0
    off_edge = 0
    for (i, j), (pts_i, pts_j, idx_i, idx_j) in matches.items():
        K = len(idx_i)
        if K == 0:
            continue
        idx_i_int = idx_i.astype(np.int64, copy=False)
        idx_j_int = idx_j.astype(np.int64, copy=False)
        node_i = i * max_kp + idx_i_int  # (K,)
        node_j = j * max_kp + idx_j_int  # (K,)

        # 엣지 (i-측 관측 ↔ j-측 관측).
        edges_a[off_edge:off_edge + K] = node_i
        edges_b[off_edge:off_edge + K] = node_j
        off_edge += K

        # i-측 관측 K 개 기록.
        s = off_obs
        e = s + K
        obs_img[s:e] = i
        obs_kp[s:e] = idx_i_int
        obs_px[s:e] = pts_i[:, 0]
        obs_py[s:e] = pts_i[:, 1]
        off_obs = e

        # j-측 관측 K 개 기록.
        s = off_obs
        e = s + K
        obs_img[s:e] = j
        obs_kp[s:e] = idx_j_int
        obs_px[s:e] = pts_j[:, 0]
        obs_py[s:e] = pts_j[:, 1]
        off_obs = e

    # 실제 사용한 부분만 자르기 (혹시 K=0 페어가 있었을 경우 대비).
    obs_img = obs_img[:off_obs]
    obs_kp = obs_kp[:off_obs]
    obs_px = obs_px[:off_obs]
    obs_py = obs_py[:off_obs]
    edges_a = edges_a[:off_edge]
    edges_b = edges_b[:off_edge]

    # 각 관측의 node_id (image, keypoint 조합).
    obs_node = obs_img.astype(np.int64) * max_kp + obs_kp

    t1 = time.perf_counter()

    # -----------------------------------------------------------------------
    # 2) connected_components 로 Union-Find 일괄 처리
    # -----------------------------------------------------------------------
    # 그래프 노드는 obs_node 의 unique 값들. 노드 ID 가 sparse (수십만 범위
    # 중 일부만 사용) 이므로 unique 매핑으로 0..n_unique-1 로 압축.
    # 압축하지 않으면 csgraph 가 max(node_id)+1 만큼 노드를 할당해 메모리
    # 폭발 (예: 10만 장 * 8000 = 8억 노드).
    all_nodes = np.concatenate([edges_a, edges_b, obs_node])
    unique_nodes, inv = np.unique(all_nodes, return_inverse=True)
    n_nodes = unique_nodes.shape[0]

    n_edges = edges_a.shape[0]
    edges_a_small = inv[:n_edges]
    edges_b_small = inv[n_edges:2 * n_edges]
    obs_node_small = inv[2 * n_edges:]  # 각 관측의 압축된 노드 ID

    # 무방향 그래프 sparse adjacency 행렬. 데이터는 dummy 1.
    data = np.ones(n_edges, dtype=np.uint8)
    graph = coo_matrix(
        (data, (edges_a_small, edges_b_small)),
        shape=(n_nodes, n_nodes),
    )
    # connected_components 는 directed=False 면 자동으로 양방향 처리.
    n_comp, labels = connected_components(
        graph, directed=False, return_labels=True,
    )

    t2 = time.perf_counter()

    # -----------------------------------------------------------------------
    # 3) component 별 관측 그룹화 — 전역 unique + argsort + split
    # -----------------------------------------------------------------------
    # 중요: 한 (image, keypoint) 가 여러 페어를 통해 여러 관측으로 등장할 수
    # 있다. 원본 dict 구현은 ``obs[(i, ki)] = ...`` 로 자동 중복 제거 후 검사
    # 했으므로, 본 함수도 unique 처리 후 검사한다.
    #
    # 성능 핵심: 그룹마다 ``np.unique`` 를 부르면 Python 호출 오버헤드가 폭발
    # (그룹 수십만 개 × unique 호출 비용 = 전체 시간의 60%). 대신 전역적으로
    # ``(label, node_id)`` 쌍에 대해 한 번만 unique 를 호출한다.

    # 각 관측에 component label 부여.
    obs_label = labels[obs_node_small]  # (total_obs,)

    # (label, node_id) 결합 키 — int64 한 단어로 패킹.
    # n_nodes < 2^32 가정 (수억 노드까지 안전).
    combined = obs_label.astype(np.int64) * (n_nodes + 1) + obs_node_small

    # 전역 unique — 같은 (label, node) 가 여러 관측에 등장하면 첫 번째만 남김.
    _, first_idx = np.unique(combined, return_index=True)
    first_idx = np.sort(first_idx)  # 원래 순서 보존.

    u_label = obs_label[first_idx]
    u_img = obs_img[first_idx]
    u_kp = obs_kp[first_idx]
    u_px = obs_px[first_idx]
    u_py = obs_py[first_idx]

    # label 별 정렬 후 split.
    sort_idx = np.argsort(u_label, kind="stable")
    sorted_labels = u_label[sort_idx]
    sorted_img = u_img[sort_idx]
    sorted_kp = u_kp[sort_idx]
    sorted_px = u_px[sort_idx]
    sorted_py = u_py[sort_idx]

    # 그룹 경계: label 이 바뀌는 인덱스.
    group_starts = np.concatenate([
        [0],
        np.nonzero(np.diff(sorted_labels))[0] + 1,
        [len(sorted_labels)],
    ])
    n_groups = len(group_starts) - 1

    t3a = time.perf_counter()

    # -----------------------------------------------------------------------
    # 4) 그룹별 필터링 (충돌/짧음/김) — 벡터화
    # -----------------------------------------------------------------------
    # 그룹별 길이 검사는 벡터로 한 번에. 충돌 검사는 그룹 내 image id 유일성을
    # bincount 로 일괄 처리.
    group_sizes = np.diff(group_starts)  # (n_groups,)

    # 길이 기반 1차 필터.
    keep_len = (group_sizes >= min_track_len) & (group_sizes <= max_track_len)
    dropped_short = int((group_sizes < min_track_len).sum())
    dropped_long = int((group_sizes > max_track_len).sum())

    # 충돌 검사 — 그룹별로 image 가 unique 한지. 그룹 단위 numpy 작업이지만
    # 그룹마다 한 번씩만 (이미 길이 필터 통과한 그룹만).
    # 빠른 충돌 검사: 그룹의 sorted_img 정렬 후 인접 중복이 있는지.
    keep_idx = np.nonzero(keep_len)[0]
    valid_mask = np.zeros(n_groups, dtype=bool)
    dropped_conflict = 0

    tracks: list[list[tuple[int, int, float, float]]] = []
    for g in keep_idx:
        s, e = group_starts[g], group_starts[g + 1]
        img_g = sorted_img[s:e]
        n = img_g.shape[0]

        # 충돌 검사:
        #   - size=2: 두 관측이 같은 component 에 속하려면 매칭 엣지로
        #     연결됐어야 하고 매칭은 항상 다른 두 이미지 사이라 충돌 불가.
        #     → 검사 스킵으로 21만 그룹 × np.unique 호출 회피.
        #   - size>=3: 같은 image 가 두 번 이상 등장하면 충돌 (다른
        #     keypoint 가 한 track 에 섞임). 작은 배열이면 sort 후 diff 가
        #     np.unique 보다 훨씬 빠름.
        if n >= 3:
            si = np.sort(img_g)
            if (si[1:] == si[:-1]).any():
                dropped_conflict += 1
                continue

        # 통과 — track 구성. tolist + zip 이 인덱싱 루프보다 약간 빠름.
        kp_g = sorted_kp[s:e]
        px_g = sorted_px[s:e]
        py_g = sorted_py[s:e]
        track = list(zip(
            img_g.tolist(), kp_g.tolist(),
            px_g.tolist(), py_g.tolist(),
        ))
        tracks.append(track)

    t3 = time.perf_counter()

    logger.info(
        "track 생성: %d개 (제거: 짧음 %d, 김 %d, 충돌 %d)",
        len(tracks), dropped_short, dropped_long, dropped_conflict,
    )
    logger.info(
        "  build_tracks 시간: 평탄화 %.2fs + CC %.2fs + unique/정렬 %.2fs "
        "+ 필터 %.2fs (총 %.2fs, 관측 %d, 노드 %d, 그룹 %d)",
        t1 - t0, t2 - t1, t3a - t2, t3 - t3a, t3 - t0,
        off_obs, n_nodes, n_comp,
    )
    if tracks:
        lengths = np.array([len(t) for t in tracks])
        logger.info("  track 길이 분포: 평균 %.1f, 최대 %d, 2장짜리 %d개",
                    lengths.mean(), lengths.max(), int((lengths == 2).sum()))
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

    Returns:
        ``True`` 면 정사영상 저장 성공, ``False`` 면 입력 메타데이터/카메라
        파라미터에 문제가 있어 건너뜀 (예: focal_length 정보 없음, 이미지
        읽기 실패, ground_z 가 비유한 등). 호출자는 이 반환값으로 사진별
        성공/실패를 카운트할 수 있다.
    """
    # ---- 입력 가드: focal_length 정보 ------------------------------------
    f_px = compute_focal_px(meta)
    if f_px <= 0:
        logger.warning(
            "정사영상 건너뜀 (focal_length 정보 없음): %s",
            meta.origin_path,
        )
        return False

    # ---- 입력 가드: ground_z 가 유한 ------------------------------------
    if not np.isfinite(ground_z):
        logger.warning(
            "정사영상 건너뜀 (ground_z=NaN/Inf): %s",
            meta.origin_path,
        )
        return False

    # ---- 입력 가드: camera_xyz 유한 ------------------------------------
    if not np.all(np.isfinite(camera_xyz)):
        logger.warning(
            "정사영상 건너뜀 (camera_xyz 에 비유한 값): %s",
            meta.origin_path,
        )
        return False

    # ---- 이미지 로딩 ------------------------------------------------------
    img = cv2.imread(meta.origin_path, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("정사영상 건너뜀 (이미지 읽기 실패): %s",
                       meta.origin_path)
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]

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
        return False

    ground_corners = np.array(ground_corners)
    x_min, y_min = ground_corners.min(axis=0)
    x_max, y_max = ground_corners.max(axis=0)

    # ground_corners 에 NaN/Inf 가 섞이면 x_max-x_min 도 NaN → int() 가 죽는다.
    # 입력 가드에서 대부분 잡히지만 4모서리 중 일부만 비유한일 경우 대비.
    if not (np.isfinite(x_min) and np.isfinite(x_max)
            and np.isfinite(y_min) and np.isfinite(y_max)):
        logger.warning(
            "Ground projection 범위에 NaN/Inf: %s (x_min=%s, x_max=%s)",
            meta.origin_path, x_min, x_max,
        )
        return False

    # 출력 래스터 크기.
    out_w = int(np.ceil((x_max - x_min) / gsd_m))
    out_h = int(np.ceil((y_max - y_min) / gsd_m))

    # 비정상적으로 큰 출력 (자세 추정 실패로 ground projection 이 발산) 차단.
    # 솔라패널 단지 사진 한 장의 ortho 는 보통 1만 픽셀 이하.
    MAX_OUT_PIXELS = 100_000_000  # 1억 픽셀 (100 megapixel) 상한.
    if out_w * out_h > MAX_OUT_PIXELS or out_w <= 0 or out_h <= 0:
        logger.warning(
            "정사영상 건너뜀 (출력 크기 비정상: %d x %d): %s",
            out_w, out_h, meta.origin_path,
        )
        return False

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
    return True


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
    """픽셀 단위 초점거리. FocalLengthIn35mmFilm 우선 (센서크기 불필요).

    Returns
    -------
    f_px : 픽셀 단위 초점거리. **두 EXIF 필드 모두 0/없음이면 0.0 반환**
        (호출자가 0 체크로 그 사진을 제외해야 함). 0 을 그대로 사용하면
        ``simple_orthophoto`` 의 ``-(px-cx)/f_px`` 에서 0 으로 나눠 NaN 이
        발생하고 ``int(np.ceil(NaN))`` 에서 ``ValueError: cannot convert
        float NaN to integer`` 로 크래시한다.

    노트
    ----
    DJI H20T 의 thermal (Z) 채널 EXIF 는 ``FocalLengthIn35mmFilm`` 과
    ``FocalLength`` 가 모두 비어있는 경우가 있다. RGB 디렉토리만 처리하려는
    의도였다면 thermal 파일이 섞이지 않았는지 확인 (``_Z.JPG``, ``_T.JPG``
    제외 등).
    """
    if meta.focal_length_in_35mm and meta.focal_length_in_35mm > 0:
        return meta.focal_length_in_35mm / 36.0 * meta.width
    if meta.focal_length and meta.focal_length > 0:
        # FocalLengthIn35mmFilm 이 없을 때만 센서폭 13.2mm (1″ 센서) 가정 fallback.
        return meta.focal_length * meta.width / 13.2
    # 두 정보 모두 없음 — 0 반환. 호출자가 필터링해야 함.
    return 0.0


def run_pipeline(image_dir: Path,
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

    # 1-1. 메타데이터 품질 진단 — 정사영상 단계 크래시 사전 차단.
    no_focal = [Path(m.origin_path).name for m in metas
                if (not m.focal_length_in_35mm or m.focal_length_in_35mm <= 0)
                and (not m.focal_length or m.focal_length <= 0)]
    if no_focal:
        logger.warning(
            "focal_length 정보가 전혀 없는 사진 %d장: %s%s — "
            "이 사진들은 정사영상 단계에서 자동 건너뜀. "
            "(DJI H20T 의 _Z (thermal) 파일이 섞여있는지 확인)",
            len(no_focal), no_focal[:3],
            " ..." if len(no_focal) > 3 else "",
        )
    # 파일명 패턴으로 thermal 혼입 가능성 사전 경고.
    thermal_like = [Path(m.origin_path).name for m in metas
                    if Path(m.origin_path).stem.endswith(("_Z", "_T"))
                    or "_Z_" in Path(m.origin_path).name
                    or "_T_" in Path(m.origin_path).name]
    if thermal_like:
        logger.warning(
            "Thermal/Telephoto 패턴 (_Z, _T) 파일명 %d장 감지: %s%s — "
            "RGB 만 처리하려면 디렉토리를 분리하거나 글로브 패턴을 변경.",
            len(thermal_like), thermal_like[:3],
            " ..." if len(thermal_like) > 3 else "",
        )

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

    n_success = 0
    n_skipped = 0
    for i, meta in enumerate(metas):
        cam = cams_opt[i]
        ground_z = estimate_ground_z(meta)
        out = output_dir / f"{Path(meta.origin_path).stem}_ortho.tif"
        ok = simple_orthophoto(
            meta=meta,
            camera_xyz=cam[:3],
            omega=cam[3], phi=cam[4], kappa=cam[5],
            ground_z=ground_z,
            gsd_m=gsd_m,
            output_path=out,
            epsg=target_epsg,
        )
        if ok:
            n_success += 1
        else:
            n_skipped += 1
    logger.info("정사영상 완료: 성공 %d, 건너뜀 %d", n_success, n_skipped)


if __name__ == "__main__":
    # 시작 시간
    start = time.perf_counter()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline(
        image_dir=Path("./data/solar/images/RGB"),
        output_dir=Path("./workspace/output"),
        target_epsg=5186,
        gsd_m=0.05,         # 5cm/pixel (DJI Mavic 3E 100m 고도 기준 적절)
        k_neighbors=8,
    )

    # 경과 시간
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {timedelta(seconds=int(elapsed))}")