"""KD-Tree 기반 인접 페어 탐색 + tie-point 빌드.

수천 장 처리 시 풀 매칭 O(N²) 을 피하기 위해 RTK 좌표로 KD-Tree 를 만들어
각 이미지의 최근접 ``k`` 개 이웃과만 매칭. RANSAC fundamental matrix 로
outlier 제거 후 inlier 만 보존.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from solar_thermal.image.metadata import ImageMetadata

from ..crs import CRSConverter
from ..gpu_backend import free_gpu_memory
from .extract import extract_all_features
from .match import match_pair

logger = logging.getLogger(__name__)


def find_neighbor_pairs(metas: list[ImageMetadata],
                        crs: CRSConverter,
                        k_neighbors: int = 8) -> list[tuple[int, int]]:
    """RTK 좌표 KD-Tree 로 인접 페어 후보 생성.

    O(N²) 풀 매칭 대신 O(N log N) 으로 단축. 수천 장 처리 시 필수.

    Parameters
    ----------
    metas : 메타데이터 리스트
    crs : 좌표계 변환기 (WGS84 → 투영좌표)
    k_neighbors : 각 이미지당 이웃 개수 (자기 자신 제외)
    """
    if len(metas) < 2:
        logger.warning("이미지가 2장 미만 → 페어 매칭 불가")
        return []
    xy = np.array([crs.forward(m.gps.lng, m.gps.lat) for m in metas])
    tree = cKDTree(xy)
    pairs: set[tuple[int, int]] = set()
    for i in range(len(metas)):
        # k=1 이면 스칼라 반환되므로 +1 해서 자기 자신 포함 후 제거.
        _, idxs = tree.query(xy[i], k=min(k_neighbors + 1, len(metas)))
        idxs = np.atleast_1d(idxs)
        for j in idxs:
            j = int(j)
            if j != i:
                pairs.add((min(i, j), max(i, j)))
    logger.info("인접 페어 %d개 생성 (k=%d)", len(pairs), k_neighbors)
    return sorted(pairs)


def build_tie_points(metas: list[ImageMetadata],
                     pairs: list[tuple[int, int]]):
    """SIFT/ORB 특징점 추출 + 페어별 매칭 + RANSAC outlier 제거.

    Returns
    -------
    matches : ``dict[(i, j) -> (pts_i, pts_j, idx_i, idx_j)]``
        - ``pts_i, pts_j`` : (N, 2) 매칭된 픽셀 좌표
        - ``idx_i, idx_j`` : (N,)   각 매칭의 keypoint 인덱스
                             (track 생성 시 어느 keypoint 끼리 연결됐는지 추적용)
    features : ``dict[i -> (keypoints, descriptors, shape)]``
    """
    if not pairs:
        logger.warning("매칭할 페어가 없음 → tie point 생성 생략")
        return {}, {}

    # 페어에 등장하는 이미지만 추출 (불필요한 메모리/연산 절약).
    needed = sorted({i for pair in pairs for i in pair})
    paths = [Path(metas[i].origin_path) for i in needed]
    raw_feat = extract_all_features(paths)
    desc_type = raw_feat.pop("__desc_type__", "SIFT")
    # raw_feat 의 키는 0..len(needed)-1 → 원래 이미지 인덱스로 remap.
    features = {needed[i]: raw_feat[i] for i in range(len(needed))}

    matches: dict = {}
    for i, j in pairs:
        kp_i, desc_i, _ = features[i]
        kp_j, desc_j, _ = features[j]
        if desc_i is None or desc_j is None:
            continue
        good = match_pair(desc_i, desc_j, desc_type=desc_type)
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

    logger.info("유효 매칭 페어: %d (desc=%s)", len(matches), desc_type)
    free_gpu_memory()
    return matches, features


__all__ = ["find_neighbor_pairs", "build_tie_points"]