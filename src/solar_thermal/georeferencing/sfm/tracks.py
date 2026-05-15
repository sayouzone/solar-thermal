"""페어별 매칭을 다중 시점 track 으로 연결.

track 의 정의
-------------
``track`` = 동일 지상점에 대응하는 관측들의 집합
         = ``[(image_idx, keypoint_idx, px, py), ...]``

Union-Find 로 ``(image_idx, keypoint_idx)`` 노드들을 연결해, 페어별로 끊겨 있던
매칭들을 하나의 그룹으로 묶는다. 그 후 그룹별로 다음 조건을 검사:

* **충돌 검사**: 한 이미지에서 두 개 이상의 keypoint 가 같은 track 에 들어가면
  매칭 오류로 보고 track 전체를 폐기 (다른 지상점이 한 track 으로 섞이는 사고).
* **길이 필터**: ``min_track_len`` 미만은 삼각측량 불가, ``max_track_len`` 초과는
  보통 반복 텍스처 (태양광 패널 그리드 등) 의 잘못된 매칭.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class _UnionFind:
    """경로 압축 + 랭크 기반 Union-Find."""

    def __init__(self):
        self.parent: dict = {}
        self.rank: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        self.rank.setdefault(x, 0)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # 경로 압축.
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
                 max_track_len: int = 30) -> list:
    """페어별 매칭들을 연결해 track 리스트 생성.

    Parameters
    ----------
    matches : ``build_tie_points`` 의 출력
    features : ``build_tie_points`` 의 출력 (현재 함수에서는 직접 사용 안 하지만
        후속 단계와 API 일관성을 위해 받음)
    min_track_len : track 으로 인정할 최소 관측 수 (기본 2 = 삼각측량 가능 최소)
    max_track_len : 비정상적으로 긴 track 제거 임계 (보통 매칭 오류).
        하나의 점이 30 장 이상에 나타나면 의심스러움.

    Returns
    -------
    tracks : track 의 리스트.
    """
    uf = _UnionFind()

    # 1) 모든 매칭을 union — (i, kp_i) 와 (j, kp_j) 를 같은 집합으로.
    for (i, j), (_, _, idx_i, idx_j) in matches.items():
        for ki, kj in zip(idx_i, idx_j):
            uf.union((i, int(ki)), (j, int(kj)))

    # 2) 루트별로 관측을 모음. dict 키 중복 자동 제거로 동일 관측 중복 차단.
    groups: dict = {}
    for (i, j), (pts_i, pts_j, idx_i, idx_j) in matches.items():
        for n in range(len(idx_i)):
            ki, kj = int(idx_i[n]), int(idx_j[n])
            root = uf.find((i, ki))
            obs = groups.setdefault(root, {})
            obs[(i, ki)] = (i, ki, float(pts_i[n][0]), float(pts_i[n][1]))
            obs[(j, kj)] = (j, kj, float(pts_j[n][0]), float(pts_j[n][1]))

    # 3) track 필터링.
    tracks: list = []
    dropped_short = dropped_long = dropped_conflict = 0
    for root, obs in groups.items():
        images_seen = [img for (img, _kp) in obs.keys()]
        # 한 이미지에서 두 개 이상의 keypoint 가 한 track 에 들어가면 폐기.
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
                    lengths.mean(), lengths.max(), int((lengths == 2).sum()))
    return tracks


__all__ = ["build_tracks"]