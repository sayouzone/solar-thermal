"""열화상 온도 맵 기반 핫스팟 분석기.

YOLO 가 제공하는 panel bbox 내부에서:
1) 패널 평균 온도 및 분산 계산
2) (픽셀온도 - 패널평균) > ΔT threshold 인 마스크 생성
3) connected component 로 핫스팟 후보 추출
4) 면적/ΔT 기반 severity 산정 및 단일/다중 셀 여부 라벨링
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from ..config import HotspotConfig
from ..schemas import BBox, DefectType, HotspotCandidate, ThermalStats


class HotspotAnalyzer:
    def __init__(self, config: HotspotConfig) -> None:
        self.cfg = config

    def analyze_panel(
        self, ir_temp: np.ndarray, panel_bbox: BBox
    ) -> tuple[ThermalStats, list[HotspotCandidate]]:
        """단일 패널에 대해 열화상 통계 및 핫스팟 후보 추출.

        Parameters
        ----------
        ir_temp : (H, W) float32 단위 °C. NaN 허용.
        panel_bbox : RGB 좌표계의 패널 bbox.
        """

        x1, y1, x2, y2 = (int(max(0, v)) for v in panel_bbox.to_xyxy())
        H, W = ir_temp.shape[:2]
        x2 = min(W, x2)
        y2 = min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return self._empty_stats(), []

        roi = ir_temp[y1:y2, x1:x2]
        valid = np.isfinite(roi)
        if not valid.any():
            logger.debug(f"Panel ROI has no valid thermal pixels: {panel_bbox}")
            return self._empty_stats(), []

        t_values = roi[valid]
        t_mean = float(np.mean(t_values))
        t_std = float(np.std(t_values))
        t_min = float(np.min(t_values))
        t_max = float(np.max(t_values))
        delta_t = t_max - t_mean

        # 핫스팟 마스크: ΔT 초과 OR 절대 임계 초과
        mask = np.zeros_like(roi, dtype=np.uint8)
        mask[(roi - t_mean) > self.cfg.delta_t_threshold] = 255
        mask[roi > self.cfg.abs_temp_threshold] = 255
        mask[~valid] = 0

        # 노이즈 제거 (opening)
        k = max(1, int(self.cfg.morph_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # connected components
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        hotspots: list[HotspotCandidate] = []
        for i in range(1, num):  # 0 is background
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.cfg.min_area_px:
                continue
            cx = float(centroids[i, 0]) + x1
            cy = float(centroids[i, 1]) + y1
            hx1 = float(stats[i, cv2.CC_STAT_LEFT]) + x1
            hy1 = float(stats[i, cv2.CC_STAT_TOP]) + y1
            hw = float(stats[i, cv2.CC_STAT_WIDTH])
            hh = float(stats[i, cv2.CC_STAT_HEIGHT])

            region = roi[
                int(stats[i, cv2.CC_STAT_TOP]) : int(stats[i, cv2.CC_STAT_TOP] + hh),
                int(stats[i, cv2.CC_STAT_LEFT]) : int(stats[i, cv2.CC_STAT_LEFT] + hw),
            ]
            region_valid = region[np.isfinite(region)]
            if region_valid.size == 0:
                continue
            r_max = float(np.max(region_valid))
            r_mean = float(np.mean(region_valid))
            r_delta = r_max - t_mean

            # 단일 셀 vs 다중 셀 판단 (면적 기반 heuristic)
            panel_area = roi.size
            area_ratio = area / max(1, panel_area)
            if area_ratio > 0.08:
                rule_label = DefectType.HOTSPOT_MULTI_CELL
            else:
                rule_label = DefectType.HOTSPOT_SINGLE_CELL

            # severity 0..1 : ΔT / 30K 로 정규화 + 면적 가중
            severity = float(
                np.clip(r_delta / 30.0, 0.0, 1.0) * 0.7
                + np.clip(area_ratio / 0.1, 0.0, 1.0) * 0.3
            )

            hotspots.append(
                HotspotCandidate(
                    bbox=BBox(
                        x1=hx1,
                        y1=hy1,
                        x2=hx1 + hw,
                        y2=hy1 + hh,
                        score=min(1.0, severity + 0.2),
                        class_name="hotspot",
                    ),
                    stats=ThermalStats(
                        t_min=float(np.min(region_valid)),
                        t_max=r_max,
                        t_mean=r_mean,
                        t_std=float(np.std(region_valid)),
                        delta_t=r_delta,
                        hotspot_area_px=area,
                    ),
                    centroid=(cx, cy),
                    rule_label=rule_label,
                    rule_severity=severity,
                )
            )

        panel_stats = ThermalStats(
            t_min=t_min,
            t_max=t_max,
            t_mean=t_mean,
            t_std=t_std,
            delta_t=delta_t,
            hotspot_area_px=int((mask > 0).sum()),
        )
        return panel_stats, hotspots

    @staticmethod
    def _empty_stats() -> ThermalStats:
        return ThermalStats(t_min=0, t_max=0, t_mean=0, t_std=0, delta_t=0, hotspot_area_px=0)
