"""YOLO + 열화상 핫스팟 + VLM 판정을 결합하는 Fusion 분석기.

strategy
--------
* rule_priority : 룰 기반 판정을 우선. VLM 은 보조.
* vlm_priority  : VLM 판정을 우선. 룰이 결함을 탐지 못해도 VLM 이 결함이면 결함.
* ensemble      : 가중 앙상블. severity 를 3 요소의 weighted sum 으로 계산.
"""

from __future__ import annotations

import numpy as np

from ..config import FusionConfig
from ..schemas import (
    DefectType,
    HotspotCandidate,
    PanelDefect,
    ThermalStats,
    VLMVerdict,
)


class FusionAnalyzer:
    def __init__(self, config: FusionConfig) -> None:
        self.cfg = config

    def combine(
        self,
        panel_id: str,
        panel_bbox,
        panel_stats: ThermalStats,
        hotspots: list[HotspotCandidate],
        vlm_verdict: VLMVerdict | None,
    ) -> PanelDefect:
        """단일 패널의 최종 결함 판정."""

        strategy = self.cfg.strategy

        # rule 기반 라벨
        rule_label = DefectType.NONE
        if hotspots:
            rule_label = max(hotspots, key=lambda h: h.rule_severity).rule_label

        # VLM 기반 라벨
        vlm_label = vlm_verdict.defect_type if vlm_verdict else DefectType.NONE
        vlm_conf = vlm_verdict.confidence if vlm_verdict else 0.0

        # severity 계산용 정규화 값
        delta_t_norm = float(np.clip(panel_stats.delta_t / 30.0, 0.0, 1.0))
        abs_t_norm = float(np.clip((panel_stats.t_max - 60.0) / 40.0, 0.0, 1.0))

        w = self.cfg.severity_weights
        severity = (
            w.get("delta_t", 0.4) * delta_t_norm
            + w.get("abs_temp", 0.3) * abs_t_norm
            + w.get("vlm_confidence", 0.3) * vlm_conf
        )
        severity = float(np.clip(severity, 0.0, 1.0))

        # 최종 라벨 결정
        if strategy == "rule_priority":
            final = rule_label if rule_label != DefectType.NONE else vlm_label
        elif strategy == "vlm_priority":
            if vlm_label == DefectType.SHADING:
                # VLM 이 그림자라 판단하면 결함 아님으로 강제 downgrade
                final = DefectType.NONE
                severity = min(severity, 0.2)
            elif vlm_label != DefectType.NONE and vlm_conf >= 0.5:
                final = vlm_label
            else:
                final = rule_label
        elif strategy == "ensemble":
            # VLM confidence 가 높고 SHADING 이면 결함 취소
            if vlm_label == DefectType.SHADING and vlm_conf >= 0.6:
                final = DefectType.NONE
                severity = min(severity, 0.2)
            elif vlm_label != DefectType.NONE and vlm_conf >= 0.7:
                final = vlm_label
            else:
                final = rule_label
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

        is_normal = final == DefectType.NONE and severity < 0.2

        return PanelDefect(
            panel_id=panel_id,
            panel_bbox=panel_bbox,
            hotspots=hotspots,
            vlm_verdict=vlm_verdict,
            final_label=final,
            severity=severity,
            is_normal=is_normal,
        )
