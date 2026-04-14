"""Unit tests (VLM 및 YOLO 호출은 mock 으로 대체)."""

from __future__ import annotations

import numpy as np
import pytest

from solar_thermal.config import HotspotConfig
from solar_thermal.detection.hotspot import HotspotAnalyzer
from solar_thermal.fusion.analyzer import FusionAnalyzer
from solar_thermal.config import FusionConfig
from solar_thermal.schemas import BBox, DefectType, VLMVerdict
from solar_thermal.preprocessing.registration import align_ir_to_rgb
from solar_thermal.vlm.client import _parse_verdict


# ---------------------------------------------------------------------- #
# Hotspot analyzer
# ---------------------------------------------------------------------- #


def test_hotspot_detects_high_delta_t():
    cfg = HotspotConfig(
        delta_t_threshold=5.0,
        abs_temp_threshold=65.0,
        min_area_px=4,
        morph_kernel=1,
    )
    analyzer = HotspotAnalyzer(cfg)

    # 50°C 배경의 패널, 중심에 70°C 작은 핫 영역 생성
    ir = np.full((100, 100), 50.0, dtype=np.float32)
    ir[40:50, 40:50] = 70.0
    panel = BBox(x1=0, y1=0, x2=100, y2=100, class_name="panel")

    stats, hs = analyzer.analyze_panel(ir, panel)
    assert len(hs) == 1
    assert hs[0].stats.delta_t > 15.0
    assert stats.t_max >= 70.0


def test_hotspot_ignores_noise_below_area():
    cfg = HotspotConfig(min_area_px=50, morph_kernel=1, delta_t_threshold=5.0)
    analyzer = HotspotAnalyzer(cfg)
    ir = np.full((100, 100), 40.0, dtype=np.float32)
    ir[10:12, 10:12] = 80.0  # 4 px only
    panel = BBox(x1=0, y1=0, x2=100, y2=100, class_name="panel")
    _, hs = analyzer.analyze_panel(ir, panel)
    assert hs == []


# ---------------------------------------------------------------------- #
# Fusion
# ---------------------------------------------------------------------- #


def test_fusion_ensemble_downgrades_shading():
    fc = FusionConfig(strategy="ensemble")
    fusion = FusionAnalyzer(fc)
    from solar_thermal.schemas import ThermalStats

    stats = ThermalStats(t_min=20, t_max=55, t_mean=40, t_std=3, delta_t=15, hotspot_area_px=100)
    vlm = VLMVerdict(defect_type=DefectType.SHADING, confidence=0.9, rationale="", tags=[])
    defect = fusion.combine(
        panel_id="p0",
        panel_bbox=BBox(x1=0, y1=0, x2=10, y2=10),
        panel_stats=stats,
        hotspots=[],
        vlm_verdict=vlm,
    )
    assert defect.final_label == DefectType.NONE


# ---------------------------------------------------------------------- #
# VLM JSON parsing
# ---------------------------------------------------------------------- #


def test_parse_verdict_with_code_fence():
    text = """```json
{"defect_type": "hotspot_single_cell", "confidence": 0.82,
 "rationale": "ΔT≈15K", "tags": []}
```"""
    v = _parse_verdict(text)
    assert v.defect_type == DefectType.HOTSPOT_SINGLE_CELL
    assert abs(v.confidence - 0.82) < 1e-6


def test_parse_verdict_invalid_json_falls_back():
    v = _parse_verdict("not a json at all")
    assert v.defect_type == DefectType.NONE
    assert "parse_error" in v.tags


# ---------------------------------------------------------------------- #
# Registration fallback
# ---------------------------------------------------------------------- #


def test_registration_identity_returns_resized():
    rgb = np.zeros((200, 300, 3), dtype=np.uint8)
    ir = np.full((100, 150), 30.0, dtype=np.float32)
    aligned = align_ir_to_rgb(rgb, ir, method="identity")
    assert aligned.ir_temp.shape == (200, 300)
