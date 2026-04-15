"""End-to-end 태양광 패널 결함 탐지 파이프라인.

흐름
----
1. RGB / IR 이미지 로드
2. IR → RGB 정합 (homography / affine / identity)
3. YOLO 로 패널 bbox 추출
4. 각 패널에 대해:
   a. 열화상 핫스팟 분석 (ΔT, 면적, 온도)
   b. 핫스팟이 있으면 VLM 으로 결함 유형 추론
   c. Fusion 으로 최종 라벨/심각도 결정
5. InspectionReport 생성 + 선택적 시각화 저장
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from ..cloud.storage import save_bytes
from ..config import AppConfig
from ..detection import HotspotAnalyzer, YOLODetector
from ..fusion import FusionAnalyzer
from ..preprocessing import align_ir_to_rgb, load_rgb, load_thermal
from ..preprocessing.thermal import to_heatmap_bgr
from ..schemas import (
    BBox,
    DefectType,
    HotspotCandidate,
    InspectionReport,
    InspectionRequest,
    PanelDefect,
    VLMVerdict,
)
from ..vlm import VLMClient


class DefectDetectionPipeline:
    """YOLO + 열화상 + VLM 결합 결함 탐지 파이프라인."""

    def __init__(self, config: AppConfig, vlm_client: Optional[VLMClient] = None) -> None:
        self.cfg = config
        self.detector = YOLODetector(
            weights=config.detector.weights,
            device=config.detector.device,
            imgsz=config.detector.imgsz,
            conf_threshold=config.detector.conf_threshold,
            iou_threshold=config.detector.iou_threshold,
            max_det=config.detector.max_det,
        )
        self.hotspot = HotspotAnalyzer(config.thermal.hotspot)
        self.fusion = FusionAnalyzer(config.fusion)
        # VLM 은 API 키 이슈 시 None 으로 유지하여 rule-only 운영 가능
        self.vlm: Optional[VLMClient] = vlm_client
        if self.vlm is None:
            try:
                self.vlm = VLMClient(config.vlm)
            except Exception as e:
                logger.warning(f"VLM client disabled: {e}")
                self.vlm = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, request: InspectionRequest, save_visualization: bool = True) -> InspectionReport:
        t0 = time.time()
        inspection_id = request.inspection_id or str(uuid.uuid4())

        # 1) Load
        rgb = load_rgb(request.rgb_uri)
        ir_temp = load_thermal(
            request.ir_uri,
            ir_format=request.ir_format,
            temp_range=(self.cfg.thermal.min_temp_c, self.cfg.thermal.max_temp_c),
        )

        # pseudo_color 입력인 경우 temp 맵 역추정
        if request.ir_format == "pseudo_color":
            from ..preprocessing.thermal import pseudo_color_to_temp

            ir_temp = pseudo_color_to_temp(
                ir_temp,
                temp_min_c=self.cfg.thermal.min_temp_c,
                temp_max_c=self.cfg.thermal.max_temp_c,
            )

        # 2) Align IR to RGB
        aligned = align_ir_to_rgb(
            rgb=rgb,
            ir_temp=ir_temp,
            method=self.cfg.registration.method,
            orb_features=self.cfg.registration.orb_features,
            ransac_reproj_threshold=self.cfg.registration.ransac_reproj_threshold,
        )

        # 3) YOLO panel detection
        all_boxes = self.detector.predict(aligned.rgb)
        panel_boxes = self.detector.panels(all_boxes)
        logger.info(f"[{inspection_id}] Detected {len(panel_boxes)} panels")

        # 4) Per-panel analysis
        panel_defects: list[PanelDefect] = []
        heatmap_bgr = to_heatmap_bgr(aligned.ir_temp)

        for i, panel in enumerate(panel_boxes):
            panel_id = f"{inspection_id}_panel_{i:04d}"
            panel_stats, hotspots = self.hotspot.analyze_panel(aligned.ir_temp, panel)

            # VLM 호출을 위한 크롭 준비
            vlm_verdict: Optional[VLMVerdict] = None
            if self.vlm is not None:
                rgb_crop = _crop(aligned.rgb, panel)
                ir_crop_bgr = _crop(heatmap_bgr, panel)
                hs_rgb = [_crop(aligned.rgb, h.bbox, pad=8) for h in hotspots]
                hs_ir = [_crop(heatmap_bgr, h.bbox, pad=8) for h in hotspots]
                try:
                    vlm_verdict = self.vlm.analyze_panel(
                        panel_id=panel_id,
                        rgb_crop=rgb_crop,
                        ir_crop_bgr=ir_crop_bgr,
                        hotspots=hotspots,
                        hotspot_rgb_crops=hs_rgb,
                        hotspot_ir_crops=hs_ir,
                        panel_mean_temp_c=panel_stats.t_mean,
                    )
                except Exception as e:
                    logger.warning(f"[{panel_id}] VLM call failed: {e}")
                    vlm_verdict = None

            defect = self.fusion.combine(
                panel_id=panel_id,
                panel_bbox=panel,
                panel_stats=panel_stats,
                hotspots=hotspots,
                vlm_verdict=vlm_verdict,
            )
            panel_defects.append(defect)

        # 5) Visualization & report
        viz_uri: Optional[str] = None
        if save_visualization:
            viz = draw_overlay(aligned.rgb, heatmap_bgr, panel_defects)
            _, buf = cv2.imencode(".jpg", viz, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            viz_uri = save_bytes(
                buf.tobytes(),
                prefix=self.cfg.storage.output_prefix,
                filename=f"{inspection_id}.jpg",
                backend=self.cfg.storage.backend,
                bucket=self.cfg.storage.gcs_bucket or self.cfg.storage.s3_bucket,
            )

        return InspectionReport(
            inspection_id=inspection_id,
            site_id=request.site_id,
            num_panels=len(panel_defects),
            num_defective_panels=sum(1 for p in panel_defects if not p.is_normal),
            panels=panel_defects,
            visualization_uri=viz_uri,
            processing_time_ms=(time.time() - t0) * 1000.0,
        )


# ====================================================================== #
# 유틸
# ====================================================================== #


def _crop(img: np.ndarray, bbox: BBox, pad: int = 0) -> np.ndarray:
    H, W = img.shape[:2]
    x1 = int(max(0, bbox.x1 - pad))
    y1 = int(max(0, bbox.y1 - pad))
    x2 = int(min(W, bbox.x2 + pad))
    y2 = int(min(H, bbox.y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return img[y1:y2, x1:x2].copy()


def draw_overlay(
    rgb: np.ndarray,
    heatmap_bgr: np.ndarray,
    panels: list[PanelDefect],
    alpha: float = 0.35,
) -> np.ndarray:
    """RGB 위에 IR heatmap + 결함 bbox 오버레이."""

    viz = cv2.addWeighted(rgb, 1.0 - alpha, heatmap_bgr, alpha, 0)

    for p in panels:
        color = _severity_color(p.severity) if not p.is_normal else (0, 200, 0)
        x1, y1, x2, y2 = map(int, p.panel_bbox.to_xyxy())
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
        label = f"{p.final_label.value} ({p.severity:.2f})"
        cv2.putText(
            viz, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
        for h in p.hotspots:
            hx1, hy1, hx2, hy2 = map(int, h.bbox.to_xyxy())
            cv2.rectangle(viz, (hx1, hy1), (hx2, hy2), (0, 0, 255), 1)
    return viz


def _severity_color(severity: float) -> tuple[int, int, int]:
    """0 (green) → 1 (red) BGR gradient."""

    severity = float(np.clip(severity, 0.0, 1.0))
    r = int(255 * severity)
    g = int(255 * (1.0 - severity))
    return (0, g, r)
