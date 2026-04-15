"""파이프라인 전반에서 사용하는 데이터 모델."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DefectType(str, Enum):
    """검출 가능한 결함 유형 (IEC TS 62446-3 / IEA PVPS Task 13 참고)."""

    HOTSPOT_SINGLE_CELL = "hotspot_single_cell"       # 단일 셀 핫스팟
    HOTSPOT_MULTI_CELL = "hotspot_multi_cell"         # 다중 셀 핫스팟
    BYPASS_DIODE_FAIL = "bypass_diode_failure"        # 바이패스 다이오드 고장 (substring)
    PID = "potential_induced_degradation"             # PID
    DELAMINATION = "delamination"                     # 박리
    SOILING = "soiling"                               # 오염 (먼지/조분/눈)
    SHADING = "shading"                               # 그림자 (false positive 후보)
    CRACKED_CELL = "cracked_cell"                     # 셀 크랙
    JUNCTION_BOX = "junction_box_overheat"            # 접속함 과열
    NONE = "none"


class BBox(BaseModel):
    """축 정렬 바운딩 박스 (xyxy, 픽셀)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = Field(0.0, ge=0.0, le=1.0)
    class_name: str = ""

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class ThermalStats(BaseModel):
    """ROI 내부 열화상 통계."""

    t_min: float
    t_max: float
    t_mean: float
    t_std: float
    delta_t: float = Field(..., description="ROI 최고온 - 패널 평균온 (K)")
    hotspot_area_px: int = 0


class HotspotCandidate(BaseModel):
    """열화상 기반 핫스팟 후보."""

    bbox: BBox
    stats: ThermalStats
    # 핫스팟 중심 좌표 (원본 좌표계)
    centroid: tuple[float, float]
    # 열화상 기반 룰 판정
    rule_label: DefectType = DefectType.HOTSPOT_SINGLE_CELL
    rule_severity: float = Field(0.0, ge=0.0, le=1.0)


class VLMVerdict(BaseModel):
    """VLM 추론 결과."""

    defect_type: DefectType
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    # 임의의 보조 태그 (e.g., "likely_shadow", "needs_retest")
    tags: list[str] = Field(default_factory=list)


class PanelDefect(BaseModel):
    """패널 단위 결함 보고."""

    panel_id: str
    panel_bbox: BBox
    hotspots: list[HotspotCandidate] = Field(default_factory=list)
    vlm_verdict: Optional[VLMVerdict] = None
    final_label: DefectType = DefectType.NONE
    severity: float = Field(0.0, ge=0.0, le=1.0, description="0(정상) ~ 1(심각)")
    # 결함 없음을 확신할 경우 True
    is_normal: bool = False


class InspectionRequest(BaseModel):
    """파이프라인 입력."""

    rgb_uri: str = Field(..., description="RGB 이미지 경로 또는 gs://, s3:// URI")
    ir_uri: str = Field(..., description="IR 이미지 경로 또는 URI")
    # 열화상 형식 힌트
    ir_format: Literal["radiometric_tiff", "pseudo_color", "gray16"] = "radiometric_tiff"
    site_id: Optional[str] = None
    inspection_id: Optional[str] = None
    # FLIR/DJI 메타데이터가 JPEG EXIF 에 있을 경우 사용
    ir_meta: Optional[dict] = None


class InspectionReport(BaseModel):
    """파이프라인 출력."""

    inspection_id: str
    site_id: Optional[str] = None
    num_panels: int
    num_defective_panels: int
    panels: list[PanelDefect] = Field(default_factory=list)
    # 시각화 결과 이미지 URI (오버레이 저장 시)
    visualization_uri: Optional[str] = None
    processing_time_ms: float = 0.0
    model_version: str = "0.1.0"
