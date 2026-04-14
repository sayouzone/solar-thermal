"""YAML 기반 설정 로더."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DetectorConfig(BaseModel):
    weights: str
    device: str = "cuda:0"
    imgsz: int = 1280
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300


class HotspotConfig(BaseModel):
    delta_t_threshold: float = 5.0
    abs_temp_threshold: float = 65.0
    min_area_px: int = 25
    morph_kernel: int = 3


class ThermalConfig(BaseModel):
    min_temp_c: float = -20.0
    max_temp_c: float = 120.0
    hotspot: HotspotConfig = Field(default_factory=HotspotConfig)


class RegistrationConfig(BaseModel):
    method: str = "homography"
    orb_features: int = 2000
    ransac_reproj_threshold: float = 5.0


class VLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-opus-4-6"
    max_tokens: int = 1024
    temperature: float = 0.0
    enable_prompt_caching: bool = True
    trigger_only_on_hotspot: bool = True
    max_crops_per_request: int = 4


class FusionConfig(BaseModel):
    severity_weights: dict[str, float] = Field(
        default_factory=lambda: {"delta_t": 0.4, "abs_temp": 0.3, "vlm_confidence": 0.3}
    )
    strategy: str = "ensemble"


class StorageConfig(BaseModel):
    backend: str = "local"
    gcs_bucket: str = ""
    s3_bucket: str = ""
    s3_region: str = "ap-northeast-2"
    output_prefix: str = "solar-thermal/reports"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    max_upload_mb: int = 50
    max_concurrent_jobs: int = 4


class AppConfig(BaseModel):
    detector: DetectorConfig
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    registration: RegistrationConfig = Field(default_factory=RegistrationConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def load_config(path: str | Path) -> AppConfig:
    """YAML 파일에서 AppConfig 로드."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return AppConfig(**data)
