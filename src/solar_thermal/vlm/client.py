"""Anthropic Claude 기반 VLM 클라이언트.

전략
----
* 시스템 프롬프트는 `cache_control: ephemeral` 로 캐싱되어 반복 호출 비용을 절감.
* 이미지는 base64 로 인코딩한 RGB/IR 크롭 쌍을 한 번에 전달.
* 응답은 JSON 만 받도록 강제 (프롬프트 지시 + 파싱 후 검증).
* 일시적 API 오류는 tenacity 로 재시도.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import VLMConfig
from ..schemas import DefectType, HotspotCandidate, VLMVerdict
from .prompts import ALLOWED_DEFECT_TYPES, build_system_prompt, build_user_prompt


class VLMClient:
    """Claude vision client for defect reasoning."""

    def __init__(self, config: VLMConfig, api_key: Optional[str] = None) -> None:
        # lazy import to keep cold start fast
        from anthropic import Anthropic

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = Anthropic(api_key=api_key)
        self.cfg = config

    # ---------------------------------------------------------------------
    # 공개 API
    # ---------------------------------------------------------------------
    def analyze_panel(
        self,
        panel_id: str,
        rgb_crop: np.ndarray,
        ir_crop_bgr: np.ndarray,
        hotspots: list[HotspotCandidate],
        hotspot_rgb_crops: list[np.ndarray],
        hotspot_ir_crops: list[np.ndarray],
        panel_mean_temp_c: float,
        ambient_temp_c: Optional[float] = None,
    ) -> VLMVerdict:
        """패널 단위 VLM 분석."""

        if self.cfg.trigger_only_on_hotspot and len(hotspots) == 0:
            return VLMVerdict(
                defect_type=DefectType.NONE,
                confidence=0.9,
                rationale="핫스팟이 감지되지 않아 VLM 분석을 생략.",
                tags=["vlm_skipped"],
            )

        max_crops = self.cfg.max_crops_per_request
        hotspots = hotspots[:max_crops]
        hotspot_rgb_crops = hotspot_rgb_crops[:max_crops]
        hotspot_ir_crops = hotspot_ir_crops[:max_crops]

        content: list[dict] = []
        # 1) panel overview
        content.append(_image_block(rgb_crop, "패널 RGB 전체"))
        content.append(_image_block(ir_crop_bgr, "패널 IR 컬러맵 전체"))
        # 2) per-hotspot zoom
        for i, (rgb, ir) in enumerate(zip(hotspot_rgb_crops, hotspot_ir_crops)):
            content.append(_image_block(rgb, f"핫스팟[{i}] RGB 확대"))
            content.append(_image_block(ir, f"핫스팟[{i}] IR 확대"))

        # 3) 마지막에 텍스트 프롬프트
        user_text = build_user_prompt(
            panel_id=panel_id,
            hotspots=hotspots,
            panel_mean_temp_c=panel_mean_temp_c,
            ambient_temp_c=ambient_temp_c,
        )
        content.append({"type": "text", "text": user_text})

        response_text = self._call_messages(content)
        return _parse_verdict(response_text)

    # ---------------------------------------------------------------------
    # 내부 헬퍼
    # ---------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _call_messages(self, content: list[dict]) -> str:
        resp = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system=build_system_prompt(enable_cache=self.cfg.enable_prompt_caching),
            messages=[{"role": "user", "content": content}],
        )
        # 첫 번째 text block 추출
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text
        raise RuntimeError("VLM response contained no text block")


# =============================================================================
# 유틸리티
# =============================================================================


def _image_block(img_bgr: np.ndarray, caption: str) -> dict:
    """OpenCV BGR 이미지를 Anthropic image content block 으로 인코딩."""

    # base64 png 로 인코딩 (lossless, 디버깅 용이)
    _, buf = cv2.imencode(".png", img_bgr)
    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64,
        },
    }


def _parse_verdict(text: str) -> VLMVerdict:
    """VLM JSON 응답을 VLMVerdict 로 파싱. 포맷이 깨졌을 경우 복구 시도."""

    text = text.strip()
    # ```json ... ``` 블록 제거
    if text.startswith("```"):
        text = text.strip("`")
        # 첫 줄의 'json' 제거
        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    # 가장 바깥 { } 만 추출
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse VLM JSON: {e}; raw={text[:200]}")
        return VLMVerdict(
            defect_type=DefectType.NONE,
            confidence=0.0,
            rationale="VLM 응답 파싱 실패",
            tags=["parse_error"],
        )

    dtype_str = str(payload.get("defect_type", "none")).lower()
    if dtype_str not in ALLOWED_DEFECT_TYPES:
        logger.warning(f"Unknown defect_type from VLM: {dtype_str}")
        dtype_str = "none"

    return VLMVerdict(
        defect_type=DefectType(dtype_str),
        confidence=float(payload.get("confidence", 0.0)),
        rationale=str(payload.get("rationale", "")),
        tags=list(payload.get("tags", []) or []),
    )
