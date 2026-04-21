"""
Claude Vision API 검증 레이어
==============================

자동 라벨링된 bbox를 Claude API로 2차 검증합니다.
SeongJung 님의 sayouzone/solar-thermal VLM verification 패턴과 동일한 접근.

워크플로우:
    1) YOLO 자동 라벨링 결과 로드
    2) 각 bbox를 이미지에서 crop
    3) Claude Vision으로 "이것이 solar panel인가?" 검증
    4) Confidence score + 결함 유형 반환
    5) Low-confidence / false-positive 제거

비용 절감 팁:
    - 이미 confidence가 높은 YOLO 결과는 스킵
    - Crop을 작게 (Claude 입력 비용 절감)
    - Batch 처리 대신 parallel (asyncio)

환경변수:
    ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np  # 지연 임포트

from solar_thermal.dataset.advanced_utils import DefectClassRegistry

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """VLM 검증 결과."""
    is_solar_panel: bool
    confidence: float                                    # 0~1
    defect_type: str = "none"                            # solar_panel / soiling / ...
    reasoning: str = ""
    panel_visibility: Literal["full", "partial", "occluded", "none"] = "full"
    raw_response: str = ""


@dataclass
class VerifiedBBox:
    """검증 후 bbox (원본 YOLO 정보 + 검증 결과)."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    verification: VerificationResult | None = None

    def to_yolo_line(self) -> str:
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


# ---------------------------------------------------------------------------
# Claude Vision verifier
# ---------------------------------------------------------------------------

VERIFICATION_PROMPT = """당신은 태양광 패널 드론 이미지 검수 전문가입니다.
주어진 crop 이미지를 분석하여 다음을 JSON으로 반환하세요:

{
  "is_solar_panel": true/false,
  "confidence": 0.0-1.0,
  "defect_type": "solar_panel" | "soiling" | "shading" | "cell_crack" |
                 "glass_breakage" | "delamination" | "snail_trail" |
                 "discoloration" | "none",
  "panel_visibility": "full" | "partial" | "occluded" | "none",
  "reasoning": "간단한 판단 근거 (한국어, 50자 이내)"
}

판단 기준:
- 태양광 패널의 특징적인 격자(셀) 패턴이 보이는가
- 패널 프레임이 명확한가
- crop이 단순 반사광, 그림자, 지면이라면 is_solar_panel=false
- 결함 판단은 visual 단서(변색, 균열 등)가 명확할 때만

반드시 JSON만 출력. 다른 텍스트, 마크다운 코드 블록 제외."""


class ClaudeVisionVerifier:
    """
    Claude Vision API로 bbox crop을 검증.

    API: anthropic SDK 사용 (동기), 배치 처리는 상위에서.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 300,
        crop_padding: int = 20,
        crop_max_side: int = 512,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Claude 검증 사용 시 `pip install anthropic` 필요"
            ) from e

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.crop_padding = crop_padding
        self.crop_max_side = crop_max_side

    def _crop_bbox(
        self,
        img: 'np.ndarray',
        x1: int, y1: int, x2: int, y2: int,
    ) -> bytes:
        """Bbox + padding으로 crop 후 JPEG bytes로 인코딩."""
        h, w = img.shape[:2]
        pad = self.crop_padding
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        crop = img[y1:y2, x1:x2]

        # Resize (long side <= crop_max_side)
        ch, cw = crop.shape[:2]
        max_side = max(ch, cw)
        if max_side > self.crop_max_side:
            scale = self.crop_max_side / max_side
            crop = cv2.resize(
                crop, (int(cw * scale), int(ch * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("JPEG 인코딩 실패")
        return buf.tobytes()

    def verify_crop(self, crop_jpeg: bytes) -> VerificationResult:
        """단일 crop을 Claude Vision으로 검증."""
        b64 = base64.standard_b64encode(crop_jpeg).decode("ascii")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": VERIFICATION_PROMPT},
                ],
            }],
        )

        raw = response.content[0].text if response.content else ""
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> VerificationResult:
        """Claude 응답에서 JSON 추출."""
        # 마크다운 코드 블록 제거
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning("JSON 파싱 실패, raw=%r", raw[:200])
            return VerificationResult(
                is_solar_panel=False,
                confidence=0.0,
                reasoning="(파싱 실패)",
                raw_response=raw,
            )

        return VerificationResult(
            is_solar_panel=bool(data.get("is_solar_panel", False)),
            confidence=float(data.get("confidence", 0.0)),
            defect_type=str(data.get("defect_type", "none")),
            reasoning=str(data.get("reasoning", "")),
            panel_visibility=str(data.get("panel_visibility", "full")),
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _read_yolo_labels(label_path: Path) -> list[VerifiedBBox]:
    if not label_path.exists():
        return []
    out: list[VerifiedBBox] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        out.append(VerifiedBBox(
            class_id=int(parts[0]),
            x_center=float(parts[1]),
            y_center=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        ))
    return out


def verify_labels(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    confidence_threshold: float = 0.5,
    update_class_on_defect: bool = True,
    class_names: list[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    YOLO 라벨을 Claude Vision으로 검증.

    Returns:
        stats: 전체 검증 통계
    """
    registry = DefectClassRegistry()
    class_names = class_names or registry.names_only(exclude_ir=True)
    name_to_id = {n: i for i, n in enumerate(class_names)}

    verifier = None if dry_run else ClaudeVisionVerifier()

    output_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [p for p in sorted(images_dir.iterdir()) if p.suffix in exts]

    stats = {
        "total_boxes": 0,
        "verified_boxes": 0,
        "rejected_boxes": 0,
        "class_changes": 0,
    }
    report: list[dict] = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        bboxes = _read_yolo_labels(labels_dir / f"{img_path.stem}.txt")

        kept: list[VerifiedBBox] = []
        for bbox in bboxes:
            stats["total_boxes"] += 1

            # YOLO 정규화 → pixel
            x1 = int((bbox.x_center - bbox.width / 2) * w)
            y1 = int((bbox.y_center - bbox.height / 2) * h)
            x2 = int((bbox.x_center + bbox.width / 2) * w)
            y2 = int((bbox.y_center + bbox.height / 2) * h)

            if dry_run:
                kept.append(bbox)
                continue

            try:
                crop = verifier._crop_bbox(img, x1, y1, x2, y2)
                result = verifier.verify_crop(crop)
            #except Exception as e:
            #    log.warning("검증 실패 %s: %s", img_path.name, e)
            #    kept.append(bbox)  # 실패 시 보수적으로 유지
            #    continue
            finally:
                pass

            bbox.verification = result

            # 판단 로직
            if not result.is_solar_panel:
                stats["rejected_boxes"] += 1
                log.info("  REJECT %s @ (%d,%d)-(%d,%d): %s",
                         img_path.name, x1, y1, x2, y2, result.reasoning)
                report.append({
                    "image": img_path.name,
                    "bbox": [x1, y1, x2, y2],
                    "action": "rejected",
                    "reason": result.reasoning,
                })
                continue

            if result.confidence < confidence_threshold:
                stats["rejected_boxes"] += 1
                log.info("  LOW-CONF %s conf=%.2f: %s",
                         img_path.name, result.confidence, result.reasoning)
                report.append({
                    "image": img_path.name,
                    "bbox": [x1, y1, x2, y2],
                    "action": "low_confidence",
                    "confidence": result.confidence,
                })
                continue

            # 결함 유형으로 클래스 변경
            if (update_class_on_defect
                    and result.defect_type in name_to_id
                    and result.defect_type != class_names[bbox.class_id]):
                new_id = name_to_id[result.defect_type]
                log.info("  RELABEL %s class %d → %d (%s)",
                         img_path.name, bbox.class_id, new_id,
                         result.defect_type)
                bbox.class_id = new_id
                stats["class_changes"] += 1

            stats["verified_boxes"] += 1
            kept.append(bbox)

        # 저장
        out_path = output_dir / f"{img_path.stem}.txt"
        out_path.write_text(
            "\n".join(b.to_yolo_line() for b in kept) + ("\n" if kept else "")
        )

    # Report 저장
    report_path = output_dir.parent / "verification_report.json"
    report_path.write_text(
        json.dumps({"stats": stats, "details": report},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("\n=== 검증 완료 ===")
    log.info("전체: %d / 승인: %d / 기각: %d / 클래스 변경: %d",
             stats["total_boxes"], stats["verified_boxes"],
             stats["rejected_boxes"], stats["class_changes"])
    log.info("리포트: %s", report_path)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True,
                    help="검증된 라벨 출력 디렉토리")
    ap.add_argument("--conf",   type=float, default=0.5)
    ap.add_argument("--no-relabel", action="store_true",
                    help="결함 유형 자동 re-labeling 비활성화")
    ap.add_argument("--dry-run", action="store_true",
                    help="API 호출 없이 파일 흐름만 테스트")
    args = ap.parse_args()

    verify_labels(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        confidence_threshold=args.conf,
        update_class_on_defect=not args.no_relabel,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
