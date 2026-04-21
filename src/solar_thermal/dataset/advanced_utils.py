"""
고급 라벨링 유틸리티
====================

1) HybridDetector: YOLO-World(대략 위치) + SAM2(정밀 마스크) 결합
2) NMSPostProcessor: 중복/근접 bbox 제거
3) DJIMetadataExtractor: EXIF/XMP에서 GPS/고도/Gimbal 각도 추출
4) DefectClassRegistry: IEC TS 62446-3 기반 결함 클래스 정의
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) Defect class registry (IEC TS 62446-3 aligned)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DefectClass:
    """IEC TS 62446-3 기반 결함 클래스 정의."""
    id: int
    name: str             # YOLO label
    korean: str           # UI 표시용
    severity: str         # minor / major / critical
    requires_ir: bool     # thermal IR 필요 여부
    iec_category: str     # IEC TS 62446-3 분류

    def to_dict(self) -> dict:
        return asdict(self)


# IEC TS 62446-3 §7 Table 1 기반 분류
DEFECT_CLASSES: list[DefectClass] = [
    # RGB 기반
    DefectClass(0, "solar_panel",      "정상 패널",     "none",     False, "baseline"),
    DefectClass(1, "soiling",          "오염",          "minor",    False, "soiling"),
    DefectClass(2, "shading",          "음영",          "minor",    False, "shading"),
    DefectClass(3, "cell_crack",       "셀 균열",       "major",    False, "cell_defect"),
    DefectClass(4, "glass_breakage",   "유리 파손",     "critical", False, "module_defect"),
    DefectClass(5, "delamination",     "박리",          "major",    False, "module_defect"),
    DefectClass(6, "snail_trail",      "달팽이자국",    "minor",    False, "cell_defect"),
    DefectClass(7, "discoloration",    "변색/황변",     "minor",    False, "encapsulant"),
    # Thermal IR 기반 (multi-modal 확장 시)
    DefectClass(8, "hotspot",          "핫스팟",        "major",    True,  "thermal"),
    DefectClass(9, "bypass_diode",     "바이패스다이오드", "major",  True,  "thermal"),
    DefectClass(10, "string_fault",    "스트링 불량",   "critical", True,  "thermal"),
    DefectClass(11, "pid",             "PID(전위유기열화)", "major", True,  "thermal"),
]


class DefectClassRegistry:
    """결함 클래스 조회 및 YOLO data.yaml 생성 헬퍼."""

    def __init__(self, classes: list[DefectClass] = DEFECT_CLASSES):
        self._classes = classes
        self._by_name = {c.name: c for c in classes}
        self._by_id   = {c.id:   c for c in classes}

    def get(self, key: int | str) -> DefectClass:
        if isinstance(key, int):
            return self._by_id[key]
        return self._by_name[key]

    def names_only(self, exclude_ir: bool = True) -> list[str]:
        """RGB 전용 학습 시 IR 클래스 제외."""
        if exclude_ir:
            return [c.name for c in self._classes if not c.requires_ir]
        return [c.name for c in self._classes]

    def to_yaml_names(self, exclude_ir: bool = True) -> dict[int, str]:
        names = self.names_only(exclude_ir)
        return {i: n for i, n in enumerate(names)}

    def export_schema(self, path: Path) -> None:
        """전체 분류 체계를 JSON으로 저장 (문서화용)."""
        path.write_text(
            json.dumps(
                [c.to_dict() for c in self._classes],
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# 2) NMS / duplicate removal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PixelBBox:
    """Pixel 좌표 bbox (내부 계산용)."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int = 0
    score: float = 1.0

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def iou(self, other: "PixelBBox") -> float:
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter == 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, other: "PixelBBox") -> float:
        """other가 self 안에 얼마나 들어있는지 (0~1)."""
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        return inter / other.area if other.area > 0 else 0.0


class NMSPostProcessor:
    """
    자동 라벨링 결과의 중복 제거.

    태양광 패널 자동 라벨링에서 자주 발생하는 두 가지 문제를 다룹니다:
      1) 중복 탐지: 같은 패널에 여러 bbox
      2) 포함 관계: 큰 bbox가 작은 bbox 여러 개를 포함
    """

    def __init__(
        self,
        iou_threshold: float = 0.4,
        containment_threshold: float = 0.8,
    ):
        self.iou_threshold = iou_threshold
        self.containment_threshold = containment_threshold

    def apply(self, boxes: list[PixelBBox]) -> list[PixelBBox]:
        """표준 NMS + containment 기반 중복 제거."""
        if not boxes:
            return []

        # score 내림차순 정렬
        sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
        keep: list[PixelBBox] = []

        for box in sorted_boxes:
            should_keep = True
            for kept in keep:
                if box.class_id != kept.class_id:
                    continue
                if box.iou(kept) > self.iou_threshold:
                    should_keep = False
                    break
                # 포함 관계 체크 (작은 bbox가 큰 bbox 안에 거의 들어있으면 제거)
                if kept.contains(box) > self.containment_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(box)

        return keep


# ---------------------------------------------------------------------------
# 3) DJI EXIF / XMP metadata extractor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DroneMetadata:
    """DJI 이미지에서 추출한 비행 메타데이터."""
    gps_lat: float | None = None
    gps_lon: float | None = None
    gps_alt: float | None = None          # 해발 고도 (m)
    relative_alt: float | None = None     # 지상 기준 상대 고도 (m)
    gimbal_pitch: float | None = None     # -90.0이면 정확한 nadir
    gimbal_yaw: float | None = None
    gimbal_roll: float | None = None
    flight_yaw: float | None = None
    capture_time: str | None = None
    camera_model: str | None = None
    focal_length_mm: float | None = None
    image_width: int | None = None
    image_height: int | None = None

    @property
    def is_nadir(self, tolerance: float = 5.0) -> bool:
        """정확한 수직 하향 촬영 여부 (pitch ≈ -90°)."""
        return (
            self.gimbal_pitch is not None
            and abs(self.gimbal_pitch + 90.0) < tolerance
        )


class DJIMetadataExtractor:
    """
    DJI JPEG에서 EXIF + XMP 메타데이터 추출.

    DJI drone JPEG는 표준 EXIF 외에 XMP 영역에 drone 고유 정보를 저장합니다:
        - drone-dji:RelativeAltitude
        - drone-dji:GimbalPitchDegree
        - drone-dji:GimbalYawDegree
        - drone-dji:FlightYawDegree

    의존성: pip install pillow (EXIF) / 정규식으로 XMP 파싱 (외부 lib 불필요)
    """

    # XMP 태그 추출용 정규식
    _XMP_PATTERN = re.compile(
        r'drone-dji:(\w+)\s*=\s*"([^"]+)"'
    )

    def extract(self, image_path: Path) -> DroneMetadata:
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS, GPSTAGS
        except ImportError:
            log.warning("Pillow 미설치: 메타데이터 스킵")
            return DroneMetadata()

        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # 1) EXIF
                exif_data = self._extract_exif(img, TAGS, GPSTAGS)

                # 2) XMP (DJI 전용)
                xmp_data = self._extract_xmp(image_path)

            return DroneMetadata(
                gps_lat=exif_data.get("gps_lat"),
                gps_lon=exif_data.get("gps_lon"),
                gps_alt=exif_data.get("gps_alt"),
                relative_alt=xmp_data.get("RelativeAltitude"),
                gimbal_pitch=xmp_data.get("GimbalPitchDegree"),
                gimbal_yaw=xmp_data.get("GimbalYawDegree"),
                gimbal_roll=xmp_data.get("GimbalRollDegree"),
                flight_yaw=xmp_data.get("FlightYawDegree"),
                capture_time=exif_data.get("capture_time"),
                camera_model=exif_data.get("camera_model"),
                focal_length_mm=exif_data.get("focal_length_mm"),
                image_width=width,
                image_height=height,
            )
        except Exception as e:
            log.warning("메타데이터 추출 실패 (%s): %s", image_path.name, e)
            return DroneMetadata()

    @staticmethod
    def _dms_to_decimal(dms, ref: str) -> float:
        """도분초 → 십진 좌표."""
        d, m, s = [float(x) for x in dms]
        val = d + m / 60.0 + s / 3600.0
        if ref in ("S", "W"):
            val = -val
        return val

    def _extract_exif(self, img, TAGS, GPSTAGS) -> dict:
        out: dict = {}
        exif = img.getexif()
        if not exif:
            return out

        # 일반 EXIF
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, str(tag_id))
            if tag == "Model":
                out["camera_model"] = str(value).strip()
            elif tag == "DateTime":
                out["capture_time"] = str(value)
            elif tag == "FocalLength":
                try:
                    out["focal_length_mm"] = float(value)
                except (TypeError, ValueError):
                    pass

        # GPS sub-IFD
        gps_ifd = exif.get_ifd(0x8825) if hasattr(exif, "get_ifd") else {}
        if gps_ifd:
            gps_data = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
            if "GPSLatitude" in gps_data and "GPSLatitudeRef" in gps_data:
                out["gps_lat"] = self._dms_to_decimal(
                    gps_data["GPSLatitude"], gps_data["GPSLatitudeRef"],
                )
            if "GPSLongitude" in gps_data and "GPSLongitudeRef" in gps_data:
                out["gps_lon"] = self._dms_to_decimal(
                    gps_data["GPSLongitude"], gps_data["GPSLongitudeRef"],
                )
            if "GPSAltitude" in gps_data:
                try:
                    out["gps_alt"] = float(gps_data["GPSAltitude"])
                except (TypeError, ValueError):
                    pass

        return out

    def _extract_xmp(self, image_path: Path) -> dict[str, float]:
        """
        JPEG 파일에서 XMP 패킷을 찾아 drone-dji 태그 파싱.

        XMP는 파일 내에 <?xpacket begin ... ?> ... <?xpacket end?> 로 감싸여 있으며
        상대적으로 크기가 작아 전체를 읽어도 부담이 적음.
        """
        try:
            raw = image_path.read_bytes()
        except OSError:
            return {}

        # XMP 패킷 추출
        start = raw.find(b"<?xpacket begin")
        end   = raw.find(b"<?xpacket end", start)
        if start == -1 or end == -1:
            return {}

        xmp_bytes = raw[start:end]
        try:
            xmp_text = xmp_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return {}

        result: dict[str, float] = {}
        for match in self._XMP_PATTERN.finditer(xmp_text):
            key, value = match.group(1), match.group(2)
            # "+47.5" 같은 값을 float로
            try:
                result[key] = float(value.lstrip("+"))
            except ValueError:
                pass

        return result


# ---------------------------------------------------------------------------
# 4) Hybrid detector: YOLO-World + SAM2
# ---------------------------------------------------------------------------

class HybridDetector:
    """
    YOLO-World (location proposal) + SAM2 (tight mask/bbox).

    단독 SAM2 auto-seg는 너무 많은 후보를 생성하고,
    YOLO-World는 drone nadir view 도메인에서 recall이 낮습니다.

    전략:
        1) YOLO-World로 "solar panel" 프롬프트 예측 → 대략 위치 포인트 생성
        2) 해당 포인트를 SAM2에 prompt로 전달 → 정밀 마스크 → tight bbox
        3) NMS 후처리로 중복 제거

    요구사항:
        pip install ultralytics
    """

    def __init__(
        self,
        class_id: int = 0,
        world_model: str = "yolov8s-world.pt",
        sam_model: str = "sam2_b.pt",
        prompts: Sequence[str] = ("solar panel", "photovoltaic module array"),
        world_conf: float = 0.05,
    ):
        self.class_id = class_id
        self.prompts = list(prompts)
        self.world_conf = world_conf

        try:
            from ultralytics import YOLOWorld, SAM
        except ImportError as e:
            raise ImportError(
                "Hybrid detector 사용 시 `pip install ultralytics` 필요"
            ) from e

        self.world = YOLOWorld(world_model)
        self.world.set_classes(self.prompts)
        self.sam = SAM(sam_model)
        self.nms = NMSPostProcessor(iou_threshold=0.4)

    def detect(self, image_path: Path) -> list[PixelBBox]:
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        h, w = img.shape[:2]

        # 1) YOLO-World: 프롬프트 기반 proposal
        world_results = self.world.predict(
            str(image_path), conf=self.world_conf, verbose=False
        )

        proposals: list[PixelBBox] = []
        for r in world_results:
            if r.boxes is None:
                continue
            xyxy  = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                proposals.append(PixelBBox(
                    int(x1), int(y1), int(x2), int(y2),
                    class_id=self.class_id, score=float(conf),
                ))

        if not proposals:
            return []

        # 2) SAM2: 각 proposal의 중심점을 prompt로 정밀 마스크 생성
        center_points = [
            [((p.x1 + p.x2) // 2), ((p.y1 + p.y2) // 2)]
            for p in proposals
        ]
        labels = [1] * len(center_points)  # 모두 foreground

        refined: list[PixelBBox] = []
        try:
            sam_results = self.sam(
                str(image_path),
                points=center_points,
                labels=labels,
                verbose=False,
            )
            for r in sam_results:
                if r.masks is None:
                    continue
                masks = r.masks.data.cpu().numpy()
                for mask, proposal in zip(masks, proposals):
                    ys, xs = np.where(mask > 0.5)
                    if len(xs) == 0:
                        refined.append(proposal)  # fallback
                        continue
                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())
                    # Aspect/size sanity
                    bw, bh = x2 - x1, y2 - y1
                    if bw < 20 or bh < 20:
                        continue
                    refined.append(PixelBBox(
                        x1, y1, x2, y2,
                        class_id=self.class_id, score=proposal.score,
                    ))
        except Exception as e:
            log.warning("SAM2 refinement 실패, YOLO-World 결과 사용: %s", e)
            refined = proposals

        # 3) NMS
        return self.nms.apply(refined)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--test-metadata", type=Path, help="메타데이터 추출 테스트")
    ap.add_argument("--export-schema", type=Path,
                    help="DefectClass schema를 JSON으로 저장")
    args = ap.parse_args()

    if args.test_metadata:
        extractor = DJIMetadataExtractor()
        meta = extractor.extract(args.test_metadata)
        print(json.dumps(asdict(meta), indent=2, ensure_ascii=False))

    if args.export_schema:
        registry = DefectClassRegistry()
        registry.export_schema(args.export_schema)
        print(f"Schema 저장: {args.export_schema}")
        print("\nRGB 전용 클래스:")
        for i, name in enumerate(registry.names_only(exclude_ir=True)):
            print(f"  {i}: {name}")