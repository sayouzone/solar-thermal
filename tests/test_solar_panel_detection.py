"""
추출기 동작 확인용 테스트:
1) DJI 드론 사진과 유사한 EXIF + XMP 가 포함된 합성 JPEG 생성
2) MetadataExtractor 로 읽어 GPS, 짐벌 각도, 상대고도가 올바르게 나오는지 검증
"""

from __future__ import annotations

import cv2
import json, dataclasses
import logging
import piexif
import sys
from fractions import Fraction
from pathlib import Path
from PIL import Image

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 추출기 동작 확인
from solar_thermal.detection.solar_panel_classical import ClassicalSolarPanelDetector
from solar_thermal.detection.solar_panel_yolo import YoloSolarPanelDetector
from solar_thermal.detection.solar_panel_sam_clip import SamClipSolarDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


if __name__ == "__main__":
    #out = Path("/tmp/test_drone_photo.jpg")
    #make_test_image(out)

    images = [
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217132210_0051_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217132205_0049_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/TM/DJI_20251217132209_0051_T.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/TM/DJI_20251217132207_0050_T.JPG",

        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217130231_0012_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217130225_0010_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217130223_0009_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217130220_0008_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217130217_0007_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217130214_0006_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217130212_0005_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217130209_0004_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217130204_0002_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217130200_0001_Z.JPG",
    ]

    log.info("=" * 60)
    log.info("STEP 1: Auto-labeling OpenCV")
    log.info("=" * 60)

    detector = ClassicalSolarPanelDetector()
    
    for image in images:
        #results = det.detect(image)
        #print(results)
        #print(json.dumps(d, indent=2, ensure_ascii=False, default=str))

        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image}")

        detections = detector.detect(img)
        print(f"[INFO] 탐지된 패널 후보: {len(detections)}개")
        for i, d in enumerate(detections[:10]):
            print(f"  #{i}: bbox={d.bbox}, aspect={d.aspect_ratio:.2f}, "
                f"rect={d.rectangularity:.2f}, score={d.score:.3f}")
    
    log.info("=" * 60)
    log.info("STEP 1: Auto-labeling YOLO11n Tuning")
    log.info("=" * 60)

    # 2) 사전학습 COCO 가중치로 시작 (패널 클래스 없음 — 학습 필요)
    #detector = YoloSolarPanelDetector("models/yolo11n.pt", target_classes=None, device="cpu")
    detector = YoloSolarPanelDetector("models/best.pt", target_classes=None, device="cpu")

    for image in images:
        results = detector.detect(image)
        
        print(f"[INFO] 탐지 수: {len(results)}")
        for i, d in enumerate(results[:20]):
            print(f"  #{i}: {d.class_name} conf={d.confidence:.3f} bbox={d.bbox_xyxy}")

    # 3) SAM 2.1 로 클래스 무관 세그멘테이션 → CLIP 으로 의미 필터링
    sam_ckpt = "models/sam2_hiera_large.pt"
    sam_cfg = "sam2_hiera_l.yaml"
    threshold = 0.25
    detector = SamClipSolarDetector(
        sam_checkpoint=sam_ckpt,
        sam_config=sam_cfg,
        clip_threshold=threshold,
    )

    for image in images:
        img = cv2.imread(image)
        results = detector.detect(img)
        print(f"[INFO] PV 마스크: {len(results)}개")
