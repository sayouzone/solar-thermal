"""
태양광 패널 탐지 - YOLO 기반 딥러닝 접근법
============================================
Ultralytics YOLO (v8/v11/v12 호환) 사용. 사전학습 모델 또는
커스텀 학습 가중치 모두 사용 가능.

설치:
    pip install ultralytics opencv-python

공개 PV 데이터셋 참고:
- PV03 / PV08 (중국 항공 PV 세그멘테이션)
- Duke California Solar Array (USGS 항공)
- Roboflow 'solar-panel' 공개 데이터셋들
- BDAPPV (Bradbury et al.)
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class YoloPanelDetection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


class YoloSolarPanelDetector:
    """
    YOLO 기반 PV 패널 탐지기.

    사용 예:
        # 1) 직접 학습한 가중치 사용
        det = YoloSolarPanelDetector("runs/detect/train/weights/best.pt")

        # 2) 사전학습 COCO 가중치로 시작 (패널 클래스 없음 — 학습 필요)
        det = YoloSolarPanelDetector("yolo11n.pt", target_classes=None)

        results = det.detect("aerial.jpg")
    """

    def __init__(
        self,
        weights: str,
        target_classes: Optional[List[str]] = ("solar_panel", "pv_panel", "panel"),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda:0",   # 엣지에서는 'cpu' 또는 Jetson: 'cuda:0'
        imgsz: int = 1280,         # 항공 이미지는 해상도 크게 유지
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics 패키지가 필요합니다: pip install ultralytics"
            ) from e

        self.model = YOLO(weights)
        self.target_classes = target_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.imgsz = imgsz

    def detect(self, image) -> List[YoloPanelDetection]:
        """
        Args:
            image: 파일 경로(str/Path) 또는 BGR numpy 배열
        Returns:
            탐지 결과 리스트
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections: List[YoloPanelDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            names = r.names  # {class_id: class_name}
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cls_name = names.get(cls_id, str(cls_id))

                if self.target_classes is not None:
                    if not any(t.lower() in cls_name.lower() for t in self.target_classes):
                        continue

                xyxy = box.xyxy.squeeze().tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append(
                    YoloPanelDetection(
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=float(box.conf.item()),
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )
        return detections

    @staticmethod
    def draw(image: np.ndarray, detections: List[YoloPanelDetection]) -> np.ndarray:
        vis = image.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 200, 255), -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return vis


# -------------------------------------------------------------------
# 커스텀 데이터셋으로 YOLO 학습하기 (Solar Plant 프로젝트용 템플릿)
# -------------------------------------------------------------------
TRAIN_CONFIG_YAML = """\
# data.yaml
path: ./datasets/solar_panels
train: images/train
val: images/val
test: images/test

names:
  0: solar_panel
  1: hotspot          # 열화상용 (선택)
  2: soiling          # 오염 (선택)
  3: crack            # 크랙 (선택)
"""


def train_example():
    """커스텀 태양광 데이터로 YOLO 학습 예시 — 실제 실행 전 데이터셋 준비 필요."""
    from ultralytics import YOLO

    model = YOLO("yolo11s.pt")  # 또는 yolo12s.pt, yolov8s.pt
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=1280,            # 항공 촬영은 고해상도가 유리
        batch=16,
        device=0,
        patience=20,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10,            # 항공뷰는 회전 증강 도움
        translate=0.1, scale=0.5, fliplr=0.5,
        project="runs/solar",
        name="yolo11s_pv",
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="입력 이미지 경로")
    parser.add_argument("--weights", required=True, help="YOLO 가중치(.pt) 경로")
    parser.add_argument("--out", default="yolo_result.jpg")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    detector = YoloSolarPanelDetector(
        weights=args.weights,
        conf_threshold=args.conf,
        device=args.device,
    )
    dets = detector.detect(img)
    print(f"[INFO] 탐지 수: {len(dets)}")
    for i, d in enumerate(dets[:20]):
        print(f"  #{i}: {d.class_name} conf={d.confidence:.3f} bbox={d.bbox_xyxy}")

    vis = detector.draw(img, dets)
    cv2.imwrite(args.out, vis)
    print(f"[INFO] 저장: {args.out}")


if __name__ == "__main__":
    main()
