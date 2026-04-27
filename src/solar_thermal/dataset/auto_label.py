"""
Solar Panel Auto-Labeling Pipeline for YOLO11n
==============================================

드론 촬영 태양광 패널 이미지에 대해 자동으로 pre-labeling을 수행합니다.
세 가지 전략을 제공합니다:

1. HEURISTIC: OpenCV 기반 (라이브러리 의존성 최소, 빠름, 정확도 중간)
2. SAM2:      Segment Anything 2 기반 (정확도 높음, GPU 권장)
3. YOLO_WORLD: YOLO-World open-vocabulary (zero-shot, 빠름)

사용 예:
    python auto_label.py \\
        --images /mnt/user-data/uploads \\
        --output data/labels \\
        --strategy heuristic \\
        --classes solar_panel

출력: YOLO format labels (.txt)
     각 줄: <class_id> <x_center> <y_center> <width> <height>  (모두 0~1 정규화)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes (clean architecture — separate data from logic)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    """YOLO 형식 bounding box (정규화 좌표)."""
    class_id: int
    x_center: float  # 0~1
    y_center: float  # 0~1
    width: float     # 0~1
    height: float    # 0~1

    def to_yolo_line(self) -> str:
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )

    @classmethod
    def from_xyxy(
        cls,
        x1: float, y1: float, x2: float, y2: float,
        img_w: int, img_h: int,
        class_id: int = 0,
    ) -> "BBox":
        """Pixel 좌표(x1,y1,x2,y2) → 정규화 YOLO 포맷."""
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        return cls(class_id, cx, cy, w, h)


@dataclass
class LabelResult:
    """단일 이미지 라벨링 결과."""
    image_path: Path
    image_shape: tuple[int, int]  # (H, W)
    bboxes: list[BBox] = field(default_factory=list)

    def save(self, out_dir: Path) -> Path:
        """YOLO .txt 파일로 저장."""
        out_dir.mkdir(parents=True, exist_ok=True)
        
        self.bboxes.sort(key=lambda b: b.x_center)
        txt_path = out_dir / f"{self.image_path.stem}.txt"
        txt_path.write_text(
            "\n".join(b.to_yolo_line() for b in self.bboxes) + "\n"
            if self.bboxes else ""
        )
        return txt_path


# ---------------------------------------------------------------------------
# Strategy 1: Heuristic (OpenCV only, no deep-learning deps)
# ---------------------------------------------------------------------------

class HeuristicPanelDetector:
    """
    OpenCV 기반 태양광 패널 검출.

    전략:
      1) 이미지 → HSV 변환, 파란색 계열(패널 표면) 마스크 추출
      2) Morphological closing으로 셀 간격 메꾸기
      3) Contour 검출 → 직사각형 fitting
      4) Aspect ratio / area 필터링
      5) 회전된 패널도 잡기 위해 minAreaRect 사용 후 axis-aligned bbox로 확장

    주의: drone nadir view + 균일한 조명 가정. 과노출 영역(반사)은 별도 처리.
    """

    # 파라미터 (필요시 __init__ 인자로 노출 가능)
    # HSV 파란색 (선명한 패널 표면)
    HSV_PANEL_LOWER = np.array([90,  15,  20])
    HSV_PANEL_UPPER = np.array([130, 255, 200])
    # 저채도 회색/흰색 (반사광 / 저공비행 시 패널)
    HSV_GRAY_LOWER  = np.array([0,    0,  80])
    HSV_GRAY_UPPER  = np.array([180, 40, 230])
    MIN_AREA_RATIO  = 0.001  # 이미지 면적 대비 (0.1%)
    MAX_AREA_RATIO  = 0.30   # 너무 큰 블롭(배경) 배제
    MIN_ASPECT      = 1.8    # 패널은 길쭉 (세로:가로 또는 그 반대)

    def __init__(
        self,
        class_id: int = 0,
        debug_dir: Path | None = None,
        use_gray_mask: bool = True,
    ):
        self.class_id = class_id
        self.debug_dir = debug_dir
        self.use_gray_mask = use_gray_mask

    def detect(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Returns:
            (x1, y1, x2, y2) pixel bbox 리스트
        """
        h, w = img.shape[:2]
        img_area = h * w

        # 1) HSV 마스크 조합: 파란색(선명 패널) + 회색(반사/저공비행 패널)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, self.HSV_PANEL_LOWER, self.HSV_PANEL_UPPER)
        if self.use_gray_mask:
            mask_gray = cv2.inRange(
                hsv, self.HSV_GRAY_LOWER, self.HSV_GRAY_UPPER
            )
            mask = cv2.bitwise_or(mask_blue, mask_gray)
        else:
            mask = mask_blue

        # 2) 추가: 에지 기반 보조 (격자 패턴이 있는 영역 강조)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.bitwise_and(mask, mask)  # 단순 유지 (edges는 점수에만 활용)

        # 3) Morphology:
        #   - Closing은 셀 사이 격자는 메꾸되 이웃 패널과 병합되지 않도록 작게
        #   - Opening은 작은 노이즈 제거
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # 4) Contour 검출
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes: list[tuple[int, int, int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * self.MIN_AREA_RATIO:
                continue
            if area > img_area * self.MAX_AREA_RATIO:
                continue

            # 회전된 최소 사각형으로 aspect 계산
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw == 0 or rh == 0:
                continue

            aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
            if aspect < self.MIN_ASPECT:
                continue

            # Edge 밀도 체크: 패널은 격자 때문에 edge가 많음
            x, y, bw, bh = cv2.boundingRect(cnt)
            edge_roi = edges[y:y + bh, x:x + bw]
            edge_density = edge_roi.mean() / 255.0 if edge_roi.size > 0 else 0
            if edge_density < 0.05:  # 격자 패턴이 거의 없으면 배경일 가능성
                continue

            # Axis-aligned bbox (YOLO는 회전 bbox를 지원하지 않으므로)
            boxes.append((x, y, x + bw, y + bh))

        # 디버그 오버레이 저장
        if self.debug_dir is not None:
            self._save_debug(img, mask, boxes)

        return boxes

    def _save_debug(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        boxes: Sequence[tuple[int, int, int, int]],
    ) -> None:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        overlay = img.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        out = np.hstack([
            overlay,
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        ])
        cv2.imwrite(str(self.debug_dir / "debug_last.jpg"), out)

    def label(self, image_path: Path) -> LabelResult:
        img = cv2.imread(str(image_path))
        if img is None:
            log.warning("이미지 로드 실패: %s", image_path)
            return LabelResult(image_path, (0, 0), [])

        h, w = img.shape[:2]
        pixel_boxes = self.detect(img)
        bboxes = [
            BBox.from_xyxy(x1, y1, x2, y2, w, h, self.class_id)
            for (x1, y1, x2, y2) in pixel_boxes
        ]
        return LabelResult(image_path, (h, w), bboxes)


# ---------------------------------------------------------------------------
# Strategy 2: SAM2-based (optional, requires ultralytics >= 8.2)
# ---------------------------------------------------------------------------

class SAM2PanelDetector:
    """
    Meta SAM2 + YOLO 조합. Ultralytics SAM wrapper 사용.

    요구사항:
        pip install ultralytics
        # SAM2 가중치는 최초 실행 시 자동 다운로드

    전략:
        1) SAM2로 전체 이미지 자동 세그멘테이션
        2) 각 마스크를 bbox로 변환
        3) aspect / area 휴리스틱으로 패널만 필터링
    """

    def __init__(self, class_id: int = 0, model_name: str = "sam2_b.pt"):
        self.class_id = class_id
        try:
            from ultralytics import SAM
        except ImportError as e:
            raise ImportError(
                "SAM2 strategy 사용 시 `pip install ultralytics` 필요"
            ) from e
        self.model = SAM(model_name)

    def label(self, image_path: Path) -> LabelResult:
        img = cv2.imread(str(image_path))
        if img is None:
            return LabelResult(image_path, (0, 0), [])
        h, w = img.shape[:2]

        # SAM2 automatic segmentation
        results = self.model(str(image_path), verbose=False)

        bboxes: list[BBox] = []
        for r in results:
            if r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy()  # (N, H, W)
            for m in masks:
                ys, xs = np.where(m > 0.5)
                if len(xs) == 0:
                    continue
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                bw, bh = x2 - x1, y2 - y1
                if bw < 20 or bh < 20:
                    continue
                # 패널 필터 (길쭉한 직사각형)
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if aspect < 1.8:
                    continue
                area_ratio = (bw * bh) / (h * w)
                if area_ratio < 0.001 or area_ratio > 0.3:
                    continue
                bboxes.append(
                    BBox.from_xyxy(x1, y1, x2, y2, w, h, self.class_id)
                )

        return LabelResult(image_path, (h, w), bboxes)


# ---------------------------------------------------------------------------
# Strategy 3: YOLO-World (zero-shot open vocabulary)
# ---------------------------------------------------------------------------

class YOLOWorldDetector:
    """
    YOLO-World zero-shot 검출.

    장점: 'solar panel', 'photovoltaic panel' 같은 텍스트 프롬프트로 바로 검출
    단점: 드론 nadir view 도메인에서 COCO/LVIS 기반 모델은 recall이 낮을 수 있음

    요구사항:
        pip install ultralytics
    """

    def __init__(
        self,
        class_id: int = 0,
        model_name: str = "yolov8s-world.pt",
        prompts: Sequence[str] = ("solar panel", "photovoltaic module"),
        conf: float = 0.05,
    ):
        self.class_id = class_id
        self.conf = conf
        try:
            from ultralytics import YOLOWorld
        except ImportError as e:
            raise ImportError(
                "YOLO-World strategy 사용 시 `pip install ultralytics` 필요"
            ) from e
        self.model = YOLOWorld(model_name)
        self.model.set_classes(list(prompts))

    def label(self, image_path: Path) -> LabelResult:
        img = cv2.imread(str(image_path))
        if img is None:
            return LabelResult(image_path, (0, 0), [])
        h, w = img.shape[:2]

        results = self.model.predict(
            str(image_path), conf=self.conf, verbose=False
        )
        bboxes: list[BBox] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()  # (N, 4)
            for (x1, y1, x2, y2) in xyxy:
                bboxes.append(
                    BBox.from_xyxy(x1, y1, x2, y2, w, h, self.class_id)
                )
        return LabelResult(image_path, (h, w), bboxes)

class FineTunedYOLODetector:
    """
    파인튜닝된 YOLO(v8~v12) 검출기.

    텍스트 프롬프트 개념이 없으며, 학습 시 정의된 클래스 인덱스로만 검출합니다.
    YOLO-World와 달리 `set_classes()`를 호출하지 않습니다.

    Args:
        class_id: 저장할 라벨의 클래스 ID (auto-labeling 통일용, 예: solar_panel=0).
        model_name: 파인튜닝된 가중치 경로 (예: "models/best.pt").
        conf: confidence threshold.
        iou: NMS IoU threshold.
        imgsz: 추론 해상도. 드론 nadir view는 1280 권장.
        target_class_names: 모델이 학습한 클래스 이름 중 사용할 것들.
            None이면 전부 사용. target_class_ids와 동시에 쓰지 않음.
        target_class_ids: 모델이 학습한 클래스 ID 중 사용할 것들.
            None이면 전부 사용.
        device: "cuda" / "cpu" / "mps" / None(자동).
        min_area_ratio: 이미지 대비 너무 작은 오탐 제거 (0.0이면 비활성).
    """

    def __init__(
        self,
        class_id: int = 0,
        model_name: str | Path = "models/best.pt",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 1280,
        target_class_names: Sequence[str] | None = None,
        target_class_ids: Sequence[int] | None = None,
        device: str | None = None,
        min_area_ratio: float = 0.0,
    ):
        model_path = Path(model_name)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO 가중치를 찾을 수 없습니다: {model_path.resolve()}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "FineTunedYOLODetector 사용 시 `pip install ultralytics` 필요"
            ) from e

        self.class_id = class_id
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.min_area_ratio = min_area_ratio

        self.model = YOLO(str(model_path))

        # 모델이 학습한 클래스명 -> ID 매핑
        # Ultralytics YOLO는 self.model.names = {0: "solar_panel", 1: "hotspot", ...}
        self.model_names: dict[int, str] = dict(self.model.names)

        # 필터링 대상 클래스 ID 결정
        if target_class_names is not None and target_class_ids is not None:
            raise ValueError(
                "target_class_names와 target_class_ids 중 하나만 지정하세요."
            )

        if target_class_names is not None:
            name_to_id = {v: k for k, v in self.model_names.items()}
            missing = [n for n in target_class_names if n not in name_to_id]
            if missing:
                raise ValueError(
                    f"모델에 존재하지 않는 클래스명: {missing}. "
                    f"모델의 클래스: {list(name_to_id.keys())}"
                )
            self.target_class_ids: set[int] | None = {
                name_to_id[n] for n in target_class_names
            }
        elif target_class_ids is not None:
            valid_ids = set(self.model_names.keys())
            invalid = [c for c in target_class_ids if c not in valid_ids]
            if invalid:
                raise ValueError(
                    f"모델에 존재하지 않는 클래스 ID: {invalid}. "
                    f"모델의 클래스 ID: {sorted(valid_ids)}"
                )
            self.target_class_ids = set(target_class_ids)
        else:
            self.target_class_ids = None  # 전체 사용

    def label(self, image_path: Path) -> LabelResult:
        img = cv2.imread(str(image_path))
        if img is None:
            return LabelResult(image_path, (0, 0), [])
        h, w = img.shape[:2]
        img_area = float(h * w)

        # Ultralytics에 classes 인자를 주면 내부에서 필터링해주어 더 빠름
        classes_arg = (
            sorted(self.target_class_ids) if self.target_class_ids else None
        )

        results = self.model.predict(
            str(image_path),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=classes_arg,
            verbose=False,
        )

        bboxes: list[BBox] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            xyxy = r.boxes.xyxy.cpu().numpy()                     # (N, 4)
            cls = r.boxes.cls.cpu().numpy().astype(int)            # (N,)
            conf_arr = r.boxes.conf.cpu().numpy().astype(float)    # (N,)

            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf_arr):
                # classes_arg로 이미 필터되지만, 방어적으로 한 번 더
                if self.target_class_ids is not None and c not in self.target_class_ids:
                    continue

                # 너무 작은 박스 제거
                if self.min_area_ratio > 0.0:
                    area_ratio = float((x2 - x1) * (y2 - y1)) / img_area
                    if area_ratio < self.min_area_ratio:
                        continue

                bboxes.append(
                    BBox.from_xyxy(
                        float(x1), float(y1), float(x2), float(y2),
                        w, h, self.class_id,
                        # BBox 클래스가 conf를 받는다면 아래 전달. 없으면 제거.
                        # conf=float(p),
                    )
                )

        return LabelResult(image_path, (h, w), bboxes)

    # --- 유틸 ---------------------------------------------------------------

    def describe(self) -> str:
        """디버깅용 요약."""
        targets = (
            "전체"
            if self.target_class_ids is None
            else [self.model_names[c] for c in sorted(self.target_class_ids)]
        )
        return (
            f"FineTunedYOLODetector("
            f"weights={self.model.ckpt_path}, "
            f"model_classes={self.model_names}, "
            f"target={targets}, "
            f"conf={self.conf}, iou={self.iou}, imgsz={self.imgsz})"
        )

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

STRATEGIES = {
    "heuristic":  HeuristicPanelDetector,
    "sam2":       SAM2PanelDetector,
    "yolo_world": YOLOWorldDetector,
    "finetuned":  FineTunedYOLODetector,
}


def run(
    images_dir: Path,
    output_dir: Path,
    strategy: str,
    class_id: int = 0,
    debug_dir: Path | None = None,
) -> None:
    if strategy not in STRATEGIES:
        raise ValueError(f"지원하지 않는 strategy: {strategy}")

    # Detector 생성 (heuristic만 debug_dir 지원)
    if strategy == "heuristic":
        detector = HeuristicPanelDetector(
            class_id=class_id, debug_dir=debug_dir
        )
    elif strategy == "sam2":
        detector = SAM2PanelDetector(class_id=class_id, model_name="models/sam2_b.pt")
    elif strategy == "yolo_world":
        detector = YOLOWorldDetector(class_id=class_id, model_name="models/best.pt", prompts=("solar panel", "photovoltaic module"), conf=0.05)
    elif strategy == "finetuned":
        #detector = FineTunedYOLODetector(class_id=class_id, model_name="models/best.pt", target_class_names=("solar panel", "photovoltaic module"), conf=0.25)
        detector = FineTunedYOLODetector(
            class_id=class_id,
            model_name="models/best.pt",
            target_class_names=["solar_panel"],
            conf=0.25,
            imgsz=1280,
        )
    else:
        detector = STRATEGIES[strategy](class_id=class_id)

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = [
        p for p in sorted(images_dir.iterdir())
        if p.suffix in exts and p.is_file()
    ]
    log.info("대상 이미지 %d장, strategy=%s", len(image_paths), strategy)

    total_boxes = 0
    for i, img_path in enumerate(image_paths, 1):
        result = detector.label(img_path)
        print(result, type(result))
        result.save(output_dir)
        total_boxes += len(result.bboxes)
        log.info(
            "[%d/%d] %s → %d boxes",
            i, len(image_paths), img_path.name, len(result.bboxes),
        )

    log.info(
        "완료. 총 %d개 bbox 생성, 평균 %.1f개/이미지",
        total_boxes, total_boxes / max(len(image_paths), 1),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Solar panel auto-labeler")
    ap.add_argument("--images",   type=Path, required=True, help="이미지 디렉토리")
    ap.add_argument("--output",   type=Path, required=True, help="라벨 출력 디렉토리")
    ap.add_argument(
        "--strategy", choices=list(STRATEGIES),
        default="heuristic",
    )
    ap.add_argument("--class-id", type=int, default=0)
    ap.add_argument(
        "--debug-dir", type=Path, default=None,
        help="heuristic 전략의 시각화 저장 경로",
    )
    args = ap.parse_args()

    run(
        images_dir=args.images,
        output_dir=args.output,
        strategy=args.strategy,
        class_id=args.class_id,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()