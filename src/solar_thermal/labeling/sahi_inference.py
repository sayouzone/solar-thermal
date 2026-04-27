"""
SAHI: Slicing Aided Hyper Inference
====================================

5184×3888 드론 이미지를 YOLO 기본 imgsz=1280 또는 1920으로 추론하면
module 단위 작은 객체를 놓치기 쉽습니다. SAHI는 이미지를 격자로 자른 후
각 tile에서 개별 추론, 결과를 원본 좌표계로 취합합니다.

이 구현은 ultralytics sahi 라이브러리 없이 동작 — 경량화를 위해
다음만 의존:
  - torch (YOLO 모델 로드)
  - numpy, cv2 (이미지 처리)
  - ultralytics (YOLO class)

사용 예:
    # Tile-based inference for small modules
    python sahi_inference.py \\
        --images data/unlabeled \\
        --model workspace/round_N/weights/best.pt \\
        --output data/sahi_labels \\
        --tile-size 1024 \\
        --overlap 0.2
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tile generator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tile:
    """원본 이미지의 한 영역을 지정하는 offset 정보."""
    x_offset: int
    y_offset: int
    width: int
    height: int


def generate_tiles(
    img_width: int,
    img_height: int,
    tile_size: int = 1024,
    overlap_ratio: float = 0.2,
) -> list[Tile]:
    """
    이미지를 격자로 타일링.

    Args:
        overlap_ratio: 0.2 → 20% overlap (boundary에 걸친 객체 놓치지 않도록)

    Returns:
        Tile 리스트. 각 tile은 (x_offset, y_offset, width, height).
    """
    stride = int(tile_size * (1 - overlap_ratio))
    tiles: list[Tile] = []

    # X 방향 시작점
    x_starts = list(range(0, max(img_width - tile_size, 0) + 1, stride))
    if x_starts[-1] + tile_size < img_width:
        x_starts.append(img_width - tile_size)

    # Y 방향 시작점
    y_starts = list(range(0, max(img_height - tile_size, 0) + 1, stride))
    if y_starts[-1] + tile_size < img_height:
        y_starts.append(img_height - tile_size)

    for y in y_starts:
        for x in x_starts:
            w = min(tile_size, img_width - x)
            h = min(tile_size, img_height - y)
            tiles.append(Tile(x, y, w, h))

    return tiles


# ---------------------------------------------------------------------------
# Prediction & merge
# ---------------------------------------------------------------------------

@dataclass
class MergedDetection:
    """NMS 후 원본 좌표계의 detection."""
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    def to_yolo_line(self, img_w: int, img_h: int) -> str:
        cx = ((self.x1 + self.x2) / 2.0) / img_w
        cy = ((self.y1 + self.y2) / 2.0) / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        return f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _iou(a: MergedDetection, b: MergedDetection) -> float:
    xi1 = max(a.x1, b.x1); yi1 = max(a.y1, b.y1)
    xi2 = min(a.x2, b.x2); yi2 = min(a.y2, b.y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)


def merge_tile_predictions(
    all_detections: list[MergedDetection],
    iou_threshold: float = 0.3,
) -> list[MergedDetection]:
    """타일 간 중복 제거 (경계 객체가 인접 타일에서 중복 감지되는 문제 해결)."""
    if not all_detections:
        return []

    sorted_dets = sorted(all_detections, key=lambda d: d.score, reverse=True)
    keep: list[MergedDetection] = []
    for det in sorted_dets:
        should_keep = True
        for kept in keep:
            if det.class_id != kept.class_id:
                continue
            if _iou(det, kept) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(det)
    return keep


# ---------------------------------------------------------------------------
# SAHI runner
# ---------------------------------------------------------------------------

class SAHIInference:
    """타일링 기반 추론 orchestrator."""

    def __init__(
        self,
        model_path: Path,
        tile_size: int = 1024,
        overlap_ratio: float = 0.2,
        conf: float = 0.2,
        iou_nms: float = 0.3,
        min_box_area_ratio: float = 1e-5,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("`pip install ultralytics` 필요") from e

        self.model = YOLO(str(model_path))
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.conf = conf
        self.iou_nms = iou_nms
        self.min_box_area_ratio = min_box_area_ratio

    def predict_image(self, image_path: Path) -> tuple[int, int, list[MergedDetection]]:
        """단일 이미지 타일링 추론.

        Returns:
            (image_height, image_width, merged_detections)
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return (0, 0, [])
        H, W = img.shape[:2]

        # 원본보다 이미지가 작으면 타일링 스킵
        if H <= self.tile_size and W <= self.tile_size:
            log.debug("  이미지가 tile size 이내, 타일링 스킵")
            return (H, W, self._predict_single(img, 0, 0))

        tiles = generate_tiles(W, H, self.tile_size, self.overlap_ratio)
        log.info("  %s: %d tiles", image_path.name, len(tiles))

        all_dets: list[MergedDetection] = []
        for i, t in enumerate(tiles):
            crop = img[t.y_offset:t.y_offset + t.height,
                       t.x_offset:t.x_offset + t.width]
            tile_dets = self._predict_single(crop, t.x_offset, t.y_offset)
            # 경계 근접 bbox는 tile 내부로 clipping (원본 좌표계에서)
            for d in tile_dets:
                # Tile 경계에 너무 붙은 bbox는 잘린 객체일 가능성
                # → overlap 영역에서 이웃 tile이 더 완전한 버전을 찾길 기대
                # 여기서는 일단 모두 수용, NMS가 처리
                all_dets.append(d)

        # Image-wide NMS
        merged = merge_tile_predictions(all_dets, iou_threshold=self.iou_nms)

        # Min size filter
        img_area = H * W
        merged = [
            d for d in merged
            if ((d.x2 - d.x1) * (d.y2 - d.y1)) / img_area >= self.min_box_area_ratio
        ]

        return (H, W, merged)

    def _predict_single(
        self, tile_img: np.ndarray, x_offset: int, y_offset: int,
    ) -> list[MergedDetection]:
        """단일 tile 추론 후 원본 좌표계로 변환."""
        results = self.model.predict(
            tile_img, conf=self.conf, verbose=False, imgsz=self.tile_size,
        )
        dets: list[MergedDetection] = []
        if not results or results[0].boxes is None:
            return dets
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        cnfs = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), c, cf in zip(xyxy, clss, cnfs):
            dets.append(MergedDetection(
                class_id=int(c),
                x1=int(x1 + x_offset),
                y1=int(y1 + y_offset),
                x2=int(x2 + x_offset),
                y2=int(y2 + y_offset),
                score=float(cf),
            ))
        return dets

    def run_batch(
        self,
        images_dir: Path,
        output_labels_dir: Path,
    ) -> dict[str, int]:
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        images = [p for p in sorted(images_dir.iterdir()) if p.suffix in exts]
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        stats = {"total_images": len(images), "total_detections": 0}
        for img_path in images:
            H, W, dets = self.predict_image(img_path)
            stats["total_detections"] += len(dets)
            # Save YOLO label
            label_path = output_labels_dir / f"{img_path.stem}.txt"
            label_path.write_text(
                "\n".join(d.to_yolo_line(W, H) for d in dets)
                + ("\n" if dets else "")
            )
            log.info("  %s → %d detections", img_path.name, len(dets))

        log.info(
            "완료. 총 %d 이미지, %d detections",
            stats["total_images"], stats["total_detections"],
        )
        return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images",   type=Path, required=True)
    ap.add_argument("--model",    type=Path, required=True)
    ap.add_argument("--output",   type=Path, required=True)
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--overlap",   type=float, default=0.2)
    ap.add_argument("--conf",      type=float, default=0.2)
    ap.add_argument("--iou-nms",   type=float, default=0.3)
    args = ap.parse_args()

    inferer = SAHIInference(
        model_path=args.model,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap,
        conf=args.conf,
        iou_nms=args.iou_nms,
    )
    inferer.run_batch(args.images, args.output)


if __name__ == "__main__":
    main()
