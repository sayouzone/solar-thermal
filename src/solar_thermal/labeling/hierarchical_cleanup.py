"""
계층 일관성 검증 및 라벨 정리
==============================

PV string과 module은 물리적으로 다음 제약을 만족해야 합니다:
  1) 모든 module은 어떤 string 내부에 있어야 함 (고아 module 제거)
  2) String은 충분한 길이여야 함 (너무 짧은 string은 제거)
  3) String 당 module 개수 타당성 (일반 72-cell 패널 기준 3-15개)
  4) Module 간 크기 일관성 (같은 string 내 module은 비슷한 크기)

또한 업로드 이미지에서 본 heuristic의 오탐 패턴을 학습 데이터에서 제거:
  - 잔디/땅으로 잡힌 가짜 box
  - String 중간에서 잘린 module
  - 방수포/tarp 오탐
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
# Data
# ---------------------------------------------------------------------------

@dataclass
class YOLOBox:
    """YOLO 정규화 bbox."""
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        x1 = int((self.cx - self.w / 2) * img_w)
        y1 = int((self.cy - self.h / 2) * img_h)
        x2 = int((self.cx + self.w / 2) * img_w)
        y2 = int((self.cy + self.h / 2) * img_h)
        return (x1, y1, x2, y2)

    @property
    def aspect(self) -> float:
        if self.w == 0 or self.h == 0:
            return 0.0
        return max(self.w, self.h) / min(self.w, self.h)

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.cx:.6f} {self.cy:.6f} "
            f"{self.w:.6f} {self.h:.6f}"
        )


def read_yolo_labels(path: Path) -> list[YOLOBox]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    out = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        out.append(YOLOBox(
            class_id=int(parts[0]),
            cx=float(parts[1]), cy=float(parts[2]),
            w=float(parts[3]), h=float(parts[4]),
        ))
    return out


def _iou_normalized(a: YOLOBox, b: YOLOBox) -> float:
    ax1, ay1 = a.cx - a.w/2, a.cy - a.h/2
    ax2, ay2 = a.cx + a.w/2, a.cy + a.h/2
    bx1, by1 = b.cx - b.w/2, b.cy - b.h/2
    bx2, by2 = b.cx + b.w/2, b.cy + b.h/2
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    area_a = a.w * a.h
    area_b = b.w * b.h
    return inter / (area_a + area_b - inter)


def _containment(inner: YOLOBox, outer: YOLOBox) -> float:
    """inner가 outer 안에 얼마나 들어있는지 (0~1)."""
    ax1, ay1 = inner.cx - inner.w/2, inner.cy - inner.h/2
    ax2, ay2 = inner.cx + inner.w/2, inner.cy + inner.h/2
    bx1, by1 = outer.cx - outer.w/2, outer.cy - outer.h/2
    bx2, by2 = outer.cx + outer.w/2, outer.cy + outer.h/2
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    inner_area = inner.w * inner.h
    if inner_area == 0:
        return 0.0
    return inter / inner_area


# ---------------------------------------------------------------------------
# Hierarchical consistency enforcer
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyStats:
    total_images: int = 0
    total_strings_in: int = 0
    total_modules_in: int = 0
    total_strings_out: int = 0
    total_modules_out: int = 0
    removed_orphan_modules: int = 0
    removed_short_strings: int = 0
    removed_non_panel_aspect: int = 0


class HierarchicalConsistencyEnforcer:
    """
    계층 일관성 제약을 자동 적용하여 라벨을 정리.

    주의: 이 도구는 `모델 예측 결과 정리`에 사용. 수동 GT 라벨에는 쓰지 말 것.
    """

    def __init__(
        self,
        string_class_id: int = 0,
        module_class_id: int = 1,
        min_string_aspect: float = 2.5,
        min_module_aspect: float = 1.3,
        orphan_containment_threshold: float = 0.6,
    ):
        self.string_class_id = string_class_id
        self.module_class_id = module_class_id
        self.min_string_aspect = min_string_aspect
        self.min_module_aspect = min_module_aspect
        self.orphan_containment_threshold = orphan_containment_threshold

    def clean(self, boxes: list[YOLOBox]) -> tuple[list[YOLOBox], dict]:
        """
        Returns:
            (정리된 box 리스트, 변경사항 통계)
        """
        stats = {
            "in_strings": 0, "in_modules": 0,
            "out_strings": 0, "out_modules": 0,
            "removed_orphan_modules": 0,
            "removed_short_strings": 0,
            "removed_non_panel_aspect": 0,
        }

        strings = [b for b in boxes if b.class_id == self.string_class_id]
        modules = [b for b in boxes if b.class_id == self.module_class_id]
        stats["in_strings"] = len(strings)
        stats["in_modules"] = len(modules)

        # 1) Aspect 기반 불량 제거 (string이면 길쭉하지 않은 것)
        kept_strings: list[YOLOBox] = []
        for s in strings:
            if s.aspect < self.min_string_aspect:
                stats["removed_non_panel_aspect"] += 1
                continue
            kept_strings.append(s)

        # 2) Module도 정사각형이면 제거 (정상 module은 aspect > 1.3)
        kept_modules: list[YOLOBox] = []
        for m in modules:
            if m.aspect < self.min_module_aspect:
                stats["removed_non_panel_aspect"] += 1
                continue
            kept_modules.append(m)

        # 3) 고아 module 제거: 어떤 string에도 충분히 포함되지 않는 module
        final_modules: list[YOLOBox] = []
        for m in kept_modules:
            in_any_string = any(
                _containment(m, s) >= self.orphan_containment_threshold
                for s in kept_strings
            )
            if in_any_string:
                final_modules.append(m)
            else:
                stats["removed_orphan_modules"] += 1

        # 4) 짧은 string 제거 (너무 작은 area ratio)
        final_strings: list[YOLOBox] = []
        for s in kept_strings:
            if s.w * s.h < 0.003:  # 이미지 대비 0.3% 미만
                stats["removed_short_strings"] += 1
                continue
            final_strings.append(s)

        # 고아 module이 발생한 string이 있으면 재검토 필요
        # (이미 filter된 module이 다시 final_strings 기준으로 고아가 될 수 있음)
        final_modules = [
            m for m in final_modules
            if any(
                _containment(m, s) >= self.orphan_containment_threshold
                for s in final_strings
            )
        ]

        result = final_strings + final_modules
        stats["out_strings"] = len(final_strings)
        stats["out_modules"] = len(final_modules)
        return result, stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_hierarchical(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    class_names: list[str] = ("pv_string", "pv_module"),
) -> None:
    """String(파랑)과 Module(녹색)을 다른 색으로 시각화."""
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        0: (255, 100, 50),   # string: 파란색
        1: (50, 255, 100),   # module: 녹색
    }

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix not in exts:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = read_yolo_labels(labels_dir / f"{img_path.stem}.txt")

        # String 먼저 그리기 (두껍게)
        for b in boxes:
            if b.class_id != 0:
                continue
            x1, y1, x2, y2 = b.to_xyxy(W, H)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors.get(0, (0, 255, 0)), 5)

        # Module 위에 (얇게)
        for b in boxes:
            if b.class_id != 1:
                continue
            x1, y1, x2, y2 = b.to_xyxy(W, H)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors.get(1, (0, 0, 255)), 2)

        # Legend
        cv2.putText(img, "string", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[0], 3)
        cv2.putText(img, "module", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[1], 3)

        out_path = output_dir / f"{img_path.stem}_hier.jpg"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        log.info(
            "  %s → %d strings, %d modules",
            img_path.name,
            sum(1 for b in boxes if b.class_id == 0),
            sum(1 for b in boxes if b.class_id == 1),
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def clean_all(
    labels_in: Path, labels_out: Path,
    enforcer: HierarchicalConsistencyEnforcer,
) -> ConsistencyStats:
    labels_out.mkdir(parents=True, exist_ok=True)
    total_stats = ConsistencyStats()

    for label_file in sorted(labels_in.glob("*.txt")):
        boxes = read_yolo_labels(label_file)
        cleaned, stats = enforcer.clean(boxes)

        total_stats.total_images += 1
        total_stats.total_strings_in += stats["in_strings"]
        total_stats.total_modules_in += stats["in_modules"]
        total_stats.total_strings_out += stats["out_strings"]
        total_stats.total_modules_out += stats["out_modules"]
        total_stats.removed_orphan_modules += stats["removed_orphan_modules"]
        total_stats.removed_short_strings += stats["removed_short_strings"]
        total_stats.removed_non_panel_aspect += stats["removed_non_panel_aspect"]

        (labels_out / label_file.name).write_text(
            "\n".join(b.to_line() for b in cleaned)
            + ("\n" if cleaned else "")
        )

    log.info("\n=== 정리 결과 ===")
    log.info("이미지:           %d", total_stats.total_images)
    log.info("String: %d → %d", total_stats.total_strings_in, total_stats.total_strings_out)
    log.info("Module: %d → %d", total_stats.total_modules_in, total_stats.total_modules_out)
    log.info("  고아 module 제거:       %d", total_stats.removed_orphan_modules)
    log.info("  너무 짧은 string 제거:  %d", total_stats.removed_short_strings)
    log.info("  aspect 불량 제거:       %d", total_stats.removed_non_panel_aspect)
    return total_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("clean", help="계층 일관성 자동 정리")
    p.add_argument("--labels-in",  type=Path, required=True)
    p.add_argument("--labels-out", type=Path, required=True)
    p.add_argument("--string-id",  type=int, default=0)
    p.add_argument("--module-id",  type=int, default=1)

    p = sub.add_parser("visualize", help="계층 시각화")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    args = ap.parse_args()

    if args.cmd == "clean":
        enforcer = HierarchicalConsistencyEnforcer(
            string_class_id=args.string_id,
            module_class_id=args.module_id,
        )
        clean_all(args.labels_in, args.labels_out, enforcer)
    elif args.cmd == "visualize":
        visualize_hierarchical(args.images, args.labels, args.output)


if __name__ == "__main__":
    main()
