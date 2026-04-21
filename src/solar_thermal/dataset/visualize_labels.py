"""
라벨 검수 도구
==============

1) YOLO 라벨을 이미지 위에 그려 시각적으로 검수
2) Label Studio 포맷으로 변환 (수동 수정 후 다시 YOLO로 export)

사용 예:
    # 시각화
    python visualize_labels.py visualize \\
        --images /mnt/user-data/uploads \\
        --labels data/labels \\
        --classes solar_panel \\
        --output data/visualized

    # Label Studio 가져오기용 JSON 생성
    python visualize_labels.py to-labelstudio \\
        --images /mnt/user-data/uploads \\
        --labels data/labels \\
        --classes solar_panel \\
        --output data/labelstudio_tasks.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

COLORS = [
    (  0, 255,   0),  # green  - solar_panel
    (  0,   0, 255),  # red    - hotspot
    (255, 200,   0),  # cyan   - soiling
    (255,   0, 255),  # magenta
    (  0, 255, 255),  # yellow
]


def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        out.append((cls_id, cx, cy, w, h))
    return out


def visualize(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    classes: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [p for p in sorted(images_dir.iterdir()) if p.suffix in exts]

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        labels = _read_yolo_labels(labels_dir / f"{img_path.stem}.txt")

        for (cls_id, cx, cy, bw, bh) in labels:
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = COLORS[cls_id % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            label_text = classes[cls_id] if cls_id < len(classes) else str(cls_id)
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, label_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
            )

        out_path = output_dir / f"{img_path.stem}_vis.jpg"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        log.info("%s → %d boxes", img_path.name, len(labels))


def to_labelstudio(
    images_dir: Path,
    labels_dir: Path,
    output_json: Path,
    classes: list[str],
    image_url_prefix: str = "/data/local-files/?d=",
) -> None:
    """
    YOLO 라벨을 Label Studio import JSON으로 변환.

    Label Studio에서 예측(pre-annotation)으로 자동 로드되므로
    작업자는 검수/수정만 하면 됩니다.
    """
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [p for p in sorted(images_dir.iterdir()) if p.suffix in exts]

    tasks = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        labels = _read_yolo_labels(labels_dir / f"{img_path.stem}.txt")

        predictions_results = []
        for (cls_id, cx, cy, bw, bh) in labels:
            # Label Studio: percent units, x/y = top-left corner
            x_pct = (cx - bw / 2) * 100
            y_pct = (cy - bh / 2) * 100
            w_pct = bw * 100
            h_pct = bh * 100
            label_name = (
                classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"
            )
            predictions_results.append({
                "from_name": "label",
                "to_name":   "image",
                "type":      "rectanglelabels",
                "original_width":  w,
                "original_height": h,
                "value": {
                    "x": x_pct,
                    "y": y_pct,
                    "width":  w_pct,
                    "height": h_pct,
                    "rotation": 0,
                    "rectanglelabels": [label_name],
                },
            })

        task = {
            "data": {
                "image": f"{image_url_prefix}{img_path.name}",
            },
            "predictions": [{
                "model_version": "auto_label_v1",
                "result": predictions_results,
            }] if predictions_results else [],
        }
        tasks.append(task)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Label Studio tasks 저장: %s (%d tasks)", output_json, len(tasks))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_vis = sub.add_parser("visualize")
    p_vis.add_argument("--images",  type=Path, required=True)
    p_vis.add_argument("--labels",  type=Path, required=True)
    p_vis.add_argument("--output",  type=Path, required=True)
    p_vis.add_argument("--classes", nargs="+", required=True)

    p_ls = sub.add_parser("to-labelstudio")
    p_ls.add_argument("--images",  type=Path, required=True)
    p_ls.add_argument("--labels",  type=Path, required=True)
    p_ls.add_argument("--output",  type=Path, required=True)
    p_ls.add_argument("--classes", nargs="+", required=True)
    p_ls.add_argument(
        "--url-prefix", default="/data/local-files/?d=",
        help="Label Studio local storage prefix",
    )

    args = ap.parse_args()

    if args.cmd == "visualize":
        visualize(args.images, args.labels, args.output, args.classes)
    elif args.cmd == "to-labelstudio":
        to_labelstudio(
            args.images, args.labels, args.output,
            args.classes, args.url_prefix,
        )


if __name__ == "__main__":
    main()
