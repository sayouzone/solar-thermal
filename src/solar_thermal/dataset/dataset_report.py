"""
데이터셋 품질 리포트 생성
=========================

드론 이미지 + 라벨 + 메타데이터를 종합하여 품질 리포트를 생성합니다:

  1) 비행 궤적 (GPS) → matplotlib plot
  2) 라벨 개수 분포 (히스토그램)
  3) Bbox 크기 분포 (패널 크기 통계)
  4) Gimbal pitch 분포 (nadir 여부)
  5) 고도 분포
  6) 이미지당 평균 라벨 수 / 라벨 없는 이미지 비율

YOLO 학습 전 데이터 품질 검수에 사용.

사용 예:
    python dataset_report.py \\
        --images /mnt/user-data/uploads \\
        --labels data/labels \\
        --output data/report
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
import matplotlib.pyplot as plt

from solar_thermal.dataset.advanced_utils import DJIMetadataExtractor

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    out = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        out.append((
            int(parts[0]),
            float(parts[1]), float(parts[2]),
            float(parts[3]), float(parts[4]),
        ))
    return out


def generate_report(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    class_names: list[str] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = DJIMetadataExtractor()

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [p for p in sorted(images_dir.iterdir()) if p.suffix in exts]

    per_image: list[dict] = []
    all_widths: list[float] = []
    all_heights: list[float] = []
    class_counts: dict[int, int] = {}

    print(labels_dir)
    for img_path in images:
        meta = extractor.extract(img_path)
        labels = _read_yolo_labels(labels_dir / f"{img_path.stem}.txt")
        print(labels)

        for (cid, _, _, w, h) in labels:
            all_widths.append(w)
            all_heights.append(h)
            class_counts[cid] = class_counts.get(cid, 0) + 1

        per_image.append({
            "filename": img_path.name,
            "num_labels": len(labels),
            "meta": asdict(meta),
        })

    # Aggregate stats
    total_labels = sum(p["num_labels"] for p in per_image)
    unlabeled = sum(1 for p in per_image if p["num_labels"] == 0)
    stats = {
        "total_images":         len(per_image),
        "total_labels":         total_labels,
        "avg_labels_per_image": total_labels / max(len(per_image), 1),
        "unlabeled_images":     unlabeled,
        "unlabeled_ratio":      unlabeled / max(len(per_image), 1),
        "class_counts":         class_counts,
    }

    # --- Plots -------------------------------------------------------------

    # 1) 비행 궤적 (GPS)
    gps_points = [
        (p["meta"]["gps_lon"], p["meta"]["gps_lat"], p["num_labels"])
        for p in per_image
        if p["meta"]["gps_lon"] is not None
        and p["meta"]["gps_lat"] is not None
    ]
    if gps_points:
        fig, ax = plt.subplots(figsize=(10, 8))
        lons = [g[0] for g in gps_points]
        lats = [g[1] for g in gps_points]
        cnts = [g[2] for g in gps_points]
        sc = ax.scatter(
            lons, lats, c=cnts, cmap="viridis",
            s=120, edgecolors="white", linewidths=1.2,
        )
        # 궤적 연결
        ax.plot(lons, lats, "-", color="gray", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Drone flight trajectory ({len(gps_points)} captures)")
        plt.colorbar(sc, label="# labels per image")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "01_trajectory.png", dpi=120)
        plt.close()

    # 2) 라벨 개수 분포
    fig, ax = plt.subplots(figsize=(8, 5))
    counts_per_img = [p["num_labels"] for p in per_image]
    ax.hist(counts_per_img, bins=max(5, max(counts_per_img) + 1),
            edgecolor="black", alpha=0.8, color="#4C72B0")
    ax.set_xlabel("# labels per image")
    ax.set_ylabel("# images")
    ax.set_title("Label count distribution")
    ax.axvline(
        stats["avg_labels_per_image"], color="red",
        linestyle="--", label=f"avg={stats['avg_labels_per_image']:.1f}",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "02_label_count_dist.png", dpi=120)
    plt.close()

    # 3) Bbox 크기 (normalized)
    if all_widths:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(all_widths, bins=30, color="#55A868", edgecolor="black", alpha=0.8)
        ax1.set_xlabel("bbox width (normalized)")
        ax1.set_ylabel("count")
        ax1.set_title("Bbox width distribution")
        ax1.grid(alpha=0.3)

        ax2.hist(all_heights, bins=30, color="#C44E52", edgecolor="black", alpha=0.8)
        ax2.set_xlabel("bbox height (normalized)")
        ax2.set_ylabel("count")
        ax2.set_title("Bbox height distribution")
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "03_bbox_size.png", dpi=120)
        plt.close()

    # 4) Gimbal pitch 분포
    pitches = [
        p["meta"]["gimbal_pitch"] for p in per_image
        if p["meta"]["gimbal_pitch"] is not None
    ]
    if pitches:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(pitches, bins=30, color="#8172B2", edgecolor="black", alpha=0.8)
        ax.axvline(-90, color="red", linestyle="--", label="Ideal nadir (-90°)")
        ax.set_xlabel("Gimbal pitch (degrees)")
        ax.set_ylabel("# images")
        ax.set_title("Gimbal pitch distribution")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "04_gimbal_pitch.png", dpi=120)
        plt.close()

    # 5) 고도 분포
    alts = [
        p["meta"]["relative_alt"] for p in per_image
        if p["meta"]["relative_alt"] is not None
    ]
    if alts:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(alts, bins=20, color="#CCB974", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Relative altitude (m)")
        ax.set_ylabel("# images")
        ax.set_title("Flight altitude distribution")
        ax.axvline(sum(alts) / len(alts), color="red", linestyle="--",
                   label=f"avg={sum(alts)/len(alts):.1f}m")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "05_altitude.png", dpi=120)
        plt.close()

    # 6) 클래스 분포
    if class_counts and class_names:
        fig, ax = plt.subplots(figsize=(10, 5))
        ids_sorted = sorted(class_counts.keys())
        names = [
            class_names[i] if i < len(class_names) else f"class_{i}"
            for i in ids_sorted
        ]
        counts = [class_counts[i] for i in ids_sorted]
        ax.bar(names, counts, color="#4C72B0", edgecolor="black", alpha=0.8)
        ax.set_xlabel("class")
        ax.set_ylabel("count")
        ax.set_title("Class distribution")
        ax.grid(alpha=0.3, axis="y")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "06_class_dist.png", dpi=120)
        plt.close()

    # JSON 리포트 저장
    (output_dir / "report.json").write_text(
        json.dumps({"stats": stats, "per_image": per_image},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 텍스트 요약
    summary_lines = [
        "=" * 60,
        "Dataset Quality Report",
        "=" * 60,
        f"Total images:           {stats['total_images']}",
        f"Total labels:           {stats['total_labels']}",
        f"Avg labels/image:       {stats['avg_labels_per_image']:.2f}",
        f"Unlabeled images:       {stats['unlabeled_images']} "
        f"({stats['unlabeled_ratio']*100:.1f}%)",
        "",
        "Class distribution:",
    ]
    for cid in sorted(class_counts.keys()):
        name = (class_names[cid] if class_names and cid < len(class_names)
                else f"class_{cid}")
        summary_lines.append(f"  [{cid}] {name}: {class_counts[cid]}")

    if pitches:
        summary_lines += [
            "",
            f"Gimbal pitch: mean={sum(pitches)/len(pitches):.1f}°, "
            f"min={min(pitches):.1f}°, max={max(pitches):.1f}°",
        ]
    if alts:
        summary_lines += [
            f"Altitude:     mean={sum(alts)/len(alts):.1f}m, "
            f"min={min(alts):.1f}m, max={max(alts):.1f}m",
        ]

    summary = "\n".join(summary_lines)
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    log.info(summary)
    log.info("\n리포트 저장 완료: %s", output_dir)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--classes", nargs="+", default=["solar_panel"])
    args = ap.parse_args()

    generate_report(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        class_names=args.classes,
    )


if __name__ == "__main__":
    main()
