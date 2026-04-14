"""YOLO 모델을 태양광 데이터셋으로 파인튜닝.

권장 공개 데이터셋
------------------
* InfraredSolarModules (RaptorMaps, 11 classes)
* PVEL-AD (Electroluminescence defect dataset)
* Photovoltaic Thermal Images Dataset (Mendeley)

data.yaml 예시:
    path: /data/solar
    train: images/train
    val: images/val
    names:
      0: panel
      1: cell
      2: candidate_hotspot
      3: bypass_diode
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="data.yaml path")
    parser.add_argument("--base", default="yolov8m.pt", help="Pretrained base checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/solar")
    parser.add_argument("--name", default="yolo_solar_v1")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=20,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        mosaic=0.5,
        mixup=0.1,
        hsv_h=0.0,  # thermal imagery 에는 hue 변화 비활성화
        hsv_s=0.3,
        hsv_v=0.4,
    )
    print("Done. Best weights at:", Path(args.project) / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
