"""YOLO 모델을 태양광 데이터셋으로 파인튜닝.

사용법
------
    python scripts/train_yolo.py --data configs/dataset.yaml

    # 또는 샘플 템플릿을 복사해서 수정
    cp configs/dataset.example.yaml configs/dataset.yaml

data.yaml 요구 필드
-------------------
* path   : 데이터셋 루트 디렉터리 (절대경로 권장)
* train  : path 기준 train 이미지 디렉터리
* val    : path 기준 val 이미지 디렉터리
* names  : 클래스 이름 매핑 (YOLODetector 와 일치해야 함)

권장 공개 데이터셋
------------------
* InfraredSolarModules (RaptorMaps, 12 classes, IR only)
* PVEL-AD (Electroluminescence defect dataset)
* Mendeley Photovoltaic Thermal Images
* Roboflow Universe: "solar panel" / "photovoltaic defect" 검색
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _validate_data_yaml(data_path: str) -> None:
    """학습 전 data.yaml 의 존재 및 기본 경로 검증."""

    p = Path(data_path)
    if not p.is_file():
        sys.exit(
            f"[error] data.yaml not found: {data_path}\n"
            f"        Copy the template first:\n"
            f"          cp configs/dataset.example.yaml configs/dataset.yaml\n"
            f"        Then edit `path`, `train`, `val` to point to your dataset."
        )

    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        sys.exit(f"[error] failed to parse {data_path}: {e}")

    required = ("path", "train", "val", "names")
    missing = [k for k in required if k not in data]
    if missing:
        sys.exit(f"[error] {data_path} is missing required keys: {missing}")

    root = Path(data["path"])
    if not root.is_dir():
        sys.exit(
            f"[error] dataset root does not exist: {root}\n"
            f"        Fix the `path:` value in {data_path} or mount/download the dataset."
        )

    for key in ("train", "val"):
        sub = root / data[key]
        if not sub.exists():
            sys.exit(
                f"[error] {key} directory not found: {sub}\n"
                f"        Expected images under {sub}/ and labels under "
                f"{root / 'labels' / data[key].split('/')[-1]}/"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data.yaml (see configs/dataset.example.yaml)",
    )
    parser.add_argument("--base", default="yolov8m.pt", help="Pretrained base checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/solar")
    parser.add_argument("--name", default="yolo_solar_v1")
    args = parser.parse_args()

    _validate_data_yaml(args.data)

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
