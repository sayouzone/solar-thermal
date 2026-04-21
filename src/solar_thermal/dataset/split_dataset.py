"""
YOLO 데이터셋 분할 및 설정 파일 생성
====================================

입력: images/ 와 labels/ 디렉토리
출력: YOLO 표준 구조
    dataset/
      ├── images/{train,val,test}/
      ├── labels/{train,val,test}/
      └── data.yaml

사용 예:
    python split_dataset.py \\
        --images /mnt/user-data/uploads \\
        --labels data/labels \\
        --output dataset \\
        --split 0.7 0.2 0.1 \\
        --classes solar_panel
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.7
    val: float   = 0.2
    test: float  = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"split 합계 != 1.0 (got {total})")


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    classes: list[str],
    split: SplitConfig = SplitConfig(),
    seed: int = 42,
    copy_unlabeled: bool = False,
) -> Path:
    """
    이미지/라벨 쌍을 train/val/test로 분할 후 YOLO 표준 구조로 복사.

    Args:
        copy_unlabeled: 라벨 파일이 없는 이미지도 포함할지 (negative sample용)

    Returns:
        생성된 data.yaml 경로
    """
    rng = random.Random(seed)

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    all_images = [p for p in sorted(images_dir.iterdir()) if p.suffix in exts]

    # 라벨 존재 여부 필터링
    pairs: list[tuple[Path, Path | None]] = []
    for img in all_images:
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists() and lbl.stat().st_size > 0:
            pairs.append((img, lbl))
        elif copy_unlabeled:
            pairs.append((img, None))  # 빈 라벨로 처리

    if not pairs:
        raise RuntimeError("유효한 이미지-라벨 쌍이 없습니다.")

    log.info("유효 샘플 수: %d", len(pairs))

    # 셔플 후 분할
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * split.train)
    n_val   = int(n * split.val)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    # 디렉토리 생성 및 복사
    for split_name, items in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_src, lbl_src in items:
            shutil.copy2(img_src, img_out / img_src.name)
            if lbl_src is not None:
                shutil.copy2(lbl_src, lbl_out / lbl_src.name)
            else:
                # 빈 라벨 생성 (negative sample)
                (lbl_out / f"{img_src.stem}.txt").touch()

        log.info("  %s: %d 장", split_name, len(items))

    # data.yaml 작성
    yaml_path = output_dir / "data.yaml"
    yaml_content = {
        "path":  str(output_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": {i: c for i, c in enumerate(classes)},
        "nc":    len(classes),
    }
    yaml_path.write_text(
        yaml.safe_dump(yaml_content, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    log.info("data.yaml 생성: %s", yaml_path)
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images",  type=Path, required=True)
    ap.add_argument("--labels",  type=Path, required=True)
    ap.add_argument("--output",  type=Path, required=True)
    ap.add_argument(
        "--split", type=float, nargs=3,
        default=[0.7, 0.2, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    ap.add_argument(
        "--classes", nargs="+", required=True,
        help="클래스 이름 (예: solar_panel defect_hotspot)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--copy-unlabeled", action="store_true",
        help="라벨 없는 이미지도 negative sample로 포함",
    )
    args = ap.parse_args()

    split_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        classes=args.classes,
        split=SplitConfig(*args.split),
        seed=args.seed,
        copy_unlabeled=args.copy_unlabeled,
    )


if __name__ == "__main__":
    main()
