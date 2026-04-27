"""
End-to-End 파이프라인 실행
===========================

전체 흐름:
    1. 자동 라벨링 (heuristic/sam2/yolo_world)
    2. 시각화 → 수동 검수
    3. (Label Studio로 수정 후 re-export)
    4. 데이터셋 분할 & data.yaml 생성
    5. YOLO11n 학습 시작

사용:
    python run_pipeline.py \
        --images /mnt/user-data/uploads \
        --work-dir ./workspace \
        --strategy heuristic \
        --classes solar_panel \
        --train-epochs 100
    
    python scripts/run_training.py \
        --images data/solar/images/RGB \
        --work-dir ./workspace \
        --strategy heuristic \
        --classes solar_panel \
        --skip-train
    
    python scripts/run_training.py \
        --images data/solar/images/RGB \
        --work-dir ./workspace \
        --strategy heuristic \
        --classes solar_panel \
        --skip-train

    python scripts/run_training.py \
        --images data/solar/images/RGB \
        --work-dir ./workspace \
        --strategy sam2 \
        --classes solar_panel \
        --steps auto_label visualize

    python scripts/run_training.py \
        --images data/solar/images/RGB \
        --work-dir ./workspace \
        --strategy sam2 \
        --classes solar_panel \
        --steps verify report visualize split \
        --skip-train

    python scripts/run_training.py \
        --images data/solar/images/RGB \
        --work-dir ./workspace \
        --strategy sam2 \
        --classes solar_panel \
        --steps split train \
        --train-epochs 100

    python scripts/run_training.py \
        --images data/solar/images/RGB \
        --work-dir ./workspace \
        --strategy finetune \
        --classes solar_panel \
        --steps inference \
        --train-epochs 100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from ultralytics import YOLO

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solar_thermal.dataset.auto_label import run as run_auto_label
from solar_thermal.dataset.split_dataset import split_dataset, SplitConfig
from solar_thermal.dataset.vlm_verify import verify_labels
from solar_thermal.dataset.dataset_report import generate_report
from solar_thermal.dataset.visualize_labels import visualize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images",       type=Path, required=True)
    ap.add_argument("--work-dir",     type=Path, default=Path("workspace"))
    ap.add_argument("--strategy",     default="heuristic",
                    choices=["heuristic", "sam2", "yolo_world", "finetuned"])
    ap.add_argument("--classes",      nargs="+", default=["solar_panel"])
    ap.add_argument("--steps",        nargs="+",
                    choices=["auto_label", "verify", "report", "visualize", "split", "train", "inference", "all"],
                    default=["auto_label", "verify", "report", "visualize", "split"],)
    ap.add_argument("--conf",         type=float, default=0.5)
    ap.add_argument("--no-relabel",   action="store_true",
                    help="결함 유형 자동 re-labeling 비활성화")
    ap.add_argument("--dry-run",      action="store_true",
                    help="API 호출 없이 파일 흐름만 테스트")
    ap.add_argument("--split",        type=float, nargs=3,
                    default=[0.7, 0.2, 0.1])
    ap.add_argument("--skip-train",   action="store_true",
                    help="YOLO 학습은 스킵하고 데이터셋까지만 준비")
    ap.add_argument("--train-epochs", type=int, default=100)
    ap.add_argument("--train-imgsz",  type=int, default=1280,
                    help="드론 nadir view는 고해상도가 유리 (기본 640 → 1280)")
    ap.add_argument("--train-batch",  type=int, default=8)
    ap.add_argument("--model",        default="models/yolo11n.pt")
    args = ap.parse_args()

    work = args.work_dir
    labels_dir    = work / "labels"
    inference_dir = work / "inference"
    visual_dir    = work / "visualized"
    dataset_dir   = work / "dataset"
    output_dir    = work / "output"
    debug_dir     = work / "debug"


    # 1) Auto-label
    if "auto_label" in args.steps or "all" in args.steps:
        log.info("=" * 60)
        log.info("STEP 1: Auto-labeling (strategy=%s)", args.strategy)
        log.info("=" * 60)
        run_auto_label(
            images_dir=args.images,
            output_dir=labels_dir,
            strategy=args.strategy,
            class_id=0,
            debug_dir=debug_dir if args.strategy == "heuristic" else None,
        )

    # 2) Verifying labels by Claude Vision (OPTIONAL)
    if "verify" in args.steps or "all" in args.steps:
        log.info("=" * 60)
        log.info("OPTION: Verifying labels by Claude Vision")
        log.info("=" * 60)
        verify_labels(
            images_dir=args.images,
            labels_dir=labels_dir,
            output_dir=dataset_dir,
            confidence_threshold=args.conf,
            update_class_on_defect=not args.no_relabel,
            dry_run=args.dry_run,
        )
        log.info("검수용 이미지: %s", dataset_dir)

    # 3) Visualize → 수동 검수
    if "visualize" in args.steps or "all" in args.steps:
        log.info("=" * 60)
        log.info("STEP 2: Visualizing labels for review")
        log.info("=" * 60)
        visualize(args.images, labels_dir, visual_dir, args.classes)
        log.info("검수용 이미지: %s", visual_dir)
        log.info("→ 잘못된 bbox는 Label Studio에서 수동 수정을 권장합니다.")

    # 4) Report dataset quality
    if "report" in args.steps or "all" in args.steps:
        log.info("=" * 60)
        log.info("STEP 4: Report dataset quality")
        log.info("=" * 60)
        generate_report(
            images_dir=args.images,
            labels_dir=labels_dir,
            output_dir=output_dir,
            class_names=args.classes,
        )
        log.info("리포트 경로: %s", visual_dir)

    # 5) Split dataset
    yaml_path = None
    if "split" in args.steps or "all" in args.steps:
        log.info("=" * 60)
        log.info("STEP 3: Splitting dataset")
        log.info("=" * 60)
        yaml_path = split_dataset(
            images_dir=args.images,
            labels_dir=labels_dir,
            output_dir=dataset_dir,
            classes=args.classes,
            split=SplitConfig(*args.split),
        )
        log.info("학습데이터 분할: %s", yaml_path)

    # 6) Train YOLO11n (optional)
    if "train" in args.steps or "all" in args.steps:
        if not yaml_path:
            return

        log.info("=" * 60)
        log.info("STEP 6: Training YOLO11n")
        log.info("=" * 60)

        model = YOLO(args.model)
        model.train(
            data=str(yaml_path),
            epochs=args.train_epochs,
            imgsz=args.train_imgsz,
            batch=args.train_batch,
            project=str(work / "runs"),
            name="solar_panel_yolo11n",
            patience=20,
            # 드론 이미지용 augmentation 튜닝
            degrees=10.0,     # 회전 (항공 촬영 각도 변동)
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.5,       # nadir view는 수직 flip도 유효
            mosaic=1.0,
            mixup=0.1,
        )
    elif "inference" not in args.steps:
        log.info("학습 스킵. 수동 학습 명령어:")
        log.info(
            "  yolo detect train data=%s model=%s epochs=%d imgsz=%d batch=%d",
            yaml_path, args.model, args.train_epochs,
            args.train_imgsz, args.train_batch,
        )
        return

    # 7) Infer from custom YOLO11n (optional)
    if "inference" in args.steps or "all" in args.steps:
        log.info("=" * 60)
        log.info("STEP 7: Infer from custom YOLO11n (optional)")
        log.info("=" * 60)

        run_auto_label(
            images_dir=args.images,
            output_dir=inference_dir,
            strategy=args.strategy,
            class_id=0,
            debug_dir=debug_dir if args.strategy == "heuristic" else None,
        )


if __name__ == "__main__":
    main()
