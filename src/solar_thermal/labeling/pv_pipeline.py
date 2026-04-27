"""
PV Hierarchical Pipeline Orchestrator (v2)
===========================================

사용자 답변을 기다리지 않고 합리적 기본값으로 실행 가능한
end-to-end runner. 모드를 바꿔가며 워크플로 전체를 커버.

모드:
    bootstrap   : SAM2로 부트스트랩 라벨 생성 (human review 대기열 생성)
                  Label Studio 없이 빠르게 시작하고 싶을 때
    train       : 수동/자동 라벨로 YOLO11n 학습
    predict     : 학습된 모델로 전체 데이터셋 예측 (SAHI 타일링 포함)
    full-cycle  : bootstrap → train → predict → clean → visualize 전체 실행

사용 예:
    # 처음 시작: SAM2로 5장 부트스트랩, 검수 후 학습
    python pv_pipeline.py bootstrap \\
        --images /mnt/user-data/uploads \\
        --output workspace/v2 \\
        --n-bootstrap 5

    # 전체 자동 (데모)
    python pv_pipeline.py full-cycle \\
        --images /mnt/user-data/uploads \\
        --output workspace/v2
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Local imports (작성한 모듈들)
from pv_detector import (
    HierarchicalPVDetector, PVStringDetector, PVModuleSplitter,
)
from active_learning import cmd_seed, cmd_predict, cmd_select
from sahi_inference import SAHIInference
from hierarchical_cleanup import (
    HierarchicalConsistencyEnforcer, clean_all, visualize_hierarchical,
)


CLASSES = ["pv_string", "pv_module"]


# ---------------------------------------------------------------------------
# Bootstrap: SAM2 기반 초기 라벨 생성 + human review 준비
# ---------------------------------------------------------------------------

def bootstrap(
    images_dir: Path,
    output_dir: Path,
    n_bootstrap: int = 5,
    detect_modules: bool = True,
) -> Path:
    """
    다양성을 고려해 N장 샘플링 → SAM2 + Module splitter로 초기 라벨 생성.

    이 라벨들은 Label Studio로 import 되어 human review 후 seed가 됨.
    """
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    all_images = [p for p in sorted(images_dir.iterdir()) if p.suffix in exts]
    if not all_images:
        raise RuntimeError(f"{images_dir}에 이미지 없음")

    # Random sampling (다양성 확보용 단순 구현)
    rng = random.Random(42)
    sample = rng.sample(all_images, min(n_bootstrap, len(all_images)))
    log.info("Bootstrap 샘플: %d장", len(sample))

    # Output 디렉토리 구조
    bootstrap_dir = output_dir / "bootstrap"
    bootstrap_images = bootstrap_dir / "images"
    bootstrap_labels = bootstrap_dir / "labels_auto"
    review_images = bootstrap_dir / "human_review_images"
    bootstrap_images.mkdir(parents=True, exist_ok=True)
    bootstrap_labels.mkdir(parents=True, exist_ok=True)
    review_images.mkdir(parents=True, exist_ok=True)

    # SAM2 detector (lazy import)
    try:
        detector = HierarchicalPVDetector(
            string_detector=PVStringDetector(string_class_id=0),
            module_splitter=PVModuleSplitter(module_class_id=1),
            detect_modules=detect_modules,
        )
    except ImportError:
        log.error(
            "SAM2 사용 불가 (ultralytics 미설치). "
            "수동 라벨링만 진행: LABEL_STUDIO_GUIDE.md 참고"
        )
        # 이미지만 복사 (라벨은 수동으로 만들 것)
        for img in sample:
            shutil.copy2(img, bootstrap_images / img.name)
            shutil.copy2(img, review_images / img.name)
        return bootstrap_dir

    # 각 샘플 처리
    class_map = {"pv_string": 0, "pv_module": 1}
    for img_path in sample:
        log.info("  처리 중: %s", img_path.name)
        result = detector.detect(img_path)

        # 이미지 복사
        shutil.copy2(img_path, bootstrap_images / img_path.name)
        shutil.copy2(img_path, review_images / img_path.name)

        # 라벨 저장 (YOLO 형식)
        lines = result.to_yolo_lines(class_map=class_map)
        label_path = bootstrap_labels / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        log.info(
            "    → %d strings, %d modules",
            len(result.strings), len(result.modules),
        )

    log.info("\n=== Bootstrap 완료 ===")
    log.info("자동 라벨: %s", bootstrap_labels)
    log.info("이미지:     %s", bootstrap_images)
    log.info("\n다음 단계:")
    log.info("  1. Label Studio에서 %s 불러오기", review_images)
    log.info("  2. %s 의 YOLO 라벨을 pre-annotation으로 import", bootstrap_labels)
    log.info("  3. 수정 후 export → workspace/seed_labels/")
    log.info("  4. python pv_pipeline.py train --seed-labels workspace/seed_labels")
    return bootstrap_dir


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    images_dir: Path,
    seed_labels_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    imgsz: int = 1920,
    batch: int = 4,
    model: str = "yolo11n.pt",
) -> Path:
    """초기 manual seed → YOLO11n 부트스트랩."""
    cmd_seed(
        images_dir=images_dir,
        seed_labels_dir=seed_labels_dir,
        output_dir=output_dir / "train_round",
        classes=CLASSES,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        model=model,
    )
    weights = output_dir / "train_round" / "weights" / "best.pt"
    log.info("학습 완료: %s", weights)
    return weights


# ---------------------------------------------------------------------------
# Predict (SAHI + cleanup)
# ---------------------------------------------------------------------------

def predict(
    images_dir: Path,
    model_path: Path,
    output_dir: Path,
    use_sahi: bool = True,
    tile_size: int = 1024,
    overlap: float = 0.2,
    conf: float = 0.2,
    imgsz: int = 1920,
) -> Path:
    """학습된 모델로 전체 예측. SAHI 타일링 + 계층 일관성 정리."""
    raw_labels = output_dir / "predict" / "labels_raw"
    clean_labels = output_dir / "predict" / "labels_clean"
    visualized = output_dir / "predict" / "visualized"

    # 1) SAHI 타일링 추론 (module 단위는 필수)
    if use_sahi:
        log.info("SAHI 타일링 추론 시작 (tile=%d, overlap=%.1f)",
                 tile_size, overlap)
        inferer = SAHIInference(
            model_path=model_path,
            tile_size=tile_size,
            overlap_ratio=overlap,
            conf=conf,
        )
        inferer.run_batch(images_dir, raw_labels)
    else:
        log.info("일반 추론 (no SAHI, imgsz=%d)", imgsz)
        cmd_predict(images_dir, model_path, output_dir / "predict",
                    conf=conf, imgsz=imgsz)
        raw_labels = output_dir / "predict" / "predicted_labels"

    # 2) 계층 일관성 정리
    log.info("\n계층 일관성 정리...")
    enforcer = HierarchicalConsistencyEnforcer(
        string_class_id=0, module_class_id=1,
    )
    clean_all(raw_labels, clean_labels, enforcer)

    # 3) 시각화
    log.info("\n시각화 생성...")
    visualize_hierarchical(images_dir, clean_labels, visualized)
    log.info("시각화: %s", visualized)
    return clean_labels


# ---------------------------------------------------------------------------
# Full cycle demo
# ---------------------------------------------------------------------------

def full_cycle(
    images_dir: Path,
    output_dir: Path,
    n_bootstrap: int = 5,
) -> None:
    """
    End-to-end 데모. 실제 프로덕션에서는 bootstrap 후 human review 단계가 필요.
    여기서는 자동 라벨을 그대로 seed로 사용하여 전체 흐름만 보여줌.
    """
    log.info("=" * 60)
    log.info("STEP 1: Bootstrap with SAM2")
    log.info("=" * 60)
    bootstrap_dir = bootstrap(images_dir, output_dir, n_bootstrap=n_bootstrap)

    log.info("\n" + "=" * 60)
    log.info("STEP 2: Train YOLO11n on auto-labeled seed")
    log.info("=" * 60)
    log.warning(
        "⚠️  실제 사용 시: bootstrap 후 human review 권장 "
        "(LABEL_STUDIO_GUIDE.md 참고)"
    )
    weights = train(
        images_dir=bootstrap_dir / "images",
        seed_labels_dir=bootstrap_dir / "labels_auto",
        output_dir=output_dir,
    )

    log.info("\n" + "=" * 60)
    log.info("STEP 3: Predict on full dataset with SAHI")
    log.info("=" * 60)
    predict(
        images_dir=images_dir,
        model_path=weights,
        output_dir=output_dir,
    )

    log.info("\n🎉 Full cycle 완료: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bootstrap", help="SAM2로 초기 라벨 생성")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-bootstrap", type=int, default=5)
    p.add_argument("--no-modules", action="store_true",
                   help="string만 검출 (module splitting 스킵)")

    p = sub.add_parser("train", help="Seed 라벨로 YOLO11n 학습")
    p.add_argument("--images",      type=Path, required=True)
    p.add_argument("--seed-labels", type=Path, required=True)
    p.add_argument("--output",      type=Path, required=True)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--imgsz",       type=int, default=1920)
    p.add_argument("--batch",       type=int, default=4)
    p.add_argument("--model",       default="yolo11n.pt")

    p = sub.add_parser("predict", help="SAHI 타일링 추론 + 정리")
    p.add_argument("--images",    type=Path, required=True)
    p.add_argument("--model",     type=Path, required=True)
    p.add_argument("--output",    type=Path, required=True)
    p.add_argument("--no-sahi",   action="store_true")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--overlap",   type=float, default=0.2)
    p.add_argument("--conf",      type=float, default=0.2)
    p.add_argument("--imgsz",     type=int, default=1920)

    p = sub.add_parser("full-cycle", help="bootstrap → train → predict 일괄")
    p.add_argument("--images",       type=Path, required=True)
    p.add_argument("--output",       type=Path, required=True)
    p.add_argument("--n-bootstrap",  type=int, default=5)

    args = ap.parse_args()

    if args.cmd == "bootstrap":
        bootstrap(
            args.images, args.output,
            n_bootstrap=args.n_bootstrap,
            detect_modules=not args.no_modules,
        )
    elif args.cmd == "train":
        train(
            args.images, args.seed_labels, args.output,
            epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, model=args.model,
        )
    elif args.cmd == "predict":
        predict(
            args.images, args.model, args.output,
            use_sahi=not args.no_sahi,
            tile_size=args.tile_size, overlap=args.overlap,
            conf=args.conf, imgsz=args.imgsz,
        )
    elif args.cmd == "full-cycle":
        full_cycle(args.images, args.output, n_bootstrap=args.n_bootstrap)


if __name__ == "__main__":
    main()
