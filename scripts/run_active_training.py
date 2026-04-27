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
    # Round 0: 5장 수동 라벨링 후 초기 학습
    python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --seed-labels ./workspace/labels_seed \
        --output ./workspace/round_0

    # Round 1+: 자동 라벨 → 샘플 선별
    python scripts/run_active_training.py iterate \
        --images data/solar/images/RGB \
        --model ./runs/detect/workspace/round_0/weights/weights/best.pt \
        --output ./workspace/round_1 \
        --select-top 20

    # Round 2: 100장 라벨링 보정 후 학습
    python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --seed-labels ./workspace/labels_seed_r2 \
        --output ./workspace/round_2


    # Round 3+: 자동 라벨 → 샘플 선별
    python scripts/run_active_training.py iterate \
        --images data/solar/images/RGB \
        --model ./runs/detect/workspace/round_2/weights/weights/best.pt \
        --output ./workspace/round_3 \
        --select-top 20


    # Round 2: 100장 라벨링 보정 후 학습
    python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --device mps \
        --seed-labels ./workspace/labels_seed_r2 \
        --output ./workspace/round_2

    # Round 3+: 자동 라벨 → 샘플 선별
    python scripts/run_active_training.py iterate \
        --images data/solar/images/RGB \
        --model ./runs/detect/workspace/round_2/weights5/weights/best.pt \
        --output ./workspace/round_3 \
        --select-top 20

    # Round 3+: 자동 라벨 → 샘플 선별
    python scripts/run_active_training.py iterate \
        --images data/solar/images/RGB \
        --model ./runs/detect/workspace/round_2/weights5/weights/best.pt \
        --device mps \
        --output ./workspace/round_3 \
        --select-top 20
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

from solar_thermal.labeling.active_learning import cmd_seed, cmd_iterate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # seed
    p = sub.add_parser("seed", help="수동 seed 라벨로 초기 YOLO 학습")
    p.add_argument("--images",      type=Path, required=True)
    p.add_argument("--seed-labels", type=Path, required=True)
    p.add_argument("--output",      type=Path, required=True)
    p.add_argument("--classes",     nargs="+", default=["pv_string", "pv_module", "other"])
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--imgsz",       type=int, default=1280)
    p.add_argument("--batch",       type=int, default=4)
    p.add_argument("--model",       default="models/yolo11n.pt")
    p.add_argument("--device",      default="mps", choices=["cpu", "gpu", "mps"])
    p.add_argument("--val-ratio",   type=float, default=0.2)

    # predict
    p = sub.add_parser("predict", help="현재 모델로 예측 + uncertainty 점수")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--model",  type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conf",   type=float, default=0.2)
    p.add_argument("--imgsz",  type=int, default=1280)

    # select
    p = sub.add_parser("select", help="uncertainty 높은 샘플 human review 큐로")
    p.add_argument("--output",     type=Path, required=True,
                   help="predict 단계와 동일한 output 디렉토리")
    p.add_argument("--top-n",      type=int, default=20)

    # iterate
    p = sub.add_parser("iterate", help="predict + select 자동 실행")
    p.add_argument("--images",     type=Path, required=True)
    p.add_argument("--model",      type=Path, required=True)
    p.add_argument("--device",     default="mps", choices=["cpu", "gpu", "mps"])
    p.add_argument("--output",     type=Path, required=True)
    p.add_argument("--classes",    nargs="+", default=["pv_string", "pv_module", "other"])
    p.add_argument("--select-top", type=int, default=20)
    p.add_argument("--conf",       type=float, default=0.2)
    p.add_argument("--imgsz",      type=int, default=1280)

    args = ap.parse_args()

    if args.cmd == "seed":
        cmd_seed(
            args.images, args.seed_labels, args.output, args.classes,
            val_ratio=args.val_ratio, epochs=args.epochs,
            imgsz=args.imgsz, batch=args.batch, model=args.model,
            device=args.device,
        )
    elif args.cmd == "predict":
        cmd_predict(args.images, args.model, args.output,
                    conf=args.conf, imgsz=args.imgsz)
    elif args.cmd == "select":
        # Re-load samples from saved report
        report_path = args.output / "uncertainty_report.json"
        if not report_path.exists():
            raise FileNotFoundError(
                f"{report_path} 없음. 먼저 `predict` 실행하세요."
            )
        report = json.loads(report_path.read_text())
        # Reconstruct UncertaintySample for select
        samples = [
            UncertaintySample(
                image_path=args.output.parent / "unlabeled" / r["image"],
                predictions=[], score=r["score"],
            )
            for r in report
        ]
        cmd_select(samples, args.output, top_n=args.top_n)
    elif args.cmd == "iterate":
        cmd_iterate(
            args.images, args.model, args.output,
            classes=args.classes, select_top=args.select_top,
            conf=args.conf, imgsz=args.imgsz,
            device=args.device,
        )


if __name__ == "__main__":
    main()
