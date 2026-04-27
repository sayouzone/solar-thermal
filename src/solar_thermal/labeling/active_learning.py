"""
PV Self-Training Active Learning Loop
======================================

현실적인 워크플로우: 수동 라벨링은 최소화하고 모델이 점점 개선되도록.

루프:
    Round 0:
      - 소수(5-20장) 이미지를 Label Studio에서 수동 라벨링
      - YOLO11n 초기 학습 (과적합 OK, 부트스트랩)

    Round 1+:
      1) 현재 모델로 unlabeled 이미지 예측
      2) Uncertainty scoring (low-confidence, disagreement, IoU 이상)
      3) 가장 informative 한 샘플 N개 선별 → human-in-the-loop
      4) 수정된 라벨 추가 → 재학습
      5) 품질 메트릭 (mAP) 수렴할 때까지 반복

이 스크립트는 3가지 모드로 동작:

    seed       : 초기 수동 라벨 → YOLO 학습 가능 상태
    predict    : 현재 모델로 unlabeled 이미지에 예측 라벨 생성
    select     : 예측 중 high-value 샘플을 human 검수 대기열로 이동
    iterate    : 전체 루프 자동 실행

사용 예:
    # Round 0: 5장 수동 라벨링 후 초기 학습
    python active_learning.py seed \
        --images data/unlabeled \
        --seed-labels data/manual_seed \
        --output workspace/round_0

    # Round 1+: 자동 라벨 → 샘플 선별
    python active_learning.py iterate \\
        --images data/unlabeled \\
        --model workspace/round_0/weights/best.pt \\
        --output workspace/round_1 \\
        --select-top 20
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import random
import time
import yaml
from collections import defaultdict
from datetime import timedelta
from statistics import mean

#from ultralytics import YOLO
try:
    from ultralytics import YOLO
except ImportError:
    log.warning(
        "ultralytics 미설치"
    )

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class UncertaintySample:
    """예측 결과 + uncertainty score."""
    image_path: Path
    predictions: list[tuple[int, float, float, float, float, float]]
    # (class_id, cx, cy, w, h, confidence)
    score: float = 0.0          # 높을수록 human 검수 우선순위 높음
    score_breakdown: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Uncertainty scoring
# ---------------------------------------------------------------------------

class UncertaintyScorer:
    """
    예측 결과의 uncertainty를 평가하여 sample selection에 사용.

    고려 요소 (weighted sum):
      1) Low confidence: 평균 box confidence가 낮을수록 uncertain
      2) Prediction count anomaly: 너무 많거나 너무 적은 예측
      3) Overlap anomaly: 겹치는 box가 많으면 confused
      4) Small boxes: 작은 box는 놓치기 쉬움 (이미 잡힌 것도 불확실)

    최고점이 "가장 학습에 도움될 샘플"이 되도록 설계.
    """

    def __init__(
        self,
        expected_count_range: tuple[int, int] = (3, 20),
        conf_weight: float = 0.4,
        count_weight: float = 0.2,
        overlap_weight: float = 0.2,
        size_weight: float = 0.2,
    ):
        self.expected_count_range = expected_count_range
        self.conf_weight = conf_weight
        self.count_weight = count_weight
        self.overlap_weight = overlap_weight
        self.size_weight = size_weight

    def score(
        self, predictions: list[tuple[int, float, float, float, float, float]],
    ) -> tuple[float, dict]:
        """
        Returns:
            (total_score [0~1], breakdown dict)
        """
        if not predictions:
            # 빈 예측 = 매우 uncertain (모델이 아무것도 못 찾음)
            return 1.0, {"reason": "empty_prediction"}

        # 1) 평균 confidence 낮을수록 점수 ↑
        confs = [p[5] for p in predictions]
        avg_conf = float(np.mean(confs))
        conf_score = 1.0 - avg_conf

        # 2) 예측 개수 이상
        n = len(predictions)
        lo, hi = self.expected_count_range
        if n < lo:
            count_score = (lo - n) / max(lo, 1)
        elif n > hi:
            count_score = min(1.0, (n - hi) / max(hi, 1))
        else:
            count_score = 0.0

        # 3) Overlap ratio: IoU > 0.3인 pair 비율
        overlap_score = self._compute_overlap_score(predictions)

        # 4) Small box 비율: 너무 작은 bbox가 많으면 불안정
        sizes = [p[3] * p[4] for p in predictions]  # 정규화 area
        small_ratio = sum(1 for s in sizes if s < 0.002) / max(len(sizes), 1)
        size_score = small_ratio

        total = (
            self.conf_weight * conf_score
            + self.count_weight * count_score
            + self.overlap_weight * overlap_score
            + self.size_weight * size_score
        )
        breakdown = {
            "avg_conf": avg_conf,
            "n_predictions": n,
            "conf_score": conf_score,
            "count_score": count_score,
            "overlap_score": overlap_score,
            "size_score": size_score,
            "total": total,
        }
        return min(1.0, max(0.0, total)), breakdown

    @staticmethod
    def _compute_overlap_score(
        predictions: list[tuple[int, float, float, float, float, float]],
    ) -> float:
        """예측 box 간 평균 IoU overlap score."""
        if len(predictions) < 2:
            return 0.0

        # 정규화 YOLO → (x1, y1, x2, y2)
        def to_xyxy(p):
            _, cx, cy, w, h, _ = p
            return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)

        boxes = [to_xyxy(p) for p in predictions]
        n_overlap = 0
        total_pairs = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                total_pairs += 1
                if _iou_xyxy(boxes[i], boxes[j]) > 0.3:
                    n_overlap += 1
        return n_overlap / max(total_pairs, 1)


def _iou_xyxy(a, b) -> float:
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def aspect_ratio(p: Prediction) -> float:
    """긴 변 / 짧은 변. 1에 가까우면 정사각형, 클수록 길쭉."""
    w, h = p[3], p[4]
    if w == 0 or h == 0:
        return 0.0
    return max(w, h) / min(w, h)


def iou(a: Prediction, b: Prediction) -> float:
    _, acx, acy, aw, ah, _ = a
    _, bcx, bcy, bw, bh, _ = b
    ax1, ay1 = acx - aw / 2, acy - ah / 2
    ax2, ay2 = acx + aw / 2, acy + ah / 2
    bx1, by1 = bcx - bw / 2, bcy - bh / 2
    bx2, by2 = bcx + bw / 2, bcy + bh / 2
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


def dedup_by_aspect(
    predictions: list[Prediction],
    iou_threshold: float = 0.4,
    aspect_low: float = 3.0,
    aspect_high: float = 4.0,
    string_class_id: int = 0,
    module_class_id: int = 1,
    middle_prefer: int | None = 1,
    verbose: bool = False,
) -> list[Prediction]:
    """
    충돌하는 prediction을 aspect ratio로 분류하여 중복 제거.

    Args:
        predictions: (class_id, cx, cy, w, h, confidence) 튜플 리스트
        iou_threshold: 충돌로 간주할 IoU 임계값
        aspect_low:  이 값 미만이면 module 우선 (string 박스 제거)
        aspect_high: 이 값 초과이면 string 우선 (module 박스 제거)
        middle_prefer: 중간 영역에서 우선할 class_id (None이면 모두 유지)

    Returns:
        정리된 prediction 리스트 (원본 순서 유지)

    Tie-breaking:
        선호 클래스가 그룹에 여러 개면 confidence 가장 높은 1개만 유지.
    """
    if not predictions:
        return []

    n = len(predictions)

    # IoU Union-Find
    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if iou(predictions[i], predictions[j]) >= iou_threshold:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    keep: set[int] = set()

    for indices in groups.values():
        if len(indices) == 1:
            keep.add(indices[0])
            continue

        group_aspect = mean(aspect_ratio(predictions[i]) for i in indices)

        if group_aspect < aspect_low:
            preferred = module_class_id
            decision = f"aspect={group_aspect:.2f}<{aspect_low} → module"
        elif group_aspect > aspect_high:
            preferred = string_class_id
            decision = f"aspect={group_aspect:.2f}>{aspect_high} → string"
        else:
            preferred = middle_prefer
            label = "유지" if middle_prefer is None else f"class={middle_prefer}"
            decision = f"aspect={group_aspect:.2f} (중간) → {label}"

        if verbose:
            print(f"  그룹 [{','.join(str(i) for i in indices)}]: {decision}")

        if preferred is None:
            keep.update(indices)
            continue

        # 선호 클래스 중 confidence 최고 1개만
        preferred_indices = [
            i for i in indices if predictions[i][0] == preferred
        ]
        if preferred_indices:
            best = max(preferred_indices, key=lambda i: predictions[i][5])
            keep.add(best)
        else:
            best = max(indices, key=lambda i: predictions[i][5])
            keep.add(best)

    return [predictions[i] for i in range(n) if i in keep]

# ---------------------------------------------------------------------------
# Workflow commands
# ---------------------------------------------------------------------------

def cmd_seed(
    images_dir: Path,
    seed_labels_dir: Path,
    output_dir: Path,
    classes: list[str],
    val_ratio: float = 0.2,
    epochs: int = 50,
    imgsz: int = 1280,
    batch: int = 4,
    model: str = "yolo11n.pt",
    device: str = "mps",
    amp: bool = False,     # ← 추가
) -> Path:
    """
    초기 manual label로 YOLO 부트스트랩.

    Args:
        seed_labels_dir: 수동 라벨링 완료된 YOLO .txt 파일 디렉토리
                         이 디렉토리의 .txt와 같은 이름의 이미지만 학습에 사용.
    """
    # 시작 시간
    start = time.perf_counter()
        
    # 1) seed label이 있는 이미지만 필터
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    seed_stems = {p.stem for p in seed_labels_dir.iterdir() if p.suffix == ".txt"}
    if not seed_stems:
        raise RuntimeError(f"{seed_labels_dir}에 seed 라벨(.txt)이 없습니다.")

    labeled_images = [
        p for p in images_dir.iterdir()
        if p.suffix in exts and p.stem in seed_stems
    ]
    log.info("Seed 이미지 %d장 (수동 라벨 완료)", len(labeled_images))

    # 2) train/val 분할
    rng = random.Random(42)
    rng.shuffle(labeled_images)
    n_val = max(1, int(len(labeled_images) * val_ratio))
    val_imgs = labeled_images[:n_val]
    train_imgs = labeled_images[n_val:]

    dataset_dir = output_dir / "dataset"
    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_out = dataset_dir / "images" / split
        lbl_out = dataset_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, img_out / img.name)
            label_file = seed_labels_dir / f"{img.stem}.txt"
            shutil.copy2(label_file, lbl_out / label_file.name)
        log.info("  %s: %d장", split, len(imgs))

    # 3) data.yaml
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path":  str(dataset_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "names": {i: c for i, c in enumerate(classes)},
        "nc":    len(classes),
    }, allow_unicode=True), encoding="utf-8")

    # 4) 학습
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log.warning(
            "ultralytics 미설치. 수동 학습 명령:\n"
            "  yolo detect train data=%s model=%s epochs=%d imgsz=%d batch=%d",
            yaml_path, model, epochs, imgsz, batch,
        )
        return yaml_path
    """

    yolo = YOLO(model)
    yolo.train(
        data=str(yaml_path),
        device=device,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(output_dir),
        amp=amp,
        name="weights",
        patience=15,
        # Seed 단계: overfitting 허용 (데이터가 적으므로 강한 augmentation 자제)
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.5,
        mixup=0.0,
    )

    # 경과 시간
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {timedelta(seconds=int(elapsed))}")

    return yaml_path


def cmd_predict(
    images_dir: Path,
    model_path: Path,
    output_dir: Path,
    device: str = "mps",
    conf: float = 0.2,
    imgsz: int = 1280,
    apply_aspect_dedup: bool = True,   # 종복 제거 여부
) -> list[UncertaintySample]:
    """현재 모델로 unlabeled 이미지 예측."""
    yolo = YOLO(str(model_path))
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [p for p in sorted(images_dir.iterdir()) if p.suffix in exts]

    labels_dir = output_dir / "predicted_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    scorer = UncertaintyScorer()
    samples: list[UncertaintySample] = []

    log.info("예측 대상: %d장", len(images))
    for img_path in images:
        results = yolo.predict(
            str(img_path), conf=conf, imgsz=imgsz, verbose=False,
            device=device,
        )

        predictions: list = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            img_h, img_w = results[0].orig_shape
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            cnfs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, cf in zip(xyxy, clss, cnfs):
                cx = ((x1 + x2) / 2.0) / img_w
                cy = ((y1 + y2) / 2.0) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                predictions.append((int(c), cx, cy, w, h, float(cf)))
        
        # 추가: aspect 기반 중복 제거
        if apply_aspect_dedup and predictions:
            before = len(predictions)
            predictions = dedup_by_aspect(
                predictions,
                iou_threshold=0.4,
                aspect_low=3.0,
                aspect_high=4.0,
                middle_prefer=1,  # 모듈 우선
            )
            if len(predictions) < before:
                log.info(
                    "  %s: %d → %d (aspect dedup)",
                    img_path.name, before, len(predictions),
                )
        
        # x좌표 기준 오름차순 정렬
        predictions = sorted(predictions, key=lambda x: x[1])
        #print(predictions, type(predictions))

        # Write YOLO label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        label_path.write_text(
            "\n".join(
                f"{p[0]} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f} {p[4]:.6f}"
                for p in predictions
            ) + ("\n" if predictions else "")
        )

        # Uncertainty score
        score, breakdown = scorer.score(predictions)
        samples.append(UncertaintySample(
            image_path=img_path,
            predictions=predictions,
            score=score,
            score_breakdown=breakdown,
        ))

    # Save uncertainty report
    report = [
        {
            "image": s.image_path.name,
            "score": s.score,
            "n_predictions": len(s.predictions),
            "breakdown": s.score_breakdown,
        }
        for s in sorted(samples, key=lambda s: s.score, reverse=True)
    ]
    (output_dir / "uncertainty_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Uncertainty 리포트: %s", output_dir / "uncertainty_report.json")
    return samples


def cmd_select(
    samples: list[UncertaintySample],
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """
    Top-N uncertain 샘플을 human review 큐로 복사.
    Label Studio로 import하여 수동 수정.
    """
    review_dir = output_dir / "human_review"
    review_imgs = review_dir / "images"
    review_lbls = review_dir / "labels_predicted"  # 예측 라벨 (수정 대상)
    review_imgs.mkdir(parents=True, exist_ok=True)
    review_lbls.mkdir(parents=True, exist_ok=True)

    # Top-N (uncertainty 높은 순)
    top = sorted(samples, key=lambda s: s.score, reverse=True)[:top_n]
    #print(top, type(top))

    for s in top:
        shutil.copy2(s.image_path, review_imgs / s.image_path.name)
        # Find predicted label
        pred_label = (
            output_dir / "predicted_labels" / f"{s.image_path.stem}.txt"
        )
        if pred_label.exists():
            shutil.copy2(pred_label, review_lbls / pred_label.name)

    log.info(
        "Human review 큐: %s (%d장, score %.2f~%.2f)",
        review_dir, len(top),
        top[-1].score if top else 0, top[0].score if top else 0,
    )
    log.info(
        "다음 단계: Label Studio에서 수정 → labels_corrected/로 저장 후 "
        "다음 round seed로 사용",
    )


def cmd_iterate(
    images_dir: Path,
    model_path: Path,
    output_dir: Path,
    classes: list[str],
    device: str = "cpu",
    select_top: int = 20,
    conf: float = 0.2,
    imgsz: int = 1280,
) -> None:
    """Round 1+ 자동 실행: predict → select."""
    # 시작 시간
    start = time.perf_counter()
    
    samples = cmd_predict(images_dir, model_path, output_dir, device, conf, imgsz)
    cmd_select(samples, output_dir, top_n=select_top)
    
    # 경과 시간
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {timedelta(seconds=int(elapsed))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    p.add_argument("--device",      type=str, default="cpu")
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
        )


if __name__ == "__main__":
    main()
