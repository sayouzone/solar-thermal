"""
Solar Panel YOLO 데이터셋 부트스트랩 스크립트
============================================
한 번 실행하면:
  1) 표준 YOLO 데이터셋 디렉토리 생성
  2) 파일명 패턴 (DJI_YYYYMMDDHHMMSS_NNNN_X.JPG) 으로 flight 자동 추출
  3) flight 단위 train/val/test 분할 (데이터 누수 방지)
  4) data.yaml, 라벨링 가이드, 학습 스크립트 템플릿 동시 생성

사용:
    python bootstrap_yolo_dataset.py \
        --source /path/to/raw/DJI_photos \
        --out ./solar_panels_dataset \
        --train 0.7 --val 0.2 --test 0.1
"""

from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path
from collections import defaultdict
import random


FLIGHT_PATTERN = re.compile(r"DJI_(\d{8})(\d{6})_(\d+)_([A-Z]+)\.JPG", re.IGNORECASE)


def extract_flight_id(filename: str) -> str:
    """
    DJI_20251217130200_0001_Z.JPG  →  '20251217'
    같은 날짜 촬영 전체를 한 flight 로 간주. 하루에 여러 비행이 있으면
    시각(시간)으로 더 세분화 가능.
    """
    m = FLIGHT_PATTERN.match(filename)
    if not m:
        return "unknown"
    date_part = m.group(1)          # YYYYMMDD
    hour_part = m.group(2)[:2]      # HH — 시간 단위로 세분
    return f"{date_part}_{hour_part}h"


def split_flights(
    flights: list[str],
    train: float = 0.7,
    val: float = 0.2,
    test: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    """flight ID 리스트를 train/val/test 로 분할."""
    assert abs(train + val + test - 1.0) < 1e-6
    flights = sorted(flights)
    random.Random(seed).shuffle(flights)

    n = len(flights)
    n_train = max(1, int(n * train))
    n_val = max(1, int(n * val))
    # 나머지는 test
    assignment = {}
    for i, f in enumerate(flights):
        if i < n_train:
            assignment[f] = "train"
        elif i < n_train + n_val:
            assignment[f] = "val"
        else:
            assignment[f] = "test"
    return assignment


def create_structure(out_dir: Path):
    """YOLO 표준 구조 생성."""
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


DATA_YAML = """# YOLO 데이터셋 정의
path: {abs_path}
train: images/train
val: images/val
test: images/test

names:
  0: solar_panel

# 선택: 계층 확장 시
# names:
#   0: panel_module
#   1: panel_string
#   2: panel_array
"""

LABELING_GUIDE = """# Solar Panel 라벨링 가이드 (v1)

## bbox 작성 규칙

1. **경계**: 알루미늄 프레임 바깥선까지 포함. 마운트/가대 구조물은 제외.
2. **개별 패널 vs 스트링**: 첫 라운드는 **스트링(한 열)** 단위로 라벨.
   - 패널 간 간격 < 0.5m: 한 bbox
   - 패널 간 간격 > 1.0m: 분리
3. **그림자**: bbox 에 포함하지 말 것.
4. **반사/글레어**: 형태가 명확하면 라벨, 셀이 전혀 안 보여도 윤곽이 있으면 라벨.
5. **가림 (Occlusion)**:
   - 0~30% 가림 → 정상 라벨
   - 30~70% → `difficult=1` 속성 표시 (가능한 도구일 때)
   - 70% 이상 → 라벨 생략
6. **프레임 경계 잘림**:
   - 50% 이상 잘림 → 라벨
   - 10~50% 잘림 → 라벨 (잘린 채로 bbox 지정)
7. **모니터링용 소형 패널**: 동일 클래스로 라벨. 스케일 편차 학습에 필수.
8. **방초시트·아스팔트**: 색이 비슷해도 절대 패널로 라벨하지 말 것.

## 의심 케이스 처리

- 확신 < 80% → 라벨하지 말고 별도 폴더 이동, 2차 검토자에 패스.
- 같은 프레임에 2인 이상 라벨 시, IoU < 0.8 이면 재논의.

## 품질 기준

- 한 라벨러가 1시간에 평균 30~50장 라벨링하면 정상 속도.
- 1시간에 100장 이상이면 bbox 가 느슨할 가능성 높음 — 샘플링 검토.
"""

TRAIN_SCRIPT = """# YOLO11n 학습 스크립트
# 실행: python train.py
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=150,
        patience=25,
        imgsz=1280,
        batch=16,
        device=0,
        cache="ram",

        # 증강 - 항공 뷰 특화
        degrees=180,
        flipud=0.5, fliplr=0.5,
        scale=0.6, translate=0.15,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        mosaic=1.0, mixup=0.1, copy_paste=0.3,
        perspective=0.0001,

        # 저장/로깅
        project="runs/solar",
        name="yolo11n_v1_1280",
        exist_ok=True,
        plots=True,
        save_period=10,
    )

    # test 셋 평가
    metrics = model.val(data="data.yaml", split="test", imgsz=1280)
    print(f"Test mAP50: {metrics.box.map50:.4f}")
    print(f"Test mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
"""

EXPORT_SCRIPT = """# Jetson Orin NX 배포용 TensorRT FP16 엔진 export
# Orin 에서 직접 실행 (x86 에서 export 하면 GPU 호환 안 됨)
from ultralytics import YOLO

model = YOLO("runs/solar/yolo11n_v1_1280/weights/best.pt")

# TensorRT FP16
model.export(
    format="engine",
    half=True,
    imgsz=1280,
    device=0,
    workspace=4,   # GB, Orin NX 16GB 기준
    simplify=True,
)
# 출력: best.engine
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="원본 DJI JPG 들이 있는 폴더")
    parser.add_argument("--out", required=True, help="데이터셋 출력 루트")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true",
                        help="심볼릭 링크 대신 복사 (Windows 사용자)")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) 파일 수집 및 flight 분류
    image_files = sorted(list(source.rglob("*.JPG")) + list(source.rglob("*.jpg")))
    if not image_files:
        print(f"[ERROR] {source} 에서 JPG 파일을 찾지 못했습니다.")
        return

    by_flight: dict[str, list[Path]] = defaultdict(list)
    for f in image_files:
        fid = extract_flight_id(f.name)
        by_flight[fid].append(f)

    print(f"[INFO] 전체 이미지: {len(image_files)}")
    print(f"[INFO] 식별된 비행(flight): {len(by_flight)}")
    for fid, files in sorted(by_flight.items()):
        print(f"  {fid}: {len(files)} 장")

    # 2) flight 단위 분할
    assignment = split_flights(
        list(by_flight.keys()),
        args.train, args.val, args.test, args.seed
    )
    print(f"\n[INFO] 분할 결과:")
    split_counts = defaultdict(int)
    for fid, split in assignment.items():
        split_counts[split] += len(by_flight[fid])
        print(f"  {fid} → {split}")
    print(f"\n  train: {split_counts['train']}장")
    print(f"  val:   {split_counts['val']}장")
    print(f"  test:  {split_counts['test']}장")

    # 3) 디렉토리 구조 생성
    create_structure(out)

    # 4) 파일 배치 (심볼릭 링크 기본, 또는 복사)
    for fid, files in by_flight.items():
        split = assignment[fid]
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        for src in files:
            dst = img_dir / src.name
            if dst.exists():
                continue
            if args.copy:
                shutil.copy2(src, dst)
            else:
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dst)
            # 빈 라벨 파일 (라벨 아직 없음을 명시)
            (lbl_dir / (src.stem + ".txt")).touch(exist_ok=True)

    # 5) data.yaml
    (out / "data.yaml").write_text(
        DATA_YAML.format(abs_path=out.resolve()), encoding="utf-8"
    )
    # 6) 라벨링 가이드
    (out / "LABELING_GUIDE.md").write_text(LABELING_GUIDE, encoding="utf-8")
    # 7) 학습 스크립트
    (out / "train.py").write_text(TRAIN_SCRIPT, encoding="utf-8")
    # 8) export 스크립트
    (out / "export_engine.py").write_text(EXPORT_SCRIPT, encoding="utf-8")

    # 9) 분할 로그
    (out / "split_log.txt").write_text(
        "\n".join(f"{fid}\t{split}\t{len(by_flight[fid])}"
                 for fid, split in assignment.items()),
        encoding="utf-8"
    )

    print(f"\n[DONE] 데이터셋 부트스트랩 완료: {out}")
    print(f"  다음 단계:")
    print(f"  1. {out}/images/train 에 있는 이미지를 CVAT/Roboflow 등으로 라벨링")
    print(f"  2. 완성된 YOLO txt 를 {out}/labels/train 에 배치")
    print(f"  3. python {out}/train.py 로 학습 시작")


if __name__ == "__main__":
    main()
