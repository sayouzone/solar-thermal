# Solar Panel YOLO11n 라벨링 파이프라인 (v2)

드론 촬영 태양광 패널 이미지를 YOLO11n 학습용 데이터셋으로 변환하는 end-to-end 파이프라인입니다.
DJI Zenmuse H20T (`_Z.JPG`) nadir view에 최적화.

## 📁 구성

```
src/
├── auto_label.py         # 자동 pre-labeling (heuristic / SAM2 / YOLO-World)
├── advanced_utils.py     # Hybrid detector + NMS + DJI 메타데이터 + 결함 클래스
├── vlm_verify.py         # Claude Vision API 검증 레이어
├── split_dataset.py      # train/val/test 분할 + data.yaml
├── visualize_labels.py   # 검수 시각화 + Label Studio 변환
├── dataset_report.py     # 비행 궤적 / 라벨 분포 / 품질 리포트
└── run_pipeline.py       # End-to-end 실행

example_output/           # 실제 18장 샘플 결과
├── 01_trajectory.png     # GPS 비행 궤적
├── 02_label_count_dist.png
├── 04_gimbal_pitch.png   # nadir 여부 확인
├── 05_altitude.png
├── defect_schema.json    # IEC TS 62446-3 기반 클래스 스키마
└── summary.txt
```

## 🚀 설치

```bash
# 최소 의존성 (heuristic 전략만)
pip install opencv-python-headless pyyaml pillow matplotlib

# SAM2 / YOLO-World / 학습
pip install ultralytics

# Claude Vision 검증 레이어
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 🔄 권장 워크플로우

```
[드론 이미지]
     ↓
┌─────────────────────────────────┐
│ 1. auto_label.py (strategy=sam2)│  ← SAM2로 대략적 마스크/bbox 생성
└─────────────────────────────────┘
     ↓
┌─────────────────────────────────┐
│ 2. vlm_verify.py                │  ← Claude Vision으로 false-positive 제거
│    (Claude Haiku 4.5 권장)       │     + 결함 유형 자동 분류
└─────────────────────────────────┘
     ↓
┌─────────────────────────────────┐
│ 3. visualize_labels.py          │  ← 수동 검수용 오버레이 생성
│    → to-labelstudio             │     Label Studio로 최종 보정
└─────────────────────────────────┘
     ↓
┌─────────────────────────────────┐
│ 4. dataset_report.py            │  ← 비행 궤적, 고도, 라벨 분포 체크
└─────────────────────────────────┘
     ↓
┌─────────────────────────────────┐
│ 5. split_dataset.py             │  ← YOLO 표준 구조 + data.yaml
└─────────────────────────────────┘
     ↓
[YOLO11n 학습]
```

## 💻 사용 예시

### 전체 파이프라인 (one-shot)

```bash
python run_pipeline.py \
    --images /mnt/user-data/uploads \
    --work-dir ./workspace \
    --strategy sam2 \
    --classes solar_panel \
    --skip-train
```

### 단계별 (권장 — 중간에 검수 포함)

```bash
# 1) 자동 라벨링
python auto_label.py \
    --images /mnt/user-data/uploads \
    --output data/labels \
    --strategy sam2

# 2) Claude Vision 검증 (선택)
python vlm_verify.py \
    --images /mnt/user-data/uploads \
    --labels data/labels \
    --output data/labels_verified \
    --conf 0.5

# 3) 시각화 → 수동 검수
python visualize_labels.py visualize \
    --images /mnt/user-data/uploads \
    --labels data/labels_verified \
    --output data/visualized \
    --classes solar_panel

# 4) 품질 리포트
python dataset_report.py \
    --images /mnt/user-data/uploads \
    --labels data/labels_verified \
    --output data/report \
    --classes solar_panel

# 5) 데이터셋 분할
python split_dataset.py \
    --images /mnt/user-data/uploads \
    --labels data/labels_verified \
    --output dataset \
    --split 0.7 0.2 0.1 \
    --classes solar_panel

# 6) 학습
yolo detect train \
    data=dataset/data.yaml \
    model=yolo11n.pt \
    epochs=100 imgsz=1280 batch=8 \
    patience=20 degrees=10 flipud=0.5
```

## 🏷️ 결함 클래스 (IEC TS 62446-3 기반)

`advanced_utils.py::DefectClassRegistry`에 정의된 12개 클래스.

**RGB 전용** (drone `_Z.JPG`): `solar_panel`(0), `soiling`(1), `shading`(2),
`cell_crack`(3), `glass_breakage`(4), `delamination`(5), `snail_trail`(6),
`discoloration`(7).

**Thermal IR 전용** (drone `_T.JPG`, 별도 학습): `hotspot`(8), `bypass_diode`(9),
`string_fault`(10), `pid`(11).

## 📊 실제 테스트 결과 (18장 DJI ZH20T)

`dataset_report.py`로 생성한 리포트 요약:

```
Total images:           18
Total labels:           54 (heuristic strategy)
Avg labels/image:       3.00
Unlabeled images:       1 (5.6%)

Gimbal pitch: mean=-90.0°, min=-90.0°, max=-89.9°  ← 완벽한 nadir
Altitude:     mean=45.0m, min=44.9m, max=45.1m      ← 자동 미션 비행
GPS:          34.71°N, 126.92°E                     ← 전남 지역
Camera:       DJI Zenmuse ZH20T
```

## ⚠️ 운영 팁

### 전략 선택
- **heuristic** (OpenCV만): 속도는 빠르지만 반사광/그림자 오탐 많음 → 개발 초기 실험용
- **SAM2**: drone nadir view에서 가장 안정적 → **프로덕션 pre-labeling 추천**
- **YOLO-World**: zero-shot 실험 → 빠른 프로토타입
- **Hybrid** (`advanced_utils.HybridDetector`): YOLO-World proposal + SAM2 refinement → 최고 품질

### Claude Vision 검증 비용 절감
- `claude-haiku-4-5-20251001` 사용 (Opus/Sonnet 대비 저렴)
- `crop_max_side=512` 로 입력 크기 제한
- YOLO 자체 confidence가 매우 높은 bbox는 검증 스킵

### 드론 이미지 학습 팁
- `imgsz=1280` 이상 권장 (5184×3888 원본 → 패널이 작게 보임)
- `flipud=0.5` 사용 (nadir view는 수직 flip도 유효한 augmentation)
- `degrees=10` 회전 augmentation (gimbal pitch 미세 변동 대응)
- **RGB/IR 듀얼 학습 시**: `_Z.JPG`와 `_T.JPG`를 homography로 정렬 후 같은 라벨 공유

### Active Learning 루프 (Self-training)
1. 초기 100장 수동 라벨링
2. YOLO11n fine-tuning
3. 새 이미지에 model confidence 계산
4. confidence < 0.4인 샘플만 Label Studio로 보냄 (human-in-the-loop)
5. 재학습 → 반복

## 🔗 sayouzone/solar-thermal 통합 참고

이 파이프라인은 기존 `solar-thermal` 프로젝트의 다음 구성요소와 연동됩니다:

- **RGB-IR 레지스트레이션**: `_Z.JPG` ↔ `_T.JPG` homography 후 공유 라벨
- **VLM 검증**: `vlm_verify.py`가 기존 Claude API 검증 패턴과 동일 구조
- **IEC TS 62446-3 분류**: `DefectClassRegistry` 클래스를 학습 데이터에 직접 사용
- **Claude Code GitHub Action**: PR에서 `auto_label.py` 자동 실행 후 artifact로 Label Studio task 업로드 가능

## 라이선스 주의

- SAM2 / YOLO-World / YOLO11: Ultralytics **AGPL-3.0** — 상업적 사용 시 라이선스 확인
- Claude API: Anthropic 상업 라이선스 (API 키 발급 필요)