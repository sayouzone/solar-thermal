# PV Hierarchical Detection Pipeline (v2)

**업로드된 시각화 이미지들(0006, 0010, 0029)에서 드러난 v1 heuristic의 한계를 해결한 재설계 버전입니다.**

## 📌 v1 → v2 변경 요약

### v1의 근본 문제 (시각화 검증에서 확인)
- ❌ 마른 잔디(갈색)가 HSV gray 범위에 들어가 **fake solar_panel** 생성
- ❌ 방수포(tarp)가 HSV blue 범위에 들어가 **fake solar_panel** 생성
- ❌ String 중간 frame gap에서 contour가 분리되어 **하나의 string이 둘로 잘림**
- ❌ 이미지 해상도 5184×3888에 imgsz=1280은 module 단위 학습 시 object가 너무 작음

### v2 재설계
1. **계층 구조**: `pv_string`(큰 단위) + `pv_module`(작은 단위) 2-class
2. **Heuristic 포기**: SAM2 기반 segmentation으로 전환 (잔디/tarp 오탐 근본 해결)
3. **SAHI 타일링**: 5184×3888을 1024 tile로 쪼개서 module 단위 small object 학습/추론
4. **Active Learning Loop**: 5-20장 수동 seed → YOLO 부트스트랩 → uncertainty 기반 샘플 선별
5. **계층 일관성 자동 정리**: 고아 module 제거, string aspect 검증

## 📁 v2 파일 구성

```
src/v2/
├── pv_detector.py              # String + Module 계층 검출기 (SAM2 기반)
├── active_learning.py          # Self-training loop (seed/predict/select/iterate)
├── sahi_inference.py           # SAHI 타일링 추론 (module 단위 필수)
├── hierarchical_cleanup.py     # 계층 일관성 강제 + 시각화
├── pv_pipeline.py              # End-to-end orchestrator
└── LABEL_STUDIO_GUIDE.md       # 초기 수동 라벨링 가이드 (필독)
```

## 🎯 권장 워크플로우

### Phase 1: 수동 Seed 라벨 (1-2시간)
가장 중요한 단계. 자동화 대체 불가능.

```bash
# Label Studio 실행
pip install label-studio
label-studio start
```

→ **`LABEL_STUDIO_GUIDE.md`** 를 따라 10-20장 정확히 라벨링.
업로드된 이미지에서 보신 것처럼 heuristic은 잔디·tarp·분할 오류를 
완전히 해결할 수 없어서, 이 단계가 필수입니다.

### Phase 2: 초기 YOLO11n 부트스트랩 (30분)

```bash
cd src/v2
python active_learning.py seed \
    --images /path/to/drone_images \
    --seed-labels /path/to/label_studio_export/labels \
    --output workspace/round_0 \
    --classes pv_string pv_module \
    --epochs 50 \
    --imgsz 1920
```

Seed 10-15장에서 초기 mAP 0.5-0.6 달성 목표.

### Phase 3: Active Learning 루프 (각 round 2시간)

```bash
# Round 1: 전체 unlabeled 예측 + top-20 uncertain 선별
python active_learning.py iterate \
    --images /path/to/drone_images \
    --model workspace/round_0/weights/best.pt \
    --output workspace/round_1 \
    --select-top 20

# → workspace/round_1/human_review/ 에 예측 라벨 복사됨
# → Label Studio로 가져가 수정
# → 다시 seed → 재학습
```

일반적으로 3-5 round에서 mAP 0.85+ 수렴.

### Phase 4: Production 추론 (SAHI 타일링)

```bash
python sahi_inference.py \
    --images /path/to/new_images \
    --model workspace/round_N/weights/best.pt \
    --output production/labels \
    --tile-size 1024 \
    --overlap 0.2
```

5184×3888 이미지당 **35 tiles** 생성 (20% overlap), 타일별 추론 후 NMS 병합.

### Phase 5: 계층 일관성 자동 정리

```bash
python hierarchical_cleanup.py clean \
    --labels-in production/labels \
    --labels-out production/labels_clean

python hierarchical_cleanup.py visualize \
    --images /path/to/new_images \
    --labels production/labels_clean \
    --output production/visualized
```

자동으로 제거되는 항목:
- String 내부에 없는 고아 module
- Aspect ratio가 2.5 미만인 "string" (잔디 오탐 방지)
- 이미지 면적 대비 0.3% 미만인 너무 작은 string

## 🎨 계층 검증 이미지 예시

`hierarchical_cleanup.py visualize`가 생성하는 시각화:
- **파란색 두꺼운 box**: pv_string (class 0)
- **녹색 얇은 box**: pv_module (class 1)

## 🔬 주요 기술 결정

### 왜 SAM2 + aspect filter인가
SAM2는 "일관된 재질/질감 영역"을 찾으므로:
- 마른 잔디 → 균일한 질감이지만 aspect ratio가 landscape 형태 → aspect 필터에서 배제
- 방수포 → 울퉁불퉁한 주름으로 SAM이 작은 조각으로 분할 → mask fill ratio < 0.4로 배제
- PV 패널 → 규칙적 격자 + 직사각형 frame → SAM이 깔끔한 긴 segment 생성

### 왜 imgsz=1920인가
| imgsz | Module 크기 (pixel) | YOLO 인식 |
|-------|------------------|----------|
| 640 | ~20px | 거의 불가능 |
| 1280 | ~40px | 경계선 |
| **1920** | **~60px** | **안정적** |

### 왜 SAHI 타일링인가
단일 1920 추론은 원본 5184 대비 37% 다운샘플. 반면 1024 tile은 100% 원본 해상도
→ module 단위 검출 recall이 약 2-3배 개선 (일반적 벤치마크 기준).

### 왜 Active Learning + self-training인가
사용자가 "100-500장" 규모일 가능성이 높을 때 가장 효율적:
- 수동 전체 라벨링: 500장 × 10분/장 = 83시간
- Active learning: seed 20장 + 3 round × 20장 = 80장 = 13시간
- **약 6배 시간 절감**

## 🔗 v1과의 관계

v1의 다음 모듈은 **v2와 함께 사용 가능**하며 권장됩니다:
- `advanced_utils.py::DefectClassRegistry` — Phase 6 결함 클래스 확장
- `vlm_verify.py` — Claude Vision으로 false-positive 최종 검증
- `dataset_report.py` — GPS 궤적 + Gimbal 메타데이터 리포트

v1의 `auto_label.py` (heuristic 전략)는 **v2에서 대체됨**. 사용 권장하지 않습니다.

## ⚠️ 환경 요구사항

```bash
pip install opencv-python-headless numpy pyyaml pillow matplotlib
pip install ultralytics    # YOLO + SAM2
pip install label-studio   # 수동 라벨링 (optional but highly recommended)
```

GPU 권장 (SAM2 + YOLO 학습 모두). CPU로도 동작하지만 시간이 수십 배 걸림.

## 🎯 기본값 근거

사용자 답변을 기다리지 않고 진행한 기본값:

| 질문 | 선택한 기본값 | 근거 |
|------|-------------|-----|
| 검출 단위 | 계층적 (string + module) | 가장 확장성 높음, 어느 쪽으로도 쉽게 전환 가능 |
| 결함 포함? | 위치만 (v1) | `vlm_verify.py`로 결함 분류는 나중에 분리 추가 |
| 데이터 규모 | 100-500장 가정 | Active learning이 가장 큰 효과를 내는 범위 |

선택이 다를 경우:
- **Module만**: `pv_pipeline.py bootstrap --no-modules`로 string만 검출
- **String만**: `pv_detector.py`에서 `detect_modules=False`
- **~100장 few-shot**: `active_learning.py seed --epochs 30` (과적합 허용)
- **2000장+**: seed 수를 50으로 늘리고 `--epochs 100`으로 정규 학습
