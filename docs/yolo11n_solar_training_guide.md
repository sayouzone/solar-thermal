# YOLO11n Solar Panel Detection — 학습 데이터 준비 가이드

**프로젝트 맥락**: Sayouzone 드론 AI 엔진 / Solar Plant AI
**목표 하드웨어**: Jetson Orin NX 16GB (엣지 추론)
**초기 학습 샘플**: DJI Mavic 3E/M3M Z렌즈 (줌 렌즈) 촬영, 2025-12-17

---

## 1. 클래스 설계 — 단일 클래스로 시작, 계층적 확장

첫 iteration 은 무조건 **단일 클래스 `solar_panel`** 로 시작합니다. 세분화된 클래스는 라벨링 비용을 5~10배로 늘리는 반면, mAP 개선은 한계가 있습니다. 모델이 "패널 위치"를 먼저 안정적으로 잡아야 결함 분류가 의미 있어집니다.

| 단계 | 클래스 | 용도 |
|---|---|---|
| **v1 (MVP)** | `solar_panel` | 탐지·위치 파악 |
| **v2** | `panel_module`, `panel_string`, `panel_array` | 계층적 counting, 발전소 인벤토리 |
| **v3** | + `soiling`, `crack`, `discoloration` | 결함 탐지 (2-stage 또는 multi-task) |
| **v3 열화상** | + `hotspot`, `bypass_diode_failure` | 열화상 별도 모델 |

**어노테이션 단위 결정**: 검사 유스케이스에는 **모듈(개별 패널) 단위 bbox** 가 옳습니다. 하지만 첫 라운드는 **스트링(한 열) 단위**로 라벨링해서 빠르게 베이스라인을 만드는 걸 추천. Image 3 기준 스트링 하나 = bbox 하나, Image 1 기준 어레이당 수직으로 긴 bbox 3개씩.

## 2. 데이터 수집 — 규모와 다양성

### 최소 볼륨 (YOLO11n 2.6M 파라미터 기준)
- **MVP 베이스라인**: 300~500 라벨 이미지 → mAP50 0.6~0.7 달성 가능
- **실사용 수준**: 1,500~2,500 이미지 → mAP50 0.85+
- **프로덕션**: 5,000 이상 + 결함 라벨 별도 수천 개

### 다양성 차원 (각 축을 고르게 커버해야 일반화됨)

| 축 | 요구 범위 | 현재 샘플 커버 여부 |
|---|---|---|
| 고도 (AGL) | 30 / 50 / 80 / 120m | ❓ EXIF 없어 확인 불가 |
| 짐벌 피치 | −90° (nadir), −60° (검사용 oblique) | Nadir 만 확인됨 |
| 시각 | 오전/정오/오후 (그림자 방향 변화) | 오후 1시만 |
| 계절 | 겨울/봄/여름/가을 (식생·태양광 강도) | 겨울 (12월) 만 |
| 날씨 | 맑음/흐림/얇은 구름 | 맑음만 |
| 패널 타입 | 결정질(청색), 단결정(검정), 박막, 양면형 | ❓ |
| 설치 형태 | 지상거치, 지붕, 수상, BIPV | 지상거치만 |
| 배경 | 콘크리트, 자갈, 잔디, 흙, 방초시트, 아스팔트 | 방초시트·자갈·흙 |
| 패널 상태 | 깨끗함, 먼지, 적설, 음영, 손상 | 깨끗~약간 먼지 |

**현재 3장은 "겨울·오후·지상거치·맑음" 한 구석만 커버**. 촬영 계획을 짤 때 위 표에서 비어 있는 조합 위주로 수집하세요.

### 도메인 외 공개 데이터셋 (워밍업·사전학습용)

| 데이터셋 | 규모 | 형식 | 라이선스 | 비고 |
|---|---|---|---|---|
| **BDAPPV** (Bradbury) | ~29,000 | Seg 마스크 | CC-BY-4.0 | 위성+항공 혼합 |
| **Duke California** | ~19,000 | 픽셀 마스크 | 공개 | USGS 항공 |
| **PV03 / PV08** (중국) | 수천~수만 | Seg | 연구용 | UAV nadir |
| **Roboflow Universe** `solar-panel` 검색 | 수십 개 프로젝트 | bbox/seg 혼재 | 각기 다름 | 품질 편차 큼 |

공개 데이터로 **프리트레이닝 (warm-start)** 한 가중치에서 자체 데이터로 파인튜닝하면 300장만으로도 mAP가 한 단계 뛰는 경우가 많습니다.

## 3. 촬영 프로토콜 — 수집 단계에서 반드시 고정해야 할 것

드론 비행 계획에 아래를 문서화해 **매 비행이 재현 가능**하게 만드세요. 나중에 데이터셋을 합치거나 재라벨링할 때 결정적으로 중요합니다.

| 항목 | 권장 설정 | 이유 |
|---|---|---|
| 촬영 모드 | 자동 격자 비행(grid mission) | 중복률 일정 |
| Overlap | 전방 80% / 측방 70% | Orthomosaic 제작도 가능 |
| 속도 | 3~5 m/s | 모션블러 방지 |
| 셔터 | 1/1000 이상 | 패널 프레임 선명도 |
| ISO | 100~400 고정 | 노이즈 억제, 색상 일관성 |
| 화이트밸런스 | Daylight 고정 (Auto 금지) | 날짜 간 색상 drift 방지 |
| 파일 형식 | JPEG + DNG (RAW) 동시 저장 | 라벨 수정 시 재처리 가능 |
| EXIF/XMP 보존 | **무조건 원본 유지** | GPS·짐벌각·고도 활용 |

**핵심**: 화이트밸런스를 Auto 로 두면 같은 발전소 같은 날 촬영한 프레임끼리도 색감이 달라져 색상 증강만으로는 보정 불가능한 domain gap 이 생깁니다.

## 4. 어노테이션 전략 — 도구와 워크플로우

### 도구 비교

| 도구 | 장점 | 단점 | 추천 시나리오 |
|---|---|---|---|
| **Roboflow** | 클라우드, AL·버전관리·증강 내장, Export 바로 YOLO txt | 유료(>1K 이미지), 데이터 외부 업로드 | 소~중규모, 협업 |
| **CVAT** | 오픈소스, 팀 협업 지원, 반자동(SAM) 통합 | 셀프호스팅 필요 | 민감 데이터, 대규모 |
| **Label Studio** | Active Learning 루프 커스텀 쉬움 | UI 가 탐지보다 분류 지향 | AL 중심 파이프라인 |
| **X-AnyLabeling** | **SAM2/YOLO 내장 반자동 라벨링** | 로컬 단독 | 1인 빠른 라벨링 |

사내 데이터 유출이 걱정이면 CVAT 셀프호스팅, 혼자 빠르게 시작하려면 X-AnyLabeling + SAM2 조합이 가장 빠릅니다. SAM2 로 클릭 한 번에 마스크 생성 → bbox 자동 추출 → YOLO txt 저장까지 수 초.

### 라벨링 가이드 (Labeling Guide) — 팀에 배포해야 할 문서

팀이 2인 이상이면 반드시 작성:

1. **bbox 경계**: 패널 프레임 바깥선(알루미늄 테두리 포함). 마운트 구조물은 제외.
2. **가림(occlusion)**: 70% 이상 가려진 패널은 제외. 30~70% 는 `difficult=1` 플래그.
3. **프레임 경계 잘림**: 50% 이상 잘렸으면 라벨. 10~50% 는 판단자 재량 후 통계 기록.
4. **그림자**: 그림자는 **절대 bbox 에 포함하지 않음**.
5. **반사/글레어**: 반사가 심해 셀 패턴이 안 보여도 형태가 명확하면 라벨.
6. **모니터링용 소형 패널** (Image 1 중앙의 작은 것): 동일 클래스로 라벨. 스케일 편차 학습에 중요.
7. **군집 vs 개별**: 스트링 라벨링 기준일 경우, 패널 간 간격이 0.5m 미만이면 한 bbox, 1m 이상이면 분리.

가이드 없이 다인 라벨링하면 IoU 0.5 기준에서도 어노테이터 간 불일치로 mAP 가 상한에 막힙니다.

### Active Learning 루프 (권장)

```
[초기 200장 수동 라벨] 
  → [YOLO11n 학습] 
  → [신규 1000장에 추론] 
  → [confidence 0.3~0.6 인 것만 우선 검토(uncertainty sampling)]
  → [수정/승인 → 라벨 풀 확장]
  → [재학습]
```

이 루프를 돌리면 **순수 수동 라벨링 대비 3~5배 빠르게** 라벨이 쌓입니다. 앞서 만든 SAM2+CLIP 파이프라인이 여기서 1라운드 pseudo-label 생성기로 유용합니다.

## 5. 데이터 분할 — 사이트/비행 단위 분할이 필수

### 절대 하면 안 되는 것

```python
# ❌ 잘못된 분할 — 데이터 누수 발생
train, val = train_test_split(all_images, test_size=0.2, random_state=42)
```

항공 사진은 프레임 간 겹침이 매우 크기 때문에, 무작위 분할하면 val 셋에 train 의 거의 같은 장면이 들어가 **mAP 가 실제보다 20~30%p 높게 나옵니다**. 배포 후 성능이 폭락하는 흔한 함정.

### 올바른 분할

```
train : val : test = 70 : 20 : 10

분할 단위 = 비행(flight) 또는 사이트(site)
  - 같은 비행의 프레임들은 모두 같은 split 으로
  - 가능하면 다른 발전소를 test 로 빼두기
```

파일명 `DJI_20251217130200_0001_Z.JPG` 의 날짜·시각 접두사를 기준으로 flight ID 를 만들면 자동 분할 가능합니다.

## 6. 전처리 및 해상도 — 작은 패널도 놓치지 않으려면

### Mavic 3E Z렌즈 원본 해상도 처리

원본이 예컨대 4000×3000 이상이면, YOLO11n 기본 `imgsz=640` 은 **다운샘플로 작은 패널을 놓칩니다** (Image 1 의 모니터링용 소형 패널이 대표 케이스).

**권장 전략**:

| 전략 | imgsz | 장단점 |
|---|---|---|
| 단순 리사이즈 | 1280 또는 1536 | 간편. Orin NX 에서 여전히 실시간 가능 |
| **SAHI 타일링 추론** | 640×640 타일 슬라이드 | 작은 객체 mAP 대폭 개선, 속도 ⅓~¼로 |
| 학습시 RandomCrop | 원본에서 1280 크롭 | 데이터 증강 + 고해상도 효과 |

실용적인 조합: **학습은 1280 리사이즈 + RandomCrop, 배포는 nadir 촬영 프레임엔 리사이즈, oblique 엔 SAHI 타일링**.

## 7. 데이터 증강 — 항공 도메인 특화

YOLO 기본 증강에서 **반드시 조정할 것**:

```yaml
# data.yaml 에 반영 또는 train() 인자로 전달
hsv_h: 0.015       # 색조 — 패널의 청색 시그니처가 중요하므로 작게
hsv_s: 0.5         # 채도 — 기본값 0.7보다 약간 낮춤
hsv_v: 0.4         # 명도 — 그림자/조명 변화 시뮬레이션

degrees: 180       # 회전 — 항공 뷰는 모든 각도 허용 (기본 0 인데 반드시 올려야 함)
translate: 0.15
scale: 0.6         # 스케일 변화 큼 (30m vs 120m 고도)
fliplr: 0.5
flipud: 0.5        # 상하 플립도 허용 (항공이므로 위/아래 구분 없음)
perspective: 0.0001 # 항공은 원근 거의 없음

mosaic: 1.0        # 기본. 배경 다양성에 크게 기여
mixup: 0.1
copy_paste: 0.3    # ★ Solar Plant 에 매우 유효 — 패널을 빈 배경에 합성
```

`copy_paste=0.3` 은 항공 검사에서 특히 효과적입니다: 패널이 없는 프레임(Image 2 같은)에 라벨된 패널을 합성 — 모델이 "이 배경 자체가 패널 있는 곳" 같은 spurious correlation 을 학습하는 걸 막아줍니다.

## 8. 폴더 구조 및 data.yaml

```
solar_panels_dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── flight_20251217_site_A_0001.jpg
│   │   └── ...
│   ├── val/
│   └── test/
└── labels/           # YOLO txt — images/ 구조와 1:1 미러
    ├── train/
    │   ├── flight_20251217_site_A_0001.txt
    │   └── ...
    ├── val/
    └── test/
```

각 `.txt` 는 한 줄당 `class_id x_center y_center width height` (모두 0~1 정규화).

```yaml
# data.yaml
path: /data/solar_panels_dataset
train: images/train
val: images/val
test: images/test

names:
  0: solar_panel
```

## 9. 학습 실행 — YOLO11n 구체 명령

### 1차 베이스라인

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # COCO 프리트레인

model.train(
    data="data.yaml",
    epochs=150,
    patience=25,          # early stopping
    imgsz=1280,           # 항공은 반드시 640 이상
    batch=16,             # RTX 4090 기준, Orin 에서 학습 안 함
    device=0,
    cache="ram",          # 1만 장 이하면 RAM 캐시
    # 증강
    degrees=180, flipud=0.5, fliplr=0.5,
    scale=0.6, translate=0.15,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
    mosaic=1.0, mixup=0.1, copy_paste=0.3,
    # 저장
    project="runs/solar", name="yolo11n_v1_1280",
    exist_ok=True,
    # 로깅
    plots=True,
)
```

### 2차: 공개 데이터 warm-start 후 파인튜닝

BDAPPV 나 Duke 로 50 epoch 학습한 가중치를 만들어 두고, 자체 데이터로 파인튜닝:

```python
model = YOLO("runs/solar/pretrain_bdappv/weights/best.pt")
model.train(data="our_data.yaml", epochs=80, lr0=0.001, ...)
```

### Jetson Orin NX 배포용 export

```bash
yolo export model=best.pt format=engine half=True imgsz=1280 device=0
# → best.engine (TensorRT FP16)
```

Orin NX 16GB 에서 YOLO11n @ 1280 FP16 → **약 25~40 FPS 예상** (입력 해상도가 720p 정도로 줄면 80 FPS 이상).

## 10. 검증 지표 — mAP 만 보면 안 됨

### 반드시 추적할 지표

| 지표 | 의미 | 목표 |
|---|---|---|
| mAP50 | IoU 0.5 기준 mAP | > 0.85 |
| mAP50-95 | COCO 스타일 엄격 mAP | > 0.55 |
| Recall @ 0.25 conf | 패널 하나도 놓치지 않기 | > 0.95 |
| Precision @ 0.5 conf | False positive 억제 | > 0.90 |
| **Per-site mAP** | 학습에 없던 발전소 일반화 | > 0.75 |
| **Small object mAP** | 스케일 < 0.02 영역 | 별도 집계 |

**Per-site mAP** 는 test 셋을 발전소별로 나눠 각각 계산. 한 발전소만 압도적으로 좋고 다른 곳은 저조하면 overfitting 신호입니다.

### 오탐 케이스 로그

배포 후 운영에서 다음을 주기적으로 샘플링해 라벨 풀로 환류:
- `conf > 0.7` 이지만 사람이 "패널 아님" 판정한 프레임 → **hard negative**
- `conf < 0.3` 인데 사람이 "패널 맞음" 판정한 프레임 → **hard positive**

이 둘만 집중적으로 재학습해도 운영 3개월 후 mAP 가 수 %p 추가 상승합니다.

## 11. 체크리스트 — 학습 시작 전

```
[ ] 최소 300장 수집 완료 (다양성 축 2개 이상 커버)
[ ] 원본 EXIF/XMP 보존 확인 (GPS·고도·짐벌각 읽히는지 exif_extractor 로 검증)
[ ] 화이트밸런스·ISO 고정 확인
[ ] 라벨링 가이드 문서화 완료
[ ] 어노테이션 도구 선정 및 1차 라벨링 (파일럿 50장)
[ ] 파일럿 라벨에 대한 2인 크로스체크 (IoU > 0.8 확인)
[ ] flight/site 단위 train/val/test 분할 스크립트 작성
[ ] 공개 데이터셋 warm-start 가중치 준비 (선택)
[ ] data.yaml, 폴더 구조 확정
[ ] 베이스라인 학습 실행
[ ] 첫 검증 — per-site mAP, 작은 객체 mAP, hard negative 샘플링
```

---

## 부록 A — 현재 3장 샘플로 바로 해볼 수 있는 것

업로드된 3장은 EXIF 없고 다운샘플되었지만 라벨링 연습 + 프리뷰 추론 용도로는 충분합니다.

1. X-AnyLabeling 설치 → SAM2 모델로 3장에 bbox 자동 생성 → 검토
2. `solar_panel_yolo.py` (이전 세션에서 만든 것) 로 Roboflow 공개 `solar-panel` 사전학습 가중치로 추론 → 현재 모델이 놓치는 케이스 확인
3. 3장을 `copy_paste` 증강 테스트용으로 사용 — 패널을 Image 2(배경) 에 합성해 증강 파이프라인 검증

## 부록 B — 추가로 고려할 최신 대안

- **RF-DETR / D-FINE** (2024~2025): 엣지에서 YOLO 대비 유사 속도·더 높은 정확도. 데이터 셋업은 동일하므로 data.yaml 그대로 재사용 가능. 2차 iteration 에서 실험 권장.
- **YOLO-World / Grounding DINO**: 제로샷 탐지. 라벨이 거의 없는 초기에 pseudo-label 생성기로 유용.
- **DINOv3 embedding + k-NN**: 기존 라벨 풀에서 "비슷한 장면" 자동 검색 → 추가 수집이 필요한 도메인 식별.
