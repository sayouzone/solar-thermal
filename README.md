# solar-thermal

드론 기반 **RGB + IR 열화상** 영상에서 태양광 패널 결함을 탐지하는 클라우드 분석 서비스.
**YOLO 객체 탐지**로 패널/셀을 빠르게 식별하고, **열화상 통계**로 핫스팟을 정량화한 뒤,
**VLM(Claude)** 이 각 핫스팟의 결함 유형(셀 크랙, PID, 소일링, 바이패스 다이오드 고장, 그림자 등)을
이미지 + 통계 기반으로 추론합니다.

```
                ┌─────────┐   ┌──────────────┐   ┌────────────────┐   ┌─────────┐
  RGB + IR ───▶ │ Loader  ├──▶│ IR→RGB 정합   ├──▶│ YOLO (패널)     ├──▶│ Hotspot │
                └─────────┘   └──────────────┘   └────────────────┘   │ Analyzer│
                                                                      └────┬────┘
                                                                           │
                                                      ┌────────────────────┴──────────┐
                                                      │                               │
                                                ┌─────▼────────┐              ┌───────▼──────┐
                                                │ VLM (Claude) │─────────────▶│   Fusion     │
                                                │  - defect    │              │  - severity  │
                                                │  - confidence│              │  - label     │
                                                └──────────────┘              └───────┬──────┘
                                                                                      │
                                                                           ┌──────────▼────────┐
                                                                           │ InspectionReport  │
                                                                           │  + 오버레이 이미지    │
                                                                           └───────────────────┘
```

## 주요 기능

- **멀티 소스 입력**: radiometric TIFF, pseudo-color JPEG, 16-bit gray PNG, 클라우드 URI (`gs://`, `s3://`, `https://`)
- **자동 정합**: ORB + RANSAC homography 로 RGB 와 IR 좌표계 정렬 (실패 시 identity fallback)
- **패널 단위 판정**: YOLO 가 놓친 셀은 grid 분할로 보강
- **IEC TS 62446-3 기반 룰**: ΔT, 절대 온도, 면적으로 1차 핫스팟 판정
- **VLM 추론**: Claude vision 이 결함 유형을 판별하고 그림자(false positive) 를 걸러냄
- **Fusion**: rule_priority / vlm_priority / ensemble 전략 선택
- **프롬프트 캐싱**: 시스템 프롬프트에 `cache_control: ephemeral` 적용으로 반복 호출 비용 절감
- **FastAPI 서비스**: 동기 엔드포인트 + 비동기 job 제출
- **스토리지 어댑터**: 로컬 / GCS / S3

## 디렉터리 구조

```
solar-thermal/
├── configs/
│   └── default.yaml              # 파이프라인 설정
├── scripts/
│   ├── run_inference.py          # CLI 추론
│   └── train_yolo.py             # YOLO 파인튜닝
├── src/solar_thermal/
│   ├── api/                      # FastAPI 앱
│   ├── cloud/storage.py          # GCS / S3 / HTTP 어댑터
│   ├── detection/
│   │   ├── yolo_detector.py      # Ultralytics 래퍼
│   │   └── hotspot.py            # 열화상 핫스팟 분석
│   ├── fusion/analyzer.py        # YOLO + IR + VLM 결합
│   ├── pipeline/pipeline.py      # end-to-end 파이프라인
│   ├── preprocessing/
│   │   ├── loader.py             # RGB / IR 로더
│   │   ├── thermal.py            # raw → 온도 변환
│   │   └── registration.py       # RGB-IR 정합
│   ├── vlm/
│   │   ├── client.py             # Claude 클라이언트
│   │   └── prompts.py            # 시스템/유저 프롬프트
│   ├── config.py                 # pydantic 설정 로더
│   └── schemas.py                # 데이터 모델
├── tests/test_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

## 빠른 시작

### 1. 로컬 설치

```bash
pip install -r requirements.txt
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

### 2. 학습된 YOLO 가중치 준비

`configs/default.yaml` 의 `detector.weights` 에 로컬 경로 또는 `gs://bucket/key`
URI 를 설정합니다. 초기 개발 단계에서는 Ultralytics pre-trained 모델(`yolov8m.pt`)을
fine-tune 한 체크포인트를 사용하세요.

**(a) 데이터셋 준비**

YOLO 포맷 (이미지 + YOLO txt 라벨) 으로 아래처럼 배치합니다:

```
/data/solar/
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt   # 각 줄: <class> <cx> <cy> <w> <h>  (0..1 정규화)
```

공개 데이터셋으로 시작하려면:
- [InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules) (RaptorMaps, IR 단독)
- [PVEL-AD](https://github.com/binyisu/PVEL-AD) (EL 결함)
- Roboflow Universe — "solar panel" / "photovoltaic defect"

**(b) `data.yaml` 작성**

템플릿을 복사해서 실제 경로로 편집:

```bash
cp configs/dataset.example.yaml configs/dataset.yaml
# path, train, val, names 수정
```

**(c) 학습 실행**

```bash
python scripts/train_yolo.py --data configs/dataset.yaml --base yolov8m.pt
```

학습 전 `data.yaml`, 루트 디렉터리, `images/{train,val}` 존재 여부를 자동 검증하여
`Dataset '...' error ❌ ... does not exist` 같은 Ultralytics 내부 오류 대신 명확한
안내를 출력합니다.

### 3. CLI 추론

```bash
python scripts/run_inference.py \
    --rgb samples/rgb.jpg \
    --ir samples/ir_radiometric.tiff \
    --ir-format radiometric_tiff \
    --out report.json
```

### 4. API 서버

```bash
uvicorn solar_thermal.api.main:app --host 0.0.0.0 --port 8080
```

```bash
curl -X POST http://localhost:8080/inspect/upload \
    -F "rgb=@samples/rgb.jpg" \
    -F "ir=@samples/ir.tiff" \
    -F "ir_format=radiometric_tiff"
```

### 5. Docker

```bash
docker compose up --build
```

## 결과 스키마 (`InspectionReport`)

```json
{
  "inspection_id": "…",
  "num_panels": 24,
  "num_defective_panels": 3,
  "panels": [
    {
      "panel_id": "…_panel_0007",
      "panel_bbox": {"x1": 1200, "y1": 800, "x2": 1580, "y2": 1020},
      "hotspots": [
        {
          "bbox": {"x1": 1320, "y1": 860, "x2": 1360, "y2": 900, "score": 0.88},
          "stats": {"t_max": 74.2, "t_mean": 46.1, "delta_t": 17.5, "hotspot_area_px": 420},
          "rule_label": "hotspot_single_cell",
          "rule_severity": 0.73
        }
      ],
      "vlm_verdict": {
        "defect_type": "cracked_cell",
        "confidence": 0.82,
        "rationale": "RGB 에 미세 크랙이 관찰되며 IR 에서 단일 셀 ΔT ≈ 17K 로 …",
        "tags": []
      },
      "final_label": "cracked_cell",
      "severity": 0.76,
      "is_normal": false
    }
  ],
  "visualization_uri": "/var/cache/solar-thermal/solar-thermal/reports/….jpg",
  "processing_time_ms": 2150.4
}
```

## 클라우드 배포 가이드

### GCP Cloud Run (GPU)
1. Artifact Registry 에 이미지 푸시
2. Cloud Run (GPU) 로 배포, `GOOGLE_APPLICATION_CREDENTIALS` 는 Workload Identity 로 주입
3. `configs/default.yaml` 의 `storage.backend: gcs`, `gcs_bucket` 지정
4. YOLO 가중치는 `gs://bucket/models/yolo_solar_v1.pt` 로 레퍼런스

### AWS ECS Fargate / EKS
1. ECR 에 이미지 푸시
2. Task role 에 S3 + Bedrock 권한 부여 (Claude 를 Bedrock 으로 사용하는 경우 `vlm.provider` 확장)
3. `storage.backend: s3`

### Kubernetes
* HPA: CPU / 큐 길이 기반 scale-out
* PVC: `/var/cache/solar-thermal` 캐시 볼륨
* Secret: `ANTHROPIC_API_KEY`

## 튜닝 팁

| 현상 | 조정 |
| --- | --- |
| 그림자가 핫스팟으로 오탐 | `vlm.temperature=0.0`, `fusion.strategy=ensemble` 유지. VLM prompt 에 그림자 감별 강조됨 |
| ΔT 가 낮아도 결함 놓침 | `thermal.hotspot.delta_t_threshold` 를 3~4K 로 하향 |
| VLM 호출 비용 과다 | `vlm.trigger_only_on_hotspot=true`, `max_crops_per_request` 축소 |
| RGB-IR 어긋남 | `registration.method=homography`, `orb_features=4000` 상향. dual-sensor 카메라는 `identity` 도 가능 |
| 소형 핫스팟 누락 | `hotspot.min_area_px` 축소, `morph_kernel=1` |

## 확장 포인트

- **VLM provider 교체**: `vlm.client.VLMClient` 는 Anthropic SDK 기반. Bedrock, Vertex AI 로 대체하려면
  동일한 `analyze_panel` 인터페이스를 구현한 클래스를 작성하고 `DefectDetectionPipeline(vlm_client=...)` 로 주입.
- **YOLO → RT-DETR / DETR**: `detection.yolo_detector.YOLODetector` 를 다른 백엔드로 교체.
- **시계열 추적**: `site_id`, `inspection_id` 를 DB 에 적재하여 동일 패널의 시간에 따른 severity 추이 분석.
- **OCR**: 모듈 시리얼 넘버를 인식해 `panel_id` 를 하드웨어 ID 로 대체.

## 테스트

```bash
pytest -q
```

VLM / YOLO 는 네트워크 / 가중치 의존이므로 `tests/` 는 순수 CPU 로직만 검증합니다.
통합 테스트는 소량의 공개 샘플 이미지와 `configs/test.yaml` 로 별도 구성 권장.
