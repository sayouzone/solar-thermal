# Solar Thermal AI — RGB+IR 하이브리드 결함 탐지 파이프라인

태양광 발전소 드론 점검 영상(RGB + 열화상 IR)에 대해 **YOLO + VLM** 을 결합하여
클라우드에서 결함을 탐지/분류/리포팅하는 시스템.

## 아키텍처

```
┌────────────────────────────────────────────────────────────────────┐
│                        Drone (Edge, Jetson Orin NX)                │
│   RGB(4K) + Thermal(640x512 radiometric)  →  S3 / GCS 업로드        │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                     FastAPI Inference Gateway                      │
│                   (Auth / Job Queue / Dedup)                       │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                     Pipeline Orchestrator                          │
│   1) Frame Loader (RGB + Radiometric TIFF)                         │
│   2) RGB–IR Registration (Homography / ECC)                        │
│   3) Panel Detector  (YOLOv12 RGB)                                 │
│   4) Hotspot Detector (YOLOv12 IR + Temperature Thresholding)      │
│   5) Thermal Analyzer (ΔT, IEC TS 62446-3 class)                   │
│   6) VLM Analyzer    (Claude / Qwen2.5-VL: defect taxonomy + RCA)  │
│   7) Report Builder  (JSON + GeoJSON + PDF)                        │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    PostgreSQL + PostGIS + S3
```

## 결함 분류 체계 (IEC TS 62446-3 기반 확장)

| Class             | 열화상 패턴                       | 판정 기준                     | VLM 교차검증 필요 |
|-------------------|----------------------------------|------------------------------|------------------|
| single_cell_hs    | 단일 셀 핫스팟                    | ΔT ≥ 20°C                    | High             |
| multi_cell_hs     | 다중 셀 핫스팟                    | ΔT ≥ 10°C, n≥2               | High             |
| bypass_diode      | 1/3 스트링 전체 과열              | 셀 그룹 균일 ΔT ≥ 5°C         | Critical         |
| pid               | 패널 하단 셀 열 과열               | 하단 row ΔT ≥ 3°C            | Critical         |
| soiling           | 저온/그림자 패턴                  | ΔT ≤ -2°C, RGB 먼지/조류      | Low              |
| cell_crack        | 미세 선형 핫라인                  | 선형 ΔT ≥ 5°C                | High             |
| junction_box      | J-box 위치 고온                   | J-box ROI ΔT ≥ 15°C          | Medium           |
| shading           | 외부 그림자 (결함 아님)            | RGB에서 shadow 확인           | Required         |

## 모듈 구성

- `app/core/pipeline.py` — 전체 파이프라인 오케스트레이터
- `app/fusion/registration.py` — RGB–IR 정합 (ECC + feature-based fallback)
- `app/detectors/panel_detector.py` — YOLO 기반 패널 검출 (RGB)
- `app/detectors/hotspot_detector.py` — YOLO 기반 핫스팟 검출 (IR)
- `app/thermal/radiometric.py` — Radiometric TIFF → 온도 변환, ΔT 계산
- `app/thermal/classifier.py` — IEC TS 62446-3 분류
- `app/vlm/analyzer.py` — VLM(Claude/Qwen2.5-VL)을 통한 결함 확정 및 shading 판별
- `app/api/server.py` — FastAPI 엔드포인트
- `app/schemas/` — Pydantic 모델
