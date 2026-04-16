# 태양광 패널 탐지 Python 소스 — 접근법 비교 가이드

드론 AI 엔진 / Solar Plant AI 파이프라인에 바로 사용할 수 있도록 세 가지 접근법을 제공합니다.

## 파일 구성

| 파일 | 접근법 | 학습 필요 | 속도 | 정확도 | 주 용도 |
|---|---|---|---|---|---|
| `solar_panel_classical.py` | OpenCV (HSV + 형태학) | ❌ | 매우 빠름 (CPU 실시간) | 중간 | 프로토타입, 마스크 부트스트랩 |
| `solar_panel_yolo.py` | YOLO v8/v11/v12 | ⭕ (커스텀 학습 권장) | 빠름 (Jetson Orin NX 실시간) | 높음 | **프로덕션 엣지 추론** |
| `solar_panel_sam_clip.py` | SAM 2.1 + CLIP 제로샷 | ❌ | 느림 (GPU 필수) | 높음 (마스크) | **라벨링 자동화, Active Learning** |

## 사용 시나리오 (Sayouzone Solar Plant AI 맥락)

### 1. 데이터 수집 초기 단계 — Classical + SAM/CLIP
라벨이 전혀 없을 때: SAM2+CLIP 으로 후보 마스크를 뽑고, 사람이 검수해 골드 셋을 만든다.
Classical 은 field 에서 빠른 sanity check 용으로 사용.

### 2. 모델 학습 — YOLO
충분한 라벨(보통 500~2000장 이상)이 쌓이면 YOLO 커스텀 학습으로 전환.
항공 촬영은 `imgsz=1280~1536` 이상이 유리. `degrees=10`, `mosaic=1.0` 증강 권장.

### 3. 엣지 배포 — YOLO (Orin NX 16GB)
- FP16 엔진으로 TensorRT 변환: `yolo export model=best.pt format=engine half=True`
- Orin NX 16GB 에서 YOLO11s @ 1280 → 약 20~35 FPS 예상
- 열화상(thermal) 스트림은 별도 채널로 넣고 2채널 late-fusion 또는 early-fusion 로 핫스팟까지 동시 탐지 설계 가능

## 빠른 실행 예

```bash
# 1) 고전 CV — 의존성 최소
pip install opencv-python numpy
python solar_panel_classical.py aerial.jpg --out result.jpg

# 2) YOLO — 가중치 필요
pip install ultralytics opencv-python
python solar_panel_yolo.py aerial.jpg --weights best.pt --device cuda:0

# 3) SAM2 + CLIP — 라벨링 자동화
pip install torch torchvision open_clip_torch
pip install git+https://github.com/facebookresearch/sam2.git
python solar_panel_sam_clip.py aerial.jpg --sam-ckpt sam2_hiera_large.pt
```

## 추천 파이프라인 (드론 엔진 통합)

```
[드론 카메라] ─► RGB ─┐
                    ├─► YOLO (PV 탐지) ─► bbox + 패널 ID
         ─► Thermal ─┘                    │
                                          ▼
                              [핫스팟 탐지 모델] ─► 이상 위치
                                          ▼
                              [VLM (Edge)] ─► 진단 리포트 초안
```

## 데이터셋 참고 (학습용)

| 데이터셋 | 종류 | 규모 |
|---|---|---|
| BDAPPV (Bradbury et al.) | 위성 + 항공 RGB | ~29,000 |
| Duke California Solar Array | 항공 RGB, 픽셀 마스크 | ~19,000 |
| PV01/03/08 (중국) | UAV RGB, 세그멘테이션 | 수천~수만 |
| Roboflow `solar-panel` 공개셋들 | RGB bbox | 다수 |

결함 클래스(핫스팟, 오염, 크랙)는 **내부 촬영 데이터로 추가 라벨링**이 사실상 필수. Active Learning 루프에서 SAM2+CLIP 이 라벨링 속도를 크게 올려준다.

## HSV 튜닝 메모 (Classical 전용)

패널 색상이 이미지에 따라 달라 오탐/미탐이 나면 `ClassicalSolarPanelDetector.HSV_LOWER/UPPER` 를 조정한다.

- 결정질 실리콘(대부분): `H: 90~130, S: 20~255, V: 20~120` (현재 기본값)
- 박막(검은색에 가까움): `H: 0~180, S: 0~80, V: 0~60` 로 확장
- 흰색 프레임/반사 강한 경우: 에지 기반 접근(Canny + Hough) 병행 추천