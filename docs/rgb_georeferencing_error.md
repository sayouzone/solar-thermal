# RGB Zoom Georeferencing 오차 진단 및 보정

DJI ZH20T Zoom 카메라의 georeferencing 오차를 진단하고 보정하는 종합 시스템.

## 진단 결과

업로드 데이터 분석:

| 측정 항목 | 값 | 평가 |
|---|---|---|
| RGB Zoom 자체 LRF 오차 | 0.241m | 우수 |
| IR 자체 LRF 오차 | 0.336m | 우수 |
| RGB-IR 같은 패널 좌표 차이 | 평균 3.65m | **큰 오차** |
| RGB GPS vs IR GPS 차이 | 0.745m | 카메라 위치 오프셋 |
| RGB LRF vs IR LRF 타깃 차이 | 1.392m | 광축 차이 |

**핵심 결론**: RGB Zoom 자체는 정확합니다(LRF 오차 0.24m). 문제는 **RGB-IR 결합 시** 발생하는 systematic offset입니다.

## 발견한 오차 패턴

이미지 위치 기반 정확한 매칭 결과:

| RGB 위치 | IR 위치 | ΔE (m) | ΔN (m) |
|---|---|---|---|
| (0.79, 0.74) | (0.73, 0.72) | -0.43 | -3.58 |
| (0.79, 0.25) | (0.73, 0.22) | +1.00 | -3.60 |
| (0.92, 0.74) | (0.89, 0.72) | -0.52 | -3.52 |
| (0.92, 0.25) | (0.89, 0.21) | +0.92 | -3.58 |

**ΔN은 -3.58m로 일정** (표준편차 0.03m) → 평행 이동
**ΔE는 cy에 따라 부호 반전** → 회전 차이 존재

이는 두 카메라의 광축이 yaw 또는 roll 방향으로 미세하게 어긋나 있어서,
이미지 위쪽과 아래쪽이 반대 부호의 횡 오프셋을 만드는 것입니다.

## 5가지 해결 방안

### 방안 1: 정밀 카메라 캘리브레이션 (가장 효과적)
체커보드로 K, D를 정확히 측정. EXIF 추정 대비 50% 이상 오차 감소 기대.
별도 비행 1회 + OpenCV calibrateCamera 필요.

### 방안 2: LRF 자동 보정
DJI가 메타데이터에 포함한 LRFTargetLat/Lon을 ground truth로 활용.
**즉시 적용 가능**. RGB 자체 좌표 정확도 0.24m 수준 유지.

### 방안 3: 알려진 패널 크기로 fx 추정
실제 패널 너비(보통 1m)를 ground truth로 fx 역추정.
캘리브레이션 데이터 없을 때 차선책.

### 방안 4: RGB-IR 광학 중심 오프셋 캘리브레이션
**RGB-IR 결합에 가장 중요**. 한 번 측정 후 모든 비행에 적용.
단순 평행 이동(이번 케이스)이면 (E=0.25, N=-3.58) 적용.
회전까지 있으면 affine transformation으로 일반화 필요.

### 방안 5: 가장자리 검출 신뢰도 감소
이미지 가장자리는 광학 왜곡 영향이 큼 → 검출 confidence 자동 감점.
방안 1이 적용되면 효과 작아지지만, 그 전엔 유용한 안전장치.

## 사용법

### 즉시 적용 (캘리브레이션 없이)
```python
from src.error_correction import CalibratedGeoreferencer

rgb_gr = CalibratedGeoreferencer(
    rgb_metadata,
    auto_lrf_correction=True,    # 방안 2
    edge_confidence_decay=True,  # 방안 5
)

geo_det = rgb_gr.convert_yolo_to_geo_corrected(yolo_det, class_names)
```

### RGB-IR 통합 (방안 4)
```python
from src.error_correction import estimate_camera_offset_from_pairs

# 첫 비행에서 한 번 측정
offset = estimate_camera_offset_from_pairs(
    rgb_meta, ir_meta,
    rgb_panel_centers, ir_panel_centers,
    common_origin,
)

# 모든 비행에 적용
ir_detection_corrected = ir_detection.apply_offset(-offset)
```

### 캘리브레이션 후 (방안 1)
```python
from src.error_correction import CameraCalibration

calibration = CameraCalibration.from_yaml("h20t_zoom_calib.yaml")
rgb_gr = CalibratedGeoreferencer(
    rgb_metadata,
    calibration=calibration,
    auto_lrf_correction=True,
)
```

## 권장 우선순위

| 우선순위 | 방안 | 비용 | 효과 |
|---|---|---|---|
| 1 | 방안 4 (RGB-IR 오프셋 측정) | 0 (한 번만) | RGB-IR 차이 크게 감소 |
| 2 | 방안 2 (LRF 자동 보정) | 0 | 매 사진 자동 적용 |
| 3 | 방안 5 (가장자리 신뢰도) | 0 | 잘못된 검출 필터링 |
| 4 | 방안 1 (정밀 캘리브레이션) | 1일 작업 | 가장 큰 정확도 개선 |
| 5 | 방안 3 (패널 크기 fx 추정) | 검증 데이터 필요 | 차선책 |

## 운영에 미치는 영향

이번 데이터 기준:
- **RGB만 사용**: 셀 단위(156mm) 식별 가능, LRF 정확도 24cm
- **RGB+IR 결합 (보정 없음)**: 같은 패널이 3.65m 차이로 인식됨, 셀 매칭 불가
- **RGB+IR 결합 (방안 4 적용)**: 1m 이내 차이로 줄어들어 패널 단위 매칭 가능
- **RGB+IR 결합 (방안 1+4)**: 30cm 이내, 셀 단위 매칭 가능