# DJI 드론 단일 이미지 Georeferencing

업로드된 DJI Zenmuse H20T 사진(`DJI_20251217130206_0003_Z.JPG`)을 사용한 georeferencing 구현.

## 실행 방법
```bash
pip install exifread Pillow numpy opencv-python
python scripts/run_georeferencing.py
```

## 검증 결과 (실제 데이터 기반)
- Nadir 정렬: 0.100° (거의 완벽)
- 이미지 중심 ↔ 드론 GPS: 0.080m 오프셋
- LRF 측정값과의 수평 오차: 0.229m
- GSD: 6.74 mm/pixel
- 커버리지: 914.8 m²

## 모듈 구조
- `src/solar_thermal/georeferencing/dji/metadata.py`: EXIF + XMP 추출 (DJI 자세, RTK, LRF)
- `src/solar_thermal/georeferencing/dji/coordinates.py`: WGS84 ↔ ECEF ↔ ENU 변환
- `src/solar_thermal/georeferencing/dji/camera_pose.py`: DJI 짐벌 자세 → 카메라 좌표계
- `src/solar_thermal/georeferencing/dji/georeferencer.py`: 메인 georeferencer
- `scripts/run_georeferencing.py`: 실행 스크립트

## 핵심 개선 사항
이전 답변의 일반론적 코드를 실제 DJI 데이터에 맞게 변경:

1. **XMP 메타데이터 직접 파싱**: drone-dji 네임스페이스의 짐벌·드론·RTK·LRF 데이터 추출
2. **DJI 짐벌 자세 직접 변환**: 회전 행렬 곱셈 대신 광축 단위벡터 직접 계산 (수치 안정)
3. **LRF 자동 활용**: 지면 고도를 LRFTargetAbsAlt로 자동 설정
4. **LRF 검증**: DJI가 측정한 타깃 좌표와 우리 계산 비교로 정확도 자동 평가
5. **EXIF 기반 K 추정**: 캘리브레이션 없을 때 fallback