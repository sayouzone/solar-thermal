# IR 이미지 YOLO bbox → Georeferenced bbox 변환

DJI ZH20T thermal 이미지(`DJI_20251217130217_0007_T.JPG`)와 YOLO 라벨을
지리 좌표로 변환하는 시스템.

## 실행 방법
```bash
pip install exifread Pillow numpy opencv-python
python scripts/run_georeferencing.py
```

## 데이터 특성
- IR 이미지: 640 × 512 (DJI ZH20T thermal native)
- 카메라: 35mm 환산 58mm
- RTK-GPS 활성, LRF 거리 45.69m
- 라벨: Roboflow 처리 후 (단순 stretch는 정규화 좌표 보존)

## 검증된 결과
4개 패널 박스 변환 시 너비가 일관됨 (2.17~2.30m, 편차 < 50cm) →
georeferencing이 올바르게 동작 중.

## RGB와의 비교

| 항목 | RGB (Zoom) | IR (Thermal) | 비율 |
|---|---|---|---|
| 해상도 | 5184×3888 | 640×512 | 8.1× |
| GSD | 6.74 mm/px | 44.28 mm/px | 6.6× |
| 커버리지 | 915 m² | 640 m² | 0.7× |
| 35mm 환산 f | 47mm | 58mm | 1.23× |

IR이 망원 렌즈(58mm)지만 해상도가 낮아 GSD가 큼.
같은 객체 좌표 정확도는 RGB가 6.6배 우수.

## 핵심 발견

1. **IR과 RGB가 다른 GPS·자세를 보고함** - 같은 비행 시점에도 카메라 모듈별로
   별도 측정. RGB GPS와 IR GPS 사이 약 1.55m 차이 (광학 중심 오프셋 + 짐벌 미세 차이)

2. **단순 stretch는 정규화 좌표를 보존** - Roboflow 등의 stretch 처리가
   비율 변경만 했다면 정규화 (cx, cy, w, h)는 변하지 않음.
   crop, letterbox, rotate 등은 별도 보정 필요.

3. **IR도 동일한 georeferencing 알고리즘 적용 가능** - 광선-평면 교차 방식은
   카메라 모달리티와 무관. 메타데이터(K, R, t)만 IR 전용으로 바꾸면 됨.

## 변환 결과

| # | 클래스 | 지리 중심 | 물리 크기 |
|---|---|---|---|
| 0 | ir_panel_section | (34.7106648, 126.9223274) | 2.21m × 9.88m |
| 1 | ir_panel_section | (34.7106651, 126.9224517) | 2.17m × 12.68m |
| 2 | ir_panel_section | (34.7107051, 126.9223260) | 2.30m × 9.71m |
| 3 | ir_panel_section | (34.7107058, 126.9224511) | 2.19m × 12.72m |
| 4 | ir_anomaly_small  | (34.7106680, 126.9223117) | 1.04m × 0.41m |