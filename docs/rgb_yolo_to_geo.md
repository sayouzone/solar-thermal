# YOLO bbox → Georeferenced bbox 변환

업로드된 YOLO 라벨 파일과 DJI 사진을 결합해 지리 좌표로 변환하는 시스템.

## 실행 방법
```bash
pip install exifread Pillow numpy opencv-python
python scripts/run_georeferencing.py
```

## 핵심 모듈

### src/yolo_to_geo.py
YOLO 라벨을 georeferenced 형식으로 변환하는 핵심 모듈.

주요 클래스/함수:
- `YOLODetection`: YOLO 형식 dataclass (정규화 검증 포함)
- `parse_yolo_label_file()`: 라벨 파일 파싱 (학습용/추론 결과 모두)
- `GeoreferencedDetection`: 지리 좌표로 변환된 결과
- `convert_yolo_to_geo()`: 단일 검출 변환
- `convert_yolo_file_to_geo()`: 파일 단위 batch 변환
- `export_to_geojson()`, `export_to_csv()`: 다양한 출력 형식

## 변환 결과 예시 (실제 데이터)

7개 YOLO 라벨이 성공적으로 변환되었습니다:

| # | 클래스 | YOLO (cx, cy) | 픽셀 bbox | 물리 크기 |
|---|---|---|---|---|
| 0 | non_panel_object | (0.420, 0.683) | (1908, 1463, 2444, 3850) | 3.6m × 16.1m |
| 1 | panel_top | (0.662, 0.252) | (3188, 0, 3674, 1960) | 3.3m × 13.2m |
| 2 | panel_bottom | (0.664, 0.733) | (3215, 1871, 3671, 3831) | 3.1m × 13.2m |
| 3 | panel_top | (0.792, 0.252) | (3862, 0, 4350, 1962) | 3.3m × 13.2m |
| 4 | panel_bottom | (0.797, 0.735) | (3914, 1876, 4344, 3840) | 2.9m × 13.2m |
| 5 | panel_top | (0.920, 0.252) | (4514, 0, 5021, 1958) | 3.4m × 13.2m |
| 6 | panel_bottom | (0.923, 0.740) | (4565, 1870, 5010, 3887) | 3.0m × 13.6m |

## 주요 설계 결정

1. **정규화 검증**: 입력 시 [0,1] 범위 자동 검증으로 픽셀 좌표 혼동 방지
2. **이미지 경계 클리핑**: 박스가 이미지를 벗어나면 자동 자름
3. **4개 모서리 보존**: 카메라 기울기로 사다리꼴이 될 수 있어 단순 lat/lon 쌍이 아닌 4점 보존
4. **물리 크기 자동 계산**: 픽셀 → 미터 변환으로 운영 의미 부여
5. **가장자리 검출 플래그**: 정합 정확도가 떨어지는 영역 자동 식별
6. **다양한 출력**: GeoJSON (시각화), CSV (작업지시), JSON (시스템 통합)

## 운영 활용

### GeoJSON
[geojson.io](https://geojson.io) 또는 QGIS에서 위성 지도 위에 박스 시각화

### CSV
Excel 또는 작업지시 시스템에 직접 import 가능
- center_lat, center_lon: 작업자 GPS 안내용
- 4개 모서리: 정확한 영역 식별
- panel_id, cell_row, cell_col: 발전소 마스터 데이터 매칭 (선택)
