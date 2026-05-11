"""
오차 보정 적용 전후 비교
"""
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solar_thermal.georeferencing.dji.metadata import DJIMetadata, extract_dji_metadata, estimate_intrinsics_from_metadata
from solar_thermal.georeferencing.dji.georeferencer import DJIImageGeoreferencer
from solar_thermal.georeferencing.dji.coordinates import geodetic_to_enu, GeodeticPoint
from solar_thermal.georeferencing.yolo_to_geo import (
    YOLODetection,
    GeoreferencedDetection,
    parse_yolo_label_file,
    convert_yolo_to_geo,
    convert_yolo_file_to_geo,
    export_to_geojson,
    export_to_csv,
    _compute_polygon_area_m2,
)
from solar_thermal.georeferencing.dji.error_correction import (
    CalibratedGeoreferencer,
    estimate_camera_offset_from_pairs,
)

SOLAR_RGB_CLASSES = {
    0: "panel_string",
    1: "panel",
    2: "non_panel",
    3: "anomaly",
}

SOLAR_IR_CLASSES = {
    0: "ir_anomaly",
    1: "ir_panel",
}

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def test_corrections(
    rgb_image: str, 
    ir_image: str, 
    rgb_yolo_label: str, 
    ir_yolo_label: str):

    print_section("1. 메타데이터 비교")

    rgb_meta = extract_dji_metadata(rgb_image)
    ir_meta = extract_dji_metadata(ir_image)
 
    print_section("BEFORE: 보정 없는 원본 georeferencing")

    rgb_gr_raw = DJIImageGeoreferencer(rgb_meta)
    ir_gr_raw = DJIImageGeoreferencer(ir_meta)

    rgb_labels = parse_yolo_label_file(rgb_yolo_label, has_confidence=False)
    ir_labels = parse_yolo_label_file(ir_yolo_label, has_confidence=False)
 
    rgb_panels_raw = []
    for det in rgb_labels:
        if det.class_id != 0:
            continue
        geo_det = convert_yolo_to_geo(det, rgb_gr_raw, SOLAR_RGB_CLASSES)
        if geo_det:
            rgb_panels_raw.append(geo_det)
 
    ir_panels_raw = []
    for det in ir_labels:
        if det.class_id != 1:
            continue
        geo_det = convert_yolo_to_geo(det, ir_gr_raw, SOLAR_IR_CLASSES)
        if geo_det:
            ir_panels_raw.append(geo_det)
 
    rgb_widths_raw = [d.width_meters for d in rgb_panels_raw]
    print(f"\n  RGB 패널 너비 (보정 전):")
    print(f"    값: {[f'{w:.2f}' for w in rgb_widths_raw]} m")
    print(f"    평균: {np.mean(rgb_widths_raw):.2f}m, 표준편차: {np.std(rgb_widths_raw):.3f}m")
 
    common_origin = GeodeticPoint(
        latitude=(rgb_meta.gps_latitude + ir_meta.gps_latitude) / 2,
        longitude=(rgb_meta.gps_longitude + ir_meta.gps_longitude) / 2,
    )
 
    matches_raw = []
    for rgb_p in rgb_panels_raw:
        rgb_enu = geodetic_to_enu(rgb_p.geo_center, common_origin).to_array()[:2]
        best = float("inf")
        for ir_p in ir_panels_raw:
            ir_enu = geodetic_to_enu(ir_p.geo_center, common_origin).to_array()[:2]
            d = np.linalg.norm(rgb_enu - ir_enu)
            if d < best:
                best = d
        if best < 5.0:
            matches_raw.append(best)
 
    print(f"\n  RGB-IR 같은 패널 좌표 차이 (보정 전):")
    print(f"    평균: {np.mean(matches_raw):.3f}m")
    print(f"    최대: {np.max(matches_raw):.3f}m")
 
    rgb_lrf_raw = rgb_gr_raw.validate_with_lrf()
    print(f"\n  RGB LRF 오차 (보정 전): {rgb_lrf_raw['error_horizontal_m']:.3f}m")
 
    print_section("AFTER 방안 2: LRF 자동 보정")
 
    rgb_gr_corrected = CalibratedGeoreferencer(
        rgb_meta,
        auto_lrf_correction=True,
        edge_confidence_decay=False,
    )
    
    ir_gr_corrected = CalibratedGeoreferencer(
        ir_meta,
        auto_lrf_correction=True,
        edge_confidence_decay=False,
    )
 
    rgb_correction = rgb_gr_corrected.get_correction_summary()
    print(f"\n  RGB 적용된 보정:")
    print(f"    East offset: {rgb_correction.east_offset_m:.3f}m")
    print(f"    North offset: {rgb_correction.north_offset_m:.3f}m")
    print(f"    Method: {rgb_correction.method}")
 
    rgb_panels_corrected = []
    for det in rgb_labels:
        if det.class_id != 0:
            continue
        geo_det = rgb_gr_corrected.convert_yolo_to_geo_corrected(det, SOLAR_RGB_CLASSES)
        if geo_det:
            rgb_panels_corrected.append(geo_det)
 
    ir_panels_corrected = []
    for det in ir_labels:
        if det.class_id != 1:
            continue
        geo_det = ir_gr_corrected.convert_yolo_to_geo_corrected(det, SOLAR_IR_CLASSES)
        if geo_det:
            ir_panels_corrected.append(geo_det)
 
    matches_corrected = []
    for rgb_p in rgb_panels_corrected:
        rgb_enu = geodetic_to_enu(rgb_p.geo_center, common_origin).to_array()[:2]
        best = float("inf")
        for ir_p in ir_panels_corrected:
            ir_enu = geodetic_to_enu(ir_p.geo_center, common_origin).to_array()[:2]
            d = np.linalg.norm(rgb_enu - ir_enu)
            if d < best:
                best = d
        if best < 5.0:
            matches_corrected.append(best)
 
    print(f"\n  RGB-IR 같은 패널 좌표 차이 (보정 후):")
    print(f"    평균: {np.mean(matches_corrected):.3f}m")
    print(f"    최대: {np.max(matches_corrected):.3f}m")
 
    improvement_pct = (np.mean(matches_raw) - np.mean(matches_corrected)) / np.mean(matches_raw) * 100
    print(f"\n  개선율: {improvement_pct:.1f}%")
 
    print_section("AFTER 방안 4: RGB-IR 광학 중심 오프셋 보정")
 
    rgb_centers_raw = [p.geo_center for p in rgb_panels_raw]
    ir_centers_raw = [p.geo_center for p in ir_panels_raw]
    
    cam_offset = estimate_camera_offset_from_pairs(
        rgb_meta, ir_meta,
        rgb_centers_raw, ir_centers_raw,
        common_origin,
    )
    
    print(f"\n  추정된 RGB→IR 오프셋:")
    print(f"    East: {cam_offset['east_offset_m']:.3f}m")
    print(f"    North: {cam_offset['north_offset_m']:.3f}m")
    print(f"    매칭 쌍 수: {cam_offset['n_pairs']}")
    print(f"    신뢰도: {cam_offset['confidence']:.3f}")
    
    print(f"\n  → 이 오프셋을 IR 좌표에 빼주면 RGB와 같은 좌표계가 됨")
    print(f"  → 또는 RGB 좌표에 더해주면 IR과 같은 좌표계가 됨")
 
    print_section("종합 검증: 각 방안의 효과")
 
    print(f"\n  {'시나리오':<35} {'평균 차이(m)':<15} {'최대 차이(m)':<15}")
    print(f"  {'-'*35} {'-'*15} {'-'*15}")
    print(f"  {'1) 보정 없음 (원본)':<33} {np.mean(matches_raw):<15.3f} {np.max(matches_raw):<15.3f}")
    print(f"  {'2) 방안 2 (LRF 보정)':<33} {np.mean(matches_corrected):<15.3f} {np.max(matches_corrected):<15.3f}")
    
    rgb_panels_with_offset = []
    east_off = cam_offset["east_offset_m"]
    north_off = cam_offset["north_offset_m"]
    
    matches_method4 = []
    for rgb_p in rgb_panels_corrected:
        rgb_enu = geodetic_to_enu(rgb_p.geo_center, common_origin).to_array()[:2]
        rgb_enu_shifted = rgb_enu + np.array([east_off, north_off])
        best = float("inf")
        for ir_p in ir_panels_corrected:
            ir_enu = geodetic_to_enu(ir_p.geo_center, common_origin).to_array()[:2]
            d = np.linalg.norm(rgb_enu_shifted - ir_enu)
            if d < best:
                best = d
        if best < 5.0:
            matches_method4.append(best)
    
    if matches_method4:
        print(f"  {'3) 방안 2 + 4 (LRF + 카메라 오프셋)':<33} {np.mean(matches_method4):<15.3f} {np.max(matches_method4):<15.3f}")
 
    print_section("권장 사용법")
    print("""
  즉시 적용 가능 (캘리브레이션 없이):
    rgb_gr = CalibratedGeoreferencer(
        rgb_metadata,
        auto_lrf_correction=True,        # 방안 2: 매 사진 자동 보정
        edge_confidence_decay=True,      # 방안 5: 가장자리 신뢰도 감소
    )
    
    geo_det = rgb_gr.convert_yolo_to_geo_corrected(yolo_det, class_names)
 
  캘리브레이션 후 (방안 1):
    calibration = CameraCalibration.from_yaml("h20t_zoom_calib.yaml")
    rgb_gr = CalibratedGeoreferencer(
        rgb_metadata,
        calibration=calibration,
        auto_lrf_correction=True,        # 캘리브레이션 + LRF 결합 시 최대 효과
    )
 
  RGB-IR 정합 (방안 4):
    offset = estimate_camera_offset_from_pairs(
        rgb_meta, ir_meta,
        rgb_panel_centers, ir_panel_centers,
        common_origin,
    )
    # offset을 IR 좌표에 적용하거나 RGB에 역으로 적용
    """)

def main():
    rgb_path = "data/solar/images/RGB/DJI_20251217130217_0007_Z.JPG"
    ir_path = "data/solar/images/TM/DJI_20251217130217_0007_T.JPG"
    rgb_yolo_label = "workspace/labels_s100_m_d/DJI_20251217130217_0007_Z.txt"
    ir_yolo_label = "workspace/ir_rf/DJI_20251217130217_0007_T_JPG.rf.a9de1d4093d48a9aff0dcb47ad3fb589.txt"
    output_path = "workspace/claude/output"

    test_corrections(rgb_image=rgb_path, ir_image=ir_path, rgb_yolo_label=rgb_yolo_label, ir_yolo_label=ir_yolo_label)

if __name__ == "__main__":
    main()
