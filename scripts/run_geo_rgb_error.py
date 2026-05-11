"""
같은 비행 시점의 RGB(Zoom)와 IR 이미지에서
같은 패널을 georeferencing한 결과 비교
 
목적: RGB Zoom의 좌표 오차 원인 진단
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


def compare_geo(
    rgb_image: str, 
    ir_image: str, 
    rgb_yolo_label: str, 
    ir_yolo_label: str):

    print_section("1. 메타데이터 비교")

    rgb_meta = extract_dji_metadata(rgb_image)
    ir_meta = extract_dji_metadata(ir_image)
    print(f"\n  {'항목':<25} {'RGB Zoom':<22} {'IR Thermal':<22}")
    print(f"  {'-'*25} {'-'*22} {'-'*22}")
    print(f"  {'해상도':<25} {rgb_meta.image_width}×{rgb_meta.image_height:<16} "
          f"{ir_meta.image_width}×{ir_meta.image_height}")
    print(f"  {'35mm 환산 초점':<22} {rgb_meta.focal_length_35mm} mm{'':16} "
          f"{ir_meta.focal_length_35mm} mm")
    print(f"  {'GPS 위도':<25} {rgb_meta.gps_latitude:<22.7f} {ir_meta.gps_latitude:<22.7f}")
    print(f"  {'GPS 경도':<24} {rgb_meta.gps_longitude:<22.7f} {ir_meta.gps_longitude:<22.7f}")
    print(f"  {'절대 고도':<24} {rgb_meta.absolute_altitude:<22.3f} {ir_meta.absolute_altitude:<22.3f}")
    print(f"  {'짐벌 yaw':<24} {rgb_meta.gimbal_yaw_deg:<22.2f} {ir_meta.gimbal_yaw_deg:<22.2f}")
    print(f"  {'짐벌 pitch':<24} {rgb_meta.gimbal_pitch_deg:<22.2f} {ir_meta.gimbal_pitch_deg:<22.2f}")
    print(f"  {'LRF 거리':<24} {rgb_meta.lrf_distance:<22.3f} {ir_meta.lrf_distance:<22.3f}")
    print(f"  {'LRF 타깃 위도':<22} {rgb_meta.lrf_target_lat:<22.7f} {ir_meta.lrf_target_lat:<22.7f}")
    print(f"  {'LRF 타깃 경도':<22} {rgb_meta.lrf_target_lon:<22.7f} {ir_meta.lrf_target_lon:<22.7f}")
 
    common_origin = GeodeticPoint(
        latitude=(rgb_meta.gps_latitude + ir_meta.gps_latitude) / 2,
        longitude=(rgb_meta.gps_longitude + ir_meta.gps_longitude) / 2,
        altitude=(rgb_meta.absolute_altitude + ir_meta.absolute_altitude) / 2,
    )
    rgb_gps_geo = GeodeticPoint(rgb_meta.gps_latitude, rgb_meta.gps_longitude, rgb_meta.absolute_altitude)
    ir_gps_geo = GeodeticPoint(ir_meta.gps_latitude, ir_meta.gps_longitude, ir_meta.absolute_altitude)
    rgb_enu = geodetic_to_enu(rgb_gps_geo, common_origin).to_array()
    ir_enu = geodetic_to_enu(ir_gps_geo, common_origin).to_array()
    gps_diff = np.linalg.norm(rgb_enu[:2] - ir_enu[:2])
    alt_diff = abs(rgb_enu[2] - ir_enu[2])
 
    print(f"\n  → GPS 위치 차이: 수평 {gps_diff:.3f}m, 수직 {alt_diff:.3f}m")
    print(f"  → 두 카메라가 자기 광학 중심 기준으로 별도 측정한 결과")

    print_section("2. RGB Zoom과 IR의 LRF 검증 결과 비교")

    rgb_gr = DJIImageGeoreferencer(rgb_meta)
    ir_gr = DJIImageGeoreferencer(ir_meta)
 
    rgb_lrf = rgb_gr.validate_with_lrf()
    ir_lrf = ir_gr.validate_with_lrf()
 
    print(f"\n  RGB Zoom 이미지 중심 → LRF 측정값과의 오차:")
    print(f"    수평 오차: {rgb_lrf['error_horizontal_m']:.3f} m")
 
    print(f"\n  IR Thermal 이미지 중심 → LRF 측정값과의 오차:")
    print(f"    수평 오차: {ir_lrf['error_horizontal_m']:.3f} m")
 
    print(f"\n  → RGB의 오차가 IR의 {rgb_lrf['error_horizontal_m']/max(ir_lrf['error_horizontal_m'], 0.001):.1f}배")

    print_section("3. 같은 패널을 RGB와 IR에서 보았을 때 좌표 차이")

    rgb_labels = parse_yolo_label_file(rgb_yolo_label, has_confidence=False)
    rgb_panels = [d for d in rgb_labels if d.class_id == 0]
    ir_labels = parse_yolo_label_file(ir_yolo_label, has_confidence=False)
    ir_panels = [d for d in ir_labels if d.class_id == 1]
 
    rgb_panel_geos = []
    for det in rgb_panels:
        geo_det = convert_yolo_to_geo(det, rgb_gr, SOLAR_RGB_CLASSES)
        if geo_det:
            rgb_panel_geos.append(geo_det)
 
    ir_panel_geos = []
    for det in ir_panels:
        geo_det = convert_yolo_to_geo(det, ir_gr, SOLAR_IR_CLASSES)
        if geo_det:
            ir_panel_geos.append(geo_det)
 
    print(f"\n  RGB 패널 검출: {len(rgb_panel_geos)}개")
    print(f"  IR 패널 검출: {len(ir_panel_geos)}개")
 
    matches = []
    used_ir = set()
    for i, rgb_det in enumerate(rgb_panel_geos):
        best_dist = float("inf")
        best_j = -1
        for j, ir_det in enumerate(ir_panel_geos):
            if j in used_ir:
                continue
            rgb_enu = geodetic_to_enu(rgb_det.geo_center, common_origin).to_array()
            ir_enu = geodetic_to_enu(ir_det.geo_center, common_origin).to_array()
            dist = np.linalg.norm(rgb_enu[:2] - ir_enu[:2])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j >= 0 and best_dist < 5.0:
            matches.append((i, best_j, best_dist))
            used_ir.add(best_j)
 
    print(f"\n  매칭된 패널 쌍: {len(matches)}개")
    print(f"\n  {'#':<3} {'RGB 중심':<35} {'IR 중심':<35} {'차이(m)':<10}")
    print(f"  {'-'*3} {'-'*35} {'-'*35} {'-'*10}")
    distances = []
    for rgb_idx, ir_idx, dist in matches:
        rgb_geo = rgb_panel_geos[rgb_idx].geo_center
        ir_geo = ir_panel_geos[ir_idx].geo_center
        rgb_str = f"({rgb_geo.latitude:.7f}, {rgb_geo.longitude:.7f})"
        ir_str = f"({ir_geo.latitude:.7f}, {ir_geo.longitude:.7f})"
        print(f"  {rgb_idx:<3} {rgb_str:<35} {ir_str:<35} {dist:<10.3f}")
        distances.append(dist)
 
    if distances:
        print(f"\n  평균 거리 차이: {np.mean(distances):.3f}m")
        print(f"  최대 거리 차이: {np.max(distances):.3f}m")
        print(f"  최소 거리 차이: {np.min(distances):.3f}m")
    
    print_section("4. 패널 박스 너비의 일관성 비교")

    rgb_widths = [d.width_meters for d in rgb_panel_geos]
    ir_widths = [d.width_meters for d in ir_panel_geos]
    rgb_heights = [d.height_meters for d in rgb_panel_geos]
    ir_heights = [d.height_meters for d in ir_panel_geos]
 
    print(f"\n  RGB 패널 박스 너비:")
    print(f"    개별 값: {[f'{w:.2f}' for w in rgb_widths]}")
    print(f"    평균: {np.mean(rgb_widths):.2f}m, 표준편차: {np.std(rgb_widths):.3f}m")
 
    print(f"\n  IR 패널 박스 너비:")
    print(f"    개별 값: {[f'{w:.2f}' for w in ir_widths]}")
    print(f"    평균: {np.mean(ir_widths):.2f}m, 표준편차: {np.std(ir_widths):.3f}m")

    print_section("5. 진단 결론")

    K_rgb, _ = estimate_intrinsics_from_metadata(rgb_meta)
    K_ir, _ = estimate_intrinsics_from_metadata(ir_meta)
    
    print(f"\n  추정된 K (EXIF 기반):")
    print(f"    RGB: fx = {K_rgb[0,0]:.1f} pixels (35mm 환산 {rgb_meta.focal_length_35mm}mm 기반)")
    print(f"    IR:  fx = {K_ir[0,0]:.1f} pixels (35mm 환산 {ir_meta.focal_length_35mm}mm 기반)")
    
    rgb_gsd = rgb_gr.compute_ground_sample_distance()
    ir_gsd = ir_gr.compute_ground_sample_distance()
    print(f"\n  계산된 GSD:")
    print(f"    RGB: {rgb_gsd*1000:.2f} mm/pixel")
    print(f"    IR:  {ir_gsd*1000:.2f} mm/pixel")
    
    print(f"\n  추정 원인:")
    print(f"    1. RGB Zoom의 35mm 환산 초점거리(47mm)는 EXIF 기록값")
    print(f"       하지만 줌 카메라는 실제 줌 위치에 따라 변동 가능")
    print(f"    2. EXIF 기반 K는 광학 중심을 이미지 중심으로 가정")
    print(f"       실제 주점은 다를 수 있음 (수십 픽셀)")
    print(f"    3. 광각 왜곡 계수가 0으로 추정됨")
    print(f"       Zoom은 망원이라 영향 작지만 0이 아님")
    print(f"    4. RGB 카메라의 광학 중심과 GPS 안테나 위치 오프셋이 IR과 다름")

def main():
    rgb_path = "data/solar/images/RGB/DJI_20251217130217_0007_Z.JPG"
    ir_path = "data/solar/images/TM/DJI_20251217130217_0007_T.JPG"
    rgb_yolo_label = "workspace/labels_s100_m_d/DJI_20251217130217_0007_Z.txt"
    ir_yolo_label = "workspace/ir_rf/DJI_20251217130217_0007_T_JPG.rf.a9de1d4093d48a9aff0dcb47ad3fb589.txt"
    output_path = "workspace/claude/output"

    compare_geo(rgb_image=rgb_path, ir_image=ir_path, rgb_yolo_label=rgb_yolo_label, ir_yolo_label=ir_yolo_label)

if __name__ == "__main__":
    main()
