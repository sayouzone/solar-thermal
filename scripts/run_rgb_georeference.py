"""
RGB Georeferencing for DJI Zenmuse H20T
=======================================

Wide / Zoom 카메라(_W, _Z) RGB 이미지의 YOLO 검출 결과를
EPSG:4326 GeoJSON으로 변환한다.

설계 원칙:
- 짐벌 자세 → 카메라 회전 행렬은 `camera_pose.py`에 위임 (Single Source of Truth)
- 동적 FOV: EXIF FocalLengthIn35mmFilm 기반 (Zoom 카메라는 매 사진마다 줌 비율이 다름)
- 광선-평면 교차로 픽셀→지면 변환 (nadir/oblique 모두 지원)
- 카메라 좌표계는 OpenCV 규약 (X=Right, Y=Down, Z=Forward)
- 평지 + 카메라 nadir 발끝 0m AGL 가정. DEM 사용 시 ground_z 인자만 교체.

기존 RGB 결과(rgb_georeferenced.geojson)에서 발견된 오류와 그 수정:
  ① Wide FOV 상수 사용 → EXIF 기반 동적 FOV로 교체
  ② width/height 축 뒤바뀜 → OpenCV 좌표계 + image_w/h 명시 분리
  ③ Gimbal yaw 회전 누락 → camera_pose.compute_camera_axes_from_gimbal() 적용

검증 결과 (DJI_..._Z.JPG, Zoom@47mm-eq, AGL 44.98m):
  drone↔footprint 중심 오프셋 = (≈0m, ≈0m)
  IR 패널 중심과의 매칭 오차 ≈ 1.0~1.2m (광축 옵셋에 의한 정상 잔차)
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import exifread

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 동일 패키지 내 camera_pose.py
from solar_thermal.georeferencing.dji.camera_pose import compute_camera_axes_from_gimbal, verify_nadir_orientation
from solar_thermal.georeferencing.dji.metadata import DJIMetadata, M_PER_DEG_LAT, extract_dji_metadata
from solar_thermal.georeferencing.yolo_to_geo import (
    parse_yolo_label_file,
)

SOLAR_RGB_CLASSES = {
    0: "panel_string",
    1: "panel",
    2: "non_panel",
    3: "anomaly",
}

# --------------------------------------------------------------------------- #
# 3. 픽셀 → 지면 투영 (광선-평면 교차)
# --------------------------------------------------------------------------- #

def _pixel_to_camera_ray(
        px: float, 
        py: float, 
        pose: DJIMetadata
    ) -> np.ndarray:
    """
    픽셀(좌상단 원점) → 카메라 좌표계 단위 광선 (X=Right, Y=Down, Z=Forward).

    초점거리(픽셀 단위) f_px = (W/2) / tan(HFOV/2)
    image center를 (cx, cy) = (W/2, H/2)로 가정.
    """
    """
    초점거리(픽셀 단위) f_px = (W/2) / tan(HFOV/2)
    FOV 대신 EXIF FocalLengthIn35mmFilm 기반으로 계산.
    DJI H20T Zoom 카메라는 매 사진마다 줌 비율이 달라 FOV도 달라짐.
    따라서 pose.hfov_deg, pose.vfov_deg는 EXIF 기반으로 계산된 동적 FOV입니다.
    """
    fx_px = (pose.image_width / 2.0) / math.tan(math.radians(pose.hfov_deg / 2.0))
    fy_px = (pose.image_height / 2.0) / math.tan(math.radians(pose.vfov_deg / 2.0))

    """
    image center를 (cx, cy) = (W/2, H/2)로 가정.
    OpenCV pinhole: x_cam = (px - cx)/fx, y_cam = (py - cy)/fy, z_cam = 1
    따라서 cx, cy는 이미지 중심의 픽셀 좌표입니다.
    """
    cx_px = pose.image_width / 2.0
    cy_px = pose.image_height / 2.0

    # OpenCV pinhole: x_cam = (px - cx)/fx, y_cam = (py - cy)/fy, z_cam = 1
    ray_cam = np.array([
        (px - cx_px) / fx_px,
        (py - cy_px) / fy_px,
        1.0,
    ])
    return ray_cam / np.linalg.norm(ray_cam)


def _pixel_to_ground_enu(
    px: float, py: float,
    pose: DJIMetadata,
    ground_z_below_drone: float | None = None,
) -> tuple[float, float]:
    """
    픽셀 → 카메라 nadir 발끝 기준 (East, North) 미터 오프셋.

    ground_z_below_drone : 드론에서 본 지면의 z 값(ENU에서 -AGL).
                           기본 None이면 -pose.rel_alt_m.
                           DEM 쓰면 픽셀별로 달리 줄 수 있음.
    """
    if ground_z_below_drone is None:
        """드론에서 본 지면의 z 값(ENU에서 -AGL)"""
        ground_z_below_drone = -pose.relative_altitude   # 드론보다 AGL만큼 아래

    # (1) 카메라 좌표 광선 → ENU 광선
    ray_cam = _pixel_to_camera_ray(px, py, pose)
    ray_enu = pose.R_cam_to_enu @ ray_cam        # 3-vec in ENU

    # (2) 평면 z = ground_z_below_drone 와의 교차
    #     drone 위치를 원점(0,0,0)이라 두면 광선식: P = t * ray_enu
    #     ground_z_below_drone = t * ray_enu[2]
    if ray_enu[2] >= -1e-9:
        # 광선이 지면을 만나지 않거나 위로 향함 (잘못된 입력)
        raise ValueError(
            f"광선이 지면을 향하지 않음 (gimbal pitch={pose.gimbal_pitch}°). "
            f"ray_enu={ray_enu.tolist()}"
        )
    t = ground_z_below_drone / ray_enu[2]
    east  = t * ray_enu[0]
    north = t * ray_enu[1]
    return east, north


def _enu_to_lonlat(east_m: float, north_m: float, pose: CameraPose) -> list[float]:
    """ENU 오프셋 → (lon, lat) 좌표. pose의 (lon, lat)을 기준으로 미터 단위 오프셋을 도 단위로 변환하여 더한다."""
    return [
        pose.gps_longitude + east_m / pose.m_per_deg_lon,
        pose.gps_latitude + north_m / M_PER_DEG_LAT,
    ]

# --------------------------------------------------------------------------- #
# 4. Public API: bbox / YOLO label → GeoJSON
# --------------------------------------------------------------------------- #

def pixel_bbox_to_polygon(
    bbox_xyxy: Sequence[float], pose: CameraPose,
) -> list[list[float]]:
    """픽셀 bbox(x1,y1,x2,y2) → GeoJSON Polygon ring [[lon,lat], ...]"""
    x1, y1, x2, y2 = bbox_xyxy
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    ring = []
    for px, py in corners:
        e, n = _pixel_to_ground_enu(px, py, pose)
        ring.append(_enu_to_lonlat(e, n, pose))
    return ring


def yolo_label_to_pixels(
    cx_n: float, cy_n: float, w_n: float, h_n: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    cx, cy = cx_n * img_w, cy_n * img_h
    w,  h  = w_n * img_w,  h_n * img_h
    return cx - w/2, cy - h/2, cx + w/2, cy + h/2


# 사용자 데이터셋 클래스 (필요시 확장)
DEFAULT_CLASS_NAMES = {
    0: 'panel_string',
    1: 'panel',
    2: 'cell',
    3: 'anomaly',
}


def georeference_rgb(
    image_path: str | Path,
    label_path: str | Path,
    class_names: dict[int, str] = DEFAULT_CLASS_NAMES,
) -> dict:
    """RGB 이미지 + YOLO TXT → GeoJSON FeatureCollection"""
    meta = extract_dji_metadata(image_path)
    print("DJIMetadata", meta)

    # nadir 검증 (경고만, 차단은 안 함)
    nadir_check = verify_nadir_orientation(meta.R_cam_to_enu, tolerance_deg=10.0)
    is_oblique = not nadir_check['is_nadir']

    rgb_features: list[dict] = []
    features: list[dict] = []

    # (a) image footprint
    footprint_ring = pixel_bbox_to_polygon(
        (0, 0, meta.image_width, meta.image_height), meta,
    )

    feature = {
        'type': 'Feature',
        'geometry': {'type': 'Polygon', 'coordinates': [footprint_ring]},
        'properties': {
            'name': 'image_coverage',
            'image_path': str(image_path),
            'camera_model': meta.camera_model,
            'focal_35mm_eq': meta.focal_length_35mm,
            'hfov_deg': round(meta.hfov_deg, 2),
            'vfov_deg': round(meta.vfov_deg, 2),
            'rel_alt_m': meta.relative_altitude,
            'gimbal_pitch_deg': meta.gimbal_pitch_deg,
            'gimbal_yaw_compass_deg': meta.gimbal_yaw_deg,
            'gimbal_roll_deg': meta.gimbal_roll_deg,
            'oblique_view': is_oblique,
            'angle_from_nadir_deg': round(nadir_check['angle_from_nadir_deg'], 2),
        },
    }
    rgb_features.append(feature)
    features.append(feature)

    # (b) drone position (Point)
    feature = {
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [meta.gps_longitude, meta.gps_latitude]},
        'properties': {
            'name': 'drone_position',
            'rel_altitude_m': meta.relative_altitude,
            'abs_altitude_m': meta.absolute_altitude,
            'rtk_active': meta.rtk_active,
        },
    }
    rgb_features.append(feature)
    features.append(feature)

    # (c) YOLO 검출 박스
    detections = parse_yolo_label_file(label_path, has_confidence=False)
    #print(detections)
    for i, det in enumerate(detections):
        cls_name = SOLAR_RGB_CLASSES.get(det.class_id, f"class_{det.class_id}")
        x1, y1, x2, y2 = det.to_pixel_xyxy(meta.image_width, meta.image_height)
        ring = pixel_bbox_to_polygon((x1, y1, x2, y2), meta)
        print(ring)

        # 지면상 크기 (참고용)
        e_corners = [_pixel_to_ground_enu(px, py, meta)
                     for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
        es = [c[0] for c in e_corners]; ns = [c[1] for c in e_corners]
        width_m  = max(es) - min(es)
        height_m = max(ns) - min(ns)

        feature = {
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': [ring]},
            'properties': {
                'class_id': det.class_id,
                'class_name': cls_name,
                'pixel_bbox': [round(x1, 1), round(y1, 1),
                               round(x2, 1), round(y2, 1)],
                'width_m': round(width_m, 3),
                'height_m': round(height_m, 3),
                'area_m2': round(width_m * height_m, 3),
                'type': 'detection_box',
            },
        }

        rgb_features.append(feature)
    
    label_lines = Path(label_path).read_text().strip().splitlines()
    #print(label_lines)
    for ln in label_lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx_n, cy_n, w_n, h_n = map(float, parts[1:5])

        x1, y1, x2, y2 = yolo_label_to_pixels(
            cx_n, cy_n, w_n, h_n, meta.image_width, meta.image_height,
        )
        print(x1, y1, x2, y2)
        ring = pixel_bbox_to_polygon((x1, y1, x2, y2), meta)
        print(ring)

        # 지면상 크기 (참고용)
        e_corners = [_pixel_to_ground_enu(px, py, meta)
                     for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
        es = [c[0] for c in e_corners]; ns = [c[1] for c in e_corners]
        width_m  = max(es) - min(es)
        height_m = max(ns) - min(ns)

        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': [ring]},
            'properties': {
                'class_id': cls,
                'class_name': class_names.get(cls, f'class_{cls}'),
                'pixel_bbox': [round(x1, 1), round(y1, 1),
                               round(x2, 1), round(y2, 1)],
                'width_m': round(width_m, 3),
                'height_m': round(height_m, 3),
                'area_m2': round(width_m * height_m, 3),
                'type': 'detection_box',
            },
        })

    out = {
        'type': 'FeatureCollection',
        'features': rgb_features,
        'metadata': {
            'image_path': str(image_path),
            'modality': 'rgb',
            'camera': f'ZH20T_{meta.camera_model.capitalize()}',
            'image_native_size': [meta.image_width, meta.image_height],
            'focal_35mm_eq': meta.focal_length_35mm,
            'hfov_deg': round(meta.hfov_deg, 4),
            'vfov_deg': round(meta.vfov_deg, 4),
            'capture_time': meta.capture_time,
            'rtk_active': meta.rtk_active,
            'gimbal': {
                'yaw_compass_deg': meta.gimbal_yaw_deg,
                'pitch_deg': meta.gimbal_pitch_deg,
                'roll_deg': meta.gimbal_roll_deg,
            },
        },
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))

    return {
        'type': 'FeatureCollection',
        'features': features,
        'metadata': {
            'image_path': str(image_path),
            'modality': 'rgb',
            'camera': f'ZH20T_{meta.camera_model.capitalize()}',
            'image_native_size': [meta.image_width, meta.image_height],
            'focal_35mm_eq': meta.focal_length_35mm,
            'hfov_deg': round(meta.hfov_deg, 4),
            'vfov_deg': round(meta.vfov_deg, 4),
            'capture_time': meta.capture_time,
            'rtk_active': meta.rtk_active,
            'gimbal': {
                'yaw_compass_deg': meta.gimbal_yaw_deg,
                'pitch_deg': meta.gimbal_pitch_deg,
                'roll_deg': meta.gimbal_roll_deg,
            },
        },
    }


# --------------------------------------------------------------------------- #
# 5. CLI
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python rgb_georeference.py <image.jpg> <labels.txt> [out.geojson]")
        sys.exit(1)
    out = georeference_rgb(sys.argv[1], sys.argv[2])
    out_path = sys.argv[3] if len(sys.argv) > 3 else 'rgb_georeferenced_fixed.geojson'
    Path(out_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    n_det = sum(1 for f in out['features']
                if f['properties'].get('type') == 'detection_box')
    print(f"✓ saved: {out_path}  (detections={n_det})")
