import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
from pathlib import Path

import numpy as np

# ============================================================
# 1. EXIF·XMP 메타데이터 추출
# ============================================================

@dataclass
class DJIMetadata:
    """DJI 드론 사진의 메타데이터"""
    # GPS (RTK 시 1cm 정확도)
    latitude: float
    longitude: float
    absolute_altitude_m: float
    relative_altitude_m: float       # 이륙 지점 기준 상대 고도
    
    # 드론 자세
    flight_yaw_deg: float
    flight_pitch_deg: float
    flight_roll_deg: float
    
    # 짐벌 자세 (월드 기준 절대값)
    gimbal_yaw_deg: float
    gimbal_pitch_deg: float
    gimbal_roll_deg: float
    
    # 이미지
    image_width: int
    image_height: int
    
    # RTK 정확도
    rtk_flag: int                    # 50 = Fixed
    rtk_std_lat_m: float
    rtk_std_lon_m: float
    rtk_std_hgt_m: float
    
    # 카메라 모델
    camera_model: str
    timestamp: str


def extract_xmp(filepath: Path) -> Optional[str]:
    """JPEG 파일에서 XMP 블록 추출"""
    with open(filepath, 'rb') as f:
        data = f.read()
    
    start = data.find(b'<x:xmpmeta')
    end = data.find(b'</x:xmpmeta>')
    
    if start == -1 or end == -1:
        return None
    
    return data[start:end + len(b'</x:xmpmeta>')].decode('utf-8', errors='ignore')


def parse_xmp_attribute(xmp: str, attr: str) -> Optional[str]:
    """XMP 속성 값 파싱"""
    match = re.search(rf'{attr}="([^"]*)"', xmp)
    if match:
        return match.group(1)
    
    match = re.search(rf'<[^>]*:{attr}>([^<]*)</', xmp)
    if match:
        return match.group(1)
    
    return None


def parse_dji_metadata(filepath: Path) -> DJIMetadata:
    """DJI 이미지에서 메타데이터 추출 (XMP 우선)"""
    import exifread
    
    with open(filepath, 'rb') as f:
        exif_tags = exifread.process_file(f, details=False)
    
    xmp = extract_xmp(filepath)
    if xmp is None:
        raise ValueError(f"XMP not found in {filepath}")
    
    return DJIMetadata(
        latitude=float(parse_xmp_attribute(xmp, "GpsLatitude")),
        longitude=float(parse_xmp_attribute(xmp, "GpsLongitude")),
        absolute_altitude_m=float(parse_xmp_attribute(xmp, "AbsoluteAltitude")),
        relative_altitude_m=float(parse_xmp_attribute(xmp, "RelativeAltitude")),
        flight_yaw_deg=float(parse_xmp_attribute(xmp, "FlightYawDegree")),
        flight_pitch_deg=float(parse_xmp_attribute(xmp, "FlightPitchDegree")),
        flight_roll_deg=float(parse_xmp_attribute(xmp, "FlightRollDegree")),
        gimbal_yaw_deg=float(parse_xmp_attribute(xmp, "GimbalYawDegree")),
        gimbal_pitch_deg=float(parse_xmp_attribute(xmp, "GimbalPitchDegree")),
        gimbal_roll_deg=float(parse_xmp_attribute(xmp, "GimbalRollDegree")),
        image_width=int(str(exif_tags["EXIF ExifImageWidth"])),
        image_height=int(str(exif_tags["EXIF ExifImageLength"])),
        rtk_flag=int(parse_xmp_attribute(xmp, "RtkFlag") or "0"),
        rtk_std_lat_m=float(parse_xmp_attribute(xmp, "RtkStdLat") or "999"),
        rtk_std_lon_m=float(parse_xmp_attribute(xmp, "RtkStdLon") or "999"),
        rtk_std_hgt_m=float(parse_xmp_attribute(xmp, "RtkStdHgt") or "999"),
        camera_model=str(exif_tags.get("Image Model", "Unknown")),
        timestamp=str(exif_tags.get("EXIF DateTimeOriginal", "")),
    )

# ============================================================
# 2. 카메라 내부 파라미터 (캘리브레이션)
# ============================================================

@dataclass
class CameraIntrinsics:
    """카메라 내부 파라미터"""
    K: np.ndarray
    D: np.ndarray
    image_width: int
    image_height: int


def estimate_zh20t_zoom_intrinsics(width: int, height: int,
                                    focal_length_35mm: float = 47.0) -> CameraIntrinsics:
    """
    DJI ZH20T Zoom 카메라의 추정 내부 파라미터
    
    실제 운영에선 캘리브레이션 결과를 사용해야 하지만,
    여기선 35mm 등가 초점거리로부터 추정
    
    35mm equivalent focal length f_35mm = 47mm
    Sensor: 1/1.7" (실제 7.6 × 5.7mm)
    Diagonal_35mm = 43.27mm
    Diagonal_sensor = 9.5mm  (1/1.7" 대각선)
    
    f_x_pixel = (f_35mm / Diagonal_35mm) * Diagonal_sensor / sensor_width * image_width
    """
    # 35mm 등가 → 실제 픽셀 초점거리 계산
    # 간단 추정: f_pixel = (f_35mm / 36mm) * image_width
    fx = fy = (focal_length_35mm / 36.0) * width
    
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1.0]
    ])
    D = np.zeros(5)  # 왜곡 계수 (실제로는 캘리브레이션 필요)
    
    return CameraIntrinsics(K=K, D=D, image_width=width, image_height=height)


def estimate_zh20t_thermal_intrinsics(width: int, height: int) -> CameraIntrinsics:
    """
    DJI ZH20T Thermal 카메라 추정 내부 파라미터
    
    공식 사양: FOV 40.6° (대각선)
    해상도: 640 × 512
    """
    # FOV로부터 초점거리 계산
    # f = (width/2) / tan(FOV_h / 2)
    # FOV_diag 40.6° → FOV_h ≈ 32.5° (4:3.2 종횡비)
    fov_h_deg = 32.5
    fx = (width / 2) / np.tan(np.radians(fov_h_deg / 2))
    fy = fx  # 정사각 픽셀 가정
    
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1.0]
    ])
    D = np.zeros(5)
    
    return CameraIntrinsics(K=K, D=D, image_width=width, image_height=height)

# ============================================================
# 4. 카메라 회전 행렬 (DJI 자세 → 월드 회전)
# ============================================================

def dji_gimbal_to_camera_rotation(
    gimbal_yaw_deg: float,
    gimbal_pitch_deg: float,
    gimbal_roll_deg: float
) -> np.ndarray:
    """
    DJI 짐벌 자세 → 카메라 회전 행렬 (R_camera_to_world)
    
    DJI 관습:
    - GimbalYaw: 북에서 시계방향 (북=0°, 동=90°)
        XMP에선 -180~180. -90.5° = 거의 서쪽 향함.
    - GimbalPitch: 수평선 기준 (수평=0°, 직하방=-90°)
    - GimbalRoll: 카메라 광축 회전
    
    이 코드의 World 좌표계: ENU (East-North-Up)
    카메라 좌표계 (OpenCV): X=오른쪽, Y=아래, Z=앞 (광축)
    
    Nadir 촬영(pitch=-90°)일 때:
    - 카메라 Z축은 아래(-Up) 방향
    - 카메라 X축은 yaw에 따라 결정
    """
    yaw = np.radians(gimbal_yaw_deg)
    pitch = np.radians(gimbal_pitch_deg)
    roll = np.radians(gimbal_roll_deg)
    
    # ENU에서 짐벌의 forward 벡터 (카메라가 보는 방향)
    # gimbal pitch=0이면 수평 (yaw 방향), pitch=-90이면 down(-Up)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    
    # ENU 좌표계에서 yaw가 북에서 동으로 시계방향이라는 것은 
    # ENU의 N축에서 E축으로 회전. yaw=0 → +N, yaw=90° → +E
    # forward (cam Z): pitch와 yaw로 결정
    forward = np.array([
        cp * sy,    # East
        cp * cy,    # North
        sp,         # Up (pitch=-90이면 sp=-1, 즉 -Up = down)
    ])
    
    # right (cam X): yaw 방향으로 90° 시계방향
    right = np.array([
        cy,         # East
        -sy,        # North
        0,          # Up
    ])
    
    # down (cam Y): forward × right 또는 별도 계산
    # OpenCV 카메라: X=right, Y=down (image의 v 방향), Z=forward
    # down = forward × right (right-handed)는 아니고
    # 표준 OpenCV에서 Y는 Z×X (right와 forward의 외적의 음수)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    
    # roll 적용 (광축 기준 회전)
    # right와 down을 광축 기준으로 회전
    if abs(roll) > 1e-9:
        right_rolled = cr * right + sr * down
        down_rolled = -sr * right + cr * down
        right = right_rolled
        down = down_rolled
    
    # R_camera_to_world: 카메라 좌표 벡터를 월드 좌표로 변환
    # 카메라 X(right) → world right
    # 카메라 Y(down) → world down
    # 카메라 Z(forward) → world forward
    R_camera_to_world = np.column_stack([right, down, forward])
    
    return R_camera_to_world
