# src/homography.py
"""
평면 가정 호모그래피 계산
"""
import cv2
import numpy as np


def compute_plane_homography(
    K_rgb: np.ndarray,
    K_ir: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    plane_normal: np.ndarray,
    plane_distance: float
) -> np.ndarray:
    """
    평면 가정 호모그래피 (RGB pixel -> IR pixel)
    
    H = K_ir × (R - t × n^T / d) × K_rgb^(-1)
    
    plane_normal: 카메라 좌표계 기준 평면 법선 (보통 [0,0,1] for nadir)
    plane_distance: 평면까지 거리 (mm)
    """
    n = plane_normal.reshape(3, 1)
    t_vec = t.reshape(3, 1)
    
    H_camera = R - (t_vec @ n.T) / plane_distance
    
    H = K_ir @ H_camera @ np.linalg.inv(K_rgb)
    
    H = H / H[2, 2]
    
    return H


def compute_homography_for_drone_flight(
    calibration: dict,
    drone_altitude_m: float,
    panel_tilt_deg: float = 30.0,
    drone_pitch_deg: float = 0.0
) -> np.ndarray:
    """
    실제 드론 비행 조건에 맞는 호모그래피
    """
    panel_tilt_rad = np.radians(panel_tilt_deg)
    drone_pitch_rad = np.radians(drone_pitch_deg)
    
    plane_normal_world = np.array([
        0,
        np.sin(panel_tilt_rad),
        -np.cos(panel_tilt_rad)
    ])
    
    R_drone = cv2.Rodrigues(
        np.array([drone_pitch_rad, 0, 0])
    )[0]
    plane_normal_camera = R_drone.T @ plane_normal_world
    
    distance_mm = drone_altitude_m * 1000
    
    return compute_plane_homography(
        calibration['K_rgb'],
        calibration['K_ir'],
        calibration['R'],
        calibration['t'],
        plane_normal_camera,
        distance_mm
    )


def compute_ir_visible_region_in_rgb(
    H_rgb_to_ir: np.ndarray,
    rgb_size: tuple,
    ir_size: tuple
) -> np.ndarray:
    """
    RGB 이미지에서 IR이 실제 커버하는 영역 (다각형)
    """
    H_ir_to_rgb = np.linalg.inv(H_rgb_to_ir)
    
    ir_w, ir_h = ir_size
    ir_corners = np.array([
        [[0, 0]],
        [[ir_w, 0]],
        [[ir_w, ir_h]],
        [[0, ir_h]]
    ], dtype=np.float32)
    
    rgb_corners = cv2.perspectiveTransform(ir_corners, H_ir_to_rgb)
    
    return rgb_corners.reshape(-1, 2)