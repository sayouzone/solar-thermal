# src/validation.py
"""
캘리브레이션 결과 저장과 시각적 검증
"""
import cv2
import numpy as np
import yaml
from pathlib import Path


def save_calibration(
    intrinsics_rgb: dict,
    intrinsics_ir: dict,
    stereo: dict,
    config: dict,
    output_path: Path
):
    """캘리브레이션 결과를 YAML로 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'metadata': {
            'camera_model': 'DJI_Zenmuse_H20T',
            'rgb_camera': 'zoom',
            'calibration_date': str(np.datetime64('now')),
            'pattern_size': list(config['pattern_size']),
            'square_size_mm': config['square_size_mm'],
            'n_pairs_used': config['n_pairs_used'],
        },
        'rgb': {
            'resolution': list(intrinsics_rgb['image_size']),
            'K': intrinsics_rgb['K'].tolist(),
            'D': intrinsics_rgb['D'].tolist(),
            'rms_px': float(intrinsics_rgb['rms']),
        },
        'ir': {
            'resolution': list(intrinsics_ir['image_size']),
            'K': intrinsics_ir['K'].tolist(),
            'D': intrinsics_ir['D'].tolist(),
            'rms_px': float(intrinsics_ir['rms']),
        },
        'stereo': {
            'R': stereo['R'].tolist(),
            't_mm': stereo['t'].tolist(),
            'baseline_mm': stereo['baseline_mm'],
            'rms_px': stereo['rms'],
        },
        'quality': {
            'rgb_intrinsic_rms_px': float(intrinsics_rgb['rms']),
            'ir_intrinsic_rms_px': float(intrinsics_ir['rms']),
            'stereo_rms_px': stereo['rms'],
            'baseline_mm': stereo['baseline_mm'],
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_calibration(yaml_path: Path) -> dict:
    """저장된 캘리브레이션 로드"""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    
    return {
        'K_rgb': np.array(data['rgb']['K']),
        'D_rgb': np.array(data['rgb']['D']),
        'K_ir': np.array(data['ir']['K']),
        'D_ir': np.array(data['ir']['D']),
        'R': np.array(data['stereo']['R']),
        't': np.array(data['stereo']['t_mm']),
        'rgb_size': tuple(data['rgb']['resolution']),
        'ir_size': tuple(data['ir']['resolution']),
        'metadata': data['metadata']
    }


def visualize_calibration_quality(
    calibration: dict,
    sample_rgb: np.ndarray,
    sample_ir: np.ndarray,
    drone_altitude_m: float,
    output_path: Path
):
    """캘리브레이션 품질 시각화"""
    from src.homography import compute_homography_for_drone_flight
    
    H = compute_homography_for_drone_flight(
        calibration, drone_altitude_m
    )
    
    rgb_undistorted = cv2.undistort(
        sample_rgb, calibration['K_rgb'], calibration['D_rgb']
    )
    ir_undistorted = cv2.undistort(
        sample_ir, calibration['K_ir'], calibration['D_ir']
    )
    
    rgb_h, rgb_w = rgb_undistorted.shape[:2]
    ir_warped = cv2.warpPerspective(
        ir_undistorted, np.linalg.inv(H), (rgb_w, rgb_h)
    )
    
    if len(ir_warped.shape) == 2:
        ir_warped = cv2.cvtColor(ir_warped, cv2.COLOR_GRAY2BGR)
    
    overlay = cv2.addWeighted(rgb_undistorted, 0.6, ir_warped, 0.4, 0)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def compute_alignment_error(
    rgb_corners: np.ndarray,
    ir_corners: np.ndarray,
    H_rgb_to_ir: np.ndarray
) -> dict:
    """
    검증용 페어에서 정합 오차 측정
    
    체커보드 코너를 ground truth로 사용
    """
    rgb_in_ir = cv2.perspectiveTransform(
        rgb_corners.reshape(-1, 1, 2), H_rgb_to_ir
    ).reshape(-1, 2)
    
    ir_pts = ir_corners.reshape(-1, 2)
    
    errors = np.linalg.norm(rgb_in_ir - ir_pts, axis=1)
    
    return {
        'mean_error_px': float(errors.mean()),
        'median_error_px': float(np.median(errors)),
        'max_error_px': float(errors.max()),
        'std_error_px': float(errors.std()),
        'n_corners': len(errors)
    }
