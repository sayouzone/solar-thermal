# src/stereo.py
"""
RGB-IR 카메라 간 상대 위치 추정
"""
import cv2
import numpy as np


def calibrate_stereo(
    object_points: list,
    rgb_corners: list,
    ir_corners: list,
    K_rgb: np.ndarray,
    D_rgb: np.ndarray,
    K_ir: np.ndarray,
    D_ir: np.ndarray,
    rgb_size: tuple,
    ir_size: tuple
) -> dict:
    """
    스테레오 캘리브레이션
    
    내부 파라미터(K, D)는 고정하고 R, t만 추정
    
    Returns:
        {
            'R': rotation RGB->IR (3x3),
            't': translation RGB->IR (3,),
            'E': essential matrix,
            'F': fundamental matrix,
            'rms': RMS error
        }
    """
    flags = cv2.CALIB_FIX_INTRINSIC
    
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100, 1e-6
    )
    
    rms, _, _, _, _, R, t, E, F = cv2.stereoCalibrate(
        object_points,
        rgb_corners,
        ir_corners,
        K_rgb, D_rgb,
        K_ir, D_ir,
        rgb_size,
        flags=flags,
        criteria=criteria
    )
    
    return {
        'R': R,
        't': t.flatten(),
        'E': E,
        'F': F,
        'rms': float(rms),
        'baseline_mm': float(np.linalg.norm(t))
    }


def validate_stereo_calibration(stereo_result: dict) -> dict:
    """
    스테레오 결과의 합리성 검증
    """
    issues = []
    warnings = []
    
    baseline = stereo_result['baseline_mm']
    if baseline > 100:
        issues.append(f"Baseline too large: {baseline:.1f}mm (expected ~50mm for H20T)")
    elif baseline < 30:
        issues.append(f"Baseline too small: {baseline:.1f}mm")
    elif baseline > 70:
        warnings.append(f"Baseline larger than expected: {baseline:.1f}mm")
    
    R = stereo_result['R']
    angle = np.degrees(np.arccos((np.trace(R) - 1) / 2))
    if angle > 5:
        issues.append(f"Rotation too large: {angle:.2f}° (cameras should be near-parallel)")
    elif angle > 2:
        warnings.append(f"Rotation larger than expected: {angle:.2f}°")
    
    if stereo_result['rms'] > 1.0:
        issues.append(f"High reprojection error: {stereo_result['rms']:.3f}px")
    elif stereo_result['rms'] > 0.5:
        warnings.append(f"Moderate reprojection error: {stereo_result['rms']:.3f}px")
    
    return {
        'baseline_mm': baseline,
        'rotation_angle_deg': angle,
        'rms_px': stereo_result['rms'],
        'issues': issues,
        'warnings': warnings,
        'overall_quality': 'good' if not issues else ('marginal' if not warnings else 'failed')
    }
