# src/intrinsic.py
"""
각 카메라의 내부 파라미터 추정
"""
import cv2
import numpy as np
from typing import Optional


def calibrate_intrinsic(
    object_points: list,
    image_points: list,
    image_size: tuple,
    initial_K: Optional[np.ndarray] = None
) -> dict:
    """
    내부 파라미터 캘리브레이션
    
    image_size: (width, height)
    
    Returns:
        {
            'K': intrinsic matrix (3x3),
            'D': distortion coefficients (5,),
            'rvecs': rotation vectors per image,
            'tvecs': translation vectors per image,
            'rms': RMS reprojection error,
            'per_view_errors': error per image
        }
    """
    flags = (
        cv2.CALIB_RATIONAL_MODEL +
        cv2.CALIB_FIX_K3
    )
    
    if initial_K is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        K_init = initial_K.copy()
        D_init = np.zeros(5)
    else:
        K_init = None
        D_init = None
    
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        K_init,
        D_init,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    )
    
    per_view_errors = compute_per_view_reprojection_error(
        object_points, image_points, rvecs, tvecs, K, D
    )
    
    return {
        'K': K,
        'D': D[:5].flatten(),
        'rvecs': rvecs,
        'tvecs': tvecs,
        'rms': rms,
        'per_view_errors': per_view_errors,
        'image_size': image_size
    }


def compute_per_view_reprojection_error(
    object_points: list,
    image_points: list,
    rvecs: list,
    tvecs: list,
    K: np.ndarray,
    D: np.ndarray
) -> list:
    """이미지별 reprojection error (이상치 식별용)"""
    errors = []
    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        error = np.sqrt(np.mean(np.sum((imgp - projected) ** 2, axis=2)))
        errors.append(float(error))
    return errors


def remove_outliers(
    object_points: list,
    image_points_list: list,
    per_view_errors: list,
    threshold_multiplier: float = 2.0
) -> tuple:
    """
    Reprojection error가 평균 + N×표준편차를 넘는 이미지 제거
    """
    errors = np.array(per_view_errors)
    mean_err = errors.mean()
    std_err = errors.std()
    threshold = mean_err + threshold_multiplier * std_err
    
    keep_mask = errors <= threshold
    
    filtered_obj = [op for op, k in zip(object_points, keep_mask) if k]
    filtered_imgs = []
    for imgs in image_points_list:
        filtered = [pt for pt, k in zip(imgs, keep_mask) if k]
        filtered_imgs.append(filtered)
    
    removed_indices = np.where(~keep_mask)[0].tolist()
    
    return filtered_obj, filtered_imgs, removed_indices