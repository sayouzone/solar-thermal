# src/corner_detection.py
"""
RGB와 IR 모두에서 체커보드 코너 검출
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def preprocess_ir_for_detection(ir_image: np.ndarray) -> np.ndarray:
    """
    IR 이미지를 체커보드 검출에 적합하게 전처리
    
    R-JPEG에서 직접 추출한 정규화 IR이 가장 좋지만,
    여기서는 일반 IR JPEG도 처리할 수 있도록 작성
    """
    if len(ir_image.shape) == 3:
        gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = ir_image
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return blurred


def detect_chessboard_corners(
    image: np.ndarray,
    pattern_size: tuple,
    is_thermal: bool = False
) -> tuple[bool, Optional[np.ndarray]]:
    """
    체커보드 내부 코너 검출
    
    Returns:
        (success, corners) - corners shape: (N, 1, 2)
    """
    if is_thermal:
        gray = preprocess_ir_for_detection(image)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FAST_CHECK
    )
    
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    
    if not found:
        return False, None
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    if is_thermal:
        win_size = (5, 5)
    else:
        win_size = (11, 11)
    
    corners_refined = cv2.cornerSubPix(
        gray, corners, win_size, (-1, -1), criteria
    )
    
    return True, corners_refined


def collect_corner_pairs(
    rgb_dir: Path,
    ir_dir: Path,
    pattern_size: tuple,
    visualize: bool = False,
    output_dir: Optional[Path] = None
) -> dict:
    """
    같은 시퀀스의 RGB-IR 페어에서 코너 검출
    
    Returns:
        {
            'rgb_corners': List[ndarray],
            'ir_corners': List[ndarray],
            'object_points': List[ndarray],
            'used_files': List[tuple],
            'failed_files': List[tuple]
        }
    """
    rgb_files = sorted(rgb_dir.glob("*_Z.JPG"))
    
    rgb_corners_list = []
    ir_corners_list = []
    used_files = []
    failed_files = []
    
    for rgb_file in rgb_files:
        seq_id = rgb_file.stem.replace("_Z", "")
        ir_file = ir_dir / f"{seq_id}_T.JPG"
        
        if not ir_file.exists():
            failed_files.append((rgb_file, ir_file, "ir_missing"))
            continue
        
        rgb_img = cv2.imread(str(rgb_file))
        ir_img = cv2.imread(str(ir_file))
        
        if rgb_img is None or ir_img is None:
            failed_files.append((rgb_file, ir_file, "load_failed"))
            continue
        
        rgb_ok, rgb_corners = detect_chessboard_corners(
            rgb_img, pattern_size, is_thermal=False
        )
        if not rgb_ok:
            failed_files.append((rgb_file, ir_file, "rgb_no_corners"))
            continue
        
        ir_ok, ir_corners = detect_chessboard_corners(
            ir_img, pattern_size, is_thermal=True
        )
        if not ir_ok:
            failed_files.append((rgb_file, ir_file, "ir_no_corners"))
            continue
        
        rgb_corners_list.append(rgb_corners)
        ir_corners_list.append(ir_corners)
        used_files.append((rgb_file, ir_file))
        
        if visualize and output_dir:
            visualize_detection(
                rgb_img, rgb_corners, pattern_size,
                output_dir / f"{seq_id}_rgb_corners.jpg"
            )
            visualize_detection(
                ir_img, ir_corners, pattern_size,
                output_dir / f"{seq_id}_ir_corners.jpg"
            )
    
    object_points = create_object_points(pattern_size, square_size=30.0)
    object_points_list = [object_points] * len(rgb_corners_list)
    
    return {
        'rgb_corners': rgb_corners_list,
        'ir_corners': ir_corners_list,
        'object_points': object_points_list,
        'used_files': used_files,
        'failed_files': failed_files
    }


def create_object_points(pattern_size: tuple, square_size: float) -> np.ndarray:
    """
    체커보드의 3D 좌표 생성 (z=0 평면 가정)
    
    pattern_size: (cols, rows) of inner corners
    square_size: in mm
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def visualize_detection(
    image: np.ndarray,
    corners: np.ndarray,
    pattern_size: tuple,
    output_path: Path
):
    """검출 결과 시각화"""
    vis = image.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners, True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)