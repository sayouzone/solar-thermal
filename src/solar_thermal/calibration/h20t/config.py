# src/config.py
"""
H20T 캘리브레이션 설정
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CalibrationConfig:
    rgb_dir: Path = Path("data/rgb")
    ir_dir: Path = Path("data/ir")
    
    pattern_inner_corners: tuple = (8, 5)
    pattern_square_size_mm: float = 30.0
    
    rgb_resolution: tuple = (5184, 3888)
    ir_resolution: tuple = (640, 512)
    
    rgb_termination: tuple = (
        3,  # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        100, 0.001
    )
    
    output_path: Path = Path("output/calibration_h20t_v1.yaml")
    
    min_acceptable_pairs: int = 15
    max_reprojection_error_px: float = 1.0
