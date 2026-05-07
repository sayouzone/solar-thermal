# scripts/run_calibration.py
"""
H20T 전체 캘리브레이션 실행
"""
import logging
from pathlib import Path
import sys

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solar_thermal.calibraton.h20t.config import CalibrationConfig
from solar_thermal.calibraton.h20t.corner_detection import collect_corner_pairs
from solar_thermal.calibraton.h20t.intrinsic import (
    calibrate_intrinsic, remove_outliers
)
from solar_thermal.calibraton.h20t.stereo import (
    calibrate_stereo, validate_stereo_calibration
)
from solar_thermal.calibraton.h20t.validation import save_calibration


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    log = logging.getLogger(__name__)
    
    config = CalibrationConfig()
    
    log.info("=" * 60)
    log.info("Step 1: Detecting chessboard corners in RGB-IR pairs")
    log.info("=" * 60)
    
    detection_result = collect_corner_pairs(
        rgb_dir=config.rgb_dir,
        ir_dir=config.ir_dir,
        pattern_size=config.pattern_inner_corners,
        visualize=True,
        output_dir=Path("output/corner_visualization")
    )
    
    n_pairs = len(detection_result['used_files'])
    log.info(f"Successfully detected {n_pairs} pairs")
    log.info(f"Failed: {len(detection_result['failed_files'])} pairs")
    
    if n_pairs < config.min_acceptable_pairs:
        log.error(
            f"Too few pairs ({n_pairs} < {config.min_acceptable_pairs}). "
            "Take more calibration images and retry."
        )
        sys.exit(1)
    
    log.info("=" * 60)
    log.info("Step 2: RGB intrinsic calibration")
    log.info("=" * 60)
    
    intrinsics_rgb = calibrate_intrinsic(
        detection_result['object_points'],
        detection_result['rgb_corners'],
        config.rgb_resolution
    )
    log.info(f"RGB RMS reprojection error: {intrinsics_rgb['rms']:.4f} px")
    
    log.info("=" * 60)
    log.info("Step 3: IR intrinsic calibration")
    log.info("=" * 60)
    
    intrinsics_ir = calibrate_intrinsic(
        detection_result['object_points'],
        detection_result['ir_corners'],
        config.ir_resolution
    )
    log.info(f"IR RMS reprojection error: {intrinsics_ir['rms']:.4f} px")
    
    if intrinsics_rgb['rms'] > config.max_reprojection_error_px or \
       intrinsics_ir['rms'] > config.max_reprojection_error_px:
        log.warning("High reprojection error - removing outlier images")
        
        all_errors = (
            intrinsics_rgb['per_view_errors'] +
            intrinsics_ir['per_view_errors']
        )
        max_per_pair = [
            max(r, i) for r, i in zip(
                intrinsics_rgb['per_view_errors'],
                intrinsics_ir['per_view_errors']
            )
        ]
        
        filtered_obj, filtered_imgs, removed = remove_outliers(
            detection_result['object_points'],
            [detection_result['rgb_corners'], detection_result['ir_corners']],
            max_per_pair
        )
        
        log.info(f"Removed {len(removed)} outlier images")
        log.info("Re-calibrating without outliers...")
        
        intrinsics_rgb = calibrate_intrinsic(
            filtered_obj, filtered_imgs[0], config.rgb_resolution
        )
        intrinsics_ir = calibrate_intrinsic(
            filtered_obj, filtered_imgs[1], config.ir_resolution
        )
        
        log.info(f"RGB RMS after filtering: {intrinsics_rgb['rms']:.4f} px")
        log.info(f"IR RMS after filtering: {intrinsics_ir['rms']:.4f} px")
        
        detection_result['object_points'] = filtered_obj
        detection_result['rgb_corners'] = filtered_imgs[0]
        detection_result['ir_corners'] = filtered_imgs[1]
    
    log.info("=" * 60)
    log.info("Step 4: Stereo calibration")
    log.info("=" * 60)
    
    stereo = calibrate_stereo(
        detection_result['object_points'],
        detection_result['rgb_corners'],
        detection_result['ir_corners'],
        intrinsics_rgb['K'], intrinsics_rgb['D'],
        intrinsics_ir['K'], intrinsics_ir['D'],
        config.rgb_resolution,
        config.ir_resolution
    )
    
    log.info(f"Stereo RMS: {stereo['rms']:.4f} px")
    log.info(f"Baseline: {stereo['baseline_mm']:.2f} mm")
    
    validation = validate_stereo_calibration(stereo)
    log.info(f"Calibration quality: {validation['overall_quality']}")
    
    for issue in validation['issues']:
        log.error(f"ISSUE: {issue}")
    for warning in validation['warnings']:
        log.warning(f"WARNING: {warning}")
    
    log.info("=" * 60)
    log.info("Step 5: Saving calibration")
    log.info("=" * 60)
    
    save_calibration(
        intrinsics_rgb=intrinsics_rgb,
        intrinsics_ir=intrinsics_ir,
        stereo=stereo,
        config={
            'pattern_size': config.pattern_inner_corners,
            'square_size_mm': config.pattern_square_size_mm,
            'n_pairs_used': len(detection_result['object_points']),
        },
        output_path=config.output_path
    )
    
    log.info(f"Calibration saved to: {config.output_path}")
    log.info("Calibration complete!")


if __name__ == "__main__":
    main()
