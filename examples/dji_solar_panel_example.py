"""
examples/solar_panel_example.py
태양광 패널 검사에서의 georeferencing 예시
"""
import json
import numpy as np
import logging
import sys
from pathlib import Path

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solar_thermal.georeferencing.dji.georeference import process_image


def main():
    logging.basicConfig(level=logging.INFO)
    
    output_dir = Path("workspace/claude/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rgb_path = Path("data/solar/images/RGB/DJI_20251217130217_0007_Z.JPG")
    ir_path = Path("data/solar/images/TM/DJI_20251217130217_0007_T.JPG")
    
    rgb_result = process_image(rgb_path, is_thermal=False, output_dir=output_dir)
    ir_result = process_image(ir_path, is_thermal=True, output_dir=output_dir)
    
    # 두 이미지 비교
    print(f"\n\n{'='*70}")
    print("RGB와 IR 이미지 커버리지 비교")
    print('='*70)
    print(f"\n{'항목':<30} {'RGB':>20} {'IR':>20}")
    print(f"{'-'*70}")
    print(f"{'GSD (mm/pixel)':<30} {rgb_result['coverage']['gsd_m']*1000:>18.1f} "
          f"{ir_result['coverage']['gsd_m']*1000:>18.1f}")
    print(f"{'지상 너비 (m)':<30} {rgb_result['coverage']['width_m']:>20.2f} "
          f"{ir_result['coverage']['width_m']:>20.2f}")
    print(f"{'지상 높이 (m)':<30} {rgb_result['coverage']['height_m']:>20.2f} "
          f"{ir_result['coverage']['height_m']:>20.2f}")
    print(f"{'면적 (m²)':<30} {rgb_result['coverage']['area_m2']:>20.1f} "
          f"{ir_result['coverage']['area_m2']:>20.1f}")
    
    # JSON 결과 저장
    summary_path = output_dir / "georeferencing_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "rgb": rgb_result,
            "ir": ir_result,
        }, f, indent=2, default=str)
    print(f"\n결과 요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
