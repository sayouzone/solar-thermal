"""CLI 로 단일 RGB+IR 이미지 쌍에 대해 파이프라인을 실행.

Usage
-----
    python scripts/run_inference.py \
        --config configs/default.yaml \
        --rgb path/to/rgb.jpg \
        --ir path/to/ir.tiff \
        --ir-format radiometric_tiff \
        --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solar_thermal.config import load_config  # noqa: E402
from solar_thermal.pipeline import DefectDetectionPipeline  # noqa: E402
from solar_thermal.schemas import InspectionRequest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rgb", required=True, help="RGB image path or URI")
    parser.add_argument("--ir", required=True, help="IR image path or URI")
    parser.add_argument(
        "--ir-format",
        default="radiometric_tiff",
        choices=["radiometric_tiff", "pseudo_color", "gray16"],
    )
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--inspection-id", default=None)
    parser.add_argument("--out", default="report.json")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = DefectDetectionPipeline(cfg)

    req = InspectionRequest(
        rgb_uri=args.rgb,
        ir_uri=args.ir,
        ir_format=args.ir_format,
        site_id=args.site_id,
        inspection_id=args.inspection_id,
    )
    report = pipeline.run(req, save_visualization=not args.no_viz)

    Path(args.out).write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Saved report: {args.out}\n"
        f"  panels={report.num_panels}, defective={report.num_defective_panels}\n"
        f"  viz={report.visualization_uri}"
    )


if __name__ == "__main__":
    main()
