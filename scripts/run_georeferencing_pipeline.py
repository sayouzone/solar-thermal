"""GCP-Free Georeferencing 파이프라인 CLI 진입점.

Usage
-----
::

python scripts/run_georeferencing_pipeline.py \
    --image-dir ./data/solar/images/RGB \
    --output-dir ./workspace/output \
    --epsg 5186 \
    --gsd 0.05 \
    --k-neighbors 8

또는 환경변수::

    GEOREF_DISABLE_GPU=1 python scripts/run_georeferencing_pipeline.py ...

로 GPU 가속을 강제 비활성화 (벤치마크/디버깅).

본 스크립트는 ``solar_thermal.georeferencing`` 패키지의 얇은 wrapper.
실제 로직은 패키지 내부에 모듈화돼 있으므로, 라이브러리로 import 해서
다른 워크플로우에도 재사용 가능.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solar_thermal.georeferencing import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GCP-Free Georeferencing 파이프라인 (RTK-PPK 기반)",
    )
    p.add_argument("--image-dir", type=Path,
                   default=Path("./data/solar/images/RGB"),
                   help="DJI JPG 디렉토리")
    p.add_argument("--output-dir", type=Path,
                   default=Path("./workspace/output"),
                   help="GeoTIFF 출력 디렉토리")
    p.add_argument("--epsg", type=int, default=5186,
                   help="출력 좌표계 EPSG (기본: 5186 / Korea 2000 Central)")
    p.add_argument("--gsd", type=float, default=0.05,
                   help="출력 픽셀 크기 (m/pixel). 기본 0.05 = 5cm/pixel")
    p.add_argument("--k-neighbors", type=int, default=8,
                   help="KD-Tree 인접 페어 수 (기본 8)")
    p.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    start = time.perf_counter()
    run_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        target_epsg=args.epsg,
        gsd_m=args.gsd,
        device=args.device,
        k_neighbors=args.k_neighbors,
    )
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {timedelta(seconds=int(elapsed))}")


if __name__ == "__main__":
    main()