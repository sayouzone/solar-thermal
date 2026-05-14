"""
DJI 드론 이미지 메타데이터 추출 모듈.

DJI Zenmuse H20T 등에서 촬영한 JPG 파일의 EXIF + drone-dji XMP를 파싱해
태양광 패널 결함 탐지 파이프라인에서 사용하는 표준 메타데이터 dict 로 반환한다.

지원 카메라 (drone-dji:ImageSource):
    - ZoomCamera       : RGB 5184x3888, focal_length 10.14, 35mm 환산 47mm
    - WideCamera       : RGB 광각 (필요시 자동 인식)
    - InfraredCamera   : 열화상 640x512, focal_length 13.5, 35mm 환산 58mm

파일명 컨벤션: ``DJI_<YYYYMMDDHHMMSS>_<seq>_Z.JPG`` (RGB),
            ``DJI_<YYYYMMDDHHMMSS>_<seq>_T.JPG`` (IR).

참고: drone-dji XMP 네임스페이스는 http://www.dji.com/drone-dji/1.0/
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from dataclasses import dataclass

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solar_thermal.image.metadata import ImageMetadata, extract_metadata

# -----------------------------------------------------------------------------
# RGB / IR pair handling
# -----------------------------------------------------------------------------

# DJI Zenmuse H20T / M3T 파일 컨벤션:
#   DJI_<YYYYMMDDHHMMSS>_<seq>_Z.JPG  →  RGB (ZoomCamera)
#   DJI_<YYYYMMDDHHMMSS>_<seq>_W.JPG  →  RGB (WideCamera)
#   DJI_<YYYYMMDDHHMMSS>_<seq>_T.JPG  →  IR  (InfraredCamera)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="DJI 드론 이미지 메타데이터 추출기")
    parser.add_argument("image", type=Path, help="대상 JPG 경로 (RGB 또는 IR)")
    parser.add_argument(
        "--origin-path",
        type=str,
        default=None,
        help="원본 캡처 경로 (없으면 image 경로 사용)",
    )
    args = parser.parse_args()

    image_path = Path(args.origin_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    
    # 지원 확장자 목록
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    # 이미지 파일 목록 수집
    if image_path.is_dir():
        image_files = sorted([
            p for p in image_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])
    else:
        # 단일 파일인 경우
        image_files = [image_path]

    print(f"총 이미지 수: {len(image_files)}")
    for img in image_files:
        full_path = image_path / img.name
        meta = extract_metadata(full_path, origin_path=args.origin_path)
        print(json.dumps(meta.to_dict(), ensure_ascii=False, indent=4))
