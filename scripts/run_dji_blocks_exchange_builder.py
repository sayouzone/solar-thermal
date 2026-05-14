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
from solar_thermal.image.blocks_exchange import build_blocks_exchange

# -----------------------------------------------------------------------------
# RGB / IR pair handling
# -----------------------------------------------------------------------------

# DJI Zenmuse H20T / M3T 파일 컨벤션:
#   DJI_<YYYYMMDDHHMMSS>_<seq>_Z.JPG  →  RGB (ZoomCamera)
#   DJI_<YYYYMMDDHHMMSS>_<seq>_W.JPG  →  RGB (WideCamera)
#   DJI_<YYYYMMDDHHMMSS>_<seq>_T.JPG  →  IR  (InfraredCamera)

# -----------------------------------------------------------------------------
# JSON → DJIImageMetadata 어댑터
# -----------------------------------------------------------------------------
 
 
def _dict_to_metadata(item: dict[str, Any]) -> ImageMetadata:
    """``image_list.json`` 의 한 항목을 ``DJIImageMetadata`` 로 변환."""
    gps_d = item["gps"]
    xmp_d = item.get("xmp") or {}
    geo_d = item.get("geo_desc") or {}
    pos_info_d = item.get("pos_info") or {}
 
    return DJIImageMetadata(
        id=item["id"],
        path=item.get("path", ""),
        origin_path=item.get("origin_path", ""),
        gps=GpsInfo(
            altitude=float(gps_d["altitude"]),
            lat=float(gps_d["lat"]),
            lng=float(gps_d["lng"]),
        ),
        position=list(item["position"]),
        relative_height=float(item.get("relative_height", 0.0)),
        orientation=list(item["orientation"]),
        pos_sigma=list(item.get("pos_sigma", [0.03, 0.03, 0.06])),
        height=int(item["height"]),
        width=int(item["width"]),
        camera_model=str(item["camera_model"]),
        camera_maker=str(item.get("camera_maker", "")),
        rtk_flag=int(item.get("rtk_flag", 0)),
        focal_length=float(item["focal_length"]),
        focal_length_in_35mm=int(item["focal_length_in_35mm"]),
        capture_time=int(item.get("capture_time", 0)),
        xmp=XmpInfo(
            bandName=xmp_d.get("bandName", ""),
            captureUUID=xmp_d.get("captureUUID", ""),
            droneID=xmp_d.get("droneID", ""),
            cameraMaker=xmp_d.get("cameraMaker", ""),
            cameraModel=xmp_d.get("cameraModel", ""),
        ),
        camera_sn=str(item.get("camera_sn", "")),
        rtk_std=list(item.get("rtk_std", [0.0, 0.0, 0.0])),
        pos_info=PosInfo(
            pos=list(pos_info_d.get("pos", item["position"])),
            pos_sigma=list(pos_info_d.get("pos_sigma", item.get("pos_sigma", [0.03, 0.03, 0.06]))),
            orientation=list(pos_info_d.get("orientation", item["orientation"])),
            id=str(pos_info_d.get("id", item["id"])),
        ),
        thumbnailPath=item.get("thumbnailPath", ""),
        orientation_type=item.get("orientation_type", "YPR"),
        geo_desc=GeoDesc(
            cs_type=geo_d.get("cs_type", "GEO_CS"),
            geo_cs=geo_d.get("geo_cs", "EPSG:4326"),
        ),
        ppk=item.get("ppk"),
        velocity=list(item.get("velocity", [0, 0, 0])),
        dewarp_flag=bool(item.get("dewarp_flag", True)),
        pre_calib_param=list(item.get("pre_calib_param", [None] * 9)),
        isImported=bool(item.get("isImported", True)),
        aux_img=item.get("aux_img"),
        sub_camera_sn=item.get("sub_camera_sn", ""),
        lens_sn=item.get("lens_sn", ""),
        lens_position=item.get("lens_position", ""),
        pre_calib_conf=int(item.get("pre_calib_conf", 0)),
        drone_model=item.get("drone_model", ""),
        payload_model=item.get("payload_model", ""),
    )
 
 
# -----------------------------------------------------------------------------
# 변환 본체
# -----------------------------------------------------------------------------
 
def _image_path_for_xml(
    origin_path: str,
    mode: str,
    relative_to: Path | None,
) -> str:
    """XML 내부에 기록할 ImagePath 결정."""
    if mode == "absolute":
        return str(Path(origin_path).resolve())
    if mode == "relative" and relative_to is not None:
        return str(Path(origin_path).resolve().relative_to(relative_to.resolve()))
    # 기본: 파일명만 (DJI Terra 표준)
    return Path(origin_path).name

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
    parser.add_argument(
        "--epsg", type=int, default=None,
        help="출력 좌표계 EPSG (생략 시 첫 사진 좌표에서 UTM zone 자동 산출)",
    )
    parser.add_argument(
        "--image-path", choices=("name", "absolute", "relative"), default="name",
        help="XML 내부에 기록할 ImagePath 형식 (기본 name = 파일명만)",
    )
    parser.add_argument("--relative-to", type=Path, default=None)
    parser.add_argument(
        "--include-ir", action="store_true",
        help="IR(InfraredCamera) 사진도 별도 Photogroup 으로 포함",
    )
    parser.add_argument(
        "--block-name", default="DJI AT Default: Block 1",
        help="Block 의 Name 필드",
    )
    parser.add_argument(
        "--block-description", default="Result of Aero Triangulation of Block 1",
        help="Block 의 Description 필드",
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
    metas: list[tuple[Path, ImageMetadata]] = []
    for img in image_files:
        full_path = image_path / img.name
        meta = extract_metadata(full_path, origin_path=args.origin_path)
        #print(json.dumps(meta.to_dict(), ensure_ascii=False, indent=4))
        if "InfraredCamera" in meta.camera_model:
            continue
        metas.append((Path(full_path) , meta))
        if "DJI_20251217130450_0064_Z.JPG" == img.name:
            break

    if not metas:
        raise ValueError("처리할 사진이 없습니다 (include_ir=True 옵션 확인).")

    xml_str = build_blocks_exchange(
        metas, 
        epsg=args.epsg, 
        block_name=args.block_name,
        block_description=args.block_description,
        image_path_in_xml=args.image_path,
        relative_to=args.relative_to,
        include_ir=args.include_ir,
    )
    print(xml_str)
