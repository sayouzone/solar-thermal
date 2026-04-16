"""
EXIF + XMP 메타데이터 추출기
============================
드론 검사 워크플로우(태양광 / 건물 / 파이프라인)를 위한 종합 추출기.

추출 대상:
  - 표준 EXIF: 카메라/렌즈/노출/타임스탬프
  - GPS: 위도/경도/고도 (10진수로 변환)
  - XMP: DJI 드론 커스텀 필드 (상대고도, 짐벌/비행 자세각)
  - 썸네일, 제조사/모델

설치:
    pip install Pillow exifread

권장 (DJI XMP 완전 지원):
    pip install pyexiv2
    # 또는 exiftool 바이너리 + subprocess 래퍼 사용
"""

from __future__ import annotations
import re
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


# =============================================================================
# 데이터 모델
# =============================================================================

@dataclass
class GpsInfo:
    latitude: Optional[float] = None      # 10진수 (남반구는 음수)
    longitude: Optional[float] = None     # 10진수 (서반구는 음수)
    altitude_m: Optional[float] = None    # 해발 고도 (m)
    altitude_ref: Optional[str] = None    # "above_sea_level" / "below_sea_level"
    timestamp_utc: Optional[str] = None
    speed: Optional[float] = None
    direction: Optional[float] = None     # 진행 방향 (도)


@dataclass
class DroneInfo:
    """DJI 등 드론 XMP 커스텀 필드."""
    relative_altitude_m: Optional[float] = None   # 이륙 지점 기준 고도
    absolute_altitude_m: Optional[float] = None
    gimbal_roll: Optional[float] = None
    gimbal_pitch: Optional[float] = None
    gimbal_yaw: Optional[float] = None
    flight_roll: Optional[float] = None
    flight_pitch: Optional[float] = None
    flight_yaw: Optional[float] = None
    flight_x_speed: Optional[float] = None
    flight_y_speed: Optional[float] = None
    flight_z_speed: Optional[float] = None
    camera_model: Optional[str] = None
    drone_model: Optional[str] = None


@dataclass
class ExifRecord:
    """한 사진의 전체 메타데이터 패키지."""
    filepath: str
    filename: str
    make: Optional[str] = None
    model: Optional[str] = None
    lens_model: Optional[str] = None
    datetime_original: Optional[str] = None    # 촬영 시각 (로컬)
    datetime_digitized: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    orientation: Optional[int] = None
    exposure_time: Optional[str] = None        # "1/1000" 형식
    f_number: Optional[float] = None
    iso: Optional[int] = None
    focal_length_mm: Optional[float] = None
    focal_length_35mm: Optional[float] = None
    gps: GpsInfo = field(default_factory=GpsInfo)
    drone: DroneInfo = field(default_factory=DroneInfo)
    raw_exif: dict[str, Any] = field(default_factory=dict)   # 선택적 원본 덤프

    def to_dict(self, include_raw: bool = False) -> dict:
        d = asdict(self)
        if not include_raw:
            d.pop("raw_exif", None)
        return d


# =============================================================================
# 유틸리티: GPS 도/분/초 → 10진수
# =============================================================================

def dms_to_decimal(dms, ref: str) -> Optional[float]:
    """
    (도, 분, 초) 튜플을 10진수 도로 변환.
    ref: 'N'/'S'/'E'/'W' — 'S' 나 'W' 면 음수.
    """
    if dms is None:
        return None
    try:
        d, m, s = [float(x) for x in dms]
    except (TypeError, ValueError):
        return None
    decimal = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def _to_float(value) -> Optional[float]:
    """PIL / exifread 의 Fraction/Ratio 객체를 float 로."""
    if value is None:
        return None
    try:
        # PIL IFDRational, Fraction 등
        return float(value)
    except (TypeError, ValueError):
        pass
    # "num/den" 문자열
    if isinstance(value, str) and "/" in value:
        try:
            num, den = value.split("/")
            return float(num) / float(den) if float(den) != 0 else None
        except ValueError:
            return None
    return None


def _to_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# =============================================================================
# 추출기: Pillow 기반 (기본 EXIF + GPS)
# =============================================================================

class PillowExifExtractor:
    """
    표준 EXIF + GPS 만 필요할 때. Pillow 만 있으면 됨.
    XMP(드론 커스텀 필드)는 별도 파서 사용.
    """

    def extract(self, path: str | Path) -> ExifRecord:
        from PIL import Image, ExifTags

        path = Path(path)
        record = ExifRecord(filepath=str(path), filename=path.name)

        with Image.open(path) as img:
            record.image_width, record.image_height = img.size
            exif_raw = img._getexif() or {}

        # 태그 ID → 이름 변환
        exif = {}
        for tag_id, value in exif_raw.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            exif[tag] = value

        record.make = exif.get("Make")
        record.model = exif.get("Model")
        record.lens_model = exif.get("LensModel")
        record.datetime_original = exif.get("DateTimeOriginal")
        record.datetime_digitized = exif.get("DateTimeDigitized")
        record.orientation = _to_int(exif.get("Orientation"))

        # 노출 정보
        exp = exif.get("ExposureTime")
        if exp is not None:
            exp_f = _to_float(exp)
            if exp_f and exp_f > 0:
                record.exposure_time = (
                    f"1/{int(round(1.0 / exp_f))}" if exp_f < 1 else f"{exp_f:.2f}"
                )

        record.f_number = _to_float(exif.get("FNumber"))
        record.iso = _to_int(exif.get("ISOSpeedRatings"))
        record.focal_length_mm = _to_float(exif.get("FocalLength"))
        record.focal_length_35mm = _to_float(exif.get("FocalLengthIn35mmFilm"))

        # GPS
        gps_raw = exif.get("GPSInfo")
        if gps_raw:
            gps = {ExifTags.GPSTAGS.get(k, str(k)): v for k, v in gps_raw.items()}
            record.gps.latitude = dms_to_decimal(
                gps.get("GPSLatitude"), gps.get("GPSLatitudeRef", "N")
            )
            record.gps.longitude = dms_to_decimal(
                gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", "E")
            )

            alt = _to_float(gps.get("GPSAltitude"))
            if alt is not None:
                alt_ref = gps.get("GPSAltitudeRef", 0)
                # 0 = above sea level, 1 = below
                if isinstance(alt_ref, bytes):
                    alt_ref = alt_ref[0] if alt_ref else 0
                if _to_int(alt_ref) == 1:
                    alt = -alt
                    record.gps.altitude_ref = "below_sea_level"
                else:
                    record.gps.altitude_ref = "above_sea_level"
                record.gps.altitude_m = alt

            record.gps.speed = _to_float(gps.get("GPSSpeed"))
            record.gps.direction = _to_float(gps.get("GPSImgDirection"))

            # UTC 타임스탬프 조합
            ts_date = gps.get("GPSDateStamp")   # "2024:08:15"
            ts_time = gps.get("GPSTimeStamp")   # (h, m, s) 튜플
            if ts_date and ts_time:
                try:
                    h, m, s = [_to_float(x) for x in ts_time]
                    record.gps.timestamp_utc = (
                        f"{ts_date.replace(':', '-')}T"
                        f"{int(h):02d}:{int(m):02d}:{int(s):02d}Z"
                    )
                except Exception:
                    pass

        return record


# =============================================================================
# XMP 파서 - DJI 드론 커스텀 필드
# =============================================================================

class XmpExtractor:
    """
    JPEG 의 APP1 세그먼트에서 XMP 패킷을 찾아 파싱.
    정규식 기반이라 외부 라이브러리 불필요. DJI 드론의
    drone-dji:* / Camera:* 네임스페이스 필드를 주로 대상으로 한다.
    """

    XMP_START = b"<x:xmpmeta"
    XMP_END = b"</x:xmpmeta>"

    # (출력 필드명, XMP 속성명, 타입 변환기) — DJI 드론 표준
    FIELD_MAP = [
        ("relative_altitude_m", "RelativeAltitude", float),
        ("absolute_altitude_m", "AbsoluteAltitude", float),
        ("gimbal_roll", "GimbalRollDegree", float),
        ("gimbal_pitch", "GimbalPitchDegree", float),
        ("gimbal_yaw", "GimbalYawDegree", float),
        ("flight_roll", "FlightRollDegree", float),
        ("flight_pitch", "FlightPitchDegree", float),
        ("flight_yaw", "FlightYawDegree", float),
        ("flight_x_speed", "FlightXSpeed", float),
        ("flight_y_speed", "FlightYSpeed", float),
        ("flight_z_speed", "FlightZSpeed", float),
    ]

    def _read_xmp_packet(self, path: Path) -> Optional[str]:
        data = path.read_bytes()
        start = data.find(self.XMP_START)
        if start == -1:
            return None
        end = data.find(self.XMP_END, start)
        if end == -1:
            return None
        return data[start : end + len(self.XMP_END)].decode("utf-8", errors="ignore")

    @staticmethod
    def _extract(xmp: str, attr: str) -> Optional[str]:
        """
        XMP 속성 추출. DJI 는 보통 attribute 형식(drone-dji:RelativeAltitude="+1.23")
        또는 element 형식(<drone-dji:RelativeAltitude>+1.23</drone-dji:RelativeAltitude>)
        둘 다 존재. 양쪽 모두 시도.
        """
        # attribute 형식 — 접두사 무관하게 attr 이름만 매칭
        patt_attr = rf'[A-Za-z\-]+:{re.escape(attr)}\s*=\s*"([^"]+)"'
        m = re.search(patt_attr, xmp)
        if m:
            return m.group(1)

        # element 형식
        patt_elem = rf"<[A-Za-z\-]+:{re.escape(attr)}>([^<]+)</[A-Za-z\-]+:{re.escape(attr)}>"
        m = re.search(patt_elem, xmp)
        if m:
            return m.group(1).strip()
        return None

    def extract(self, path: str | Path) -> DroneInfo:
        path = Path(path)
        info = DroneInfo()
        xmp = self._read_xmp_packet(path)
        if xmp is None:
            return info

        for field_name, xmp_attr, caster in self.FIELD_MAP:
            raw = self._extract(xmp, xmp_attr)
            if raw is None:
                continue
            try:
                # DJI 값은 종종 "+1.23" 처럼 부호 접두사 포함
                setattr(info, field_name, caster(raw.strip()))
            except ValueError:
                pass

        # 드론/카메라 모델 — 네임스페이스 다양하므로 모두 시도
        for field_name, attr in [("drone_model", "Model"), ("camera_model", "CameraModel")]:
            val = self._extract(xmp, attr)
            if val:
                setattr(info, field_name, val)

        return info


# =============================================================================
# 통합 추출기
# =============================================================================

class MetadataExtractor:
    """
    Pillow + XMP 결합. 드론 사진 한 장에서 가능한 모든 메타데이터 추출.
    """

    def __init__(self, parse_xmp: bool = True, include_raw: bool = False):
        self.pillow = PillowExifExtractor()
        self.xmp = XmpExtractor() if parse_xmp else None
        self.include_raw = include_raw

    def extract(self, path: str | Path) -> ExifRecord:
        record = self.pillow.extract(path)
        if self.xmp is not None:
            record.drone = self.xmp.extract(path)
        return record

    def extract_batch(self, paths: list[str | Path]) -> list[ExifRecord]:
        out = []
        for p in paths:
            try:
                out.append(self.extract(p))
            except Exception as e:
                print(f"[WARN] {p}: {e}")
        return out


# =============================================================================
# 선택: exiftool 래퍼 (가장 완전한 XMP/제조사 태그 지원)
# =============================================================================

class ExifToolWrapper:
    """
    외부 exiftool 바이너리가 설치되어 있을 때 사용 (brew/apt/choco install exiftool).
    DJI 의 thermal 이미지 메타(R-JPEG, T-JPEG), 파나소닉/소니 특수 태그까지 모두 읽힘.
    """

    def __init__(self, binary: str = "exiftool"):
        self.binary = binary

    def available(self) -> bool:
        try:
            subprocess.run(
                [self.binary, "-ver"],
                capture_output=True, check=True, timeout=5
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def extract(self, path: str | Path) -> dict:
        """모든 태그를 JSON 으로 덤프."""
        result = subprocess.run(
            [self.binary, "-j", "-n", "-G", str(path)],
            capture_output=True, check=True, text=True,
        )
        data = json.loads(result.stdout)
        return data[0] if data else {}


# =============================================================================
# CLI: 폴더 → CSV + GeoJSON
# =============================================================================

def _write_csv(records: list[ExifRecord], out: Path):
    import csv
    if not records:
        return
    rows = [r.to_dict() for r in records]
    # 플랫 딕셔너리로 펼침
    flat_rows = []
    for row in rows:
        flat = {}
        for k, v in row.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
            else:
                flat[k] = v
        flat_rows.append(flat)
    keys = sorted({k for row in flat_rows for k in row.keys()})
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_rows)


def _write_geojson(records: list[ExifRecord], out: Path):
    features = []
    for r in records:
        if r.gps.latitude is None or r.gps.longitude is None:
            continue
        props = r.to_dict()
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r.gps.longitude, r.gps.latitude,
                                r.gps.altitude_m if r.gps.altitude_m is not None else 0],
            },
            "properties": props,
        })
    geojson = {"type": "FeatureCollection", "features": features}
    out.write_text(json.dumps(geojson, ensure_ascii=False, indent=2, default=str))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="드론 사진 EXIF+XMP 추출기")
    parser.add_argument("target", help="이미지 파일 또는 폴더")
    parser.add_argument("--csv", help="결과 CSV 경로")
    parser.add_argument("--geojson", help="결과 GeoJSON 경로 (QGIS/Kepler 에서 바로 열림)")
    parser.add_argument("--json", help="결과 JSON 경로")
    parser.add_argument("--ext", default="jpg,jpeg,JPG,JPEG",
                        help="대상 확장자 (폴더 모드, 콤마 구분)")
    parser.add_argument("--no-xmp", action="store_true", help="XMP 파싱 생략")
    args = parser.parse_args()

    target = Path(args.target)
    if target.is_file():
        paths = [target]
    else:
        exts = {"." + e.lower().lstrip(".") for e in args.ext.split(",")}
        paths = [p for p in target.rglob("*") if p.suffix.lower() in exts]

    if not paths:
        print("[ERROR] 대상 파일이 없습니다.")
        return

    ext = MetadataExtractor(parse_xmp=not args.no_xmp)
    records = ext.extract_batch(paths)

    print(f"[INFO] {len(records)}개 파일 처리 완료")
    for r in records[:3]:
        lat, lon, alt = r.gps.latitude, r.gps.longitude, r.gps.altitude_m
        rel_alt = r.drone.relative_altitude_m
        gp = r.drone.gimbal_pitch
        print(f"  - {r.filename}  GPS=({lat}, {lon}) alt={alt}m "
              f"rel_alt={rel_alt}m gimbal_pitch={gp}")

    if args.csv:
        _write_csv(records, Path(args.csv))
        print(f"[INFO] CSV 저장: {args.csv}")
    if args.geojson:
        _write_geojson(records, Path(args.geojson))
        print(f"[INFO] GeoJSON 저장: {args.geojson}")
    if args.json:
        Path(args.json).write_text(
            json.dumps([r.to_dict() for r in records],
                       ensure_ascii=False, indent=2, default=str)
        )
        print(f"[INFO] JSON 저장: {args.json}")


if __name__ == "__main__":
    main()
