"""
추출기 동작 확인용 테스트:
1) DJI 드론 사진과 유사한 EXIF + XMP 가 포함된 합성 JPEG 생성
2) MetadataExtractor 로 읽어 GPS, 짐벌 각도, 상대고도가 올바르게 나오는지 검증
"""

from __future__ import annotations

import json, dataclasses
import piexif
import sys
from fractions import Fraction
from pathlib import Path
from PIL import Image

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 추출기 동작 확인
from solar_thermal.detection.exif_extractor import MetadataExtractor

def _rat(x: float) -> tuple[int, int]:
    """float → (numerator, denominator) for EXIF rational."""
    f = Fraction(x).limit_denominator(1_000_000)
    return (f.numerator, f.denominator)


def make_test_image(path: Path):
    # 1) 빈 이미지
    img = Image.new("RGB", (1920, 1080), (30, 30, 40))

    # 2) EXIF: 카메라, GPS (서울 광화문 근처, 해발 150m)
    lat, lon, alt = 37.5796, 126.9770, 150.0

    def deg_to_dms_rational(deg: float):
        deg_abs = abs(deg)
        d = int(deg_abs)
        m_full = (deg_abs - d) * 60
        m = int(m_full)
        s = (m_full - m) * 60
        return [_rat(d), _rat(m), _rat(s)]

    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
        piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(lat),
        piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
        piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(lon),
        piexif.GPSIFD.GPSAltitudeRef: 0,
        piexif.GPSIFD.GPSAltitude: _rat(alt),
    }

    zeroth_ifd = {
        piexif.ImageIFD.Make: b"DJI",
        piexif.ImageIFD.Model: b"FC7303",   # DJI Mavic 3 계열
        piexif.ImageIFD.Orientation: 1,
    }
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: b"2024:08:15 14:23:45",
        piexif.ExifIFD.ExposureTime: _rat(1 / 1000),
        piexif.ExifIFD.FNumber: _rat(2.8),
        piexif.ExifIFD.ISOSpeedRatings: 100,
        piexif.ExifIFD.FocalLength: _rat(12.29),
        piexif.ExifIFD.FocalLengthIn35mmFilm: 24,
    }
    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd,
                 "1st": {}, "thumbnail": None}
    exif_bytes = piexif.dump(exif_dict)

    # 3) 중간 저장 (EXIF 포함)
    img.save(path, "jpeg", exif=exif_bytes, quality=85)

    # 4) XMP 패킷 삽입 (DJI 형식 시뮬레이션)
    xmp_packet = b"""<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 5.1.2">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
   xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/">
   <drone-dji:AbsoluteAltitude>+150.00</drone-dji:AbsoluteAltitude>
   <drone-dji:RelativeAltitude>+82.30</drone-dji:RelativeAltitude>
   <drone-dji:GimbalRollDegree>+0.00</drone-dji:GimbalRollDegree>
   <drone-dji:GimbalPitchDegree>-89.90</drone-dji:GimbalPitchDegree>
   <drone-dji:GimbalYawDegree>+45.20</drone-dji:GimbalYawDegree>
   <drone-dji:FlightRollDegree>+1.20</drone-dji:FlightRollDegree>
   <drone-dji:FlightPitchDegree>-2.50</drone-dji:FlightPitchDegree>
   <drone-dji:FlightYawDegree>+45.00</drone-dji:FlightYawDegree>
   <drone-dji:FlightXSpeed>+0.10</drone-dji:FlightXSpeed>
   <drone-dji:FlightYSpeed>+0.20</drone-dji:FlightYSpeed>
   <drone-dji:FlightZSpeed>+0.00</drone-dji:FlightZSpeed>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

    # JPEG APP1 세그먼트로 XMP 삽입 (간단히 EXIF 뒤에 추가)
    data = path.read_bytes()
    # APP1 XMP 세그먼트: 0xFFE1 + length + namespace + xmp_packet
    xmp_ns = b"http://ns.adobe.com/xap/1.0/\x00"
    app1_payload = xmp_ns + xmp_packet
    seg_length = len(app1_payload) + 2
    app1_segment = b"\xff\xe1" + seg_length.to_bytes(2, "big") + app1_payload

    # EXIF APP1 세그먼트 뒤에 XMP APP1 삽입
    # SOI(0xFFD8) 를 지나 첫 APP1 끝을 찾음
    insert_pos = 2  # SOI 다음
    if data[2:4] == b"\xff\xe1":  # 기존 EXIF APP1
        exif_len = int.from_bytes(data[4:6], "big")
        insert_pos = 4 + exif_len  # 기존 APP1 끝 바로 뒤

    new_data = data[:insert_pos] + app1_segment + data[insert_pos:]
    path.write_bytes(new_data)
    print(f"[OK] 테스트 이미지 생성: {path}  ({len(new_data)} bytes)")


if __name__ == "__main__":
    #out = Path("/tmp/test_drone_photo.jpg")
    #make_test_image(out)

    images = [
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217132210_0051_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/RGB/DJI_20251217132205_0049_Z.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/TM/DJI_20251217132209_0051_T.JPG",
        "/Users/seongjungkim/Downloads/태양광 발전소/TM/DJI_20251217132207_0050_T.JPG"
    ]

    ext = MetadataExtractor()

    for image in images:
        out = Path(image)
        rec = ext.extract(out)

        d = rec.to_dict()
        print(json.dumps(d, indent=2, ensure_ascii=False, default=str))

    """
    # 핵심 필드 검증
    print("\n=== 핵심 필드 검증 ===")
    assert rec.make == "DJI", f"Make={rec.make}"
    assert rec.model.startswith("FC7303"), f"Model={rec.model}"
    assert rec.gps.latitude is not None and abs(rec.gps.latitude - 37.5796) < 0.001, rec.gps.latitude
    assert rec.gps.longitude is not None and abs(rec.gps.longitude - 126.9770) < 0.001, rec.gps.longitude
    assert rec.gps.altitude_m is not None and abs(rec.gps.altitude_m - 150.0) < 0.1, rec.gps.altitude_m
    assert rec.drone.relative_altitude_m is not None and abs(rec.drone.relative_altitude_m - 82.3) < 0.01
    assert rec.drone.gimbal_pitch is not None and abs(rec.drone.gimbal_pitch - (-89.90)) < 0.01
    assert rec.drone.gimbal_yaw is not None and abs(rec.drone.gimbal_yaw - 45.20) < 0.01
    assert rec.iso == 100
    assert rec.f_number == 2.8
    assert rec.focal_length_mm == 12.29
    print("모든 검증 통과 ✓")
    """