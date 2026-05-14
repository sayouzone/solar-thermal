"""
드론 사진 Georeferencing 파이프라인
====================================

항공삼각측량(AT) 워크플로우를 Python으로 구현한 실용 예제.

워크플로우:
    1. EXIF/XMP 메타데이터 추출 (DJI 드론 GPS/자세값)
    2. RTK/PPK 좌표 파싱
    3. GCP(지상기준점) 로딩 및 사진 픽셀과 매칭
    4. SfM 기반 Tie Point 추출 (OpenCV)
    5. Bundle Block Adjustment (간략화된 형태)
    6. 정사영상(Orthophoto) 생성 - GeoTIFF 출력

주요 라이브러리:
    - exifread / piexif : EXIF 추출
    - opencv-python      : SfM, 특징점 매칭, 호모그래피
    - numpy / scipy      : 번들 조정 최소제곱
    - rasterio           : GeoTIFF 입출력
    - pyproj             : 좌표계 변환 (WGS84 ↔ UTM/EPSG:5186)

작성: Sayouzone Solar-Thermal 프로젝트 참고용
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as RioGCP
from pyproj import Transformer
from scipy.optimize import least_squares

import exifread
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 메타데이터 추출 (DJI EXIF/XMP)
# ---------------------------------------------------------------------------
@dataclass
class DroneImageMeta:
    """드론 사진 한 장의 외부표정요소(EOP) + 내부표정요소(IOP)."""
    path: Path
    # GPS (WGS84)
    lat: float
    lon: float
    abs_alt: float          # 절대 표고 (m, MSL)
    rel_alt: float | None   # 이륙지점 기준 상대 고도
    # 자세 (degrees) - DJI XMP 표준
    gimbal_yaw: float
    gimbal_pitch: float
    gimbal_roll: float
    flight_yaw: float
    flight_pitch: float
    flight_roll: float
    # 카메라 내부 파라미터
    focal_len_mm: float
    width: int
    height: int
    # RTK 플래그
    rtk_flag: int = 0       # 0=None, 16/50=RTK Float/Fixed


def _to_float(rational) -> float:
    """EXIF의 IFDRational/튜플을 float으로."""
    if isinstance(rational, (list, tuple)) and len(rational) == 3:
        # 위경도 [도, 분, 초] → 십진도
        print(rational, type(rational))
        d, m, s = [float(x.num) / float(x.den) for x in rational]
        return d + m / 60.0 + s / 3600.0
    elif isinstance(rational, (list, tuple)):
        print(rational, type(rational))
        return rational[0]
    return float(rational.num) / float(rational.den)


def extract_metadata(image_path: Path) -> DroneImageMeta:
    """DJI 드론 사진에서 EXIF + XMP를 모두 파싱.

    DJI Zenmuse H20T 같은 멀티센서 카메라는 RGB와 IR 채널별로 별도 EXIF를 가짐.
    XMP는 EXIF 뒤쪽 APP1 세그먼트에 XML로 들어 있어서 별도 파싱이 필요.
    """
    # --- EXIF 파싱 -----------------------------------------------------------
    with open(image_path, "rb") as fp:
        tags = exifread.process_file(fp, details=False)

    lat = _to_float(tags["GPS GPSLatitude"].values)
    if str(tags.get("GPS GPSLatitudeRef", "N")) == "S":
        lat = -lat
    lon = _to_float(tags["GPS GPSLongitude"].values)
    if str(tags.get("GPS GPSLongitudeRef", "E")) == "W":
        lon = -lon
    alt = _to_float(tags["GPS GPSAltitude"].values)

    focal = _to_float(tags["EXIF FocalLength"].values)
    width = int(str(tags["EXIF ExifImageWidth"]))
    height = int(str(tags["EXIF ExifImageLength"]))

    # --- XMP 파싱 (DJI 전용 태그) -------------------------------------------
    xmp_data = _extract_xmp(image_path)
    ns = {"drone-dji": "http://www.dji.com/drone-dji/1.0/"}

    def _xmp(tag: str, default: float = 0.0) -> float:
        node = xmp_data.find(f".//drone-dji:{tag}", ns) if xmp_data is not None else None
        return float(node.text) if node is not None and node.text else default

    return DroneImageMeta(
        path=image_path,
        lat=lat,
        lon=lon,
        abs_alt=alt,
        rel_alt=_xmp("RelativeAltitude"),
        gimbal_yaw=_xmp("GimbalYawDegree"),
        gimbal_pitch=_xmp("GimbalPitchDegree"),
        gimbal_roll=_xmp("GimbalRollDegree"),
        flight_yaw=_xmp("FlightYawDegree"),
        flight_pitch=_xmp("FlightPitchDegree"),
        flight_roll=_xmp("FlightRollDegree"),
        focal_len_mm=focal,
        width=width,
        height=height,
        rtk_flag=int(_xmp("RtkFlag")),
    )


def _extract_xmp(image_path: Path) -> ET.Element | None:
    """JPG 바이너리에서 XMP XML 블록을 잘라내 ElementTree로 반환."""
    raw = image_path.read_bytes()
    start = raw.find(b"<x:xmpmeta")
    end = raw.find(b"</x:xmpmeta>")
    if start == -1 or end == -1:
        return None
    return ET.fromstring(raw[start : end + len(b"</x:xmpmeta>")].decode("utf-8", "ignore"))


# ---------------------------------------------------------------------------
# 2. 좌표계 변환 (WGS84 → UTM52N 또는 한국 EPSG:5186)
# ---------------------------------------------------------------------------
class CRSConverter:
    """WGS84 (위경도) ↔ 투영좌표계 (m).

    한국에서는 EPSG:5186 (Korea 2000 / Central Belt 2010) 흔히 사용.
    글로벌은 UTM Zone 52N (EPSG:32652) 등.
    """

    def __init__(self, target_epsg: int = 5186):
        self.to_proj = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
        self.to_wgs = Transformer.from_crs(f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True)
        self.target_epsg = target_epsg

    def forward(self, lon: float, lat: float) -> tuple[float, float]:
        x, y = self.to_proj.transform(lon, lat)
        return x, y

    def inverse(self, x: float, y: float) -> tuple[float, float]:
        lon, lat = self.to_wgs.transform(x, y)
        return lon, lat


# ---------------------------------------------------------------------------
# 3. GCP (지상기준점) 데이터 모델
# ---------------------------------------------------------------------------
@dataclass
class GCP:
    """지상기준점.

    world: 실측 좌표 (X, Y, Z) — 투영좌표계 단위 m
    pixels: { image_path : (px, py) } — 사진별 픽셀 위치
    """
    name: str
    world: np.ndarray            # shape (3,)
    pixels: dict[Path, tuple[float, float]] = field(default_factory=dict)


def load_gcps_csv(csv_path: Path, crs: CRSConverter) -> list[GCP]:
    """CSV 포맷: name,lat,lon,alt"""
    gcps: list[GCP] = []
    import csv
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            x, y = crs.forward(float(row["lon"]), float(row["lat"]))
            z = float(row["alt"])
            gcps.append(GCP(name=row["name"], world=np.array([x, y, z])))
    return gcps


# ---------------------------------------------------------------------------
# 4. SfM - Tie Point 추출 & 매칭
# ---------------------------------------------------------------------------
def extract_features(image_path: Path, max_features: int = 8000):
    """SIFT로 특징점/디스크립터 추출.

    OpenCV 4.4+ 부터 SIFT가 메인 모듈로 편입되어 라이선스 걱정 없음.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc, img.shape


def match_pair(desc1, desc2, ratio: float = 0.75):
    """Lowe ratio test로 강건한 매칭."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return good


def build_tie_points(images: list[Path]) -> dict:
    """모든 이미지 쌍에 대해 SIFT + RANSAC 매칭.

    실무에서는 N장(N>100) 풀 페어 매칭은 O(N²)라 너무 느려서
    GPS 기반 인접 사진만 매칭하거나, Sequential/Vocabulary tree 매칭 사용.
    여기서는 데모용 단순 구현.
    """
    features = {p: extract_features(p) for p in images}
    matches = {}

    for i, img_i in enumerate(images):
        kp_i, desc_i, _ = features[img_i]
        for img_j in images[i + 1:]:
            kp_j, desc_j, _ = features[img_j]
            good = match_pair(desc_i, desc_j)
            if len(good) < 30:
                continue

            pts_i = np.float32([kp_i[m.queryIdx].pt for m in good])
            pts_j = np.float32([kp_j[m.trainIdx].pt for m in good])

            # RANSAC으로 outlier 제거 (Fundamental matrix)
            _, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 1.0, 0.99)
            inliers = mask.ravel().astype(bool)
            matches[(img_i, img_j)] = (pts_i[inliers], pts_j[inliers])

    logger.info("총 %d 페어에서 tie point 생성", len(matches))
    return matches


# ---------------------------------------------------------------------------
# 5. Bundle Block Adjustment (Collinearity Equation)
# ---------------------------------------------------------------------------
def rotation_matrix(omega: float, phi: float, kappa: float) -> np.ndarray:
    """사진측량 표준 ω-φ-κ 회전행렬 (라디안 입력)."""
    co, so = np.cos(omega), np.sin(omega)
    cp, sp = np.cos(phi),   np.sin(phi)
    ck, sk = np.cos(kappa), np.sin(kappa)
    R = np.array([
        [cp * ck,                  -cp * sk,                 sp],
        [co * sk + so * sp * ck,    co * ck - so * sp * sk,  -so * cp],
        [so * sk - co * sp * ck,    so * ck + co * sp * sk,   co * cp],
    ])
    return R


def project_point(world_xyz: np.ndarray,
                  camera_xyz: np.ndarray,
                  omega: float, phi: float, kappa: float,
                  f: float, cx: float, cy: float) -> np.ndarray:
    """공선조건(Collinearity Equation):

        x - cx = -f * (r11*(X-Xc) + r12*(Y-Yc) + r13*(Z-Zc))
                       / (r31*(X-Xc) + r32*(Y-Yc) + r33*(Z-Zc))
        y - cy = -f * (r21*(X-Xc) + r22*(Y-Yc) + r23*(Z-Zc))
                       / (r31*(X-Xc) + r32*(Y-Yc) + r33*(Z-Zc))

    카메라 렌즈 중심 - 사진점 - 지상점이 하나의 직선(Bundle) 위에 있다는 조건.
    """
    R = rotation_matrix(omega, phi, kappa)
    diff = world_xyz - camera_xyz
    num_x = R[0] @ diff
    num_y = R[1] @ diff
    den = R[2] @ diff
    x = cx - f * num_x / den
    y = cy - f * num_y / den
    return np.array([x, y])


def bundle_adjustment(initial_cameras: np.ndarray,
                      initial_points: np.ndarray,
                      observations: list[tuple[int, int, np.ndarray]],
                      f: float, cx: float, cy: float,
                      fixed_gcp_indices: set[int] | None = None):
    """간략화한 번들 조정.

    Args:
        initial_cameras: shape (n_cam, 6) — [Xc, Yc, Zc, ω, φ, κ]
        initial_points:  shape (n_pts, 3) — [X, Y, Z]
        observations:    [(cam_idx, pt_idx, observed_xy), ...]
        fixed_gcp_indices: GCP는 좌표를 고정 (제약조건)

    Returns:
        최적화된 카메라 외부표정 + 지상점 좌표
    """
    n_cam = len(initial_cameras)
    n_pts = len(initial_points)
    fixed = fixed_gcp_indices or set()

    def pack(cams, pts):
        return np.concatenate([cams.ravel(), pts.ravel()])

    def unpack(x):
        cams = x[: n_cam * 6].reshape(n_cam, 6)
        pts = x[n_cam * 6 :].reshape(n_pts, 3)
        return cams, pts

    def residuals(x):
        cams, pts = unpack(x)
        res = []
        for cam_idx, pt_idx, obs in observations:
            cam = cams[cam_idx]
            predicted = project_point(
                pts[pt_idx],
                cam[:3],
                cam[3], cam[4], cam[5],
                f, cx, cy,
            )
            res.extend(predicted - obs)
        # GCP 고정 제약 (소프트 제약, 큰 가중치)
        for idx in fixed:
            res.extend(1e6 * (pts[idx] - initial_points[idx]))
        return np.array(res)

    x0 = pack(initial_cameras, initial_points)
    result = least_squares(
        residuals, x0,
        method="trf",
        loss="huber",       # outlier에 강건
        max_nfev=200,
        verbose=2,
    )
    return unpack(result.x), result.cost


# ---------------------------------------------------------------------------
# 6. 정사영상 (Orthophoto) 생성 - GeoTIFF
# ---------------------------------------------------------------------------
def write_geotiff_with_gcps(image_path: Path,
                            output_path: Path,
                            gcp_pixel_world: list[tuple[float, float, float, float, float]],
                            epsg: int = 5186):
    """GCP를 이용해 사진 한 장을 GeoTIFF로 좌표 부착(georeference).

    완전한 정사영상은 DEM 기반 reprojection이 필요하지만,
    여기서는 평지 가정 하에 affine/polynomial transform으로 단순화.

    gcp_pixel_world: [(px, py, X, Y, Z), ...]
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # rasterio GCP 객체 변환
    rio_gcps = [
        RioGCP(row=py, col=px, x=X, y=Y, z=Z)
        for (px, py, X, Y, Z) in gcp_pixel_world
    ]
    transform = from_gcps(rio_gcps)

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=h, width=w, count=3,
        dtype=img_rgb.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        compress="lzw",
    ) as dst:
        for i in range(3):
            dst.write(img_rgb[:, :, i], i + 1)

    logger.info("GeoTIFF 저장 완료: %s (EPSG:%d)", output_path, epsg)


# ---------------------------------------------------------------------------
# 7. End-to-End 파이프라인
# ---------------------------------------------------------------------------
def run_pipeline(image_dir: Path,
                 gcp_csv: Path,
                 output_dir: Path,
                 target_epsg: int = 5186):
    """전체 georeferencing 파이프라인 실행."""
    output_dir.mkdir(parents=True, exist_ok=True)
    crs = CRSConverter(target_epsg=target_epsg)

    # 1. 메타데이터 추출
    images = sorted(image_dir.glob("*.JPG"))
    metas = [extract_metadata(p) for p in images]
    logger.info("이미지 %d장 메타데이터 추출 완료", len(metas))

    # 2. 카메라 초기 위치 (RTK 좌표를 투영좌표로)
    initial_cameras = []
    for m in metas:
        X, Y = crs.forward(m.lon, m.lat)
        Z = m.abs_alt
        # 라디안 변환 + 사진측량 컨벤션 (φ=pitch, ω=roll, κ=yaw)
        omega = np.deg2rad(m.gimbal_roll)
        phi = np.deg2rad(m.gimbal_pitch + 90)  # nadir보정: pitch=-90이 수직하방
        kappa = np.deg2rad(m.gimbal_yaw)
        initial_cameras.append([X, Y, Z, omega, phi, kappa])
    initial_cameras = np.array(initial_cameras)

    # 3. GCP 로딩
    gcps = load_gcps_csv(gcp_csv, crs)
    logger.info("GCP %d점 로딩", len(gcps))

    # 4. Tie point 추출
    tie_matches = build_tie_points(images)

    # 5. (실제 구현시) tie point들을 3D로 triangulation → bundle adjustment
    #    여기서는 분량상 호출 인터페이스만 시연.
    # observations, initial_points = triangulate_initial(tie_matches, initial_cameras, ...)
    # (cams_opt, pts_opt), cost = bundle_adjustment(
    #     initial_cameras, initial_points, observations,
    #     f=metas[0].focal_len_mm, cx=metas[0].width/2, cy=metas[0].height/2,
    #     fixed_gcp_indices={0, 1, 2, ...},
    # )

    # 6. 사진별 GeoTIFF 생성 (GCP가 매칭된 사진만)
    for meta in metas:
        gcp_pw = [
            (px, py, *gcp.world)
            for gcp in gcps
            if (pixel := gcp.pixels.get(meta.path)) is not None
            for px, py in [pixel]
        ]
        if len(gcp_pw) >= 4:  # affine 최소 3, polynomial은 4점 이상
            out = output_dir / f"{meta.path.stem}_geo.tif"
            write_geotiff_with_gcps(meta.path, out, gcp_pw, target_epsg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline(
        image_dir=Path("./data/solar/images/RGB"),
        gcp_csv=Path("./workspace/gcp.csv"),
        output_dir=Path("./workspace/output"),
        target_epsg=5186,  # 한국 중부 원점
    )
