"""
DJI DIRP SDK Python Wrapper for Solar Panel Thermal Inspection
================================================================

DJI Zenmuse H20T R-JPEG 파일에서 픽셀별 온도값을 추출하고
태양광 패널 검사에 적합한 방사율 보정을 수행하는 모듈.

[사용 전 준비]
1. DJI Thermal SDK 다운로드:
   https://www.dji.com/downloads/softwares/dji-thermal-sdk
   현재 최신: dji_thermal_sdk_v1.5_20240507

2. SDK 압축 해제 후 운영체제에 맞는 라이브러리 경로 지정:
   - Linux:   {SDK}/linux/release_x64/libdirp.so
   - Windows: {SDK}/windows/release_x64/libdirp.dll

3. 의존 라이브러리:
   pip install numpy pillow matplotlib opencv-python

[작성자] sayouzone / SeongJung Kim
"""

from __future__ import annotations

import ctypes as ct
import os
import platform
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ============================================================
# DIRP SDK ctypes 구조체 / enum 정의
# ============================================================

class DirpRet(IntEnum):
    """DIRP SDK 반환 코드"""
    SUCCESS = 0
    INVALID_PARAMS = -1
    INVALID_RAW = -2
    INVALID_HEADER = -3
    INVALID_CURVE_LUT = -4
    INVALID_RJPEG = -5
    INVALID_HANDLE = -6
    INVALID_FORMAT = -7
    BUFFER_OVERFLOW = -8
    FILE_IO_FAIL = -9
    SDK_NOT_INIT = -10


class DirpVerbose(IntEnum):
    NONE = 0
    DEBUG = 1
    DETAIL = 2


class DirpResolution(ct.Structure):
    _fields_ = [("width", ct.c_int32),
                ("height", ct.c_int32)]


class DirpMeasurementParams(ct.Structure):
    """
    방사선측정 파라미터 (온도 변환에 직접 영향)

    distance:    피사체까지 거리 [m]   (1 ~ 25)
    humidity:    상대 습도 [%]         (20 ~ 100)
    emissivity:  방사율                (0.10 ~ 1.00)
    reflection:  반사 겉보기 온도 [°C] (-40.0 ~ 500.0)
    """
    _fields_ = [
        ("distance",   ct.c_float),
        ("humidity",   ct.c_float),
        ("emissivity", ct.c_float),
        ("reflection", ct.c_float),
    ]


# ============================================================
# 측정 파라미터 데이터클래스 (사용자 친화적 인터페이스)
# ============================================================

@dataclass
class MeasurementParams:
    """방사선측정 파라미터.

    태양광 패널 검사 시 권장값 (실리콘 + 유리 표면 기준):
        emissivity   = 0.85 ~ 0.95  (모듈 표면 방사율)
        reflection   = ambient_temp + 5  (반사 보정값)
        humidity     = 현장 측정값 (보통 40~80%)
        distance     = LRF 측정값 또는 비행 고도
    """
    distance: float = 5.0       # m
    humidity: float = 70.0      # %
    emissivity: float = 0.95    # 0~1
    reflection: float = 25.0    # °C

    def __post_init__(self) -> None:
        # SDK 허용 범위 검증
        if not 1.0 <= self.distance <= 25.0:
            raise ValueError(f"distance must be 1~25m, got {self.distance}")
        if not 20.0 <= self.humidity <= 100.0:
            raise ValueError(f"humidity must be 20~100%, got {self.humidity}")
        if not 0.10 <= self.emissivity <= 1.00:
            raise ValueError(f"emissivity must be 0.1~1.0, got {self.emissivity}")
        if not -40.0 <= self.reflection <= 500.0:
            raise ValueError(f"reflection must be -40~500°C, got {self.reflection}")

    def to_ctypes(self) -> DirpMeasurementParams:
        return DirpMeasurementParams(
            distance=self.distance,
            humidity=self.humidity,
            emissivity=self.emissivity,
            reflection=self.reflection,
        )


# ============================================================
# 태양광 패널 표준 방사율 프리셋
# ============================================================

SOLAR_PANEL_PRESETS = {
    # 일반 결정질 실리콘 모듈 (유리 커버) — IEC TS 62446-3 권장값
    "crystalline_si":      0.85,
    # 박막형 (a-Si, CIGS, CdTe)
    "thin_film":           0.90,
    # AR(반사방지) 코팅 강화유리 표면
    "ar_coated_glass":     0.92,
    # 일반 강화유리 (단면)
    "tempered_glass":      0.94,
    # 후면 백시트 (PE/PET)
    "backsheet":           0.95,
}


# ============================================================
# DJI DIRP SDK 래퍼
# ============================================================

class DJIThermalSDK:
    """DJI DIRP SDK ctypes 래퍼 (싱글톤 라이브러리 핸들)."""

    _instance: Optional["DJIThermalSDK"] = None
    _lib: Optional[ct.CDLL] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, sdk_root: str | os.PathLike):
        if self._lib is not None:
            return  # 이미 초기화됨

        sdk_root = Path(sdk_root)
        system = platform.system().lower()       # 'linux' / 'windows'
        arch = "x64" if platform.architecture()[0] == "64bit" else "x86"
        ext = "so" if system == "linux" else "dll"

        lib_dir = sdk_root / system / f"release_{arch}"
        libdirp = lib_dir / f"libdirp.{ext}"

        if not libdirp.is_file():
            raise FileNotFoundError(
                f"libdirp not found at {libdirp}\n"
                f"DJI Thermal SDK: https://www.dji.com/downloads/softwares/dji-thermal-sdk"
            )

        # 의존 라이브러리(libv_dirp, libv_girp, libv_iirp)도 같은 폴더에 있어야 함
        # Linux: LD_LIBRARY_PATH, Windows: PATH 처리
        if system == "linux":
            os.environ["LD_LIBRARY_PATH"] = (
                str(lib_dir) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
            )
        else:
            os.add_dll_directory(str(lib_dir))  # type: ignore[attr-defined]

        self._lib = ct.CDLL(str(libdirp))
        self._setup_signatures()

    # --- ctypes 함수 시그니처 등록 -------------------------------
    def _setup_signatures(self) -> None:
        L = self._lib

        # int dirp_set_verbose_level(dirp_verbose_level_e level)
        L.dirp_set_verbose_level.argtypes = [ct.c_int]
        L.dirp_set_verbose_level.restype = ct.c_int

        # int dirp_create_from_rjpeg(const uint8_t* data, int32_t size, DIRP_HANDLE* h)
        L.dirp_create_from_rjpeg.argtypes = [ct.c_char_p, ct.c_int32, ct.POINTER(ct.c_void_p)]
        L.dirp_create_from_rjpeg.restype = ct.c_int

        # int dirp_destroy(DIRP_HANDLE h)
        L.dirp_destroy.argtypes = [ct.c_void_p]
        L.dirp_destroy.restype = ct.c_int

        # int dirp_get_rjpeg_resolution(DIRP_HANDLE h, dirp_resolution_t* res)
        L.dirp_get_rjpeg_resolution.argtypes = [ct.c_void_p, ct.POINTER(DirpResolution)]
        L.dirp_get_rjpeg_resolution.restype = ct.c_int

        # int dirp_get_measurement_params(DIRP_HANDLE h, dirp_measurement_params_t* p)
        L.dirp_get_measurement_params.argtypes = [ct.c_void_p, ct.POINTER(DirpMeasurementParams)]
        L.dirp_get_measurement_params.restype = ct.c_int

        # int dirp_set_measurement_params(DIRP_HANDLE h, dirp_measurement_params_t* p)
        L.dirp_set_measurement_params.argtypes = [ct.c_void_p, ct.POINTER(DirpMeasurementParams)]
        L.dirp_set_measurement_params.restype = ct.c_int

        # int dirp_measure_ex(DIRP_HANDLE h, float* temp, int32_t size)
        # ※ measure_ex는 float 단위 °C, 일반 measure는 int16 (0.1°C 단위)
        L.dirp_measure_ex.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int32]
        L.dirp_measure_ex.restype = ct.c_int

        L.dirp_measure.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int16), ct.c_int32]
        L.dirp_measure.restype = ct.c_int

    @property
    def lib(self) -> ct.CDLL:
        assert self._lib is not None, "SDK not initialised"
        return self._lib


# ============================================================
# R-JPEG 처리 메인 클래스
# ============================================================

class RJPEGProcessor:
    """R-JPEG 파일 1장을 열어 온도 행렬을 추출하는 컨텍스트 매니저."""

    def __init__(self, rjpeg_path: str | os.PathLike, sdk: DJIThermalSDK):
        self.path = Path(rjpeg_path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sdk = sdk
        self._handle: Optional[ct.c_void_p] = None
        self._width: int = 0
        self._height: int = 0

    def __enter__(self) -> "RJPEGProcessor":
        # 파일 전체를 메모리로 읽어 SDK에 전달
        data = self.path.read_bytes()
        buf = ct.c_char_p(data)
        size = ct.c_int32(len(data))
        handle = ct.c_void_p()

        ret = self.sdk.lib.dirp_create_from_rjpeg(buf, size, ct.byref(handle))
        if ret != DirpRet.SUCCESS:
            raise RuntimeError(
                f"dirp_create_from_rjpeg failed: {DirpRet(ret).name} "
                f"(파일이 R-JPEG가 아니거나 손상됨)"
            )
        self._handle = handle

        # 해상도 조회
        res = DirpResolution()
        if self.sdk.lib.dirp_get_rjpeg_resolution(handle, ct.byref(res)) != 0:
            raise RuntimeError("dirp_get_rjpeg_resolution failed")
        self._width, self._height = res.width, res.height
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self.sdk.lib.dirp_destroy(self._handle)
            self._handle = None

    # --- 속성 접근 ---------------------------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        """(height, width) — numpy 관례"""
        return self._height, self._width

    def get_params(self) -> MeasurementParams:
        """파일에 내장된 현재 측정 파라미터 조회"""
        p = DirpMeasurementParams()
        if self.sdk.lib.dirp_get_measurement_params(self._handle, ct.byref(p)) != 0:
            raise RuntimeError("dirp_get_measurement_params failed")
        return MeasurementParams(
            distance=p.distance, humidity=p.humidity,
            emissivity=p.emissivity, reflection=p.reflection,
        )

    def set_params(self, params: MeasurementParams) -> None:
        """방사율 등 파라미터를 변경하여 재계산 (핵심: 보정 적용 단계)"""
        c_p = params.to_ctypes()
        ret = self.sdk.lib.dirp_set_measurement_params(self._handle, ct.byref(c_p))
        if ret != DirpRet.SUCCESS:
            raise RuntimeError(f"dirp_set_measurement_params failed: {DirpRet(ret).name}")

    # --- 온도 추출 ---------------------------------------------
    def measure_temperature(self, dtype: str = "float32") -> np.ndarray:
        """
        픽셀별 절대 온도(°C) 행렬 반환.

        dtype:
            'float32' → dirp_measure_ex 사용, 단위 °C (실수)
            'int16'   → dirp_measure 사용, 단위 0.1°C (정수, ÷10 필요)

        반환: np.ndarray shape=(H, W)
        """
        n_pixels = self._width * self._height

        if dtype == "float32":
            buf = (ct.c_float * n_pixels)()
            byte_size = ct.sizeof(buf)
            ret = self.sdk.lib.dirp_measure_ex(self._handle, buf, byte_size)
            if ret != DirpRet.SUCCESS:
                raise RuntimeError(f"dirp_measure_ex failed: {DirpRet(ret).name}")
            arr = np.frombuffer(buf, dtype=np.float32).copy()

        elif dtype == "int16":
            buf = (ct.c_int16 * n_pixels)()
            byte_size = ct.sizeof(buf)
            ret = self.sdk.lib.dirp_measure(self._handle, buf, byte_size)
            if ret != DirpRet.SUCCESS:
                raise RuntimeError(f"dirp_measure failed: {DirpRet(ret).name}")
            arr = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 10.0

        else:
            raise ValueError("dtype must be 'float32' or 'int16'")

        return arr.reshape(self._height, self._width)


# ============================================================
# 고수준 헬퍼 — 한 줄 사용 인터페이스
# ============================================================

def extract_temperature(
    rjpeg_path: str | os.PathLike,
    sdk_root: str | os.PathLike,
    params: Optional[MeasurementParams] = None,
) -> Tuple[np.ndarray, MeasurementParams, MeasurementParams]:
    """
    R-JPEG에서 온도 행렬을 추출. (보정 전후 파라미터를 함께 반환)

    Returns:
        temp_celsius : np.ndarray (H, W) float32 — 픽셀별 °C
        original_params : 파일 원본 파라미터
        applied_params  : 실제 적용된 파라미터 (params 인자가 None이면 원본과 동일)
    """
    sdk = DJIThermalSDK(sdk_root)

    with RJPEGProcessor(rjpeg_path, sdk) as proc:
        original = proc.get_params()
        if params is not None:
            proc.set_params(params)
            applied = params
        else:
            applied = original
        temp = proc.measure_temperature(dtype="float32")

    return temp, original, applied


# ============================================================
# 태양광 패널 검사용 유틸리티
# ============================================================

def detect_hotspots(
    temp: np.ndarray,
    panel_mask: Optional[np.ndarray] = None,
    delta_threshold: float = 5.0,
    abs_threshold: Optional[float] = None,
) -> dict:
    """
    핫스팟 검출 (IEC TS 62446-3 기반 간이 진단).

    delta_threshold : 패널 평균 대비 ΔT [°C] 임계값
                      - 5°C  : 셀 단위 결함 의심
                      - 10°C : 모듈 단위 심각 결함 (string/diode 의심)

    Returns 통계 dict
    """
    if panel_mask is None:
        # 패널 영역을 추정하지 않고 이미지 전체에서 가장 따뜻한 상위 70%
        # (실제 운용에서는 RGB 정합 + 패널 분할 마스크 사용 권장)
        threshold = np.percentile(temp, 30)
        panel_mask = temp > threshold

    panel_temps = temp[panel_mask]
    mean_t = float(panel_temps.mean())
    std_t  = float(panel_temps.std())

    delta = temp - mean_t
    hotspot_mask = (delta > delta_threshold) & panel_mask
    if abs_threshold is not None:
        hotspot_mask &= temp > abs_threshold

    return {
        "panel_mean_C":     mean_t,
        "panel_std_C":      std_t,
        "panel_min_C":      float(panel_temps.min()),
        "panel_max_C":      float(panel_temps.max()),
        "hotspot_count":    int(hotspot_mask.sum()),
        "hotspot_max_dT":   float(delta[hotspot_mask].max()) if hotspot_mask.any() else 0.0,
        "hotspot_mask":     hotspot_mask,
        "delta_T_map":      delta,
    }


def save_temperature_tiff(temp: np.ndarray, out_path: str | os.PathLike) -> None:
    """온도 행렬을 16-bit GeoTIFF 호환 TIFF로 저장 (0.1°C 단위)."""
    from PIL import Image
    arr = (temp * 10.0).round().astype(np.int16)
    Image.fromarray(arr, mode="I;16").save(out_path)


def visualize(
    temp: np.ndarray,
    out_path: str | os.PathLike,
    cmap: str = "inferno",
    title: Optional[str] = None,
    hotspots: Optional[dict] = None,
) -> None:
    """온도 맵 시각화 (matplotlib)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    im = ax.imshow(temp, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature [°C]")

    if hotspots is not None and hotspots["hotspot_mask"].any():
        # 핫스팟 컨투어 오버레이
        ax.contour(hotspots["hotspot_mask"], levels=[0.5], colors="cyan", linewidths=1.0)

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CLI 데모 (직접 실행)
# ============================================================

if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="DJI R-JPEG 온도 추출 / 방사율 보정")
    parser.add_argument("rjpeg",       help="R-JPEG 파일 (예: DJI_..._T.JPG)")
    parser.add_argument("--sdk-root",  required=True, help="DJI Thermal SDK 루트 디렉토리")
    parser.add_argument("--emissivity", type=float, default=0.95,
                        help="방사율 (기본 0.95, 결정질 Si=0.85, AR글래스=0.92)")
    parser.add_argument("--distance",   type=float, default=None,
                        help="피사체 거리 [m] (미지정 시 파일 원본 유지)")
    parser.add_argument("--humidity",   type=float, default=None, help="상대습도 [%%]")
    parser.add_argument("--reflection", type=float, default=None, help="반사 온도 [°C]")
    parser.add_argument("--hotspot-dt", type=float, default=5.0,
                        help="핫스팟 ΔT 임계값 [°C]")
    parser.add_argument("--out-tiff",  default=None, help="출력 TIFF 경로 (16-bit, 0.1°C)")
    parser.add_argument("--out-png",   default=None, help="시각화 PNG 경로")
    args = parser.parse_args()

    sdk = DJIThermalSDK(args.sdk_root)

    # 1) 원본 파라미터 조회 (보정 적용 전 비교용)
    with RJPEGProcessor(args.rjpeg, sdk) as proc:
        original = proc.get_params()
        h, w = proc.shape
    print(f"[ORIGINAL] {w}×{h}  {original}")

    # 2) 사용자가 지정한 항목만 덮어쓴 새 파라미터 구성
    new_params = MeasurementParams(
        distance   = args.distance   if args.distance   is not None else original.distance,
        humidity   = args.humidity   if args.humidity   is not None else original.humidity,
        emissivity = args.emissivity,
        reflection = args.reflection if args.reflection is not None else original.reflection,
    )
    print(f"[APPLIED ] {new_params}")

    # 3) 보정 적용 후 온도 추출
    temp, _, _ = extract_temperature(args.rjpeg, args.sdk_root, params=new_params)

    # 4) 통계
    stats = {
        "shape":        list(temp.shape),
        "min_C":        float(temp.min()),
        "max_C":        float(temp.max()),
        "mean_C":       float(temp.mean()),
        "median_C":     float(np.median(temp)),
        "std_C":        float(temp.std()),
    }
    print("[STATS]", json.dumps(stats, indent=2))

    # 5) 핫스팟 진단
    diag = detect_hotspots(temp, delta_threshold=args.hotspot_dt)
    print(f"[HOTSPOT] count={diag['hotspot_count']}, "
          f"panel mean={diag['panel_mean_C']:.2f}°C, "
          f"max ΔT={diag['hotspot_max_dT']:.2f}°C")

    # 6) 출력 파일
    if args.out_tiff:
        save_temperature_tiff(temp, args.out_tiff)
        print(f"[SAVED] {args.out_tiff}")
    if args.out_png:
        visualize(temp, args.out_png,
                  title=f"ε={new_params.emissivity:.2f}",
                  hotspots=diag)
        print(f"[SAVED] {args.out_png}")
