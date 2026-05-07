"""
End-to-End Solar Panel Inspection Pipeline
============================================

DJI Zenmuse H20T로 동시 촬영된 RGB(_Z.JPG)와 IR R-JPEG(_T.JPG) 한 쌍을 입력받아
IEC TS 62446-3 표준 결함 보고서를 생성하는 통합 파이프라인.

[파이프라인 5단계]
  1. R-JPEG → 픽셀 단위 온도(°C) 추출 (방사율 보정 포함)
  2. RGB ↔ IR 정합 (호모그래피 추정)
  3. 패널 분할 (IR 기반 자동 분할)
  4. IEC 62446-3 결함 분류 (CoA 1/2/3)
  5. 보고서 생성 (JSON + 시각화 PNG)

[입력 파일]
  - RGB: DJI_*_Z.JPG  (5184×3888)
  - IR : DJI_*_T.JPG  (640×512, R-JPEG)

[작성자] sayouzone / SeongJung Kim
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dji_thermal_extractor import (
    DJIThermalSDK, RJPEGProcessor, MeasurementParams,
    SOLAR_PANEL_PRESETS, extract_temperature,
)
from rgb_ir_coregistration import coregister_rgb_ir, ZH20T_ZOOM_WIDE, ZH20T_THERMAL
from iec62446_classifier import (
    classify_defects, segment_panels, visualize_classification,
    generate_inspection_report, InspectionConditions,
)


def parse_xmp_metadata(rjpeg_path: str | Path) -> dict:
    """exiftool로 R-JPEG XMP에서 정합·보정에 필요한 메타데이터 추출."""
    import subprocess
    out = subprocess.run(
        ["exiftool", "-j", "-LRFTargetDistance", "-RelativeAltitude",
         "-GPSLatitude", "-GPSLongitude", "-DateTimeOriginal",
         "-Emissivity", "-RelativeHumidity", "-AmbientTemperature",
         "-ObjectDistance", str(rjpeg_path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)[0]
    return {
        "lrf_distance_m":   float(data.get("LRFTargetDistance", 5.0)),
        "altitude_rel_m":   float(str(data.get("RelativeAltitude", 0)).replace("+", "")),
        "gps_lat":          str(data.get("GPSLatitude", "")),
        "gps_lon":          str(data.get("GPSLongitude", "")),
        "datetime":         data.get("DateTimeOriginal", ""),
        "original_emissivity": float(data.get("Emissivity", 100)) / 100.0,
        "original_distance":   float(data.get("ObjectDistance", 5.0)),
    }


def run_pipeline(
    rgb_path: str | Path,
    ir_path: str | Path,
    sdk_root: str | Path,
    out_dir: str | Path = "inspection_output",
    panel_preset: str = "ar_coated_glass",
    irradiance_wm2: float = 800.0,
    wind_speed_ms: float = 2.0,
    ambient_temp_c: float | None = None,    # None이면 XMP에서 읽음
    delta_t_threshold: float = 3.0,
) -> dict:
    """end-to-end 실행 후 최종 보고서 dict 반환."""
    rgb_path = Path(rgb_path)
    ir_path  = Path(ir_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Solar Panel Inspection Pipeline")
    print(f"  RGB: {rgb_path.name}")
    print(f"  IR : {ir_path.name}")
    print("=" * 70)

    # ─────────────────────────────────────────────
    # STEP 0: XMP 메타데이터 파싱
    # ─────────────────────────────────────────────
    print("\n[0] XMP 메타데이터 파싱")
    meta = parse_xmp_metadata(ir_path)
    print(f"    LRF 거리: {meta['lrf_distance_m']:.1f}m")
    print(f"    상대 고도: {meta['altitude_rel_m']:.1f}m")
    print(f"    GPS: {meta['gps_lat']}, {meta['gps_lon']}")
    print(f"    원본 ε={meta['original_emissivity']:.2f}, "
          f"d={meta['original_distance']:.1f}m")

    if ambient_temp_c is None:
        # 외기온도가 지정되지 않으면 일사량/계절을 고려한 합리적 기본값 사용
        ambient_temp_c = 15.0   # 외부에서 받지 않을 때만 fallback

    # ─────────────────────────────────────────────
    # STEP 1: R-JPEG → 온도 추출 (방사율 보정)
    # ─────────────────────────────────────────────
    print("\n[1] R-JPEG 온도 추출 (방사율 보정)")
    emissivity = SOLAR_PANEL_PRESETS[panel_preset]
    correction = MeasurementParams(
        distance=meta["lrf_distance_m"],
        humidity=70.0,
        emissivity=emissivity,
        reflection=ambient_temp_c,
    )
    ir_temp, original_params, applied_params = extract_temperature(
        ir_path, sdk_root, params=correction,
    )
    print(f"    프리셋 '{panel_preset}' (ε={emissivity:.2f}) 적용")
    print(f"    온도 범위: {ir_temp.min():.1f}°C ~ {ir_temp.max():.1f}°C "
          f"(평균 {ir_temp.mean():.1f}°C)")

    # 16-bit TIFF 저장 (0.1°C 단위)
    temp_tiff = out_dir / "temperature_corrected.tiff"
    Image.fromarray((ir_temp * 10).round().astype(np.int16),
                    mode="I;16").save(temp_tiff)

    # ─────────────────────────────────────────────
    # STEP 2: RGB-IR 정합
    # ─────────────────────────────────────────────
    print("\n[2] RGB-IR 정합")

    # IR 시각화 이미지 (호모그래피 추정용 입력)
    # — R-JPEG 자체에는 시각화된 RGB 레이어가 들어있으므로 그대로 사용
    ir_visual = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
    ir_visual_path = out_dir / "ir_visual_for_registration.png"
    cv2.imwrite(str(ir_visual_path), ir_visual)

    reg = coregister_rgb_ir(
        rgb_path=rgb_path, ir_visual_path=ir_visual_path,
        distance_m=meta["lrf_distance_m"],
        rgb_spec=ZH20T_ZOOM_WIDE, ir_spec=ZH20T_THERMAL,
        use_ecc=True, use_features=True,
    )
    print(f"    방법: {reg.method}, 신뢰도: {reg.confidence:.3f}, "
          f"RMSE: {reg.rmse_px:.2f}px")

    # RGB → IR 좌표계 워핑 (시각화용)
    rgb_full = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)[:, :, ::-1]
    rgb_warped = reg.warp_rgb_to_ir(rgb_full)

    # ─────────────────────────────────────────────
    # STEP 3: 패널 분할
    # ─────────────────────────────────────────────
    print("\n[3] 패널 분할")
    panel_labels, panel_bboxes = segment_panels(ir_temp, method="otsu")
    print(f"    검출된 패널: {len(panel_bboxes)}개")

    # ─────────────────────────────────────────────
    # STEP 4: IEC 62446-3 결함 분류
    # ─────────────────────────────────────────────
    print("\n[4] IEC TS 62446-3 결함 분류")
    conditions = InspectionConditions(
        irradiance_wm2=irradiance_wm2,
        wind_speed_ms=wind_speed_ms,
        ambient_temp_c=ambient_temp_c,
    )
    defects, summary = classify_defects(
        ir_temp,
        panel_labels=panel_labels, panel_bboxes=panel_bboxes,
        conditions=conditions, delta_t_threshold=delta_t_threshold,
    )
    print(f"    결함 검출: {summary['defective_panels']}건 / "
          f"{summary['total_panels']}개 패널 "
          f"({summary['defect_rate_pct']}%)")
    print(f"    CoA1: {summary['by_coa']['CoA_1']}, "
          f"CoA2: {summary['by_coa']['CoA_2']}, "
          f"CoA3: {summary['by_coa']['CoA_3']}")
    if summary["warnings"]:
        for w in summary["warnings"]:
            print(f"    ⚠ {w}")

    # ─────────────────────────────────────────────
    # STEP 5: 보고서 + 시각화
    # ─────────────────────────────────────────────
    print("\n[5] 보고서 생성")
    visualize_classification(
        ir_temp, defects,
        out_path=str(out_dir / "defect_classification_overlay.png"),
        rgb_overlay=rgb_warped,
        panel_labels=panel_labels,
    )

    site_info = {
        "rgb_file":  rgb_path.name,
        "ir_file":   ir_path.name,
        "datetime":  meta["datetime"],
        "gps":       {"lat": meta["gps_lat"], "lon": meta["gps_lon"]},
        "altitude_relative_m": meta["altitude_rel_m"],
        "lrf_distance_m":      meta["lrf_distance_m"],
        "camera":              "DJI Zenmuse H20T",
        "inspection_datetime": datetime.now().isoformat(timespec="seconds"),
        "calibration": {
            "applied_emissivity": applied_params.emissivity,
            "applied_distance_m": applied_params.distance,
            "applied_reflection_c": applied_params.reflection,
            "panel_preset": panel_preset,
        },
        "registration": reg.to_dict(),
        "environmental_conditions": asdict(conditions),
    }
    report = generate_inspection_report(defects, summary, site_info=site_info)

    report_path = out_dir / "inspection_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"    → {out_dir}/inspection_report.json")
    print(f"    → {out_dir}/defect_classification_overlay.png")
    print(f"    → {out_dir}/temperature_corrected.tiff")
    print("\n완료")
    return report


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DJI ZH20T → IEC 62446-3 통합 검사 파이프라인",
    )
    parser.add_argument("--rgb",       required=True, help="RGB JPG (..._Z.JPG)")
    parser.add_argument("--ir",        required=True, help="R-JPEG (..._T.JPG)")
    parser.add_argument("--sdk-root",  required=True, help="DJI Thermal SDK 디렉토리")
    parser.add_argument("--out",       default="inspection_output")
    parser.add_argument("--preset",    default="ar_coated_glass",
                        choices=list(SOLAR_PANEL_PRESETS.keys()))
    parser.add_argument("--irradiance", type=float, default=800.0,
                        help="일사량 W/m² (현장 측정값)")
    parser.add_argument("--wind",       type=float, default=2.0)
    parser.add_argument("--ambient",    type=float, default=None,
                        help="외기온도 °C (생략 시 15°C)")
    parser.add_argument("--delta-t",    type=float, default=3.0,
                        help="패널 내부 핫스팟 검출 임계 ΔT [K]")
    args = parser.parse_args()

    run_pipeline(
        rgb_path=args.rgb, ir_path=args.ir, sdk_root=args.sdk_root,
        out_dir=args.out, panel_preset=args.preset,
        irradiance_wm2=args.irradiance, wind_speed_ms=args.wind,
        ambient_temp_c=args.ambient, delta_t_threshold=args.delta_t,
    )
