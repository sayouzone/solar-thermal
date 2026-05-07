"""
DJI R-JPEG 처리 실전 예제
=========================

업로드된 DJI_20251217131600_0310_T.JPG 처럼 ε=1.00, distance=5m 으로
잘못 설정된 R-JPEG에서 정확한 온도를 복구하는 워크플로우.

[실행 전]
1. DJI Thermal SDK 다운로드 후 압축 해제
2. SDK_ROOT 변수를 실제 경로로 수정
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solar_thermal.dji.thermal_extractor import (
    DJIThermalSDK, RJPEGProcessor, MeasurementParams,
    SOLAR_PANEL_PRESETS, extract_temperature, detect_hotspots,
)

# ===============================================
# 사용자 환경에 맞게 수정
# ===============================================
SDK_ROOT  = "/path/to/dji_thermal_sdk_v1.5_20240507"
RJPEG     = "DJI_20251217131600_0310_T.JPG"

# 업로드 이미지의 XMP에서 추출한 실제 촬영 조건
ACTUAL_DISTANCE_M    = 45.5   # LRF 측정값 (XMP: LRFTargetDistance=45.457)
ACTUAL_HUMIDITY_PCT  = 70.0   # 동절기 한국 평균
ACTUAL_AMBIENT_C     = 5.0    # 12월 17일 13시경 한국 남부 (XMP의 21°C는 센서 내부 온도)
SOLAR_PANEL_EMISS    = SOLAR_PANEL_PRESETS["ar_coated_glass"]   # = 0.92


def main():
    sdk = DJIThermalSDK(SDK_ROOT)

    # ─────────────────────────────────────────────
    # STEP 1: 원본 (잘못된) 파라미터로 온도 추출
    # ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: 원본 파라미터로 온도 추출 (보정 전)")
    print("=" * 60)

    with RJPEGProcessor(RJPEG, sdk) as proc:
        original = proc.get_params()
        temp_raw = proc.measure_temperature("float32")
    print(f"  원본 파라미터: {original}")
    print(f"  온도 통계: min={temp_raw.min():.2f}°C, "
          f"max={temp_raw.max():.2f}°C, mean={temp_raw.mean():.2f}°C")

    # ─────────────────────────────────────────────
    # STEP 2: 정확한 파라미터로 보정 적용
    # ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 2: 태양광 패널용 파라미터로 보정 (ε=0.92)")
    print("=" * 60)

    corrected_params = MeasurementParams(
        distance   = ACTUAL_DISTANCE_M,
        humidity   = ACTUAL_HUMIDITY_PCT,
        emissivity = SOLAR_PANEL_EMISS,
        reflection = ACTUAL_AMBIENT_C,
    )
    temp_corr, _, _ = extract_temperature(RJPEG, SDK_ROOT, params=corrected_params)
    print(f"  보정 파라미터: {corrected_params}")
    print(f"  온도 통계: min={temp_corr.min():.2f}°C, "
          f"max={temp_corr.max():.2f}°C, mean={temp_corr.mean():.2f}°C")

    delta = temp_corr - temp_raw
    print(f"  보정 효과: 평균 ΔT={delta.mean():+.2f}°C, "
          f"최대 ΔT={delta.max():+.2f}°C")

    # ─────────────────────────────────────────────
    # STEP 3: 방사율 민감도 분석 (sweep)
    # ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 3: 방사율 민감도 분석")
    print("=" * 60)

    emissivities = [0.80, 0.85, 0.90, 0.92, 0.95, 1.00]
    sweep_results = []
    for eps in emissivities:
        p = MeasurementParams(
            distance=ACTUAL_DISTANCE_M, humidity=ACTUAL_HUMIDITY_PCT,
            emissivity=eps, reflection=ACTUAL_AMBIENT_C,
        )
        t, _, _ = extract_temperature(RJPEG, SDK_ROOT, params=p)
        sweep_results.append({
            "emissivity": eps,
            "min_C": float(t.min()),
            "max_C": float(t.max()),
            "mean_C": float(t.mean()),
        })
        print(f"  ε={eps:.2f}  →  min={t.min():6.2f}°C  "
              f"max={t.max():6.2f}°C  mean={t.mean():6.2f}°C")

    # ─────────────────────────────────────────────
    # STEP 4: 핫스팟 진단 (IEC TS 62446-3 간이 적용)
    # ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 4: 핫스팟 검출")
    print("=" * 60)

    # 5°C ΔT — 셀 단위 결함 의심
    diag_cell = detect_hotspots(temp_corr, delta_threshold=5.0)
    # 10°C ΔT — 모듈 단위 심각 결함
    diag_module = detect_hotspots(temp_corr, delta_threshold=10.0)

    print(f"  패널 평균 온도: {diag_cell['panel_mean_C']:.2f}°C "
          f"(σ={diag_cell['panel_std_C']:.2f})")
    print(f"  ΔT > 5°C  픽셀 (셀 결함 의심):    {diag_cell['hotspot_count']:>8d} px")
    print(f"  ΔT > 10°C 픽셀 (모듈 심각 결함): {diag_module['hotspot_count']:>8d} px")
    print(f"  최대 ΔT: {diag_cell['hotspot_max_dT']:+.2f}°C")

    # ─────────────────────────────────────────────
    # STEP 5: 시각화 (보정 전후 비교 + ΔT 맵)
    # ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 5: 시각화 저장")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

    # (a) 원본
    im0 = axes[0, 0].imshow(temp_raw, cmap="inferno")
    axes[0, 0].set_title(f"원본 (ε={original.emissivity:.2f}, "
                         f"d={original.distance:.1f}m)")
    axes[0, 0].set_axis_off()
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, label="°C")

    # (b) 보정 후
    im1 = axes[0, 1].imshow(temp_corr, cmap="inferno")
    axes[0, 1].set_title(f"보정 후 (ε={SOLAR_PANEL_EMISS:.2f}, "
                         f"d={ACTUAL_DISTANCE_M:.1f}m)")
    axes[0, 1].set_axis_off()
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, label="°C")

    # (c) ΔT 맵 (보정 효과)
    im2 = axes[1, 0].imshow(delta, cmap="RdBu_r",
                            vmin=-abs(delta).max(), vmax=abs(delta).max())
    axes[1, 0].set_title("보정 효과 ΔT [°C]")
    axes[1, 0].set_axis_off()
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, label="°C")

    # (d) 핫스팟 오버레이
    axes[1, 1].imshow(temp_corr, cmap="inferno")
    if diag_cell["hotspot_mask"].any():
        axes[1, 1].contour(diag_cell["hotspot_mask"],
                           levels=[0.5], colors="cyan", linewidths=1.2)
    axes[1, 1].set_title(f"핫스팟 검출 (ΔT > 5°C, "
                         f"N={diag_cell['hotspot_count']:,} px)")
    axes[1, 1].set_axis_off()

    plt.tight_layout()
    plt.savefig("thermal_analysis_report.png", bbox_inches="tight")
    print("  → thermal_analysis_report.png")

    # 보고서 JSON
    report = {
        "file": str(RJPEG),
        "resolution": list(temp_corr.shape),
        "original_params":  vars(original),
        "corrected_params": vars(corrected_params),
        "raw_stats": {
            "min_C": float(temp_raw.min()),
            "max_C": float(temp_raw.max()),
            "mean_C": float(temp_raw.mean()),
        },
        "corrected_stats": {
            "min_C": float(temp_corr.min()),
            "max_C": float(temp_corr.max()),
            "mean_C": float(temp_corr.mean()),
        },
        "emissivity_sweep": sweep_results,
        "diagnosis": {
            "panel_mean_C":   diag_cell["panel_mean_C"],
            "panel_std_C":    diag_cell["panel_std_C"],
            "hotspot_5K_px":  diag_cell["hotspot_count"],
            "hotspot_10K_px": diag_module["hotspot_count"],
            "max_dT":         diag_cell["hotspot_max_dT"],
        },
    }
    Path("thermal_analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print("  → thermal_analysis_report.json")
    print("\n완료!")


if __name__ == "__main__":
    main()
