"""
IEC TS 62446-3:2017 PV Thermal Anomaly Classifier
==================================================

태양광 패널 IR 이미지에서 IEC TS 62446-3 표준에 따라 결함을 자동 분류한다.

[IEC TS 62446-3 표준 핵심 개념]

1. Classes of Abnormalities (CoA) — 위험도 분류
   CoA 1: 즉시 조치 필요 (안전·화재 위험, 발전 손실 큼)
   CoA 2: 단기 조치 필요 (성능 영향 있음, 모니터링)
   CoA 3: 장기 모니터링 (영향 미미, 추적 관찰)

2. ΔT 정의
   ΔT_1 : 같은 모듈 내 셀 간 온도차
   ΔT_2 : 같은 어레이 내 모듈 간 온도차 (1000 W/m² 기준 정규화)

3. 표준 IR 패턴 분류 (Annex C)
   A. Module-level patterns (모듈 단위)
      - Single module fully heated:        모듈 전체 발열
      - Module patchwork pattern:          모듈 내 patchwork
      - Bypass diode active:               우회 다이오드 작동
      - Substring(s) heated:               부분 스트링 발열
   B. Cell-level patterns (셀 단위)
      - Single hot cell (hotspot):         셀 1개 고온 (단락/파손)
      - Multiple hotspots in module:       다수 셀 발열 (PID, 균열)
      - Cell-area hotspot (sub-cell):      셀 일부분 발열

4. 환경 조건 (수행 가능 조건)
   - 일사량 ≥ 600 W/m² (이상적 700 W/m² 이상)
   - 풍속 < 4 m/s (8 m/s 절대 한계)
   - 모듈 정상 작동 상태 (load 연결)
   - 청정 상태 (먼지/오염 없음)

[작성자] sayouzone / SeongJung Kim
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# IEC TS 62446-3 분류 체계
# ============================================================

class CoA(Enum):
    """Class of Abnormality - 위험도 분류"""
    CoA_1 = 1   # Immediate action — 즉시 조치
    CoA_2 = 2   # Short-term action — 단기 조치
    CoA_3 = 3   # Long-term monitoring — 장기 모니터링
    NORMAL = 0  # 정상


class DefectPattern(Enum):
    """IEC 62446-3 Annex C 기반 표준 결함 패턴."""
    NORMAL                = "normal"
    HOTSPOT_SINGLE        = "hotspot_single_cell"          # 단일 셀 고온
    HOTSPOT_MULTIPLE      = "hotspot_multiple_cells"       # 다수 셀 고온
    SUB_CELL_HOTSPOT      = "sub_cell_hotspot"             # 셀 일부 고온
    SUBSTRING_HEATED      = "substring_heated"             # 부분 스트링 발열
    BYPASS_DIODE_ACTIVE   = "bypass_diode_active"          # 우회 다이오드 작동
    MODULE_FULLY_HEATED   = "module_fully_heated"          # 모듈 전체 발열
    PATCHWORK_PATTERN     = "patchwork_pattern"            # Patchwork 패턴
    JUNCTION_BOX_HOT      = "junction_box_hot"             # 접속함 발열
    PID_LIKE              = "pid_like"                     # PID 의심 패턴
    SOILING               = "soiling"                      # 오염 (검출 시 분석 제외)


@dataclass
class DefectClassification:
    """단일 결함 분류 결과."""
    pattern: DefectPattern
    coa: CoA
    delta_t: float                          # 셀/모듈 간 ΔT [K]
    delta_t_normalized: float               # 1000 W/m² 정규화 ΔT [K]
    severity_score: float                   # 0~1 정규화 위험도
    affected_area_pct: float                # 모듈 내 영향 면적 비율 [%]
    description: str                        # 한국어 설명
    recommendation: str                     # 권장 조치
    bbox: Optional[Tuple[int, int, int, int]] = None     # (x, y, w, h)
    centroid: Optional[Tuple[float, float]] = None       # (cx, cy)
    max_temp: float = 0.0
    mean_temp: float = 0.0


# ============================================================
# 환경 조건 검증 (IEC TS 62446-3 §4)
# ============================================================

@dataclass
class InspectionConditions:
    """촬영 조건 (검사 유효성 판정용)."""
    irradiance_wm2: float = 800.0     # 일사량 W/m²
    wind_speed_ms: float = 2.0        # 풍속 m/s
    ambient_temp_c: float = 25.0      # 외기 온도 °C
    sky_clear: bool = True            # 청천 여부
    modules_loaded: bool = True       # 발전 중 여부

    def validate(self) -> Tuple[bool, List[str]]:
        """IEC 62446-3 § 4 환경 조건 검증."""
        warnings: List[str] = []
        if self.irradiance_wm2 < 600:
            warnings.append(
                f"일사량 {self.irradiance_wm2}W/m² < 600W/m² "
                "(IEC 62446-3 미충족)"
            )
        elif self.irradiance_wm2 < 700:
            warnings.append(
                f"일사량 {self.irradiance_wm2}W/m² < 700W/m² (권장 미달)"
            )
        if self.wind_speed_ms > 8:
            warnings.append(f"풍속 {self.wind_speed_ms}m/s > 8m/s (검사 부적합)")
        elif self.wind_speed_ms > 4:
            warnings.append(f"풍속 {self.wind_speed_ms}m/s > 4m/s (권장 초과)")
        if not self.sky_clear:
            warnings.append("청천 조건 미충족 (구름 낀 상태)")
        if not self.modules_loaded:
            warnings.append("모듈 무부하 상태 (발전 중 아님)")
        is_valid = self.irradiance_wm2 >= 600 and self.wind_speed_ms <= 8 \
                   and self.modules_loaded
        return is_valid, warnings


def normalize_delta_t(
    delta_t_measured: float,
    irradiance_wm2: float,
    target_irradiance: float = 1000.0,
) -> float:
    """
    측정된 ΔT를 표준 일사량(1000 W/m²) 기준으로 정규화.
    IEC TS 62446-3 § 5: ΔT_norm = ΔT_meas × (G_target / G_meas)
    """
    if irradiance_wm2 < 100:
        return delta_t_measured  # 비정상 일사 — 정규화 무효
    return delta_t_measured * (target_irradiance / irradiance_wm2)


# ============================================================
# 패널 분할 (간이 — 운용에서는 학습 모델로 교체 권장)
# ============================================================

def segment_panels(
    ir_temp: np.ndarray,
    method: str = "otsu",
    min_panel_area: int = 800,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    IR 온도 맵에서 패널 영역 분할.

    Args:
        ir_temp: float32 (H, W) °C
        method:  'otsu' | 'adaptive'
        min_panel_area: 노이즈 제거용 최소 면적

    Returns:
        labels: int32 (H, W), 각 패널에 1, 2, ..., N 번호
        bboxes: 각 패널 (x, y, w, h)

    [한계] 이는 단순 임계값 기반 — 실제 운용에서는 RGB 이미지 + YOLO/SAM 으로
    패널 분할 후 정합으로 IR에 투영하는 것이 권장됨.
    """
    # 정규화
    img = ir_temp.copy()
    img = (img - img.min()) / (img.ptp() + 1e-9)
    img8 = (img * 255).astype(np.uint8)

    if method == "otsu":
        _, mask = cv2.threshold(img8, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = cv2.adaptiveThreshold(
            img8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=51, C=-2,
        )

    # 모폴로지 — 패널 사이 갭 분리
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 연결 요소 라벨링
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    bboxes: List[Tuple[int, int, int, int]] = []
    out_labels = np.zeros_like(labels)
    new_id = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_panel_area:
            continue
        # 종횡비로 패널 후보 검증 (정상 패널 ~ 2:1)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 10:   # 너무 가는 띠는 그림자/배선
            continue
        new_id += 1
        out_labels[labels == i] = new_id
        bboxes.append((int(x), int(y), int(w), int(h)))

    return out_labels, bboxes


# ============================================================
# 결함 패턴 분류 핵심 로직
# ============================================================

def _analyze_panel(
    panel_temp: np.ndarray,
    panel_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    array_mean_t: float,
    delta_t_threshold: float,
) -> Optional[DefectClassification]:
    """단일 패널 영역 분석 → 결함 패턴 + CoA 분류."""
    x, y, w, h = bbox
    valid = panel_temp[panel_mask]
    if valid.size < 50:
        return None

    panel_mean = float(valid.mean())
    panel_std  = float(valid.std())
    panel_max  = float(valid.max())
    panel_min  = float(valid.min())

    # ── 1) 모듈 단위 ΔT (어레이 평균 대비)
    delta_t_module = panel_mean - array_mean_t

    # ── 2) 셀 단위 ΔT (모듈 내 최대 - 평균)
    delta_t_cell = panel_max - panel_mean

    # ── 3) 패턴 분류 결정 트리
    # 각 패널 내부에서 임계값을 초과한 픽셀(결함 후보) 영역 계산
    hot_mask = (panel_temp - panel_mean > delta_t_threshold) & panel_mask
    affected_pct = float(hot_mask.sum() / panel_mask.sum() * 100.0)

    # 연결 컴포넌트로 hotspot 개수 추정
    hot_u8 = hot_mask.astype(np.uint8) * 255
    num_hot, hot_labels, hot_stats, _ = cv2.connectedComponentsWithStats(hot_u8, connectivity=8)
    hot_blobs = [hot_stats[i, cv2.CC_STAT_AREA] for i in range(1, num_hot)
                 if hot_stats[i, cv2.CC_STAT_AREA] >= 5]
    n_hotspots = len(hot_blobs)

    # 분류 결정 (IEC 62446-3 Annex C 매트릭스 기반)
    pattern, coa, desc, rec = _classify_pattern(
        delta_t_module=delta_t_module,
        delta_t_cell=delta_t_cell,
        affected_pct=affected_pct,
        n_hotspots=n_hotspots,
        panel_std=panel_std,
    )

    if pattern == DefectPattern.NORMAL:
        return None

    severity = _severity_score(pattern, delta_t_cell, delta_t_module, affected_pct)

    # 질량 중심
    ys, xs = np.where(hot_mask)
    if xs.size > 0:
        centroid = (float(xs.mean()), float(ys.mean()))
    else:
        centroid = (x + w / 2.0, y + h / 2.0)

    return DefectClassification(
        pattern=pattern,
        coa=coa,
        delta_t=max(delta_t_cell, delta_t_module),
        delta_t_normalized=0.0,  # 호출 측에서 일사량으로 정규화
        severity_score=severity,
        affected_area_pct=affected_pct,
        description=desc,
        recommendation=rec,
        bbox=bbox,
        centroid=centroid,
        max_temp=panel_max,
        mean_temp=panel_mean,
    )


def _classify_pattern(
    delta_t_module: float,    # 모듈 평균 - 어레이 평균
    delta_t_cell: float,      # 모듈 최대 - 모듈 평균
    affected_pct: float,      # 모듈 내 발열 면적 %
    n_hotspots: int,          # 발열 영역 수
    panel_std: float,
) -> Tuple[DefectPattern, CoA, str, str]:
    """
    IEC TS 62446-3 Annex C 매트릭스 기반 결정 트리.

    ΔT 임계값은 표준 + Tsanakas et al. 2016 / Mantel et al. 2019 권장값:
        ΔT_cell ≥ 20K  → CoA 1 (즉시)
        ΔT_cell 10~20K → CoA 2 (단기)
        ΔT_cell 5~10K  → CoA 3 (모니터링)
    """
    # 1) 모듈 전체 발열 (영향 면적 ≥ 80%)
    if affected_pct >= 80 and delta_t_module > 5:
        if delta_t_module >= 15:
            return (DefectPattern.MODULE_FULLY_HEATED, CoA.CoA_1,
                    f"모듈 전체 발열 (ΔT={delta_t_module:.1f}K)",
                    "모듈 단락 회로/역방향 바이어스 의심. 즉시 격리 후 인버터·MPPT 점검.")
        elif delta_t_module >= 8:
            return (DefectPattern.MODULE_FULLY_HEATED, CoA.CoA_2,
                    f"모듈 전체 발열 (ΔT={delta_t_module:.1f}K)",
                    "스트링 단위 검사 및 I-V 곡선 측정 권장.")
        else:
            return (DefectPattern.MODULE_FULLY_HEATED, CoA.CoA_3,
                    f"경미한 모듈 전체 발열 (ΔT={delta_t_module:.1f}K)",
                    "다음 정기 검사 시 재측정.")

    # 2) Substring 발열 (1/2~2/3 영역, 다이오드 작동 의심)
    if 25 <= affected_pct <= 70 and delta_t_module > 8 and n_hotspots <= 3:
        if delta_t_cell >= 15:
            return (DefectPattern.BYPASS_DIODE_ACTIVE, CoA.CoA_1,
                    f"Bypass diode 작동 (영향 {affected_pct:.0f}%, ΔT={delta_t_cell:.1f}K)",
                    "다이오드 단락/모듈 손상 의심. 모듈 교체 검토.")
        else:
            return (DefectPattern.SUBSTRING_HEATED, CoA.CoA_2,
                    f"Substring 발열 (영향 {affected_pct:.0f}%)",
                    "Bypass diode 동작 확인 및 모듈 단위 정밀 검사.")

    # 3) 셀 단위 다중 핫스팟 (PID/균열 의심)
    if n_hotspots >= 4 and delta_t_cell > 5:
        if delta_t_cell >= 20:
            return (DefectPattern.HOTSPOT_MULTIPLE, CoA.CoA_1,
                    f"다수 셀 핫스팟 ({n_hotspots}개, ΔT={delta_t_cell:.1f}K)",
                    "PID 또는 다수 셀 균열. EL 검사 및 절연 저항 측정.")
        elif delta_t_cell >= 10:
            return (DefectPattern.PID_LIKE, CoA.CoA_2,
                    f"PID 의심 패턴 ({n_hotspots}개 핫스팟)",
                    "PID 회복 처리 또는 추가 진단(I-V, EL).")
        else:
            return (DefectPattern.HOTSPOT_MULTIPLE, CoA.CoA_3,
                    f"경미한 다중 핫스팟 ({n_hotspots}개)",
                    "장기 모니터링.")

    # 4) 단일 셀 핫스팟
    if n_hotspots == 1 and delta_t_cell > 5:
        if delta_t_cell >= 20:
            return (DefectPattern.HOTSPOT_SINGLE, CoA.CoA_1,
                    f"단일 셀 고온 핫스팟 (ΔT={delta_t_cell:.1f}K)",
                    "셀 단락/파손 위험. 화재 위험 — 즉시 차단 후 모듈 교체.")
        elif delta_t_cell >= 10:
            return (DefectPattern.HOTSPOT_SINGLE, CoA.CoA_2,
                    f"단일 셀 핫스팟 (ΔT={delta_t_cell:.1f}K)",
                    "EL/I-V 검사로 셀 상태 진단 권장.")
        else:
            return (DefectPattern.HOTSPOT_SINGLE, CoA.CoA_3,
                    f"경미한 셀 핫스팟 (ΔT={delta_t_cell:.1f}K)",
                    "다음 정기 검사 시 재측정.")

    # 5) Sub-cell hotspot (작은 영역, 균열 의심)
    if n_hotspots >= 2 and delta_t_cell > 3 and panel_std > 1.5:
        return (DefectPattern.SUB_CELL_HOTSPOT, CoA.CoA_3,
                f"Sub-cell 발열 ({n_hotspots}개, ΔT={delta_t_cell:.1f}K)",
                "셀 균열 가능성. 추적 관찰.")

    # 6) Patchwork (경미한 불균일)
    if 5 < affected_pct < 25 and panel_std > 2.0 and delta_t_cell > 4:
        return (DefectPattern.PATCHWORK_PATTERN, CoA.CoA_3,
                f"Patchwork 패턴 (불균일 ΔT={delta_t_cell:.1f}K)",
                "재료 결함 또는 소일링. 청소 후 재검사.")

    # 7) 정상
    return (DefectPattern.NORMAL, CoA.NORMAL, "정상 동작", "조치 불필요")


def _severity_score(
    pattern: DefectPattern,
    dt_cell: float,
    dt_module: float,
    affected_pct: float,
) -> float:
    """0~1 사이 위험도 점수 (보고서 정렬용)."""
    if pattern == DefectPattern.NORMAL:
        return 0.0
    base_weights = {
        DefectPattern.HOTSPOT_SINGLE:     0.7,   # 단일이지만 화재 위험
        DefectPattern.HOTSPOT_MULTIPLE:   0.6,
        DefectPattern.SUB_CELL_HOTSPOT:   0.3,
        DefectPattern.SUBSTRING_HEATED:   0.7,
        DefectPattern.BYPASS_DIODE_ACTIVE: 0.85,
        DefectPattern.MODULE_FULLY_HEATED: 0.95,
        DefectPattern.PATCHWORK_PATTERN:  0.25,
        DefectPattern.JUNCTION_BOX_HOT:   0.9,
        DefectPattern.PID_LIKE:           0.5,
        DefectPattern.SOILING:            0.15,
    }
    base = base_weights.get(pattern, 0.5)
    dt_factor = min(1.0, max(dt_cell, dt_module) / 25.0)
    area_factor = min(1.0, affected_pct / 100.0)
    return float(np.clip(0.5 * base + 0.3 * dt_factor + 0.2 * area_factor, 0, 1))


# ============================================================
# 메인 분류 파이프라인
# ============================================================

def classify_defects(
    ir_temp: np.ndarray,
    panel_labels: Optional[np.ndarray] = None,
    panel_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    conditions: Optional[InspectionConditions] = None,
    delta_t_threshold: float = 3.0,
) -> Tuple[List[DefectClassification], dict]:
    """
    IR 온도 맵에서 IEC 62446-3 결함 분류 수행.

    Args:
        ir_temp: float32 (H, W) °C — DIRP SDK로 추출한 온도 행렬
        panel_labels, panel_bboxes: 사전 분할된 패널 마스크 (없으면 자동 분할)
        conditions: 측정 환경 (정규화/유효성 검증용)
        delta_t_threshold: 패널 내부 핫스팟 검출 ΔT [K]

    Returns:
        defects: 검출된 결함 리스트
        summary: 요약 통계
    """
    if conditions is None:
        conditions = InspectionConditions()

    # 환경 조건 검증
    is_valid, warnings = conditions.validate()

    # 패널 자동 분할
    if panel_labels is None or panel_bboxes is None:
        panel_labels, panel_bboxes = segment_panels(ir_temp)

    # 어레이 평균 (모든 패널 영역의 평균)
    array_mask = panel_labels > 0
    array_mean_t = float(ir_temp[array_mask].mean()) if array_mask.any() else float(ir_temp.mean())

    defects: List[DefectClassification] = []
    for pid, bbox in enumerate(panel_bboxes, start=1):
        panel_mask = panel_labels == pid
        if not panel_mask.any():
            continue
        result = _analyze_panel(
            panel_temp=ir_temp,
            panel_mask=panel_mask,
            bbox=bbox,
            array_mean_t=array_mean_t,
            delta_t_threshold=delta_t_threshold,
        )
        if result is not None:
            # 일사량 정규화
            result.delta_t_normalized = normalize_delta_t(
                result.delta_t, conditions.irradiance_wm2,
            )
            defects.append(result)

    # 위험도 정렬
    defects.sort(key=lambda d: d.severity_score, reverse=True)

    summary = {
        "total_panels":     len(panel_bboxes),
        "defective_panels": len(defects),
        "defect_rate_pct":  round(100 * len(defects) / max(len(panel_bboxes), 1), 2),
        "by_coa": {
            "CoA_1": sum(d.coa == CoA.CoA_1 for d in defects),
            "CoA_2": sum(d.coa == CoA.CoA_2 for d in defects),
            "CoA_3": sum(d.coa == CoA.CoA_3 for d in defects),
        },
        "by_pattern": {
            p.value: sum(d.pattern == p for d in defects) for p in DefectPattern
            if sum(d.pattern == p for d in defects) > 0
        },
        "array_mean_temp_c": array_mean_t,
        "conditions_valid":  is_valid,
        "warnings":          warnings,
    }
    return defects, summary


# ============================================================
# 보고서 생성
# ============================================================

def visualize_classification(
    ir_temp: np.ndarray,
    defects: List[DefectClassification],
    out_path: str,
    rgb_overlay: Optional[np.ndarray] = None,
    panel_labels: Optional[np.ndarray] = None,
) -> None:
    """결함 분류 결과 시각화."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(14, 10), dpi=120)

    # 배경: RGB 오버레이가 있으면 50% 알파, 아니면 IR 온도맵
    if rgb_overlay is not None:
        ax.imshow(rgb_overlay)
        ax.imshow(ir_temp, cmap="inferno", alpha=0.55)
    else:
        im = ax.imshow(ir_temp, cmap="inferno")
        plt.colorbar(im, ax=ax, fraction=0.04, label="°C")

    # CoA 색상
    coa_colors = {
        CoA.CoA_1: "#ff0000",   # 적색
        CoA.CoA_2: "#ffa500",   # 주황
        CoA.CoA_3: "#ffff00",   # 노랑
    }

    for d in defects:
        if d.bbox is None:
            continue
        x, y, w, h = d.bbox
        color = coa_colors.get(d.coa, "white")
        rect = Rectangle((x, y), w, h, linewidth=2.0,
                         edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        label = f"{d.coa.name}\n{d.pattern.value}\nΔT={d.delta_t:.1f}K"
        ax.text(x, y - 5, label, color=color, fontsize=7,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.65, pad=1.5, edgecolor="none"))

    # 범례
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="none", edgecolor=c, label=k.name, linewidth=2)
               for k, c in coa_colors.items()]
    ax.legend(handles=handles, loc="upper right",
              facecolor="white", framealpha=0.9, fontsize=9)

    ax.set_title(f"IEC TS 62446-3 결함 분류 — {len(defects)}건 검출")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_inspection_report(
    defects: List[DefectClassification],
    summary: dict,
    site_info: Optional[dict] = None,
) -> dict:
    """IEC 62446-3 § 6 요구사항을 따르는 JSON 검사 보고서."""
    return {
        "standard": "IEC TS 62446-3:2017",
        "site": site_info or {},
        "summary": summary,
        "defects": [
            {
                "id": i + 1,
                "pattern": d.pattern.value,
                "coa": d.coa.name,
                "delta_t_K": round(d.delta_t, 2),
                "delta_t_normalized_K": round(d.delta_t_normalized, 2),
                "affected_area_pct": round(d.affected_area_pct, 2),
                "severity_score": round(d.severity_score, 3),
                "max_temp_C": round(d.max_temp, 2),
                "mean_temp_C": round(d.mean_temp, 2),
                "bbox_xywh": list(d.bbox) if d.bbox else None,
                "centroid": list(d.centroid) if d.centroid else None,
                "description": d.description,
                "recommendation": d.recommendation,
            }
            for i, d in enumerate(defects)
        ],
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse, json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="IEC 62446-3 PV 결함 분류")
    parser.add_argument("--temp-tiff", required=True,
                        help="16-bit TIFF 온도 맵 (0.1°C 단위, dji_thermal_extractor 출력)")
    parser.add_argument("--irradiance", type=float, default=800.0)
    parser.add_argument("--wind",       type=float, default=2.0)
    parser.add_argument("--ambient",    type=float, default=15.0)
    parser.add_argument("--delta-t",    type=float, default=3.0)
    parser.add_argument("--out-vis",    default="defect_classification.png")
    parser.add_argument("--out-report", default="inspection_report.json")
    args = parser.parse_args()

    # 온도 맵 로드
    from PIL import Image
    arr16 = np.array(Image.open(args.temp_tiff))
    ir_temp = arr16.astype(np.float32) / 10.0   # 0.1°C → °C

    conditions = InspectionConditions(
        irradiance_wm2=args.irradiance,
        wind_speed_ms=args.wind,
        ambient_temp_c=args.ambient,
    )
    defects, summary = classify_defects(
        ir_temp, conditions=conditions,
        delta_t_threshold=args.delta_t,
    )
    print(f"[검출] 패널 {summary['total_panels']}개 중 결함 {summary['defective_panels']}건 "
          f"({summary['defect_rate_pct']}%)")
    print(f"  CoA1={summary['by_coa']['CoA_1']}, "
          f"CoA2={summary['by_coa']['CoA_2']}, "
          f"CoA3={summary['by_coa']['CoA_3']}")
    if summary["warnings"]:
        for w in summary["warnings"]:
            print(f"  ⚠ {w}")

    visualize_classification(ir_temp, defects, args.out_vis)
    report = generate_inspection_report(defects, summary)
    Path(args.out_report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  → {args.out_vis}")
    print(f"  → {args.out_report}")
