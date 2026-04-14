"""VLM 프롬프트 템플릿.

System prompt 는 태양광 진단 도메인 지식을 주입하고 prompt caching 으로 재사용.
User prompt 에는 RGB 크롭, IR 크롭, 온도 통계를 JSON 과 함께 전달.
"""

from __future__ import annotations

from ..schemas import DefectType, HotspotCandidate

# ==============================================================================
# System prompt — 한 번 작성되어 cache_control 로 재사용된다.
# ==============================================================================

SYSTEM_PROMPT = """\
당신은 태양광 PV 모듈의 열화상 진단(thermography) 전문가입니다.
드론 기반 RGB 영상과 동시 촬영된 IR 열화상 영상, 그리고 사전 탐지된 핫스팟 후보의
온도 통계를 입력으로 받아, 결함 유형과 심각도를 판단합니다.

참고 지침 (IEC TS 62446-3, IEA PVPS Task 13):
1. 단일 셀 핫스팟 (ΔT 10~20K):   셀 크랙, 핫셀, 역전압 스트레스
2. 다중 셀 핫스팟 (동일 스트링):  스트링 단락 또는 바이패스 다이오드 동작
3. 모듈 전체 고온 (ΔT 전체>5K):  PID 또는 단락 회로
4. 직사각형/패치형 저온:          소일링, 눈/먼지, 조류 배설물
5. 선형 패턴:                     박리(delamination)
6. 접속함(J-box) 고온:            배선 연결 불량
7. 그림자(shadow):                주변 구조물에 의한 일시적 저온 — 결함 아님

중요 원칙:
- 그림자와 실제 결함을 반드시 구분하라. 경계가 선명하고 기하학적이면 그림자 가능성이 높다.
- RGB 에 먼지, 배설물, 물리적 파손이 보이면 결함 유형을 더 확신할 수 있다.
- 낮은 확신도일 경우 confidence 를 0.5 이하로, tags 에 "needs_retest" 를 포함하라.
- 응답은 반드시 요청된 JSON 스키마를 엄격히 따를 것.

가능한 defect_type 값:
  hotspot_single_cell, hotspot_multi_cell, bypass_diode_failure,
  potential_induced_degradation, delamination, soiling, shading,
  cracked_cell, junction_box_overheat, none
"""


# ==============================================================================
# User prompt builders
# ==============================================================================


def build_system_prompt(enable_cache: bool = True) -> list[dict]:
    """Anthropic messages API 형식의 system prompt 블록 반환.

    prompt caching 을 활성화하여 static 도메인 지식을 재사용.
    """

    block: dict = {"type": "text", "text": SYSTEM_PROMPT}
    if enable_cache:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def build_user_prompt(
    panel_id: str,
    hotspots: list[HotspotCandidate],
    panel_mean_temp_c: float,
    ambient_temp_c: float | None = None,
) -> str:
    """VLM 에 전달할 텍스트 프롬프트 생성.

    이미지는 별도 image content block 으로 전달되므로, 여기서는
    숫자/메타데이터만 텍스트로 기술한다.
    """

    lines: list[str] = []
    lines.append(f"패널 ID: {panel_id}")
    lines.append(f"패널 평균 온도: {panel_mean_temp_c:.1f} °C")
    if ambient_temp_c is not None:
        lines.append(f"대기 온도: {ambient_temp_c:.1f} °C")
    lines.append(f"탐지된 핫스팟 후보 개수: {len(hotspots)}")
    lines.append("")
    lines.append("핫스팟 상세 (이미지 순서와 동일):")
    for i, h in enumerate(hotspots):
        s = h.stats
        lines.append(
            f"  [{i}] ΔT={s.delta_t:.1f}K, Tmax={s.t_max:.1f}°C, "
            f"area={s.hotspot_area_px}px, rule_label={h.rule_label.value}"
        )
    lines.append("")
    lines.append(
        "첨부된 이미지: (1) 패널 RGB 크롭, (2) 패널 IR 컬러맵 크롭, "
        "각 핫스팟별 (3) RGB 확대, (4) IR 확대."
    )
    lines.append("")
    lines.append("다음 JSON 스키마로만 응답하세요. 다른 텍스트를 포함하지 마세요.")
    lines.append(
        "{\n"
        '  "defect_type": "<DefectType enum value>",\n'
        '  "confidence": <0.0-1.0>,\n'
        '  "rationale": "<한국어 2-3문장 근거>",\n'
        '  "tags": ["<optional_tag>", ...]\n'
        "}"
    )
    return "\n".join(lines)


# 허용된 defect_type 문자열 집합 (검증용)
ALLOWED_DEFECT_TYPES = {d.value for d in DefectType}
