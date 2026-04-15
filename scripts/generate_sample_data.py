"""RGB + Radiometric TIFF 합성 샘플 데이터 생성기.

드론으로 촬영한 태양광 패널 array 를 모사한 스타일라이즈드 RGB 이미지와,
동일 좌표계의 radiometric TIFF (16-bit, centi-Kelvin) 쌍을 만듭니다.
패널 일부에 의도된 핫스팟(단일/다중 셀, 소일링 저온, 그림자) 을 삽입하여
파이프라인 검증에 사용할 수 있습니다.

TIFF 포맷
---------
* dtype    : uint16
* encoding : centi-Kelvin (예: 50°C → 32315)
    loader.py 의 heuristic 이 이 컨벤션을 자동 감지해 °C 로 역변환합니다.

사용법
------
    python scripts/generate_sample_data.py --out samples/
    python scripts/run_inference.py \
        --rgb samples/sample_rgb.jpg \
        --ir  samples/sample_ir.tiff \
        --ir-format radiometric_tiff \
        --out samples/report.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile


# 패널 array 기하 파라미터
ROWS = 3            # 패널 array 행 개수
COLS = 4            # 패널 array 열 개수
PANEL_W = 320       # 패널 픽셀 폭
PANEL_H = 180       # 패널 픽셀 높이
CELL_ROWS = 6
CELL_COLS = 12
GAP = 20            # 패널 간 간격 (px)
MARGIN = 40         # 이미지 외곽 여백 (px)

# 온도 파라미터 (°C)
AMBIENT_C = 28.0
PANEL_BASE_C = 45.0      # 정상 패널 평균 온도
NOISE_STD_C = 0.6


def _image_size() -> tuple[int, int]:
    W = MARGIN * 2 + COLS * PANEL_W + (COLS - 1) * GAP
    H = MARGIN * 2 + ROWS * PANEL_H + (ROWS - 1) * GAP
    return H, W


def _panel_bbox(r: int, c: int) -> tuple[int, int, int, int]:
    x1 = MARGIN + c * (PANEL_W + GAP)
    y1 = MARGIN + r * (PANEL_H + GAP)
    return x1, y1, x1 + PANEL_W, y1 + PANEL_H


def _draw_panel_rgb(canvas: np.ndarray, r: int, c: int, soiled: bool = False) -> None:
    """하나의 패널을 RGB canvas 에 그린다."""

    x1, y1, x2, y2 = _panel_bbox(r, c)

    # 프레임 (알루미늄)
    cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (180, 180, 180), -1)
    # 셀 영역 바탕 (짙은 파랑/남색)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 40, 20), -1)

    # 셀 그리드
    cell_w = PANEL_W / CELL_COLS
    cell_h = PANEL_H / CELL_ROWS
    for i in range(1, CELL_COLS):
        x = int(x1 + i * cell_w)
        cv2.line(canvas, (x, y1), (x, y2), (50, 30, 15), 1)
    for j in range(1, CELL_ROWS):
        y = int(y1 + j * cell_h)
        cv2.line(canvas, (x1, y), (x2, y), (50, 30, 15), 1)

    # 은색 busbar (각 셀 내부 세로 2줄)
    for i in range(CELL_COLS):
        cx = int(x1 + (i + 0.5) * cell_w)
        cv2.line(canvas, (cx - 3, y1), (cx - 3, y2), (160, 160, 160), 1)
        cv2.line(canvas, (cx + 3, y1), (cx + 3, y2), (160, 160, 160), 1)

    # 소일링 표현 (먼지 얼룩)
    if soiled:
        overlay = canvas.copy()
        for _ in range(80):
            cx = np.random.randint(x1, x2)
            cy = np.random.randint(y1, y2)
            rad = np.random.randint(3, 12)
            cv2.circle(overlay, (cx, cy), rad, (60, 100, 130), -1)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, dst=canvas)


def _make_rgb(defect_map: dict[tuple[int, int], str]) -> np.ndarray:
    H, W = _image_size()
    # 하늘색 bluish-gray 배경 (지면 + 마운팅)
    canvas = np.full((H, W, 3), (85, 90, 90), dtype=np.uint8)

    for r in range(ROWS):
        for c in range(COLS):
            soiled = defect_map.get((r, c)) == "soiling"
            _draw_panel_rgb(canvas, r, c, soiled=soiled)

    # 그림자 시뮬레이션: 한 패널을 비스듬하게 어둡게
    if (0, 3) in defect_map and defect_map[(0, 3)] == "shading":
        x1, y1, x2, y2 = _panel_bbox(0, 3)
        # 왼쪽 절반을 사다리꼴 그림자로
        shadow_poly = np.array(
            [[x1, y1], [x1 + PANEL_W // 2, y1], [x1 + PANEL_W // 3, y2], [x1, y2]],
            dtype=np.int32,
        )
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [shadow_poly], (30, 30, 30))
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)

    # 약간의 가우시안 블러로 드론 압축 효과
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)
    return canvas


def _make_thermal(defect_map: dict[tuple[int, int], str]) -> np.ndarray:
    """°C 단위 온도맵 생성 후 centi-Kelvin uint16 TIFF 로 변환할 준비."""

    H, W = _image_size()
    rng = np.random.default_rng(seed=42)

    # 배경 (지면/프레임) 은 대기 근처
    temp = np.full((H, W), AMBIENT_C, dtype=np.float32)
    temp += rng.normal(0, NOISE_STD_C, size=temp.shape)

    for r in range(ROWS):
        for c in range(COLS):
            x1, y1, x2, y2 = _panel_bbox(r, c)
            label = defect_map.get((r, c), "normal")
            panel = np.full((y2 - y1, x2 - x1), PANEL_BASE_C, dtype=np.float32)

            # 패널 내부 온도 변동 (edge 쪽 약간 낮음)
            yy, xx = np.mgrid[0 : y2 - y1, 0 : x2 - x1]
            edge_d = np.minimum.reduce(
                [yy, xx, (y2 - y1 - 1) - yy, (x2 - x1 - 1) - xx]
            ).astype(np.float32)
            panel -= np.clip(3.0 - edge_d * 0.2, 0, 3.0)

            if label == "hotspot_single_cell":
                # 하나의 셀을 15K 가열
                cy = int((y2 - y1) * 0.4)
                cx = int((x2 - x1) * 0.3)
                ch = int((y2 - y1) / CELL_ROWS)
                cw = int((x2 - x1) / CELL_COLS)
                panel[cy : cy + ch, cx : cx + cw] += 18.0

            elif label == "hotspot_multi_cell":
                # 한 string (가로 1줄) 을 12K 가열
                row = int((y2 - y1) * 0.66)
                ch = int((y2 - y1) / CELL_ROWS)
                panel[row : row + ch, :] += 12.0

            elif label == "bypass_diode":
                # 3셀 서브스트링이 뜨거움
                cy = int((y2 - y1) * 0.2)
                ch = int((y2 - y1) / CELL_ROWS)
                cw = int((x2 - x1) / CELL_COLS)
                panel[cy : cy + ch, : cw * 3] += 10.0

            elif label == "soiling":
                # 먼지 영역은 오히려 저온 (복사 감소) — 약 -4K
                yy2 = int((y2 - y1) * 0.5)
                xx2 = int((x2 - x1) * 0.5)
                panel[yy2:, xx2:] -= 4.0

            elif label == "shading":
                # 그림자 쪽 영역이 저온 (-10K)
                panel[:, : (x2 - x1) // 2] -= 10.0

            panel += rng.normal(0, NOISE_STD_C, size=panel.shape)
            temp[y1:y2, x1:x2] = panel

    return temp


def _temp_c_to_centi_kelvin_u16(temp_c: np.ndarray) -> np.ndarray:
    ck = (temp_c + 273.15) * 100.0
    ck = np.clip(ck, 0, 65535)
    return ck.astype(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", default="samples", help="output directory")
    parser.add_argument("--prefix", default="sample", help="filename prefix")
    parser.add_argument("--preview", action="store_true", help="also save a JET heatmap preview")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 3x4 array 에 4 개 결함 + 1 개 그림자 심기
    defects: dict[tuple[int, int], str] = {
        (0, 0): "hotspot_single_cell",
        (1, 2): "hotspot_multi_cell",
        (2, 1): "bypass_diode",
        (2, 3): "soiling",
        (0, 3): "shading",
    }

    rgb = _make_rgb(defects)
    temp_c = _make_thermal(defects)
    ck_u16 = _temp_c_to_centi_kelvin_u16(temp_c)

    rgb_path = out / f"{args.prefix}_rgb.jpg"
    tiff_path = out / f"{args.prefix}_ir.tiff"
    cv2.imwrite(str(rgb_path), rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    tifffile.imwrite(
        str(tiff_path),
        ck_u16,
        photometric="minisblack",
        description="radiometric_tiff; unit=centi-Kelvin",
    )

    # 메타 요약
    meta = {
        "rgb": str(rgb_path),
        "ir_tiff": str(tiff_path),
        "image_size_hw": list(rgb.shape[:2]),
        "temp_range_c": [float(temp_c.min()), float(temp_c.max())],
        "ir_encoding": "uint16 centi-Kelvin (FLIR convention)",
        "panel_grid": {"rows": ROWS, "cols": COLS},
        "ground_truth": {f"{r}_{c}": label for (r, c), label in defects.items()},
    }
    (out / f"{args.prefix}_meta.json").write_text(
        __import__("json").dumps(meta, indent=2, ensure_ascii=False)
    )

    if args.preview:
        # INFERNO 컬러맵으로 열화상 프리뷰 저장
        lo = float(np.percentile(temp_c, 1))
        hi = float(np.percentile(temp_c, 99))
        norm = np.clip((temp_c - lo) / max(1e-6, hi - lo), 0, 1)
        gray = (norm * 255).astype(np.uint8)
        heat = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(out / f"{args.prefix}_ir_preview.jpg"), heat)

    print(f"[ok] wrote {rgb_path}")
    print(f"[ok] wrote {tiff_path}  ({ck_u16.dtype}, range={ck_u16.min()}..{ck_u16.max()})")
    print(
        f"[ok] temp_c range = [{temp_c.min():.1f}, {temp_c.max():.1f}] °C, "
        f"mean = {temp_c.mean():.1f} °C"
    )
    print(f"[ok] ground-truth labels: {meta['ground_truth']}")


if __name__ == "__main__":
    main()
