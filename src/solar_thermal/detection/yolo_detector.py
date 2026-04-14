"""Ultralytics YOLO 기반 태양광 패널/셀 검출기.

학습된 모델은 아래 클래스를 가진다고 가정:
    0: panel            # 전체 패널 (module)
    1: cell             # 개별 셀
    2: candidate_hotspot # (선택) 시각적으로 이상이 보이는 영역
    3: bypass_diode     # (선택) 바이패스 다이오드 영역

모델이 panel 클래스만 가지고 있어도 사용 가능하며,
셀 단위 정보는 panel bbox 에서 grid-based 분할로 대체할 수 있다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from ..cloud.storage import ensure_local
from ..schemas import BBox


class YOLODetector:
    """Ultralytics YOLO wrapper."""

    def __init__(
        self,
        weights: str,
        device: str = "cuda:0",
        imgsz: int = 1280,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
    ) -> None:
        # ultralytics 는 heavy dependency 이므로 lazy import
        from ultralytics import YOLO

        local_weights = ensure_local(weights)
        logger.info(f"Loading YOLO weights from {local_weights}")
        self.model = YOLO(str(local_weights))
        self.device = device
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        # 클래스 이름 맵
        self.names: dict[int, str] = self.model.names

    def predict(self, image: np.ndarray) -> list[BBox]:
        """단일 이미지 추론.

        Parameters
        ----------
        image : (H, W, 3) BGR uint8

        Returns
        -------
        list[BBox]
        """

        result = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )[0]

        boxes_out: list[BBox] = []
        if result.boxes is None:
            return boxes_out

        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            boxes_out.append(
                BBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=float(c),
                    class_name=self.names.get(int(k), str(int(k))),
                )
            )
        return boxes_out

    def panels(self, boxes: list[BBox]) -> list[BBox]:
        """panel 클래스만 필터링."""

        return [b for b in boxes if b.class_name.lower() in {"panel", "module"}]


def split_panel_into_cells(
    panel: BBox, rows: int = 6, cols: int = 12
) -> list[BBox]:
    """패널 bbox 를 균등 grid 로 분할하여 셀 bbox 추정.

    YOLO 가 cell 클래스를 가지고 있지 않을 때 fallback.
    기본 6x12 는 72-cell 모듈을 가정.
    """

    x1, y1, x2, y2 = panel.to_xyxy()
    w = (x2 - x1) / cols
    h = (y2 - y1) / rows
    cells: list[BBox] = []
    for r in range(rows):
        for c in range(cols):
            cells.append(
                BBox(
                    x1=x1 + c * w,
                    y1=y1 + r * h,
                    x2=x1 + (c + 1) * w,
                    y2=y1 + (r + 1) * h,
                    score=panel.score,
                    class_name="cell",
                )
            )
    return cells
