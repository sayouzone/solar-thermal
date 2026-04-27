"""
PV String & Module 계층적 검출기
==================================

드론 nadir view에서 태양광 패널을 두 가지 단위로 검출:
  - pv_string: 세로로 긴 1개 줄 (여러 모듈의 직렬 배열)
  - pv_module: 개별 패널 프레임 (string 내부의 subdivision)

전략:
  1) String 검출 (큰 단위):
     - SAM2 automatic segmentation
     - Elongation (세로 aspect ratio > 3) 필터
     - NMS 후 string 후보 확정
  2) Module 검출 (작은 단위):
     - 각 string 내부에서 밝기 프로파일(vertical projection) 분석
     - 어두운 프레임 경계(module gap)에서 분할
     - 또는 학습된 YOLO module-level model 사용 (self-training 이후)

주의: heuristic은 HSV gray 범위에서 잔디를 오탐하므로 제거.
      SAM2 → 밝기/형태 필터로 훨씬 안정적.

요구사항: pip install ultralytics opencv-python numpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures (hierarchical)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PixelBBox:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int = 0
    score: float = 1.0

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def aspect_long(self) -> float:
        """긴 쪽 / 짧은 쪽. 패널은 >= 3 정도."""
        w, h = self.width, self.height
        if w == 0 or h == 0:
            return 0.0
        return max(w, h) / min(w, h)

    def iou(self, other: "PixelBBox") -> float:
        xi1 = max(self.x1, other.x1); yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2); yi2 = min(self.y2, other.y2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter == 0:
            return 0.0
        return inter / (self.area + other.area - inter)

    def contains(self, other: "PixelBBox", threshold: float = 0.8) -> bool:
        """other가 self 내부에 threshold 이상 들어있는가."""
        xi1 = max(self.x1, other.x1); yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2); yi2 = min(self.y2, other.y2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if other.area == 0:
            return False
        return (inter / other.area) >= threshold


@dataclass
class HierarchicalDetection:
    """한 이미지의 계층적 검출 결과: string과 module."""
    strings: list[PixelBBox] = field(default_factory=list)
    modules: list[PixelBBox] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)  # (H, W)

    def assign_modules_to_strings(self) -> dict[int, list[int]]:
        """각 string에 포함된 module 인덱스 반환."""
        mapping: dict[int, list[int]] = {i: [] for i in range(len(self.strings))}
        for m_idx, module in enumerate(self.modules):
            best_string_idx = -1
            best_overlap = 0.0
            for s_idx, string_box in enumerate(self.strings):
                xi1 = max(string_box.x1, module.x1)
                yi1 = max(string_box.y1, module.y1)
                xi2 = min(string_box.x2, module.x2)
                yi2 = min(string_box.y2, module.y2)
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                if module.area == 0:
                    continue
                overlap = inter / module.area
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_string_idx = s_idx
            if best_string_idx >= 0 and best_overlap > 0.5:
                mapping[best_string_idx].append(m_idx)
        return mapping

    def to_yolo_lines(
        self,
        class_map: dict[str, int] | None = None,
    ) -> list[str]:
        """
        YOLO 포맷 라인 리스트로 변환.

        Args:
            class_map: {"pv_string": 0, "pv_module": 1}
                       None이면 각 bbox의 class_id 그대로 사용
        """
        H, W = self.image_shape
        if H == 0 or W == 0:
            return []
        lines = []
        for box in list(self.strings) + list(self.modules):
            if class_map is not None:
                # 기본 규칙: strings 리스트의 것은 pv_string, modules는 pv_module
                if box in self.strings:
                    cid = class_map.get("pv_string", 0)
                else:
                    cid = class_map.get("pv_module", 1)
            else:
                cid = box.class_id
            cx = box.cx / W
            cy = box.cy / H
            w = box.width / W
            h = box.height / H
            lines.append(
                f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            )
        return lines


# ---------------------------------------------------------------------------
# NMS utility (재사용)
# ---------------------------------------------------------------------------

def nms(
    boxes: list[PixelBBox],
    iou_threshold: float = 0.3,
    containment_threshold: float = 0.85,
) -> list[PixelBBox]:
    """Greedy NMS + containment 기반 중복 제거."""
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    keep: list[PixelBBox] = []
    for box in sorted_boxes:
        should_keep = True
        for kept in keep:
            if box.iou(kept) > iou_threshold:
                should_keep = False
                break
            if kept.contains(box, threshold=containment_threshold):
                should_keep = False
                break
        if should_keep:
            keep.append(box)
    return keep


# ---------------------------------------------------------------------------
# String detector (SAM2 based, no heuristic HSV)
# ---------------------------------------------------------------------------

class PVStringDetector:
    """
    SAM2 automatic segmentation 기반 PV string 검출기.

    이전 heuristic(HSV mask)의 문제:
      - 마른 잔디가 gray 범위에 들어가 오탐
      - 방수포(tarp)가 blue 범위에 들어가 오탐
      - module 사이 프레임 간격에서 contour가 분리됨

    SAM2 기반 접근:
      - 모델이 "일관된 재질/질감 영역"을 자동 식별
      - PV 패널의 규칙적 격자 + 금속 프레임이 뚜렷한 segment를 형성
      - HSV 휴리스틱 없이 aspect ratio와 크기 필터만으로 string 선별

    요구사항: pip install ultralytics
    """

    MIN_ASPECT_STRING = 3.5      # string은 매우 길쭉
    MIN_AREA_RATIO   = 0.002    # 이미지의 0.2% 이상
    MAX_AREA_RATIO   = 0.25     # 25% 초과는 배경

    def __init__(
        self,
        sam_model: str = "sam2_b.pt",
        string_class_id: int = 0,
        min_aspect: float | None = None,
    ):
        try:
            from ultralytics import SAM
        except ImportError as e:
            raise ImportError(
                "SAM2 사용 시 `pip install ultralytics` 필요"
            ) from e
        self.sam = SAM(sam_model)
        self.string_class_id = string_class_id
        self.min_aspect = min_aspect or self.MIN_ASPECT_STRING

    def detect(self, image_path: Path) -> list[PixelBBox]:
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        H, W = img.shape[:2]
        img_area = H * W

        # SAM2 auto segmentation
        results = self.sam(str(image_path), verbose=False)

        candidates: list[PixelBBox] = []
        for r in results:
            if r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy()
            for m in masks:
                ys, xs = np.where(m > 0.5)
                if len(xs) < 100:
                    continue
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                bw, bh = x2 - x1, y2 - y1
                if bw < 20 or bh < 20:
                    continue
                # Area filter
                area_ratio = (bw * bh) / img_area
                if area_ratio < self.MIN_AREA_RATIO or area_ratio > self.MAX_AREA_RATIO:
                    continue
                # Aspect filter — string은 매우 길쭉
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if aspect < self.min_aspect:
                    continue

                # 마스크 충실도: bbox 대비 실제 mask fill ratio
                # PV 패널은 직사각형이라 fill ratio가 높음 (> 0.5)
                mask_area = len(xs)
                fill_ratio = mask_area / (bw * bh)
                if fill_ratio < 0.4:
                    continue

                candidates.append(PixelBBox(
                    x1, y1, x2, y2,
                    class_id=self.string_class_id,
                    score=fill_ratio,  # fill이 높을수록 확실
                ))

        # NMS로 중복 제거 (SAM이 여러 scale에서 같은 영역 segment하는 경우)
        return nms(candidates, iou_threshold=0.3, containment_threshold=0.85)


# ---------------------------------------------------------------------------
# Module detector (string 내부 분할)
# ---------------------------------------------------------------------------

class PVModuleSplitter:
    """
    String bbox 내부를 module 단위로 분할.

    전략:
      1) String을 세로 방향으로 rotation-aware crop
      2) 세로 pixel intensity profile 계산
      3) 프레임(어두운 가로 선)을 valley로 detect
      4) 각 valley 사이를 module 1개로 분할

    이 방법은 pre-trained YOLO가 없어도 작동하며,
    self-training으로 YOLO module detector를 학습시킨 후에는
    `YOLOModuleDetector`로 대체 가능.
    """

    def __init__(
        self,
        module_class_id: int = 1,
        expected_cells_per_module: int = 3,  # 세로 cell 개수 (일반 60셀 패널)
        min_module_height_px: int = 40,
    ):
        self.module_class_id = module_class_id
        self.expected_cells_per_module = expected_cells_per_module
        self.min_module_height_px = min_module_height_px

    def split(
        self, img: np.ndarray, string_box: PixelBBox,
    ) -> list[PixelBBox]:
        """단일 string bbox를 여러 module bbox로 분할."""
        # Crop string ROI
        x1, y1, x2, y2 = string_box.x1, string_box.y1, string_box.x2, string_box.y2
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        bh, bw = roi.shape[:2]
        is_vertical = bh > bw  # 대부분의 경우 string은 세로로 길쭉

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if is_vertical:
            # 세로 string: 가로 방향으로 평균 → 세로 프로파일
            profile = gray.mean(axis=1)  # shape (bh,)
            split_axis_len = bh
        else:
            profile = gray.mean(axis=0)
            split_axis_len = bw

        # Valley detection: module 경계는 프레임(어두운 선)으로 나타남
        valleys = self._detect_valleys(profile)

        if len(valleys) < 1:
            # 분할점 없음 → 전체를 하나의 module로 취급 (string과 같은 bbox)
            return [PixelBBox(
                string_box.x1, string_box.y1, string_box.x2, string_box.y2,
                class_id=self.module_class_id, score=string_box.score,
            )]

        # Valley 위치로 module 분할
        boundaries = [0] + list(valleys) + [split_axis_len]
        modules: list[PixelBBox] = []
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            if (b - a) < self.min_module_height_px:
                continue
            if is_vertical:
                mx1 = x1
                mx2 = x2
                my1 = y1 + a
                my2 = y1 + b
            else:
                mx1 = x1 + a
                mx2 = x1 + b
                my1 = y1
                my2 = y2
            modules.append(PixelBBox(
                int(mx1), int(my1), int(mx2), int(my2),
                class_id=self.module_class_id,
                score=string_box.score,
            ))
        return modules

    @staticmethod
    def _detect_valleys(profile: np.ndarray) -> list[int]:
        """
        1D intensity profile에서 local minima (module 간 프레임) 검출.

        방법:
          1) Smoothing으로 셀 내부 노이즈 제거 (cell 자체 격자가 아닌 module 경계만 잡기 위해)
          2) Rolling min보다 작은 위치를 valley로 선정
          3) 간격이 너무 가까운 valley는 병합
        """
        # Large smoothing kernel to remove cell-level pattern,
        # keep only module-level frame dips
        kernel_size = max(15, len(profile) // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = cv2.GaussianBlur(
            profile.astype(np.float32).reshape(-1, 1),
            (1, kernel_size), 0,
        ).ravel()

        # 1st derivative: 부호 바뀌는 곳 중 음→양 전환이 local minimum
        diff = np.diff(smoothed)
        # Sign change 찾기
        sign_change = np.where(np.diff(np.sign(diff)) > 0)[0] + 1

        if len(sign_change) == 0:
            return []

        # Filter: 프로파일 평균 대비 충분히 낮은 valley만 (진짜 프레임)
        mean_val = smoothed.mean()
        threshold_val = mean_val * 0.92  # 평균의 92% 미만인 valley만
        significant = [int(i) for i in sign_change if smoothed[i] < threshold_val]

        # 너무 가까운 valley 병합 (가장 깊은 것만 유지)
        if len(significant) > 1:
            min_gap = max(30, len(profile) // 30)
            merged: list[int] = []
            for v in significant:
                if not merged or (v - merged[-1]) >= min_gap:
                    merged.append(v)
                else:
                    # 더 깊은 것으로 교체
                    if smoothed[v] < smoothed[merged[-1]]:
                        merged[-1] = v
            significant = merged

        return significant


# ---------------------------------------------------------------------------
# Main hierarchical detector (orchestrator)
# ---------------------------------------------------------------------------

class HierarchicalPVDetector:
    """
    String → Module 순차 검출 orchestrator.

    사용법:
        detector = HierarchicalPVDetector()
        result = detector.detect("drone_image.jpg")
        result.to_yolo_lines({"pv_string": 0, "pv_module": 1})
    """

    def __init__(
        self,
        string_detector: PVStringDetector | None = None,
        module_splitter: PVModuleSplitter | None = None,
        detect_modules: bool = True,
    ):
        self.string_detector = string_detector
        self.module_splitter = module_splitter or PVModuleSplitter()
        self.detect_modules = detect_modules

    def detect(self, image_path: Path) -> HierarchicalDetection:
        img = cv2.imread(str(image_path))
        if img is None:
            return HierarchicalDetection()
        H, W = img.shape[:2]

        # 1) String detection
        if self.string_detector is None:
            # Lazy init (import heavy SAM2 only when needed)
            self.string_detector = PVStringDetector()
        strings = self.string_detector.detect(image_path)

        # 2) Module splitting (선택적)
        modules: list[PixelBBox] = []
        if self.detect_modules:
            for s in strings:
                modules.extend(self.module_splitter.split(img, s))

        return HierarchicalDetection(
            strings=strings,
            modules=modules,
            image_shape=(H, W),
        )
