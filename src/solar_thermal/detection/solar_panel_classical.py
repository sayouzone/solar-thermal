"""
태양광 패널 탐지 - 고전 컴퓨터 비전 접근법 (OpenCV)
====================================================
학습 데이터 없이 즉시 동작. 공중 촬영 / 지상 촬영 모두 사용 가능.
PV 패널의 특성(짙은 청색~검은색, 직사각형 격자, 균일한 표면)을 이용.

장점: 학습 불필요, CPU 실시간
한계: 조명/각도 변화에 민감, 흰색 모듈이나 곡면 패널은 놓칠 수 있음
권장 용도: 초기 마스크 생성, Active Learning 라벨 부트스트랩
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PanelDetection:
    """탐지된 패널 하나의 정보"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    contour: np.ndarray
    area: float
    aspect_ratio: float
    rectangularity: float  # 윤곽 영역 / 바운딩박스 영역 (1.0에 가까울수록 직사각형)
    score: float


class ClassicalSolarPanelDetector:
    """
    색상 + 형태 기반 태양광 패널 탐지기.
    """

    # HSV 색공간에서의 PV 패널 전형적 범위
    # 결정질 실리콘: 짙은 청색~검은색 (H: 90~130, S: 20~255, V: 20~120)
    HSV_LOWER = np.array([90, 20, 20])
    HSV_UPPER = np.array([130, 255, 120])

    def __init__(
        self,
        min_area_ratio: float = 0.001,   # 이미지 대비 최소 면적 비율
        max_area_ratio: float = 0.5,     # 이미지 대비 최대 면적 비율
        min_aspect_ratio: float = 1.2,   # 패널은 보통 가로가 세로보다 김
        max_aspect_ratio: float = 3.0,
        min_rectangularity: float = 0.75,
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_rectangularity = min_rectangularity

    def _color_mask(self, bgr: np.ndarray) -> np.ndarray:
        """청색/검은색 계열의 픽셀 마스크 생성."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.HSV_LOWER, self.HSV_UPPER)

        # 형태학적 연산으로 노이즈 제거 및 패널 내부 채우기
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _filter_contour(
        self, contour: np.ndarray, img_area: float
    ) -> PanelDetection | None:
        """단일 윤곽선이 패널 조건을 만족하는지 검사."""
        area = cv2.contourArea(contour)
        if not (self.min_area_ratio * img_area <= area <= self.max_area_ratio * img_area):
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            return None

        aspect = max(w, h) / min(w, h)
        if not (self.min_aspect_ratio <= aspect <= self.max_aspect_ratio):
            return None

        bbox_area = w * h
        rectangularity = area / bbox_area if bbox_area > 0 else 0.0
        if rectangularity < self.min_rectangularity:
            return None

        # 간단한 스코어: 직사각형일수록, 적절한 비율일수록 높음
        score = rectangularity * (1.0 - abs(aspect - 1.75) / 1.75)

        return PanelDetection(
            bbox=(x, y, w, h),
            contour=contour,
            area=area,
            aspect_ratio=aspect,
            rectangularity=rectangularity,
            score=float(score),
        )

    def detect(self, image: np.ndarray) -> List[PanelDetection]:
        """
        이미지에서 태양광 패널 후보를 모두 탐지.

        Args:
            image: BGR 형식의 OpenCV 이미지 (H, W, 3)

        Returns:
            탐지된 패널 리스트 (score 내림차순)
        """
        if image is None or image.size == 0:
            return []

        img_area = image.shape[0] * image.shape[1]
        mask = self._color_mask(image)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[PanelDetection] = []
        for c in contours:
            det = self._filter_contour(c, img_area)
            if det is not None:
                detections.append(det)

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    @staticmethod
    def draw(image: np.ndarray, detections: List[PanelDetection]) -> np.ndarray:
        """탐지 결과를 시각화."""
        vis = image.copy()
        for i, d in enumerate(detections):
            x, y, w, h = d.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"PV#{i} score={d.score:.2f}"
            cv2.putText(
                vis, label, (x, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        return vis


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="입력 이미지 경로")
    parser.add_argument("--out", default="detection_result.jpg", help="결과 저장 경로")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {args.image}")

    detector = ClassicalSolarPanelDetector()
    detections = detector.detect(img)
    print(f"[INFO] 탐지된 패널 후보: {len(detections)}개")
    for i, d in enumerate(detections[:10]):
        print(f"  #{i}: bbox={d.bbox}, aspect={d.aspect_ratio:.2f}, "
              f"rect={d.rectangularity:.2f}, score={d.score:.3f}")

    vis = detector.draw(img, detections)
    cv2.imwrite(args.out, vis)
    print(f"[INFO] 결과 저장: {args.out}")


if __name__ == "__main__":
    main()
