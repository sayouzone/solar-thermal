"""
태양광 패널 세그멘테이션 - SAM 2.1 + CLIP 제로샷 접근법
=======================================================
학습 데이터가 적거나 없을 때, 픽셀 단위 마스크가 필요할 때 사용.
Active Learning 의 초기 라벨 부트스트랩에 특히 유용.

파이프라인:
  1) 그리드 포인트 또는 bbox 프롬프트로 SAM 2.1 이 이미지 전체를 세그먼트
  2) 각 마스크 영역을 crop → CLIP 으로 "solar panel" 의미 유사도 계산
  3) 임계값 넘는 마스크만 유지

설치:
    pip install torch torchvision
    pip install git+https://github.com/facebookresearch/sam2.git
    pip install open_clip_torch
"""

from __future__ import annotations
import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import List


@dataclass
class SamClipDetection:
    mask: np.ndarray                # H x W 불리언 마스크
    bbox: tuple[int, int, int, int] # x, y, w, h
    clip_score: float
    sam_score: float


class SamClipSolarDetector:
    """
    SAM 2.1 로 클래스 무관 세그멘테이션 → CLIP 으로 의미 필터링.
    """

    TEXT_PROMPTS_POSITIVE = [
        "a solar panel",
        "photovoltaic module",
        "aerial view of solar panels",
        "rooftop solar array",
    ]
    TEXT_PROMPTS_NEGATIVE = [
        "a roof",
        "a road",
        "grass",
        "a car",
        "a building without solar panels",
        "water",
        "soil",
    ]

    def __init__(
        self,
        sam_checkpoint: str,
        sam_config: str = "sam2_hiera_l.yaml",
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        clip_threshold: float = 0.25,  # positive 확률이 이 값 이상이면 PV 로 간주
        min_area_px: int = 400,
    ):
        self.device = device
        self.clip_threshold = clip_threshold
        self.min_area_px = min_area_px

        # --- SAM 2.1 ---
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam_model = build_sam2(sam_config, sam_checkpoint, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam_model,
            points_per_side=32,
            pred_iou_thresh=0.85,
            stability_score_thresh=0.92,
            min_mask_region_area=self.min_area_px,
        )

        # --- CLIP ---
        import open_clip
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained, device=device
        )
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model)
        self.clip_model.eval()

        # 텍스트 임베딩 미리 계산
        all_prompts = self.TEXT_PROMPTS_POSITIVE + self.TEXT_PROMPTS_NEGATIVE
        with torch.no_grad():
            tokens = self.clip_tokenizer(all_prompts).to(device)
            text_feats = self.clip_model.encode_text(tokens)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
        self.text_feats = text_feats
        self.num_positive = len(self.TEXT_PROMPTS_POSITIVE)

    def _score_crop_with_clip(self, crop_bgr: np.ndarray) -> float:
        """crop 이 solar panel 일 확률 (softmax 합)."""
        from PIL import Image
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        img_t = self.clip_preprocess(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat = self.clip_model.encode_image(img_t)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            logits = (img_feat @ self.text_feats.T).squeeze(0) * 100
            probs = logits.softmax(dim=-1).cpu().numpy()

        positive_prob = float(probs[: self.num_positive].sum())
        return positive_prob

    def detect(self, image_bgr: np.ndarray) -> List[SamClipDetection]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)

        H, W = image_bgr.shape[:2]
        results: List[SamClipDetection] = []

        for m in masks:
            seg = m["segmentation"]                 # HxW bool
            x, y, w, h = map(int, m["bbox"])        # SAM bbox (x, y, w, h)
            if w * h < self.min_area_px:
                continue

            x2 = min(W, x + w)
            y2 = min(H, y + h)
            crop = image_bgr[y:y2, x:x2].copy()

            # 마스크 밖을 검게 가려서 CLIP 이 배경에 속지 않게
            local_mask = seg[y:y2, x:x2]
            crop[~local_mask] = 0

            score = self._score_crop_with_clip(crop)
            if score >= self.clip_threshold:
                results.append(
                    SamClipDetection(
                        mask=seg,
                        bbox=(x, y, w, h),
                        clip_score=score,
                        sam_score=float(m.get("predicted_iou", 0.0)),
                    )
                )

        results.sort(key=lambda d: d.clip_score, reverse=True)
        return results

    @staticmethod
    def draw(image_bgr: np.ndarray, detections: List[SamClipDetection]) -> np.ndarray:
        overlay = image_bgr.copy()
        rng = np.random.default_rng(42)
        for d in detections:
            color = tuple(int(c) for c in rng.integers(64, 255, size=3))
            overlay[d.mask] = (0.5 * overlay[d.mask] + 0.5 * np.array(color)).astype(np.uint8)
            x, y, w, h = d.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                overlay, f"PV {d.clip_score:.2f}", (x, max(y - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        return overlay


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--sam-ckpt", required=True, help="sam2_hiera_large.pt 등")
    parser.add_argument("--sam-cfg", default="sam2_hiera_l.yaml")
    parser.add_argument("--out", default="sam_clip_result.jpg")
    parser.add_argument("--thr", type=float, default=0.25)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    det = SamClipSolarDetector(
        sam_checkpoint=args.sam_ckpt,
        sam_config=args.sam_cfg,
        clip_threshold=args.thr,
    )
    results = det.detect(img)
    print(f"[INFO] PV 마스크: {len(results)}개")
    vis = det.draw(img, results)
    cv2.imwrite(args.out, vis)


if __name__ == "__main__":
    main()
