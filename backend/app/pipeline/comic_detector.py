import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

INPUT_SIZE = 1024


@dataclass
class ComicRegion:
    """A single speech bubble / text block detected by comic-text-detector ONNX."""

    bbox: Tuple[int, int, int, int]
    polygon: List[List[float]]
    confidence: float
    is_vertical: bool = False
    line_count: int = 1
    line_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    seg_mask: Optional[np.ndarray] = None  # Full-image uint8 mask (0 or 255) for this block


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


def _containment_ratio(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_inner = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / float(area_inner)


def _deduplicate_nested_boxes(
    boxes: List[Tuple[int, int, int, int, float]],
) -> List[Tuple[int, int, int, int, float]]:
    """
    Merge duplicate / nested YOLO detections (e.g. one column detected inside a full speech bubble).
    Each entry is (x1, y1, x2, y2, confidence).
    """
    if len(boxes) <= 1:
        return boxes

    # Sort larger area first so enclosing bubble blocks absorb partial inner duplicates
    ordered = sorted(boxes, key=lambda b: ((b[2] - b[0]) * (b[3] - b[1]), b[4]), reverse=True)
    kept: List[Tuple[int, int, int, int, float]] = []

    for cand in ordered:
        c_box = cand[:4]
        c_area = max(1, (c_box[2] - c_box[0]) * (c_box[3] - c_box[1]))
        absorbed = False
        for i, existing in enumerate(kept):
            e_box = existing[:4]
            e_area = max(1, (e_box[2] - e_box[0]) * (e_box[3] - e_box[1]))
            if _iou(c_box, e_box) >= 0.55 or (
                _containment_ratio(c_box, e_box) >= 0.78 and e_area <= c_area * 4.0
            ):
                # Keep enclosing box with max confidence
                if cand[4] > existing[4]:
                    kept[i] = (e_box[0], e_box[1], e_box[2], e_box[3], cand[4])
                absorbed = True
                break
        if not absorbed:
            kept.append(cand)

    return kept


def sort_comic_regions(
    regions: List[ComicRegion],
    img_height: int,
    manga_rtl: bool = False,
) -> List[ComicRegion]:
    """
    Order speech bubbles in natural reading order so context-aware translators (like DeepSeek)
    receive comic dialogue in exact narrative sequence:
      - Manga (vertical/Japanese): top-to-bottom bands, right-to-left within each band
      - Western comics / memes: top-to-bottom bands, left-to-right within each band
    """
    if len(regions) <= 1:
        return list(regions)

    band_h = max(72, int(img_height * 0.14))

    def _sort_key(r: ComicRegion) -> Tuple[int, int, int]:
        x1, y1, x2, y2 = r.bbox
        band = y1 // band_h
        if manga_rtl:
            return (band, -x2, y1)
        return (band, x1, y1)

    return sorted(regions, key=_sort_key)


def postprocess_comic_outputs(
    blk_raw: np.ndarray,
    seg_raw: np.ndarray,
    det_raw: np.ndarray,
    orig_shape: Tuple[int, int],
    scale: float,
    new_w: int,
    new_h: int,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    source_lang: str = "auto",
) -> List[ComicRegion]:
    """
    Post-process raw ONNX tensors (`blk`, `seg`, `det`) from comic-text-detector into
    grouped speech-bubble `ComicRegion` objects with per-block learned pixel ink masks.
    """
    h_img, w_img = orig_shape

    # 1. Un-letterbox UNet pixel segmentation mask and DBNet line probability map
    seg_crop = seg_raw[0, 0, :new_h, :new_w]
    seg_prob = cv2.resize(seg_crop, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    seg_bin_global = (seg_prob > 0.30).astype(np.uint8) * 255

    det_crop = det_raw[0, 0, :new_h, :new_w]
    det_prob = cv2.resize(det_crop, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    det_bin = (det_prob > 0.35).astype(np.uint8) * 255

    # Extract DBNet text-line bounding boxes for orientation & line-count inference
    contours, _ = cv2.findContours(det_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue
        lx, ly, lw, lh = cv2.boundingRect(cnt)
        if max(lw, lh) < 6:
            continue
        line_boxes.append((lx, ly, lx + lw, ly + lh))

    # 2. Decode YOLOv5 `blk` predictions: [cx, cy, w, h, obj_conf, cls0, cls1]
    preds = blk_raw[0]
    obj_conf = preds[:, 4]
    cls_max = preds[:, 5:].max(axis=1)
    scores = obj_conf * cls_max
    valid = scores >= conf_thresh

    raw_boxes: List[Tuple[int, int, int, int, float]] = []
    if np.any(valid):
        f_preds = preds[valid]
        f_scores = scores[valid]
        nms_boxes = [
            [
                float(p[0] - p[2] / 2.0),
                float(p[1] - p[3] / 2.0),
                float(p[2]),
                float(p[3]),
            ]
            for p in f_preds
        ]
        keep = cv2.dnn.NMSBoxes(nms_boxes, f_scores.tolist(), conf_thresh, iou_thresh)
        if len(keep) > 0:
            for idx in np.array(keep).flatten():
                x, y, bw, bh = nms_boxes[int(idx)]
                x1 = max(0, int(np.floor(x / scale)))
                y1 = max(0, int(np.floor(y / scale)))
                x2 = min(w_img, int(np.ceil((x + bw) / scale)))
                y2 = min(h_img, int(np.ceil((y + bh) / scale)))
                if (x2 - x1) < settings.ocr_min_box_height or (y2 - y1) < settings.ocr_min_box_height:
                    continue
                raw_boxes.append((x1, y1, x2, y2, float(f_scores[int(idx)])))

    merged_boxes = _deduplicate_nested_boxes(raw_boxes)

    # Recover high-confidence UNet `seg` text clusters missed by YOLOv5 `blk`
    # (e.g. standalone dramatic captions / white text directly on dark manga panels),
    # while rejecting tiny artwork specks and giant decorative SFX strokes.
    covered = np.zeros((h_img, w_img), dtype=np.uint8)
    for bx1, by1, bx2, by2, _ in merged_boxes:
        cv2.rectangle(
            covered,
            (max(0, bx1 - 10), max(0, by1 - 10)),
            (min(w_img - 1, bx2 + 10), min(h_img - 1, by2 + 10)),
            255,
            thickness=-1,
        )
    uncovered_seg = ((seg_prob > 0.65) & (covered == 0)).astype(np.uint8) * 255
    min_uncov_px = max(220, int(h_img * w_img * 0.00035))
    max_uncov_px = int(h_img * w_img * 0.06)
    if int(np.count_nonzero(uncovered_seg)) >= min_uncov_px:
        # Group nearby glyphs & wide fullwidth punctuation of the same uncovered caption cleanly
        k_grp = cv2.getStructuringElement(cv2.MORPH_RECT, (43, 21))
        grouped_uncov = cv2.dilate(uncovered_seg, k_grp)
        num_u, u_labels, u_stats, _ = cv2.connectedComponentsWithStats(grouped_uncov, connectivity=8)
        for uid in range(1, num_u):
            ux, uy, uw, uh, _ = u_stats[uid]
            if uw < 20 or uh < 18:
                continue
            comp_mask = (u_labels == uid) & (uncovered_seg > 0)
            raw_px = int(np.count_nonzero(comp_mask))
            if raw_px < min_uncov_px or raw_px > max_uncov_px:
                continue
            ys_u, xs_u = np.where(comp_mask)
            tx1, ty1 = int(xs_u.min()), int(ys_u.min())
            tx2, ty2 = int(xs_u.max()) + 1, int(ys_u.max()) + 1
            bw_u, bh_u = tx2 - tx1, ty2 - ty1
            if bw_u < 20 or bh_u < 18:
                continue
            # Reject sparse scattered artwork specks (real captions have decent ink density)
            density = raw_px / float(max(1, bw_u * bh_u))
            peak_prob = float(seg_prob[ty1:ty2, tx1:tx2].max())
            if peak_prob >= 0.85 and density >= 0.08:
                merged_boxes.append((tx1, ty1, tx2, ty2, round(min(0.85, peak_prob * 0.8), 3)))

    if not merged_boxes:
        return []

    regions: List[ComicRegion] = []
    for bx1, by1, bx2, by2, conf in merged_boxes:
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)

        # Verify the region actually contains learned text probability (filters artwork false positives)
        seg_slice = seg_prob[by1:by2, bx1:bx2]
        det_slice = det_prob[by1:by2, bx1:bx2]
        if (seg_slice.size == 0 or float(seg_slice.max()) < 0.25) and (
            det_slice.size == 0 or float(det_slice.max()) < 0.25
        ):
            continue

        # Collect DBNet line contours belonging to this speech bubble
        inside_lines: List[Tuple[int, int, int, int]] = []
        for lb in line_boxes:
            lcx = (lb[0] + lb[2]) / 2.0
            lcy = (lb[1] + lb[3]) / 2.0
            if (bx1 - 8) <= lcx <= (bx2 + 8) and (by1 - 8) <= lcy <= (by2 + 8):
                inside_lines.append(lb)

        if inside_lines:
            vert_votes = sum(1 for lb in inside_lines if (lb[3] - lb[1]) > (lb[2] - lb[0]) * 1.35)
            horiz_votes = sum(1 for lb in inside_lines if (lb[2] - lb[0]) >= (lb[3] - lb[1]) * 1.1)
            is_vertical = vert_votes > horiz_votes
        else:
            is_vertical = bh > bw * 1.45

        # Extract per-block learned pixel ink mask: keep connected components in `seg_bin_global`
        # inside a small +8px margin that actually overlap the core block bbox.
        pad = 8
        rx1 = max(0, bx1 - pad)
        ry1 = max(0, by1 - pad)
        rx2 = min(w_img, bx2 + pad)
        ry2 = min(h_img, by2 + pad)

        local_seg = seg_bin_global[ry1:ry2, rx1:rx2]
        block_seg_full = np.zeros((h_img, w_img), dtype=np.uint8)
        if np.count_nonzero(local_seg) > 0:
            num_cc, cc_labels = cv2.connectedComponents(local_seg, connectivity=8)
            core_slice = cc_labels[(by1 - ry1) : (by2 - ry1), (bx1 - rx1) : (bx2 - rx1)]
            seed_ids = set(np.unique(core_slice)) - {0}
            if seed_ids:
                kept_local = np.isin(cc_labels, list(seed_ids)).astype(np.uint8) * 255
                block_seg_full[ry1:ry2, rx1:rx2] = kept_local

        polygon = [
            [float(bx1), float(by1)],
            [float(bx2), float(by1)],
            [float(bx2), float(by2)],
            [float(bx1), float(by2)],
        ]

        regions.append(
            ComicRegion(
                bbox=(bx1, by1, bx2, by2),
                polygon=polygon,
                confidence=round(conf, 3),
                is_vertical=is_vertical,
                line_count=max(1, len(inside_lines)),
                line_bboxes=inside_lines,
                seg_mask=block_seg_full,
            )
        )

    if not regions:
        return []

    vertical_count = sum(1 for r in regions if r.is_vertical)
    manga_rtl = (source_lang == "ja") or (vertical_count >= max(1, len(regions) / 2.0))
    return sort_comic_regions(regions, img_height=h_img, manga_rtl=manga_rtl)


class ComicTextDetector:
    """
    ONNX comic/manga speech-bubble detector + UNet pixel-exact ink segmentation (`comic-text-detector`).
    Uses OpenCV DNN (`cv2.dnn.readNetFromONNX`), which runs ~14x faster than ORT CPU on this Conv2D model (~3s).
    Auto-downloads ~95 MB ONNX weights into `backend/.cache/models/comic-text-detector/` on first use.
    """

    def __init__(self) -> None:
        self._net = None
        self._lock = threading.Lock()
        self._load_failed = False

    def model_path(self) -> Path:
        return settings.models_dir / "comic-text-detector" / settings.comic_detector_hf_filename

    def is_downloaded(self) -> bool:
        p = self.model_path()
        return p.is_file() and p.stat().st_size > 10_000_000

    def _ensure_loaded(self) -> bool:
        if self._net is not None:
            return True
        if self._load_failed:
            return False

        with self._lock:
            if self._net is not None:
                return True
            target = self.model_path()
            if not self.is_downloaded():
                try:
                    from huggingface_hub import hf_hub_download

                    target.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(
                        "Downloading comic-text-detector ONNX (%s/%s)...",
                        settings.comic_detector_hf_repo,
                        settings.comic_detector_hf_filename,
                    )
                    hf_hub_download(
                        repo_id=settings.comic_detector_hf_repo,
                        filename=settings.comic_detector_hf_filename,
                        local_dir=str(target.parent),
                    )
                except Exception as exc:
                    logger.warning("Could not download comic-text-detector ONNX (%s); falling back to RapidOCR.", exc)
                    self._load_failed = True
                    return False

            try:
                logger.info("Loading comic-text-detector ONNX via OpenCV DNN...")
                self._net = cv2.dnn.readNetFromONNX(str(target))
                return True
            except Exception as exc:
                logger.warning("Failed loading comic-text-detector ONNX (%s); falling back to RapidOCR.", exc)
                self._load_failed = True
                return False

    def detect(
        self,
        image_rgb: np.ndarray,
        source_lang: str = "auto",
        conf_thresh: Optional[float] = None,
    ) -> Optional[List[ComicRegion]]:
        """
        Detect speech-bubble blocks + learned pixel ink masks.
        Returns `None` if the ONNX detector is unavailable, or a list of `ComicRegion` (possibly empty).
        """
        if not self._ensure_loaded():
            return None

        h_img, w_img = image_rgb.shape[:2]
        scale = INPUT_SIZE / float(max(h_img, w_img))
        new_w = max(1, min(INPUT_SIZE, int(round(w_img * scale))))
        new_h = max(1, min(INPUT_SIZE, int(round(h_img * scale))))

        resized = cv2.resize(
            image_rgb,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        blob = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        with self._lock:
            self._net.setInput(blob)
            blk_raw, seg_raw, det_raw = self._net.forward(["blk", "seg", "det"])

        return postprocess_comic_outputs(
            blk_raw=blk_raw,
            seg_raw=seg_raw,
            det_raw=det_raw,
            orig_shape=(h_img, w_img),
            scale=scale,
            new_w=new_w,
            new_h=new_h,
            conf_thresh=conf_thresh if conf_thresh is not None else settings.comic_detector_conf,
            iou_thresh=settings.comic_detector_iou,
            source_lang=source_lang,
        )


comic_detector = ComicTextDetector()
