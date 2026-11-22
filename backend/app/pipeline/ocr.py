import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class OCRLine:
    polygon: List[List[float]]       # 4 points [[x,y], ...]
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    height: float
    angle: float                     # degrees


@dataclass
class OCRBlock:
    id: str
    lines: List[OCRLine] = field(default_factory=list)
    polygon: List[List[float]] = field(default_factory=list)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    text: str = ""
    confidence: float = 1.0
    angle: float = 0.0
    alignment: str = "center"


class OCREngine:
    """Lazy-initialized multilingual OCR using RapidOCR (ONNX PaddleOCR)."""

    def __init__(self) -> None:
        self._engine = None

    def _ensure_loaded(self) -> None:
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            logger.info("Initializing RapidOCR ONNX engine...")
            self._engine = RapidOCR()

    def detect_and_group(
        self,
        image_rgb: np.ndarray,
        min_confidence: Optional[float] = None,
    ) -> List[OCRBlock]:
        self._ensure_loaded()
        conf_threshold = min_confidence if min_confidence is not None else settings.ocr_min_confidence

        # RapidOCR expects BGR or RGB numpy array (it handles RGB/BGR smoothly, BGR is native OpenCV)
        image_bgr = image_rgb[:, :, ::-1].copy()
        result, _ = self._engine(image_bgr)
        if not result:
            return []

        h_img, w_img = image_rgb.shape[:2]
        raw_lines: List[OCRLine] = []

        for item in result:
            box, text, score = item[0], str(item[1]).strip(), float(item[2])
            if not text or score < conf_threshold:
                continue

            pts = [[float(p[0]), float(p[1])] for p in box]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1 = max(0, int(math.floor(min(xs))))
            y1 = max(0, int(math.floor(min(ys))))
            x2 = min(w_img, int(math.ceil(max(xs))))
            y2 = min(h_img, int(math.ceil(max(ys))))

            # Compute line height and angle from top edge (p0 -> p1)
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) < 2.5:
                angle = 0.0

            left_h = math.hypot(pts[3][0] - pts[0][0], pts[3][1] - pts[0][1])
            right_h = math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
            line_h = (left_h + right_h) / 2.0

            if line_h < settings.ocr_min_box_height or (x2 - x1) < 4:
                continue

            raw_lines.append(
                OCRLine(
                    polygon=pts,
                    text=text,
                    confidence=score,
                    bbox=(x1, y1, x2, y2),
                    height=line_h,
                    angle=angle,
                )
            )

        return group_ocr_lines(raw_lines)


def _horizontal_overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    min_w = max(1, min(a[2] - a[0], b[2] - b[0]))
    return inter / min_w


def _should_merge(line_a: OCRLine, line_b: OCRLine) -> bool:
    """Decide whether two adjacent OCR lines belong to the same paragraph/caption/bubble."""
    # Must have similar orientation
    if abs(line_a.angle - line_b.angle) > 12.0:
        return False

    # Similar font size
    h_ratio = max(line_a.height, line_b.height) / max(1.0, min(line_a.height, line_b.height))
    if h_ratio > 1.65:
        return False

    avg_h = (line_a.height + line_b.height) / 2.0

    # Case 1: Vertically stacked lines (multi-line meme caption or speech bubble)
    top, bot = (line_a, line_b) if line_a.bbox[1] <= line_b.bbox[1] else (line_b, line_a)
    v_gap = bot.bbox[1] - top.bbox[3]
    if -avg_h * 0.35 <= v_gap <= avg_h * 0.95:
        overlap = _horizontal_overlap_ratio(top.bbox, bot.bbox)
        cx_diff = abs((top.bbox[0] + top.bbox[2]) / 2.0 - (bot.bbox[0] + bot.bbox[2]) / 2.0)
        left_diff = abs(top.bbox[0] - bot.bbox[0])
        right_diff = abs(top.bbox[2] - bot.bbox[2])
        if overlap >= 0.25 or cx_diff <= avg_h * 1.2 or left_diff <= avg_h * 0.8 or right_diff <= avg_h * 0.8:
            return True

    # Case 2: Horizontally adjacent fragments on the exact same line
    left, right = (line_a, line_b) if line_a.bbox[0] <= line_b.bbox[0] else (line_b, line_a)
    v_inter = max(0, min(left.bbox[3], right.bbox[3]) - max(left.bbox[1], right.bbox[1]))
    h_gap = right.bbox[0] - left.bbox[2]
    if v_inter >= avg_h * 0.55 and -avg_h * 0.2 <= h_gap <= avg_h * 1.1:
        return True

    return False


def _infer_alignment(lines: List[OCRLine], block_bbox: Tuple[int, int, int, int]) -> str:
    if len(lines) <= 1:
        return "center"
    bx1, _, bx2, _ = block_bbox
    bcx = (bx1 + bx2) / 2.0

    left_err = sum(abs(l.bbox[0] - bx1) for l in lines) / len(lines)
    right_err = sum(abs(l.bbox[2] - bx2) for l in lines) / len(lines)
    center_err = sum(abs(((l.bbox[0] + l.bbox[2]) / 2.0) - bcx) for l in lines) / len(lines)

    best = min((center_err, "center"), (left_err, "left"), (right_err, "right"), key=lambda x: x[0])
    return best[1]


def group_ocr_lines(lines: List[OCRLine]) -> List[OCRBlock]:
    if not lines:
        return []

    n = len(lines)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _should_merge(lines[i], lines[j]):
                union(i, j)

    groups: dict[int, List[OCRLine]] = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(lines[idx])

    blocks: List[OCRBlock] = []
    # Sort blocks top-to-bottom, left-to-right
    sorted_groups = sorted(
        groups.values(),
        key=lambda grp: (min(l.bbox[1] for l in grp), min(l.bbox[0] for l in grp)),
    )

    for b_idx, grp in enumerate(sorted_groups):
        # Order lines within block by row (bucketed by half-line-height) then x1
        avg_h = max(1.0, sum(l.height for l in grp) / len(grp))
        grp.sort(key=lambda l: (round(l.bbox[1] / (avg_h * 0.6)), l.bbox[0]))

        x1 = min(l.bbox[0] for l in grp)
        y1 = min(l.bbox[1] for l in grp)
        x2 = max(l.bbox[2] for l in grp)
        y2 = max(l.bbox[3] for l in grp)

        # Join text naturally: space between Latin/words, no extra space for CJK
        joined_parts: List[str] = []
        for l in grp:
            t = l.text.strip()
            if t:
                joined_parts.append(t)
        combined_text = " ".join(joined_parts)

        avg_conf = sum(l.confidence for l in grp) / len(grp)
        # Weighted average angle by line width
        total_w = sum(max(1, l.bbox[2] - l.bbox[0]) for l in grp)
        avg_angle = sum(l.angle * max(1, l.bbox[2] - l.bbox[0]) for l in grp) / total_w
        if abs(avg_angle) < 2.5:
            avg_angle = 0.0

        alignment = _infer_alignment(grp, (x1, y1, x2, y2))
        polygon = [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]
        if len(grp) == 1:
            polygon = grp[0].polygon

        blocks.append(
            OCRBlock(
                id=f"blk_{b_idx + 1}",
                lines=grp,
                polygon=polygon,
                bbox=(x1, y1, x2, y2),
                text=combined_text,
                confidence=round(avg_conf, 3),
                angle=round(avg_angle, 2),
                alignment=alignment,
            )
        )

    return blocks


ocr_engine = OCREngine()
