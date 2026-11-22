import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from app.config import settings
from app.languages import detect_script

logger = logging.getLogger(__name__)


@dataclass
class OCRLine:
    polygon: List[List[float]]       # 4 points [[x,y], ...]
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    height: float
    width: float
    angle: float                     # degrees
    is_vertical: bool = False


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
    is_vertical: bool = False
    is_comic_block: bool = False
    comic_seg_mask: Optional[np.ndarray] = None  # Learned pixel ink mask from comic-text-detector


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

            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # Detect vertical East-Asian (Japanese/Chinese) manga columns
            script = detect_script(text)
            is_vertical = (
                script.startswith("cjk")
                and bh > bw * 1.45
                and len(text) >= 2
            )

            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            angle = 0.0 if is_vertical else math.degrees(math.atan2(dy, dx))
            if abs(angle) < 2.5 or abs(angle) > 75.0:
                angle = 0.0

            left_h = math.hypot(pts[3][0] - pts[0][0], pts[3][1] - pts[0][1])
            right_h = math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
            line_h = float(bw) if is_vertical else (left_h + right_h) / 2.0

            if min(bw, bh) < settings.ocr_min_box_height:
                continue

            raw_lines.append(
                OCRLine(
                    polygon=pts,
                    text=text,
                    confidence=score,
                    bbox=(x1, y1, x2, y2),
                    height=line_h,
                    width=float(bw),
                    angle=angle,
                    is_vertical=is_vertical,
                )
            )

        return group_ocr_lines(raw_lines)

    def recognize_crop(
        self,
        crop_rgb: np.ndarray,
        offset_xy: Tuple[int, int] = (0, 0),
        min_confidence: float = 0.35,
    ) -> Tuple[str, float, List[OCRLine]]:
        """
        Recognize text inside a single speech-bubble crop using RapidOCR.
        Falls back to recognition-only mode for tight single-word stylized bubbles
        when RapidOCR's detector misses them.
        """
        self._ensure_loaded()
        if crop_rgb is None or crop_rgb.size == 0:
            return "", 0.0, []

        ox, oy = offset_xy
        ch, cw = crop_rgb.shape[:2]
        crop_bgr = crop_rgb[:, :, ::-1].copy()
        result, _ = self._engine(crop_bgr)

        raw_lines: List[OCRLine] = []
        if result:
            for item in result:
                box, text, score = item[0], str(item[1]).strip(), float(item[2])
                if not text or score < min_confidence:
                    continue
                pts = [[float(p[0]) + ox, float(p[1]) + oy] for p in box]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1 = int(math.floor(min(xs)))
                y1 = int(math.floor(min(ys)))
                x2 = int(math.ceil(max(xs)))
                y2 = int(math.ceil(max(ys)))
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                script = detect_script(text)
                is_vert = script.startswith("cjk") and bh > bw * 1.45 and len(text) >= 2
                left_h = math.hypot(pts[3][0] - pts[0][0], pts[3][1] - pts[0][1])
                right_h = math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
                line_h = float(bw) if is_vert else max(8.0, (left_h + right_h) / 2.0)
                raw_lines.append(
                    OCRLine(
                        polygon=pts,
                        text=text,
                        confidence=score,
                        bbox=(x1, y1, x2, y2),
                        height=line_h,
                        width=float(bw),
                        angle=0.0,
                        is_vertical=is_vert,
                    )
                )

        if raw_lines:
            if all(l.is_vertical for l in raw_lines):
                raw_lines.sort(key=lambda l: (-l.bbox[0], l.bbox[1]))
                combined = "".join(l.text for l in raw_lines)
            else:
                avg_h = max(1.0, sum(l.height for l in raw_lines) / len(raw_lines))
                raw_lines.sort(key=lambda l: (round(l.bbox[1] / (avg_h * 0.6)), l.bbox[0]))
                parts = [l.text for l in raw_lines if l.text]
                sample = "".join(parts)
                joiner = "" if detect_script(sample).startswith("cjk") else " "
                combined = joiner.join(parts)
            avg_conf = sum(l.confidence for l in raw_lines) / len(raw_lines)
            return combined.strip(), float(avg_conf), raw_lines

        # Safety net: recognition-only pass for tight single-line bubbles where det missed
        if cw >= bh_min(ch) and ch <= cw * 1.2:
            try:
                rec_res, _ = self._engine(crop_bgr, use_det=False, use_cls=False, use_rec=True)
                if rec_res:
                    txt, sc = str(rec_res[0][0]).strip(), float(rec_res[0][1])
                    if txt and sc >= min_confidence:
                        return txt, sc, []
            except Exception:
                pass

        return "", 0.0, []


def bh_min(h: int) -> int:
    return max(8, int(h * 0.4))


def _horizontal_overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    min_w = max(1, min(a[2] - a[0], b[2] - b[0]))
    return inter / min_w


def _vertical_overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    min_h = max(1, min(a[3] - a[1], b[3] - b[1]))
    return inter / min_h


def _should_merge(line_a: OCRLine, line_b: OCRLine) -> bool:
    """Decide whether two adjacent OCR lines/columns belong to the same caption or speech bubble."""
    if line_a.is_vertical != line_b.is_vertical:
        return False

    # Vertical East-Asian manga columns: group horizontally adjacent columns into one speech bubble
    if line_a.is_vertical:
        left, right = (line_a, line_b) if line_a.bbox[0] <= line_b.bbox[0] else (line_b, line_a)
        avg_col_w = (line_a.width + line_b.width) / 2.0
        w_ratio = max(line_a.width, line_b.width) / max(1.0, min(line_a.width, line_b.width))
        if w_ratio > 1.75:
            return False
        h_gap = right.bbox[0] - left.bbox[2]
        v_ov = _vertical_overlap_ratio(left.bbox, right.bbox)
        if -avg_col_w * 0.3 <= h_gap <= avg_col_w * 1.1 and v_ov >= 0.35:
            return True
        return False

    # Horizontal text lines
    if abs(line_a.angle - line_b.angle) > 12.0:
        return False

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
    if len(lines) <= 1 or lines[0].is_vertical:
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
    # Sort blocks top-to-bottom; for Japanese vertical manga blocks prefer right-to-left within the same row band
    sorted_groups = sorted(
        groups.values(),
        key=lambda grp: (min(l.bbox[1] for l in grp), min(l.bbox[0] for l in grp)),
    )

    for b_idx, grp in enumerate(sorted_groups):
        is_vert_block = grp[0].is_vertical
        if is_vert_block:
            # Right-to-left column reading order for Japanese/Chinese vertical manga
            grp.sort(key=lambda l: (-l.bbox[0], l.bbox[1]))
            combined_text = "".join(l.text.strip() for l in grp if l.text.strip())
        else:
            avg_h = max(1.0, sum(l.height for l in grp) / len(grp))
            grp.sort(key=lambda l: (round(l.bbox[1] / (avg_h * 0.6)), l.bbox[0]))
            parts = [l.text.strip() for l in grp if l.text.strip()]
            # Join without extra space if CJK, else space
            sample = "".join(parts)
            joiner = "" if detect_script(sample).startswith("cjk") else " "
            combined_text = joiner.join(parts)

        x1 = min(l.bbox[0] for l in grp)
        y1 = min(l.bbox[1] for l in grp)
        x2 = max(l.bbox[2] for l in grp)
        y2 = max(l.bbox[3] for l in grp)

        avg_conf = sum(l.confidence for l in grp) / len(grp)
        total_w = sum(max(1, l.bbox[2] - l.bbox[0]) for l in grp)
        avg_angle = 0.0 if is_vert_block else (
            sum(l.angle * max(1, l.bbox[2] - l.bbox[0]) for l in grp) / total_w
        )
        if abs(avg_angle) < 2.5:
            avg_angle = 0.0

        alignment = _infer_alignment(grp, (x1, y1, x2, y2))
        polygon = [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]
        if len(grp) == 1 and not is_vert_block:
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
                is_vertical=is_vert_block,
            )
        )

    return blocks


ocr_engine = OCREngine()
