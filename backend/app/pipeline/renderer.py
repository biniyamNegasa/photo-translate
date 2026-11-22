import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.languages import detect_script, get_language
from app.pipeline.fonts import FontEntry, is_raqm_available, load_font_chain, split_font_runs
from app.schemas import TextBlock


def _luminance(rgb: Tuple[int, int, int]) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _contrast_ratio(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    l1 = _luminance(c1) / 255.0
    l2 = _luminance(c2) / 255.0
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _normalize_script_punctuation(text: str, target_lang: str) -> str:
    """Use native Arabic question mark `؟` when rendering Arabic/Persian RTL text."""
    if target_lang in ("ar", "fa"):
        return text.replace("?", "؟")
    return text


def _font_ascent(font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
    if hasattr(font, "getmetrics"):
        try:
            return int(font.getmetrics()[0])
        except Exception:
            pass
    return 0


def _measure_runs(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_chain: List[FontEntry],
    stroke_width: int = 0,
    direction: str | None = None,
) -> Tuple[int, int]:
    """Measure width & height of `text` using multi-font fallback runs."""
    runs = split_font_runs(text, font_chain)
    if not runs:
        return 1, 1

    if len(runs) == 1:
        seg, font = runs[0]
        kwargs = {"font": font, "stroke_width": stroke_width}
        if direction and is_raqm_available():
            kwargs["direction"] = direction
        bbox = draw.textbbox((0, 0), seg, **kwargs)
        return max(1, int(math.ceil(bbox[2] - bbox[0]))), max(1, int(math.ceil(bbox[3] - bbox[1])))

    total_w = 0.0
    max_ascent = max(_font_ascent(f) for _, f in runs)
    min_top = 10_000.0
    max_bot = -10_000.0

    for seg, font in runs:
        y_off = max_ascent - _font_ascent(font)
        kwargs = {"font": font, "stroke_width": stroke_width}
        if direction and is_raqm_available():
            kwargs["direction"] = direction
        bbox = draw.textbbox((0, y_off), seg, **kwargs)
        adv = draw.textlength(seg, font=font, **({"direction": direction} if direction and is_raqm_available() else {}))
        total_w += max(float(adv), float(bbox[2] - bbox[0]) - stroke_width)
        min_top = min(min_top, float(bbox[1]))
        max_bot = max(max_bot, float(bbox[3]))

    total_w += stroke_width * 2
    total_h = max(1.0, max_bot - min_top)
    return max(1, int(math.ceil(total_w))), max(1, int(math.ceil(total_h)))


def _draw_runs(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font_chain: List[FontEntry],
    fill: Tuple[int, int, int, int],
    stroke_width: int = 0,
    stroke_fill: Tuple[int, int, int, int] = (0, 0, 0, 255),
    direction: str | None = None,
) -> None:
    """Draw `text` using multi-font fallback runs aligned on a shared baseline."""
    runs = split_font_runs(text, font_chain)
    if not runs:
        return

    x, y = xy
    if len(runs) == 1:
        seg, font = runs[0]
        kwargs: dict = {"font": font, "fill": fill}
        if stroke_width > 0:
            kwargs["stroke_width"] = stroke_width
            kwargs["stroke_fill"] = stroke_fill
        if direction and is_raqm_available():
            kwargs["direction"] = direction
        draw.text((x, y), seg, **kwargs)
        return

    # For RTL lines with mixed scripts, render right-to-left run order
    ordered_runs = list(reversed(runs)) if direction == "rtl" else runs
    max_ascent = max(_font_ascent(f) for _, f in ordered_runs)

    cur_x = x
    for seg, font in ordered_runs:
        y_off = max_ascent - _font_ascent(font)
        kwargs = {"font": font, "fill": fill}
        if stroke_width > 0:
            kwargs["stroke_width"] = stroke_width
            kwargs["stroke_fill"] = stroke_fill
        if direction and is_raqm_available():
            kwargs["direction"] = direction
        draw.text((cur_x, y + y_off), seg, **kwargs)
        adv = draw.textlength(seg, font=font, **({"direction": direction} if direction and is_raqm_available() else {}))
        cur_x += float(adv)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_chain: List[FontEntry],
    max_width: int,
    stroke_width: int,
    direction: str | None,
    char_wrap: bool,
) -> List[str]:
    """Wrap text into lines that fit within `max_width`."""
    paragraphs = text.splitlines() or [text]
    lines: List[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if char_wrap:
            current = ""
            for ch in para:
                trial = current + ch
                w, _ = _measure_runs(draw, trial, font_chain, stroke_width, direction)
                if w <= max_width or not current:
                    current = trial
                else:
                    lines.append(current)
                    current = ch
            if current:
                lines.append(current)
        else:
            words = para.split(" ")
            current = ""
            for word in words:
                trial = f"{current} {word}".strip() if current else word
                w, _ = _measure_runs(draw, trial, font_chain, stroke_width, direction)
                if w <= max_width or not current:
                    current = trial
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)

    return lines or [text]


def _layout_fit(
    draw: ImageDraw.ImageDraw,
    text: str,
    target_lang: str,
    box_w: int,
    box_h: int,
    bold: bool,
    meme_style: bool,
    stroke_ratio: float,
    direction: str | None,
    char_wrap: bool,
    preferred_lines: int,
) -> Tuple[List[FontEntry], List[str], int, int, int]:
    """
    Binary-search the largest font size that fits cleanly inside (box_w, box_h).
    Returns: (font_chain, lines, line_step, total_height, stroke_width)
    """
    min_size = 9
    max_size = max(min_size + 2, min(int(box_h * 0.92), int((box_h / max(1, preferred_lines)) * 1.55)))
    max_size = min(max_size, 180)

    best_result = None
    lo, hi = min_size, max_size

    while lo <= hi:
        mid = (lo + hi) // 2
        font_chain = load_font_chain(text, target_lang, size=mid, bold=bold, meme_style=meme_style)
        sw = max(1, int(round(mid * stroke_ratio))) if stroke_ratio > 0 else 0
        lines = _wrap_text(draw, text, font_chain, max(12, box_w), sw, direction, char_wrap)

        line_widths = []
        line_heights = []
        for ln in lines:
            w, h = _measure_runs(draw, ln, font_chain, sw, direction)
            line_widths.append(w)
            line_heights.append(h)

        max_lw = max(line_widths) if line_widths else 0
        base_lh = max(line_heights) if line_heights else mid
        line_spacing = max(2, int(round(mid * 0.18)))
        total_h = sum(line_heights) + line_spacing * max(0, len(lines) - 1)

        if max_lw <= box_w * 1.02 and total_h <= box_h * 1.04:
            best_result = (font_chain, lines, base_lh + line_spacing, total_h, sw)
            lo = mid + 1
        else:
            hi = mid - 1

    if best_result is not None:
        return best_result

    font_chain = load_font_chain(text, target_lang, size=min_size, bold=bold, meme_style=meme_style)
    sw = 1 if stroke_ratio > 0 else 0
    lines = _wrap_text(draw, text, font_chain, max(12, box_w), sw, direction, char_wrap)
    lh = min_size + 3
    return font_chain, lines, lh, lh * len(lines), sw


def composite_and_render(
    original_rgb: np.ndarray,
    inpainted_rgb: np.ndarray,
    block_masks: Dict[str, np.ndarray],
    blocks: List[TextBlock],
    target_lang: str,
) -> np.ndarray:
    """
    1. Pixel-exact compositing: start from `original_rgb` and blend `inpainted_rgb`
       ONLY inside masks belonging to `enabled == True` blocks.
       Disabled blocks (e.g. watermarks/logos) remain 100% original pixels!
    2. Render each enabled block's translated text with matched color, stroke/outline,
       alignment, HarfBuzz/raqm shaping, multi-font punctuation fallback, and rotation angle.
    """
    h_img, w_img = original_rgb.shape[:2]

    # Step 1: Pixel-exact mask compositing
    combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for blk in blocks:
        if not blk.enabled:
            continue
        m = block_masks.get(blk.id)
        if m is not None and m.shape[:2] == (h_img, w_img):
            combined_mask = cv2.bitwise_or(combined_mask, m)

    if np.count_nonzero(combined_mask) > 0:
        alpha = cv2.GaussianBlur(combined_mask.astype(np.float32) / 255.0, (3, 3), 0)
        alpha = np.clip(alpha * 1.1, 0.0, 1.0)[..., None]
        canvas_np = np.clip(
            inpainted_rgb.astype(np.float32) * alpha + original_rgb.astype(np.float32) * (1.0 - alpha),
            0,
            255,
        ).astype(np.uint8)
    else:
        canvas_np = original_rgb.copy()

    # Step 2: Style-matched text rendering onto RGBA overlay
    canvas_rgba = Image.fromarray(canvas_np, mode="RGB").convert("RGBA")
    lang_info = get_language(target_lang)
    direction = "rtl" if lang_info.rtl else None

    for blk in blocks:
        if not blk.enabled:
            continue
        raw_text = _normalize_script_punctuation(blk.translated_text.strip(), target_lang)
        if not raw_text:
            continue

        style = blk.style
        script = detect_script(raw_text)
        char_wrap = script.startswith("cjk") or script == "thai"

        # Preserve all-caps meme/headline convention for cased scripts
        render_text = raw_text.upper() if (style.is_uppercase and script in ("latin", "cyrillic", "greek")) else raw_text

        # Choose target box: prefer original bbox; use expanded_bbox when uniform_bg and text grew longer
        x1, y1, x2, y2 = blk.bbox
        ex1, ey1, ex2, ey2 = blk.expanded_bbox
        len_ratio = len(render_text) / max(1, len(blk.source_text.strip()))
        if style.uniform_bg and len_ratio > 1.15:
            tx1, ty1, tx2, ty2 = ex1, ey1, ex2, ey2
        else:
            tx1, ty1, tx2, ty2 = x1, y1, x2, y2

        # If rotated, compute oriented box dimensions from polygon
        angle = style.angle
        if abs(angle) >= 2.5 and len(blk.polygon) >= 4:
            pts = blk.polygon
            box_w = int(round(math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])))
            box_h = int(round(math.hypot(pts[3][0] - pts[0][0], pts[3][1] - pts[0][1])))
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
        else:
            angle = 0.0
            box_w = max(16, tx2 - tx1)
            box_h = max(14, ty2 - ty1)
            cx = (tx1 + tx2) / 2.0
            cy = (ty1 + ty2) / 2.0

        fg_color = tuple(style.fg_color)
        stroke_color = tuple(style.stroke_color)
        has_stroke = style.has_stroke
        meme_style = has_stroke and style.is_bold

        sample_patch = canvas_np[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
        local_bg = (
            tuple(int(c) for c in np.median(sample_patch.reshape(-1, 3), axis=0))
            if sample_patch.size > 0
            else style.bg_color
        )

        stroke_ratio = 0.075 if has_stroke else 0.0
        if not has_stroke and _contrast_ratio(fg_color, local_bg) < 2.2:
            stroke_color = (0, 0, 0) if _luminance(fg_color) > 128 else (255, 255, 255)
            stroke_ratio = 0.055

        pad = max(24, int(max(box_w, box_h) * 0.25))
        patch_w = box_w + pad * 2
        patch_h = box_h + pad * 2
        text_layer = Image.new("RGBA", (patch_w, patch_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        font_chain, lines, line_step, total_h, sw = _layout_fit(
            draw=draw,
            text=render_text,
            target_lang=target_lang,
            box_w=box_w,
            box_h=box_h,
            bold=style.is_bold,
            meme_style=meme_style,
            stroke_ratio=stroke_ratio,
            direction=direction,
            char_wrap=char_wrap,
            preferred_lines=blk.line_count,
        )

        cur_y = pad + max(0, (box_h - total_h) / 2.0)

        for ln in lines:
            lw, _ = _measure_runs(draw, ln, font_chain, sw, direction)
            if style.alignment == "left" and not lang_info.rtl:
                cur_x = pad
            elif style.alignment == "right" or (style.alignment == "left" and lang_info.rtl):
                cur_x = pad + max(0, box_w - lw)
            else:
                cur_x = pad + (box_w - lw) / 2.0

            _draw_runs(
                draw=draw,
                xy=(cur_x, cur_y),
                text=ln,
                font_chain=font_chain,
                fill=(*fg_color, 255),
                stroke_width=sw,
                stroke_fill=(*stroke_color, 255),
                direction=direction,
            )
            cur_y += line_step

        if abs(angle) >= 2.5:
            text_layer = text_layer.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)

        paste_x = int(round(cx - text_layer.width / 2.0))
        paste_y = int(round(cy - text_layer.height / 2.0))
        canvas_rgba.alpha_composite(text_layer, dest=(paste_x, paste_y))

    return np.array(canvas_rgba.convert("RGB"), dtype=np.uint8)
