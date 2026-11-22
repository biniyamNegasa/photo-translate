import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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


def _measure_runs_ink(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_chain: List[FontEntry],
    stroke_width: int = 0,
    direction: str | None = None,
) -> Tuple[float, float, float, float]:
    """
    Measure exact ink metrics for `text` drawn at origin `(0, 0)` across fallback font runs.
    Returns `(ink_left, ink_top, ink_width, ink_height)` so lines can be positioned
    with sub-pixel vertical centering (eliminating top-leading sag).
    """
    runs = split_font_runs(text, font_chain)
    if not runs:
        return 0.0, 0.0, 1.0, 1.0

    if len(runs) == 1:
        seg, font = runs[0]
        kwargs = {"font": font, "stroke_width": stroke_width}
        if direction and is_raqm_available():
            kwargs["direction"] = direction
        b = draw.textbbox((0, 0), seg, **kwargs)
        return float(b[0]), float(b[1]), max(1.0, float(b[2] - b[0])), max(1.0, float(b[3] - b[1]))

    ordered = list(reversed(runs)) if direction == "rtl" else runs
    max_ascent = max(_font_ascent(f) for _, f in ordered)
    cur_x = 0.0
    min_l = 1e9
    min_t = 1e9
    max_r = -1e9
    max_b = -1e9

    for seg, font in ordered:
        y_off = max_ascent - _font_ascent(font)
        kwargs = {"font": font, "stroke_width": stroke_width}
        if direction and is_raqm_available():
            kwargs["direction"] = direction
        b = draw.textbbox((cur_x, y_off), seg, **kwargs)
        min_l = min(min_l, float(b[0]))
        min_t = min(min_t, float(b[1]))
        max_r = max(max_r, float(b[2]))
        max_b = max(max_b, float(b[3]))
        adv = draw.textlength(
            seg,
            font=font,
            **({"direction": direction} if direction and is_raqm_available() else {}),
        )
        cur_x += float(adv)

    return min_l, min_t, max(1.0, max_r - min_l), max(1.0, max_b - min_t)


def _measure_runs(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_chain: List[FontEntry],
    stroke_width: int = 0,
    direction: str | None = None,
) -> Tuple[int, int]:
    _, _, w, h = _measure_runs_ink(draw, text, font_chain, stroke_width, direction)
    return max(1, int(math.ceil(w))), max(1, int(math.ceil(h)))


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
        adv = draw.textlength(
            seg,
            font=font,
            **({"direction": direction} if direction and is_raqm_available() else {}),
        )
        cur_x += float(adv)


def _wrap_balanced_words(
    draw: ImageDraw.ImageDraw,
    words: List[str],
    font_chain: List[FontEntry],
    max_width: int,
    stroke_width: int,
    direction: str | None,
) -> List[str]:
    """
    Knuth-Plass-style balanced line wrapping: minimizes raggedness across all lines
    and prevents awkward single-word orphans in speech bubbles and meme captions.
    """
    n = len(words)
    if n <= 1:
        return words

    # Pre-measure single-line check for fast exit
    full_line = " ".join(words)
    full_w, _ = _measure_runs(draw, full_line, font_chain, stroke_width, direction)
    if full_w <= max_width:
        return [full_line]

    # Limit DP complexity for very long paragraphs (>36 words) with greedy fallback
    if n > 36:
        lines: List[str] = []
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
        return lines

    # DP over word indices: dp[i] = (min_cost, next_break_idx)
    inf = 1e18
    dp: List[Tuple[float, int]] = [(inf, n)] * (n + 1)
    dp[n] = (0.0, n)

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n + 1):
            candidate = " ".join(words[i:j])
            w, _ = _measure_runs(draw, candidate, font_chain, stroke_width, direction)
            if w > max_width and j > i + 1:
                break
            slack = max(0.0, float(max_width - w)) / max(1.0, float(max_width))
            # Mildly penalize orphan single-word last lines, strongly balance non-final lines
            if j == n:
                line_cost = (slack * slack) * (0.45 if (j - i) > 1 else 0.85)
            else:
                line_cost = slack * slack
            # Slight extra penalty if single word itself overflows
            if w > max_width:
                line_cost += 4.0 + ((w - max_width) / max(1.0, float(max_width))) ** 2
            total_cost = line_cost + dp[j][0]
            if total_cost < dp[i][0]:
                dp[i] = (total_cost, j)

    result: List[str] = []
    idx = 0
    while idx < n:
        nxt = dp[idx][1]
        if nxt <= idx:
            nxt = idx + 1
        result.append(" ".join(words[idx:nxt]))
        idx = nxt
    return result


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_chain: List[FontEntry],
    max_width: int,
    stroke_width: int,
    direction: str | None,
    char_wrap: bool,
) -> List[str]:
    """Wrap text into balanced lines that fit within `max_width`."""
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
            words = [w for w in para.split(" ") if w]
            lines.extend(_wrap_balanced_words(draw, words, font_chain, max_width, stroke_width, direction))

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
    comic_style: bool = False,
    max_font_cap: int = 180,
) -> Tuple[List[FontEntry], List[str], List[Tuple[float, float, float, float]], int, int, int]:
    """
    Binary-search the largest font size that fits cleanly inside (box_w, box_h)
    using exact ink metrics and balanced line wrapping.
    Returns: (font_chain, lines, line_inks, line_gap, total_ink_h, stroke_width)
    """
    min_size = 9
    max_size = max(min_size + 2, min(int(box_h * 0.98), int((box_h / max(1, preferred_lines)) * 1.65)))
    max_size = min(max_size, max(min_size + 2, max_font_cap))

    best_result = None
    lo, hi = min_size, max_size

    while lo <= hi:
        mid = (lo + hi) // 2
        font_chain = load_font_chain(
            text, target_lang, size=mid, bold=bold, meme_style=meme_style, comic_style=comic_style
        )
        sw = max(1, int(round(mid * stroke_ratio))) if stroke_ratio > 0 else 0
        lines = _wrap_text(draw, text, font_chain, max(12, box_w), sw, direction, char_wrap)

        line_inks = [_measure_runs_ink(draw, ln, font_chain, sw, direction) for ln in lines]
        max_lw = max((ink[2] for ink in line_inks), default=0.0)
        line_gap = max(3, int(round(mid * 0.24)))
        total_ink_h = sum(ink[3] for ink in line_inks) + line_gap * max(0, len(lines) - 1)

        if max_lw <= box_w * 1.02 and total_ink_h <= box_h * 1.04:
            best_result = (font_chain, lines, line_inks, line_gap, int(math.ceil(total_ink_h)), sw)
            lo = mid + 1
        else:
            hi = mid - 1

    if best_result is not None:
        return best_result

    font_chain = load_font_chain(
        text, target_lang, size=min_size, bold=bold, meme_style=meme_style, comic_style=comic_style
    )
    sw = 1 if stroke_ratio > 0 else 0
    lines = _wrap_text(draw, text, font_chain, max(12, box_w), sw, direction, char_wrap)
    line_inks = [_measure_runs_ink(draw, ln, font_chain, sw, direction) for ln in lines]
    line_gap = 3
    total_ink_h = int(math.ceil(sum(ink[3] for ink in line_inks) + line_gap * max(0, len(lines) - 1)))
    return font_chain, lines, line_inks, line_gap, max(1, total_ink_h), sw


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
    2. Render each enabled block's translated text with exact ink-centered layout,
       balanced line wrapping, speech-bubble interior fitting, multi-font HarfBuzz shaping,
       and subtle photographic edge blending.
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
        alpha = np.clip(alpha * 1.15, 0.0, 1.0)[..., None]
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

        # Choose target box:
        # For uniform speech bubbles / solid bars, use expanded bubble interior when helpful
        # (always for vertical-to-horizontal manga bubbles so translated lines fill the oval!).
        x1, y1, x2, y2 = blk.bbox
        ex1, ey1, ex2, ey2 = blk.expanded_bbox
        len_ratio = len(render_text) / max(1, len(blk.source_text.strip()))
        has_expanded = (ex2 - ex1) > (x2 - x1) + 8 or (ey2 - ey1) > (y2 - y1) + 8
        if style.uniform_bg and has_expanded and (blk.is_vertical or len_ratio > 1.05 or blk.line_count >= 2):
            tx1, ty1, tx2, ty2 = ex1, ey1, ex2, ey2
        else:
            tx1, ty1, tx2, ty2 = x1, y1, x2, y2

        # Oriented dimensions if rotated
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

        # Respect measured stroke thickness ratio when available
        est_line_h = max(12.0, box_h / max(1, blk.line_count))
        if has_stroke and style.stroke_width > 0:
            stroke_ratio = max(0.055, min(0.11, style.stroke_width / est_line_h))
        elif has_stroke:
            stroke_ratio = 0.08
        else:
            stroke_ratio = 0.0

        if not has_stroke and _contrast_ratio(fg_color, local_bg) < 2.3:
            stroke_color = (0, 0, 0) if _luminance(fg_color) > 128 else (255, 255, 255)
            stroke_ratio = 0.055

        pad = max(24, int(max(box_w, box_h) * 0.25))
        patch_w = box_w + pad * 2
        patch_h = box_h + pad * 2
        text_layer = Image.new("RGBA", (patch_w, patch_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        # Keep font size natural on vertical-to-horizontal manga speech bubbles
        # (prevents short words like "Huh?" from blowing up inside tall ovals without cramping full sentences)
        max_font_cap = 180
        if blk.is_vertical:
            max_font_cap = max(36, min(64, int(min(box_w, box_h) * 0.26)))

        font_chain, lines, line_inks, line_gap, total_ink_h, sw = _layout_fit(
            draw=draw,
            text=render_text,
            target_lang=target_lang,
            box_w=box_w,
            box_h=box_h,
            bold=style.is_bold or (style.comic_style and script == "latin"),
            meme_style=meme_style,
            stroke_ratio=stroke_ratio,
            direction=direction,
            char_wrap=char_wrap,
            preferred_lines=1 if blk.is_vertical else blk.line_count,
            comic_style=style.comic_style,
            max_font_cap=max_font_cap,
        )

        # Sub-pixel vertical ink centering (subtract each line's actual ink_top!)
        cur_ink_y = pad + (box_h - total_ink_h) / 2.0

        for ln, (ink_l, ink_t, ink_w, ink_h) in zip(lines, line_inks):
            if style.alignment == "left" and not lang_info.rtl:
                target_ink_x = float(pad)
            elif style.alignment == "right" or (style.alignment == "left" and lang_info.rtl):
                target_ink_x = pad + max(0.0, box_w - ink_w)
            else:
                target_ink_x = pad + (box_w - ink_w) / 2.0

            draw_x = target_ink_x - ink_l
            draw_y = cur_ink_y - ink_t

            _draw_runs(
                draw=draw,
                xy=(draw_x, draw_y),
                text=ln,
                font_chain=font_chain,
                fill=(*fg_color, 255),
                stroke_width=sw,
                stroke_fill=(*stroke_color, 255),
                direction=direction,
            )
            cur_ink_y += ink_h + line_gap

        # Subtle micro-softening on textured/photographic backgrounds so vector glyphs blend naturally
        if not style.uniform_bg and box_h >= 22:
            r, g, b, a = text_layer.split()
            a = a.filter(ImageFilter.GaussianBlur(radius=0.35))
            text_layer = Image.merge("RGBA", (r, g, b, a))

        if abs(angle) >= 2.5:
            text_layer = text_layer.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)

        paste_x = int(round(cx - text_layer.width / 2.0))
        paste_y = int(round(cy - text_layer.height / 2.0))
        canvas_rgba.alpha_composite(text_layer, dest=(paste_x, paste_y))

    return np.array(canvas_rgba.convert("RGB"), dtype=np.uint8)
