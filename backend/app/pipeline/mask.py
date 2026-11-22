from typing import Tuple

import cv2
import numpy as np

from app.pipeline.ocr import OCRBlock
from app.schemas import TextStyle


def _color_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean RGB distance."""
    diff = a.astype(np.float32) - b.astype(np.float32)
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _luminance(rgb: Tuple[int, int, int]) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _find_speech_bubble_box(
    image_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    bg_color: Tuple[int, int, int],
    max_expand_ratio: float = 0.45,
) -> Tuple[int, int, int, int]:
    """
    Find the largest clean inscribed rectangle inside a uniform-color speech bubble or
    solid caption bar around `bbox`, leaving safe padding from ink borders.
    """
    h_img, w_img = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # Search window generously around bbox (clamped to image bounds)
    margin_x = min(max(40, int(bw * max_expand_ratio)), 220 if max_expand_ratio > 0.6 else 140)
    margin_y = min(max(30, int(bh * max_expand_ratio)), 140 if max_expand_ratio > 0.6 else 110)
    sx1 = max(0, x1 - margin_x)
    sy1 = max(0, y1 - margin_y)
    sx2 = min(w_img, x2 + margin_x)
    sy2 = min(h_img, y2 + margin_y)

    patch = image_rgb[sy1:sy2, sx1:sx2]
    ph, pw = patch.shape[:2]
    bg_arr = np.array(bg_color, dtype=np.float32)
    tol = 26.0

    # Pixels matching bubble background color
    bg_mask = (_color_distance(patch, bg_arr) < tol).astype(np.uint8)
    # Treat the original text bbox itself as interior so text strokes don't split the bubble
    lx1, ly1 = x1 - sx1, y1 - sy1
    lx2, ly2 = x2 - sx1, y2 - sy1
    bg_mask[ly1:ly2, lx1:lx2] = 1

    # Keep ONLY the single connected component containing the text center
    num_labels, labels = cv2.connectedComponents(bg_mask, connectivity=4)
    cx = min(pw - 1, max(0, (lx1 + lx2) // 2))
    cy = min(ph - 1, max(0, (ly1 + ly2) // 2))
    target_label = int(labels[cy, cx])
    if target_label == 0:
        return bbox

    bubble = (labels == target_label).astype(np.uint8)

    # Expand outward row/column-wise while >= 92% of the strip stays inside the bubble
    nx1, ny1, nx2, ny2 = lx1, ly1, lx2, ly2
    max_dx = min(int(bw * max_expand_ratio), margin_x - 4)
    max_dy = min(int(bh * max_expand_ratio), margin_y - 4)

    for _ in range(max_dx):
        if nx1 <= 3:
            break
        if float(np.mean(bubble[ny1:ny2, nx1 - 1])) >= 0.94:
            nx1 -= 1
        else:
            break

    for _ in range(max_dx):
        if nx2 >= pw - 3:
            break
        if float(np.mean(bubble[ny1:ny2, nx2])) >= 0.94:
            nx2 += 1
        else:
            break

    for _ in range(max_dy):
        if ny1 <= 3:
            break
        if float(np.mean(bubble[ny1 - 1, nx1:nx2])) >= 0.94:
            ny1 -= 1
        else:
            break

    for _ in range(max_dy):
        if ny2 >= ph - 3:
            break
        if float(np.mean(bubble[ny2, nx1:nx2])) >= 0.94:
            ny2 += 1
        else:
            break

    # Leave crisp visual breathing room (6px or ~6% of expanded dim) away from bubble ink border
    pad_x = min(10, max(4, int((nx2 - nx1) * 0.05)))
    pad_y = min(8, max(3, int((ny2 - ny1) * 0.05)))
    if nx1 < lx1:
        nx1 = min(lx1, nx1 + pad_x)
    if nx2 > lx2:
        nx2 = max(lx2, nx2 - pad_x)
    if ny1 < ly1:
        ny1 = min(ly1, ny1 + pad_y)
    if ny2 > ly2:
        ny2 = max(ly2, ny2 - pad_y)

    return (sx1 + nx1, sy1 + ny1, sx1 + nx2, sy1 + ny2)


def extract_block_mask_and_style(
    image_rgb: np.ndarray,
    block: OCRBlock,
) -> Tuple[np.ndarray, TextStyle, Tuple[int, int, int, int]]:
    """
    Create a stroke-complete binary erasure mask (uint8, 0 or 255) for `block`
    — capturing thick meme outlines, drop shadows, and anti-aliased fringes so LaMa
    leaves zero dark ghost halos — and infer typography style + speech-bubble bounds.
    """
    h_img, w_img = image_rgb.shape[:2]
    x1, y1, x2, y2 = block.bbox
    min_line_h = max(8.0, min(l.height for l in block.lines))

    # Generous ROI padding so thick meme strokes & drop shadows outside OCR boxes are fully visible
    pad = max(24, int(min_line_h * 0.65))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(w_img, x2 + pad)
    ry2 = min(h_img, y2 + pad)

    roi = image_rgb[ry1:ry2, rx1:rx2]
    rh, rw = roi.shape[:2]

    # Build tight union mask of all individual line polygons inside the ROI
    poly_mask = np.zeros((rh, rw), dtype=np.uint8)
    for line in block.lines:
        pts = np.array([[p[0] - rx1, p[1] - ry1] for p in line.polygon], dtype=np.int32)
        cv2.fillPoly(poly_mask, [pts], 255)

    # Check if comic-text-detector provided a learned UNet pixel ink segmentation mask for this block
    learned_roi = None
    if block.comic_seg_mask is not None and block.comic_seg_mask.shape[:2] == (h_img, w_img):
        seg_slice = block.comic_seg_mask[ry1:ry2, rx1:rx2]
        if int(np.count_nonzero(seg_slice)) >= 12:
            learned_roi = ((seg_slice > 127).astype(np.uint8)) * 255
            # Ensure poly_mask covers the full learned ink footprint (DBNet kernels can be shrunken)
            poly_mask = cv2.bitwise_or(poly_mask, learned_roi)

    # Sample true background pixels cleanly outside text strokes!
    if learned_roi is not None:
        # When we have the learned UNet ink mask, non-ink pixels inside & immediately around
        # the block bbox give an uncontaminated reading of the speech bubble / background color!
        bx_mask = np.zeros((rh, rw), dtype=np.uint8)
        bx_pad = 6
        cv2.rectangle(
            bx_mask,
            (max(0, (x1 - rx1) - bx_pad), max(0, (y1 - ry1) - bx_pad)),
            (min(rw - 1, (x2 - rx1) + bx_pad), min(rh - 1, (y2 - ry1) + bx_pad)),
            255,
            thickness=-1,
        )
        k_ink = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        ink_excl = cv2.dilate(learned_roi, k_ink)
        halo_mask = cv2.subtract(bx_mask, ink_excl)
        halo_pixels = roi[halo_mask > 0]
    else:
        halo_pixels = np.empty((0, 3), dtype=np.uint8)

    if len(halo_pixels) < 20:
        # Outer halo (6..16px beyond polygons — far enough outside thick meme outlines!) to sample true background
        inner_r = max(7, int(min_line_h * 0.22) | 1)
        outer_r = max(inner_r + 10, int(min_line_h * 0.55) | 1)
        k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_r, inner_r))
        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_r, outer_r))
        halo_mask = cv2.subtract(cv2.dilate(poly_mask, k_outer), cv2.dilate(poly_mask, k_inner))
        halo_pixels = roi[halo_mask > 0]
    if len(halo_pixels) < 10:
        halo_pixels = roi[poly_mask == 0]
    if len(halo_pixels) == 0:
        halo_pixels = roi.reshape(-1, 3)

    bg_median = np.median(halo_pixels, axis=0)
    bg_std = float(np.mean(np.std(halo_pixels.astype(np.float32), axis=0)))
    uniform_bg = bg_std < 22.0
    bg_color = (int(bg_median[0]), int(bg_median[1]), int(bg_median[2]))

    # Compute color distance from local background across the whole ROI
    dist_map = _color_distance(roi, bg_median)
    inside_dist = dist_map[poly_mask > 0]

    char_mask = np.zeros((rh, rw), dtype=np.uint8)
    fg_color = (255, 255, 255) if _luminance(bg_color) < 128 else (18, 18, 18)
    has_stroke = False
    stroke_color = (0, 0, 0)
    stroke_width = 0
    is_bold = False

    if len(inside_dist) > 20:
        # ---- Style Inference FIRST so mask dilation knows whether text has a thick outline ----
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        in_gray = gray_roi[poly_mask > 0]
        p10, p50, p90 = np.percentile(in_gray, [10, 50, 90])

        # Expand search region slightly (+6px) so strokes sitting on polygon edges are visible
        k_style = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        style_zone = cv2.dilate(poly_mask, k_style)

        d_min, d_max = float(inside_dist.min()), float(inside_dist.max())
        if d_max - d_min > 18.0:
            norm_dist = np.clip((dist_map - d_min) / (d_max - d_min) * 255.0, 0, 255).astype(np.uint8)
            otsu_vals = norm_dist[poly_mask > 0]
            thresh_val, _ = cv2.threshold(otsu_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            raw_fg = ((norm_dist >= max(thresh_val * 0.72, 24)) & (poly_mask > 0)).astype(np.uint8) * 255
        else:
            raw_fg = poly_mask.copy()

        fg_ratio = float(np.count_nonzero(raw_fg)) / max(1, int(np.count_nonzero(poly_mask)))

        # Check for classic high-contrast outlined text (white meme fill + black outline)
        bg_is_white = _luminance(bg_color) > 210.0 and (uniform_bg or bg_std < 38.0)
        white_mask = ((gray_roi >= 215) & (poly_mask > 0)).astype(np.uint8) * 255
        dark_mask = ((gray_roi <= 50) & (style_zone > 0)).astype(np.uint8) * 255
        white_frac = np.count_nonzero(white_mask) / max(1, len(in_gray))
        dark_frac = np.count_nonzero(dark_mask & poly_mask) / max(1, len(in_gray))

        if not bg_is_white and white_frac >= 0.08 and dark_frac >= 0.05:
            k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            white_ring = cv2.subtract(cv2.dilate(white_mask, k_ring), white_mask)
            ring_overlap = np.count_nonzero(cv2.bitwise_and(white_ring, dark_mask)) / max(
                1, np.count_nonzero(white_ring)
            )
            if ring_overlap >= 0.28:
                fg_color = tuple(int(c) for c in np.median(roi[white_mask > 0], axis=0))
                stroke_color = tuple(int(c) for c in np.median(roi[dark_mask > 0], axis=0))
                has_stroke = True
                # Estimate real outline thickness from dark-to-white area ratio
                est_sw = max(2, min(6, int(round(min_line_h * 0.085))))
                stroke_width = est_sw
                is_bold = True

        if not has_stroke:
            k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            core = cv2.erode(raw_fg, k_erode)
            fg_pts = roi[core > 0] if np.count_nonzero(core) >= 15 else roi[raw_fg > 0]
            if len(fg_pts) > 0:
                fg_med = np.median(fg_pts, axis=0)
                fg_color = (int(fg_med[0]), int(fg_med[1]), int(fg_med[2]))

                rim = cv2.subtract(cv2.dilate(raw_fg, k_erode), raw_fg)
                rim = cv2.bitwise_and(rim, style_zone)
                rim_pts = roi[rim > 0]
                if len(rim_pts) >= 20:
                    rim_med = np.median(rim_pts, axis=0)
                    d_fg = float(np.linalg.norm(rim_med - fg_med))
                    d_bg = float(np.linalg.norm(rim_med - bg_median))
                    if d_fg > 65.0 and d_bg > 55.0 and not uniform_bg:
                        has_stroke = True
                        stroke_color = (int(rim_med[0]), int(rim_med[1]), int(rim_med[2]))
                        stroke_width = max(2, min(5, int(round(min_line_h * 0.065))))

            is_bold = fg_ratio > 0.34 or (p90 - p10 > 180 and not uniform_bg)

        # ---- Stroke-Complete Erasure Mask (Zero Ghost Halo!) ----
        if uniform_bg and not has_stroke and 0.05 <= fg_ratio <= 0.88:
            # Speech bubble / solid bar: capture ALL foreground strokes touching the polygon
            # (within a safe local bound so we never touch the bubble's outer ink border!)
            fg_any = (dist_map > 28.0).astype(np.uint8) * 255
            k_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            safe_zone = cv2.dilate(poly_mask, k_safe)
            fg_in_zone = cv2.bitwise_and(fg_any, safe_zone)

            # Keep only connected foreground components that actually overlap `poly_mask`
            num_cc, cc_labels = cv2.connectedComponents(fg_in_zone, connectivity=8)
            seed_labels = set(np.unique(cc_labels[poly_mask > 0])) - {0}
            clean_fg = np.isin(cc_labels, list(seed_labels)).astype(np.uint8) * 255 if seed_labels else raw_fg

            if learned_roi is not None:
                # Combine learned comic UNet ink segmentation with connected foreground strokes
                # so furigana, thin kana, and exclamation marks are 100% erased inside the bubble.
                combined_fg = cv2.bitwise_or(clean_fg, learned_roi)
                k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                char_mask = cv2.dilate(combined_fg, k_dil)
                bubble_safe = ((dist_map < 30.0) | (poly_mask > 0) | (combined_fg > 0)).astype(np.uint8) * 255
                char_mask = cv2.bitwise_and(char_mask, bubble_safe)
            else:
                k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                char_mask = cv2.dilate(clean_fg, k_dil)
                # Clamp strictly inside uniform bubble pixels + original polygon so borders stay crisp
                bubble_safe = ((dist_map < 28.0) | (poly_mask > 0) | (clean_fg > 0)).astype(np.uint8) * 255
                char_mask = cv2.bitwise_and(char_mask, bubble_safe)
        elif learned_roi is not None:
            # Screentones, comic panels, or textured comic artwork:
            # Use the learned UNet pixel ink mask so LaMa ONLY inpaints actual text strokes
            # (+ clean outline margin), preserving surrounding manga artwork & panel lines!
            dil_px = (
                max(11, min(19, int(round(min_line_h * 0.28)) | 1))
                if has_stroke
                else max(9, min(15, int(round(min_line_h * 0.20)) | 1))
            )
            k_seg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px, dil_px))
            char_mask = cv2.dilate(learned_roi, k_seg)
            if has_stroke:
                fg_any = (dist_map > 26.0).astype(np.uint8) * 255
                candidate = cv2.bitwise_and(fg_any, cv2.dilate(learned_roi, k_seg))
                num_cc, cc_labels = cv2.connectedComponents(candidate, connectivity=8)
                seed_labels = set(np.unique(cc_labels[learned_roi > 0])) - {0}
                if seed_labels:
                    stroke_cc = np.isin(cc_labels, list(seed_labels)).astype(np.uint8) * 255
                    k_pad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    char_mask = cv2.bitwise_or(char_mask, cv2.dilate(stroke_cc, k_pad))
        else:
            # Outlined meme text, drop shadows, or textured/photo background:
            # Dilate generously so thick black meme borders & drop shadows are 100% INSIDE the mask!
            dil_px = max(15, min(27, int(round(min_line_h * (0.36 if has_stroke else 0.26))) | 1))
            k_poly = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px, dil_px))
            char_mask = cv2.dilate(poly_mask, k_poly)

            # If outlined meme text, also absorb any connected outline pixels touching the mask
            if has_stroke:
                fg_any = (dist_map > 26.0).astype(np.uint8) * 255
                k_extra = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px + 8, dil_px + 8))
                candidate = cv2.bitwise_and(fg_any, cv2.dilate(poly_mask, k_extra))
                num_cc, cc_labels = cv2.connectedComponents(candidate, connectivity=8)
                seed_labels = set(np.unique(cc_labels[poly_mask > 0])) - {0}
                if seed_labels:
                    stroke_cc = np.isin(cc_labels, list(seed_labels)).astype(np.uint8) * 255
                    k_pad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    char_mask = cv2.bitwise_or(char_mask, cv2.dilate(stroke_cc, k_pad))
    else:
        k_poly = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        char_mask = cv2.dilate(poly_mask, k_poly)

    alpha_chars = [ch for ch in block.text if ch.isalpha()]
    is_uppercase = len(alpha_chars) >= 3 and all(ch.isupper() for ch in alpha_chars)

    # For uniform backgrounds (speech bubbles / solid bars), compute inscribed clean bubble bounds.
    # Vertical manga bubbles expand generously horizontally so horizontal translations fill the oval!
    expand_ratio = 0.85 if block.is_vertical else 0.45
    expanded_bbox = (
        _find_speech_bubble_box(image_rgb, block.bbox, bg_color, max_expand_ratio=expand_ratio)
        if uniform_bg and abs(block.angle) < 3.0
        else block.bbox
    )

    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    full_mask[ry1:ry2, rx1:rx2] = char_mask

    comic_style = bool(block.is_comic_block and not (has_stroke and is_bold))

    style = TextStyle(
        fg_color=fg_color,
        bg_color=bg_color,
        has_stroke=has_stroke,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        is_bold=is_bold,
        is_uppercase=is_uppercase,
        angle=block.angle,
        alignment=block.alignment,
        uniform_bg=uniform_bg,
        comic_style=comic_style,
    )

    return full_mask, style, expanded_bbox
