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


def _expand_uniform_bbox(
    image_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    bg_color: Tuple[int, int, int],
    max_expand_ratio: float = 0.28,
) -> Tuple[int, int, int, int]:
    """Safely expand bounding box outward only into uniform solid-color space (e.g. speech bubbles)."""
    h_img, w_img = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    max_dx = min(int(bw * max_expand_ratio), 60)
    max_dy = min(int(bh * max_expand_ratio), 40)
    bg_arr = np.array(bg_color, dtype=np.float32)
    tol = 28.0

    # Expand left
    nx1 = x1
    for step in range(1, max_dx + 1):
        col = x1 - step
        if col < 2:
            break
        strip = image_rgb[y1:y2, col]
        if np.mean(_color_distance(strip, bg_arr) < tol) > 0.90:
            nx1 = col
        else:
            break

    # Expand right
    nx2 = x2
    for step in range(1, max_dx + 1):
        col = x2 + step - 1
        if col >= w_img - 2:
            break
        strip = image_rgb[y1:y2, col]
        if np.mean(_color_distance(strip, bg_arr) < tol) > 0.90:
            nx2 = col + 1
        else:
            break

    # Expand top
    ny1 = y1
    for step in range(1, max_dy + 1):
        row = y1 - step
        if row < 2:
            break
        strip = image_rgb[row, nx1:nx2]
        if np.mean(_color_distance(strip, bg_arr) < tol) > 0.90:
            ny1 = row
        else:
            break

    # Expand bottom
    ny2 = y2
    for step in range(1, max_dy + 1):
        row = y2 + step - 1
        if row >= h_img - 2:
            break
        strip = image_rgb[row, nx1:nx2]
        if np.mean(_color_distance(strip, bg_arr) < tol) > 0.90:
            ny2 = row + 1
        else:
            break

    # Leave a small 2px breathing margin from edges/borders
    if nx1 < x1:
        nx1 = min(x1, nx1 + 2)
    if nx2 > x2:
        nx2 = max(x2, nx2 - 2)
    if ny1 < y1:
        ny1 = min(y1, ny1 + 2)
    if ny2 > y2:
        ny2 = max(y2, ny2 - 2)

    return (nx1, ny1, nx2, ny2)


def extract_block_mask_and_style(
    image_rgb: np.ndarray,
    block: OCRBlock,
) -> Tuple[np.ndarray, TextStyle, Tuple[int, int, int, int]]:
    """
    Create a tight character/stroke-level binary mask (uint8, 0 or 255) for `block`
    and infer its typography style (fg color, stroke/outline, boldness, bg uniformity).
    Returns: (full_image_mask, TextStyle, expanded_bbox)
    """
    h_img, w_img = image_rgb.shape[:2]
    x1, y1, x2, y2 = block.bbox

    # Work on a padded local ROI for speed
    pad = max(12, int(min(x2 - x1, y2 - y1) * 0.2))
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

    # Outer halo (3..8px around line polygons) to sample local background
    k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
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

    # Compute pixel distance from local background inside the polygon
    dist_map = _color_distance(roi, bg_median)
    inside_pixels = roi[poly_mask > 0]
    inside_dist = dist_map[poly_mask > 0]

    # Adaptive character-level foreground segmentation
    char_mask = np.zeros((rh, rw), dtype=np.uint8)
    fg_color = (255, 255, 255) if _luminance(bg_color) < 128 else (18, 18, 18)
    has_stroke = False
    stroke_color = (0, 0, 0)
    stroke_width = 0
    is_bold = False

    if len(inside_dist) > 20:
        # Normalize distance map within polygon for Otsu thresholding
        d_min, d_max = float(inside_dist.min()), float(inside_dist.max())
        if d_max - d_min > 18.0:
            norm_dist = np.clip((dist_map - d_min) / (d_max - d_min) * 255.0, 0, 255).astype(np.uint8)
            otsu_vals = norm_dist[poly_mask > 0]
            thresh_val, _ = cv2.threshold(otsu_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            raw_fg = ((norm_dist >= max(thresh_val * 0.75, 25)) & (poly_mask > 0)).astype(np.uint8) * 255
        else:
            raw_fg = poly_mask.copy()

        fg_ratio = float(np.count_nonzero(raw_fg)) / max(1, int(np.count_nonzero(poly_mask)))

        # Decide whether to use tight character mask or full tight polygon mask for inpainting
        if uniform_bg and 0.08 <= fg_ratio <= 0.85:
            # Dilate character strokes slightly (4px) to cleanly remove anti-aliasing & outlines
            k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            char_mask = cv2.dilate(raw_fg, k_dil)
            # Keep mask bounded near the polygon (+6px) so we never erase nearby artwork
            k_bound = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            char_mask = cv2.bitwise_and(char_mask, cv2.dilate(poly_mask, k_bound))
        else:
            # Busy/textured background (photos, gradients, memes over scenes):
            # Cleanly cover the tight line polygons (+4px) so LaMa reconstructs the scene seamlessly
            k_poly = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            char_mask = cv2.dilate(poly_mask, k_poly)

        # ---- Style Inference (Foreground Color & Meme Stroke/Outline Detection) ----
        # Use bright/dark extremes + core eroded pixels to separate fill vs outline (classic meme text!)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        in_gray = gray_roi[poly_mask > 0]
        p10, p50, p90 = np.percentile(in_gray, [10, 50, 90])

        # Check for classic high-contrast outlined text (e.g. white meme text with black border)
        white_mask = ((gray_roi >= 215) & (poly_mask > 0)).astype(np.uint8) * 255
        dark_mask = ((gray_roi <= 45) & (poly_mask > 0)).astype(np.uint8) * 255
        white_frac = np.count_nonzero(white_mask) / max(1, len(in_gray))
        dark_frac = np.count_nonzero(dark_mask) / max(1, len(in_gray))

        # Check if dark pixels tightly wrap around white pixels (classic Impact meme style)
        # Guard: if the outer background is already bright white (e.g. black ink inside a white comic speech bubble),
        # those white pixels are the speech bubble background, NOT white meme text!
        bg_is_white = _luminance(bg_color) > 210.0 and (uniform_bg or bg_std < 38.0)
        if not bg_is_white and white_frac >= 0.08 and dark_frac >= 0.06:
            k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            white_ring = cv2.subtract(cv2.dilate(white_mask, k_ring), white_mask)
            ring_overlap = np.count_nonzero(cv2.bitwise_and(white_ring, dark_mask)) / max(
                1, np.count_nonzero(white_ring)
            )
            if ring_overlap >= 0.30:
                fg_color = tuple(int(c) for c in np.median(roi[white_mask > 0], axis=0))
                stroke_color = tuple(int(c) for c in np.median(roi[dark_mask > 0], axis=0))
                has_stroke = True
                stroke_width = max(2, int(round(min(l.height for l in block.lines) * 0.07)))
                is_bold = True

        if not has_stroke:
            # General foreground color estimation: use core eroded pixels of raw_fg
            k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            core = cv2.erode(raw_fg, k_erode)
            fg_pts = roi[core > 0] if np.count_nonzero(core) >= 15 else roi[raw_fg > 0]
            if len(fg_pts) > 0:
                fg_med = np.median(fg_pts, axis=0)
                fg_color = (int(fg_med[0]), int(fg_med[1]), int(fg_med[2]))

                # Detect general colored outline: pixels in dilated rim around core that differ from BOTH fg & bg
                rim = cv2.subtract(cv2.dilate(raw_fg, k_erode), raw_fg)
                rim = cv2.bitwise_and(rim, poly_mask)
                rim_pts = roi[rim > 0]
                if len(rim_pts) >= 20:
                    rim_med = np.median(rim_pts, axis=0)
                    d_fg = float(np.linalg.norm(rim_med - fg_med))
                    d_bg = float(np.linalg.norm(rim_med - bg_median))
                    if d_fg > 65.0 and d_bg > 55.0 and not uniform_bg:
                        has_stroke = True
                        stroke_color = (int(rim_med[0]), int(rim_med[1]), int(rim_med[2]))
                        stroke_width = max(2, int(round(min(l.height for l in block.lines) * 0.06)))

            # Bold heuristic: thicker glyphs occupy higher fraction of polygon
            is_bold = fg_ratio > 0.34 or (p90 - p10 > 180 and not uniform_bg)
    else:
        k_poly = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        char_mask = cv2.dilate(poly_mask, k_poly)

    # Check if original text was uppercase (for Latin/Cyrillic memes & headings)
    alpha_chars = [ch for ch in block.text if ch.isalpha()]
    is_uppercase = len(alpha_chars) >= 3 and all(ch.isupper() for ch in alpha_chars)

    # Expand bbox safely if background is uniform (speech bubble / solid bar)
    expanded_bbox = (
        _expand_uniform_bbox(image_rgb, block.bbox, bg_color)
        if uniform_bg and abs(block.angle) < 3.0
        else block.bbox
    )

    # Place ROI mask back into full image dimensions
    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    full_mask[ry1:ry2, rx1:rx2] = char_mask

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
    )

    return full_mask, style, expanded_bbox
