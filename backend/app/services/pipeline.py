import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.languages import detect_language_code, detect_script
from app.pipeline.comic_detector import ComicRegion, comic_detector
from app.pipeline.inpainter import inpainter
from app.pipeline.manga_ocr import manga_ocr
from app.pipeline.mask import extract_block_mask_and_style
from app.pipeline.ocr import OCRBlock, OCRLine, ocr_engine
from app.pipeline.renderer import composite_and_render
from app.pipeline.translator import translator_registry
from app.schemas import ItemStatus, TextBlock

logger = logging.getLogger(__name__)

StageCallback = Callable[[ItemStatus, str], None]

MAX_PIPELINE_DIMENSION = 8192
MAX_PIPELINE_PIXELS = 25_000_000
MAX_BLOCKS_PER_IMAGE = 150


def _build_lines_for_comic_region(
    region: ComicRegion,
    text: str,
    confidence: float,
    rapid_lines: List[OCRLine],
) -> List[OCRLine]:
    if rapid_lines:
        return rapid_lines

    if region.line_bboxes:
        out: List[OCRLine] = []
        for lx1, ly1, lx2, ly2 in region.line_bboxes:
            lw = max(1, lx2 - lx1)
            lh = max(1, ly2 - ly1)
            out.append(
                OCRLine(
                    polygon=[
                        [float(lx1), float(ly1)],
                        [float(lx2), float(ly1)],
                        [float(lx2), float(ly2)],
                        [float(lx1), float(ly2)],
                    ],
                    text=text,
                    confidence=confidence,
                    bbox=(lx1, ly1, lx2, ly2),
                    height=float(lw if region.is_vertical else lh),
                    width=float(lw),
                    angle=0.0,
                    is_vertical=region.is_vertical,
                )
            )
        return out

    bx1, by1, bx2, by2 = region.bbox
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    est_h = float(bw if region.is_vertical else bh) / max(1, region.line_count)
    return [
        OCRLine(
            polygon=region.polygon,
            text=text,
            confidence=confidence,
            bbox=region.bbox,
            height=max(8.0, est_h),
            width=float(bw),
            angle=0.0,
            is_vertical=region.is_vertical,
        )
    ]


def _is_meaningful_text(text: str, confidence: float = 1.0) -> bool:
    """Filter out OCR symbol/SFX noise (like '\\', '++t\"', or low-conf 1-letter logo hits) while keeping dialogue & '?!'."""
    t = text.strip()
    if not t:
        return False
    if t in ("?!", "!?", "!!", "??", "!!!", "…", "..."):
        return True
    alnum = [ch for ch in t if ch.isalnum()]
    if not alnum:
        return False
    # Reject symbol-heavy OCR gibberish like '++t"' or '#@x!'
    if detect_script(t) == "latin":
        if len(alnum) == 1 and len(t) >= 3:
            return False
        if len(t) == 1 and t.upper() != "I" and confidence < 0.88:
            return False
    return True


def _detect_and_recognize_blocks(
    image_rgb: np.ndarray,
    source_lang: str,
    use_comic_detector: bool = True,
) -> List[OCRBlock]:
    """
    Manga/comics-first text detection & OCR:
      1. Run `comic-text-detector` ONNX (YOLOv5 speech-bubble blocks + UNet pixel ink mask + DBNet lines).
      2. Route each bubble crop to the optimal recognizer WITHOUT hallucinating Japanese on English comics:
         - Explicit Japanese (`source_lang == "ja"`) -> quantized `manga-ocr` ONNX directly.
         - Explicit non-Japanese (`en`, `zh`, `ko`, etc.) -> `RapidOCR` on full-res bubble crops.
         - Auto-detect (`source_lang == "auto"`): run fast `RapidOCR` first; if the page is predominantly
           Western/Latin dialogue, never feed empty/SFX artwork crops to `manga-ocr`. On Japanese / vertical
           manga pages, upgrade vertical & Japanese bubbles with `manga-ocr` (stripping furigana).
      3. Gracefully fall back to full-page `RapidOCR` if the comic detector is unavailable or finds no blocks.
    """
    if use_comic_detector:
        regions = comic_detector.detect(image_rgb, source_lang=source_lang)
        if regions:
            h_img, w_img = image_rgb.shape[:2]

            def _make_crops(region: ComicRegion) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
                bx1, by1, bx2, by2 = region.bbox
                jp_x1, jp_y1 = max(0, bx1 - 6), max(0, by1 - 6)
                jp_x2, jp_y2 = min(w_img, bx2 + 6), min(h_img, by2 + 6)
                crop_jp = image_rgb[jp_y1:jp_y2, jp_x1:jp_x2]

                rp_x1, rp_y1 = max(0, bx1 - 14), max(0, by1 - 14)
                rp_x2, rp_y2 = min(w_img, bx2 + 14), min(h_img, by2 + 14)
                crop_rp = image_rgb[rp_y1:rp_y2, rp_x1:rp_x2]
                return crop_jp, crop_rp, (rp_x1, rp_y1)

            def _pass1(region: ComicRegion) -> Tuple[ComicRegion, str, float, List[OCRLine], np.ndarray, bool]:
                crop_jp, crop_rp, offset_xy = _make_crops(region)
                if source_lang == "ja":
                    m_txt = manga_ocr.recognize(crop_jp)
                    if m_txt:
                        return region, m_txt.strip(), max(region.confidence, 0.85), [], crop_jp, True
                    r_txt, r_conf, r_lines = ocr_engine.recognize_region_lines(
                        image_rgb,
                        bubble_bbox=region.bbox,
                        line_bboxes=region.line_bboxes,
                        is_vertical=region.is_vertical,
                    )
                    if not r_txt:
                        r_txt, r_conf, r_lines = ocr_engine.recognize_crop(crop_rp, offset_xy=offset_xy)
                    conf = (region.confidence + r_conf) / 2.0 if r_conf > 0 else region.confidence
                    return region, r_txt.strip(), conf, r_lines, crop_jp, True

                # Fast path: run RapidOCR recognition-only on DBNet line boxes already found by
                # comic-text-detector (skips running RapidOCR's full DBNet detector per bubble!).
                r_txt, r_conf, r_lines = ocr_engine.recognize_region_lines(
                    image_rgb,
                    bubble_bbox=region.bbox,
                    line_bboxes=region.line_bboxes,
                    is_vertical=region.is_vertical,
                )
                if not r_txt:
                    r_txt, r_conf, r_lines = ocr_engine.recognize_crop(crop_rp, offset_xy=offset_xy)
                conf = (region.confidence + r_conf) / 2.0 if r_conf > 0 else region.confidence
                return region, r_txt.strip(), conf, r_lines, crop_jp, False

            max_ocr_workers = min(max(1, settings.bubble_ocr_concurrency), len(regions))
            if max_ocr_workers > 1 and source_lang != "ja":
                with ThreadPoolExecutor(max_workers=max_ocr_workers) as ocr_pool:
                    pass1_results = list(ocr_pool.map(_pass1, regions))
            else:
                pass1_results = [_pass1(r) for r in regions]

            # Determine whether this page is predominantly Western/Latin comic dialogue vs Japanese manga
            latin_dialogue = sum(
                1
                for _, txt, conf, _, _, _ in pass1_results
                if txt and detect_script(txt) == "latin" and len(txt) >= 4 and conf >= 0.60
            )
            jp_reads = sum(
                1
                for _, txt, _, _, _, _ in pass1_results
                if txt and detect_script(txt) == "cjk_jp"
            )
            page_is_western = (source_lang == "auto") and (latin_dialogue >= 2) and (latin_dialogue >= jp_reads * 2)
            page_has_jp = (source_lang == "ja") or (jp_reads > 0)

            comic_blocks: List[OCRBlock] = []
            for region, text, rec_conf, rapid_lines, crop_jp, used_manga in pass1_results:
                if source_lang == "auto" and not used_manga and not page_is_western:
                    is_strong_latin = (
                        bool(text)
                        and detect_script(text) == "latin"
                        and len(text) >= 5
                        and rec_conf >= 0.78
                    )
                    should_try_manga = (
                        (region.is_vertical and not is_strong_latin)
                        or (bool(text) and detect_script(text) == "cjk_jp")
                        or (page_has_jp and (len(text) <= 2 or rec_conf < 0.78))
                    )
                    if should_try_manga:
                        m_txt = manga_ocr.recognize(crop_jp)
                        if m_txt and (
                            region.is_vertical
                            or detect_script(m_txt) == "cjk_jp"
                            or (text and detect_script(text) == "cjk_jp")
                        ):
                            text = m_txt.strip()
                            rapid_lines = []
                            rec_conf = max(rec_conf, 0.85)
                            if detect_script(text) == "cjk_jp":
                                page_has_jp = True

                if not _is_meaningful_text(text, confidence=rec_conf):
                    continue

                lines = _build_lines_for_comic_region(region, text, rec_conf, rapid_lines)
                comic_blocks.append(
                    OCRBlock(
                        id=f"blk_{len(comic_blocks) + 1}",
                        lines=lines,
                        polygon=region.polygon,
                        bbox=region.bbox,
                        text=text,
                        confidence=round(rec_conf, 3),
                        angle=0.0,
                        alignment="center",
                        is_vertical=region.is_vertical,
                        is_comic_block=True,
                        comic_seg_mask=region.seg_mask,
                    )
                )

            if comic_blocks:
                return comic_blocks

    return ocr_engine.detect_and_group(image_rgb)


def _fill_uniform_bubble_masks(
    image_rgb: np.ndarray,
    blocks: List[TextBlock],
    block_masks: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hybrid erasure:
      - Clean uniform-background blocks (speech bubbles, solid caption bars) are filled
        directly with their exact sampled `bg_color` (1px soft edge feather) for zero
        LaMa blur or smudge artifacts.
      - Non-uniform / outlined blocks are collected into `lama_mask` for patch-based LaMa ONNX.
    Returns: (partially_cleaned_rgb, lama_mask)
    """
    h_img, w_img = image_rgb.shape[:2]
    cleaned = image_rgb.copy()
    lama_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    for blk in blocks:
        if not blk.enabled:
            continue
        m = block_masks.get(blk.id)
        if m is None:
            continue
        if blk.style.uniform_bg:
            # Clean studio-grade solid fill inside speech bubbles and solid caption bars,
            # computed strictly inside the block's padded ROI to avoid full-image float32 copies.
            mx, my, mw, mh = cv2.boundingRect(m)
            if mw <= 0 or mh <= 0:
                continue
            pad = 4
            rx1 = max(0, mx - pad)
            ry1 = max(0, my - pad)
            rx2 = min(w_img, mx + mw + pad)
            ry2 = min(h_img, my + mh + pad)
            m_roi = m[ry1:ry2, rx1:rx2]
            alpha = cv2.GaussianBlur(m_roi.astype(np.float32) / 255.0, (3, 3), 0)
            alpha = np.clip(alpha * 1.25, 0.0, 1.0)[..., None]
            bg_fill = np.array(blk.style.bg_color, dtype=np.float32)
            roi_clean = cleaned[ry1:ry2, rx1:rx2].astype(np.float32)
            cleaned[ry1:ry2, rx1:rx2] = np.clip(
                bg_fill * alpha + roi_clean * (1.0 - alpha),
                0,
                255,
            ).astype(np.uint8)
        else:
            lama_mask = cv2.bitwise_or(lama_mask, m)

    return cleaned, lama_mask


def process_image_pipeline(
    image_rgb: np.ndarray,
    source_lang: str,
    target_lang: str,
    translator_id: str = "google",
    use_lama: bool = True,
    use_comic_detector: bool = True,
    on_stage: Optional[StageCallback] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], List[TextBlock]]:
    """
    Execute the image text-translation pipeline:
      1. Comic/Manga speech-bubble detection + learned UNet pixel ink mask (`comic-text-detector` ONNX)
         with `manga-ocr` ONNX for vertical/Japanese bubbles and `RapidOCR` for other scripts & fallback
      2. Stroke-complete mask refinement + typography & speech-bubble geometry analysis
      3. Hybrid erasure: clean solid fill for speech bubbles + full-res LaMa ONNX for screentones/scenes
      4. Pluggable API translation (DeepSeek / Addis AI / Google / DeepL)
      5. Pixel-exact compositing + comic-lettered, ink-centered, balanced HarfBuzz rendering

    Returns: (translated_rgb, inpainted_rgb, block_masks, blocks)
    """
    if image_rgb is None or image_rgb.ndim < 2:
        raise ValueError("Invalid input image array.")
    h_img, w_img = image_rgb.shape[:2]
    if (
        h_img <= 0
        or w_img <= 0
        or h_img > MAX_PIPELINE_DIMENSION
        or w_img > MAX_PIPELINE_DIMENSION
        or (h_img * w_img) > MAX_PIPELINE_PIXELS
    ):
        raise ValueError(
            f"Image dimensions ({w_img}x{h_img}) exceed maximum allowed pipeline limits "
            f"({MAX_PIPELINE_DIMENSION}px max side, {MAX_PIPELINE_PIXELS} pixels)."
        )

    if on_stage:
        on_stage(ItemStatus.DETECTING, "Detecting speech bubbles & text...")

    ocr_blocks = _detect_and_recognize_blocks(
        image_rgb=image_rgb,
        source_lang=source_lang,
        use_comic_detector=use_comic_detector,
    )
    if not ocr_blocks:
        return image_rgb.copy(), image_rgb.copy(), {}, []

    if len(ocr_blocks) > MAX_BLOCKS_PER_IMAGE:
        ocr_blocks = sorted(ocr_blocks, key=lambda b: b.confidence, reverse=True)[:MAX_BLOCKS_PER_IMAGE]

    block_masks: Dict[str, np.ndarray] = {}
    blocks: List[TextBlock] = []

    # When source_lang is "auto", detect dominant language per script across the WHOLE image
    # so short 1-3 word blocks don't get misclassified by langdetect.
    script_lang_map: Dict[str, str] = {}
    if source_lang == "auto":
        texts_by_script: Dict[str, List[str]] = {}
        for ob in ocr_blocks:
            sc = detect_script(ob.text)
            texts_by_script.setdefault(sc, []).append(ob.text)
        for sc, txts in texts_by_script.items():
            combined = " ".join(txts)
            script_lang_map[sc] = detect_language_code(combined, hint="auto")

    for ob in ocr_blocks:
        mask, style, expanded_bbox = extract_block_mask_and_style(image_rgb, ob)
        block_masks[ob.id] = mask

        if source_lang != "auto":
            detected_src = detect_language_code(ob.text, hint=source_lang)
        else:
            sc = detect_script(ob.text)
            detected_src = script_lang_map.get(sc, detect_language_code(ob.text, hint="auto"))

        blocks.append(
            TextBlock(
                id=ob.id,
                polygon=ob.polygon,
                bbox=ob.bbox,
                expanded_bbox=expanded_bbox,
                source_text=ob.text,
                translated_text=ob.text,
                source_lang=detected_src,
                confidence=ob.confidence,
                enabled=True,
                line_count=len(ob.lines),
                is_vertical=ob.is_vertical,
                style=style,
            )
        )

    translator = translator_registry.get(translator_id)

    # Pre-disable blocks whose source script cannot be translated by the selected engine
    # (e.g. untranslatable Japanese SFX on an English->Amharic page with Addis AI) so we
    # never waste LaMa passes erasing artwork that won't be translated!
    addis_supported = getattr(translator, "SUPPORTED_SOURCE_LANGS", None)
    if addis_supported:
        for blk in blocks:
            src_base = blk.source_lang.split("-")[0].lower()
            if src_base not in addis_supported and detect_script(blk.source_text) != "latin":
                blk.enabled = False

    # Hybrid erasure (CPU) and API translation (network I/O) are completely independent:
    # run them concurrently so network wait time overlaps with LaMa ONNX inpainting!
    pre_cleaned, lama_mask = _fill_uniform_bubble_masks(image_rgb, blocks, block_masks)
    needs_lama = int(np.count_nonzero(lama_mask)) > 0

    by_src: Dict[str, List[TextBlock]] = {}
    for blk in blocks:
        if blk.enabled:
            by_src.setdefault(blk.source_lang, []).append(blk)

    def _run_translations() -> Dict[str, List[str]]:
        if not by_src:
            return {}
        if len(by_src) <= 1:
            return {
                src_code: translator.translate_batch([b.source_text for b in grp], src_code, target_lang)
                for src_code, grp in by_src.items()
            }
        with ThreadPoolExecutor(max_workers=min(4, len(by_src))) as tr_pool:
            futs = {
                src_code: tr_pool.submit(
                    translator.translate_batch, [b.source_text for b in grp], src_code, target_lang
                )
                for src_code, grp in by_src.items()
            }
            return {src_code: fut.result() for src_code, fut in futs.items()}

    active_count = sum(len(g) for g in by_src.values())
    if needs_lama:
        if on_stage:
            on_stage(ItemStatus.INPAINTING, f"Erasing text & translating {active_count} block(s)...")
        with ThreadPoolExecutor(max_workers=2) as stage_pool:
            inpaint_fut = stage_pool.submit(inpainter.inpaint, pre_cleaned, lama_mask, use_lama)
            translate_fut = stage_pool.submit(_run_translations)
            inpainted_rgb = inpaint_fut.result()
            translations_by_src = translate_fut.result()
    else:
        if on_stage:
            on_stage(ItemStatus.TRANSLATING, f"Translating {active_count} text block(s)...")
        inpainted_rgb = pre_cleaned
        translations_by_src = _run_translations()

    for src_code, grp in by_src.items():
        for blk, tr in zip(grp, translations_by_src.get(src_code, [])):
            blk.translated_text = tr
            # If text stayed unchanged across a language switch (e.g. pure punctuation/SFX),
            # leave the original pixels bit-for-bit untouched instead of overwriting artwork!
            if (
                blk.source_lang.split("-")[0].lower() != target_lang.split("-")[0].lower()
                and blk.translated_text.strip() == blk.source_text.strip()
                and not any(ch.isalpha() for ch in blk.source_text)
            ):
                blk.enabled = False

    # Composite & render
    if on_stage:
        on_stage(ItemStatus.RENDERING, "Rendering style-matched text...")
    translated_rgb = composite_and_render(
        original_rgb=image_rgb,
        inpainted_rgb=inpainted_rgb,
        block_masks=block_masks,
        blocks=blocks,
        target_lang=target_lang,
    )

    return translated_rgb, inpainted_rgb, block_masks, blocks
