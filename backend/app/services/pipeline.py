import logging
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

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


def _detect_and_recognize_blocks(
    image_rgb: np.ndarray,
    source_lang: str,
    use_comic_detector: bool = True,
) -> List[OCRBlock]:
    """
    Manga/comics-first text detection & OCR:
      1. Run `comic-text-detector` ONNX (YOLOv5 speech-bubble blocks + UNet pixel ink mask + DBNet lines).
      2. Route each bubble crop to the optimal recognizer:
         - Japanese (`source_lang == "ja"`) or vertical manga columns -> quantized `manga-ocr` ONNX
           (natively reads vertical Japanese and strips furigana!).
         - Horizontal / other scripts -> `RapidOCR` on the full-resolution bubble crop
           (auto-refining with `manga-ocr` if Japanese kana is detected).
      3. Gracefully fall back to full-page `RapidOCR` if the comic detector is unavailable or finds no blocks.
    """
    if use_comic_detector:
        regions = comic_detector.detect(image_rgb, source_lang=source_lang)
        if regions:
            h_img, w_img = image_rgb.shape[:2]
            comic_blocks: List[OCRBlock] = []
            page_has_jp = source_lang == "ja"

            for idx, region in enumerate(regions):
                bx1, by1, bx2, by2 = region.bbox

                # Tight crop (+6px) for MangaOCR; looser crop (+14px) for RapidOCR
                jp_x1, jp_y1 = max(0, bx1 - 6), max(0, by1 - 6)
                jp_x2, jp_y2 = min(w_img, bx2 + 6), min(h_img, by2 + 6)
                crop_jp = image_rgb[jp_y1:jp_y2, jp_x1:jp_x2]

                rp_x1, rp_y1 = max(0, bx1 - 14), max(0, by1 - 14)
                rp_x2, rp_y2 = min(w_img, bx2 + 14), min(h_img, by2 + 14)
                crop_rp = image_rgb[rp_y1:rp_y2, rp_x1:rp_x2]

                text = ""
                rec_conf = region.confidence
                rapid_lines: List[OCRLine] = []

                prefer_manga_ocr = (source_lang == "ja") or (
                    source_lang == "auto" and region.is_vertical
                )

                if prefer_manga_ocr:
                    m_txt = manga_ocr.recognize(crop_jp)
                    if m_txt:
                        text = m_txt
                        rec_conf = max(region.confidence, 0.85)
                        if detect_script(m_txt) == "cjk_jp":
                            page_has_jp = True
                    else:
                        text, r_conf, rapid_lines = ocr_engine.recognize_crop(
                            crop_rp, offset_xy=(rp_x1, rp_y1)
                        )
                        if r_conf > 0:
                            rec_conf = (region.confidence + r_conf) / 2.0
                else:
                    text, r_conf, rapid_lines = ocr_engine.recognize_crop(
                        crop_rp, offset_xy=(rp_x1, rp_y1)
                    )
                    if r_conf > 0:
                        rec_conf = (region.confidence + r_conf) / 2.0

                    # Auto-upgrade horizontal Japanese bubbles (or weak/short RapidOCR reads on a Japanese manga page)
                    should_try_manga = source_lang == "auto" and (
                        (text and detect_script(text) == "cjk_jp")
                        or not text
                        or (page_has_jp and (len(text.strip()) <= 2 or r_conf < 0.80))
                    )
                    if should_try_manga:
                        m_txt = manga_ocr.recognize(crop_jp)
                        if m_txt and (
                            detect_script(m_txt) == "cjk_jp"
                            or (text and detect_script(text) == "cjk_jp")
                        ):
                            text = m_txt
                            rapid_lines = []
                            rec_conf = max(rec_conf, 0.85)
                            page_has_jp = True

                text = text.strip()
                if not text:
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
        m = block_masks.get(blk.id)
        if m is None:
            continue
        if blk.style.uniform_bg:
            # Clean studio-grade solid fill inside speech bubbles and solid caption bars
            alpha = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (3, 3), 0)
            alpha = np.clip(alpha * 1.25, 0.0, 1.0)[..., None]
            bg_fill = np.array(blk.style.bg_color, dtype=np.float32)
            cleaned = np.clip(
                bg_fill * alpha + cleaned.astype(np.float32) * (1.0 - alpha),
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
    if on_stage:
        on_stage(ItemStatus.DETECTING, "Detecting speech bubbles & text...")

    ocr_blocks = _detect_and_recognize_blocks(
        image_rgb=image_rgb,
        source_lang=source_lang,
        use_comic_detector=use_comic_detector,
    )
    if not ocr_blocks:
        return image_rgb.copy(), image_rgb.copy(), {}, []

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

    # Hybrid erasure: clean solid fill for uniform bubbles + LaMa ONNX for textured/outlined text
    if on_stage:
        on_stage(ItemStatus.INPAINTING, "Erasing original text...")
    pre_cleaned, lama_mask = _fill_uniform_bubble_masks(image_rgb, blocks, block_masks)
    if np.count_nonzero(lama_mask) > 0:
        inpainted_rgb = inpainter.inpaint(pre_cleaned, lama_mask, use_lama=use_lama)
    else:
        inpainted_rgb = pre_cleaned

    # Translate grouped blocks
    if on_stage:
        on_stage(ItemStatus.TRANSLATING, f"Translating {len(blocks)} text block(s)...")

    translator = translator_registry.get(translator_id)
    by_src: Dict[str, List[TextBlock]] = {}
    for blk in blocks:
        by_src.setdefault(blk.source_lang, []).append(blk)

    for src_code, grp in by_src.items():
        src_texts = [b.source_text for b in grp]
        translated_texts = translator.translate_batch(src_texts, src_code, target_lang)
        for blk, tr in zip(grp, translated_texts):
            blk.translated_text = tr

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
