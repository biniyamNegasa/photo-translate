import logging
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.languages import detect_language_code, detect_script
from app.pipeline.inpainter import inpainter
from app.pipeline.mask import extract_block_mask_and_style
from app.pipeline.ocr import ocr_engine
from app.pipeline.renderer import composite_and_render
from app.pipeline.translator import translator_registry
from app.schemas import ItemStatus, TextBlock

logger = logging.getLogger(__name__)

StageCallback = Callable[[ItemStatus, str], None]


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
    on_stage: Optional[StageCallback] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], List[TextBlock]]:
    """
    Execute the image text-translation pipeline:
      1. Multilingual OCR + horizontal & vertical manga column grouping
      2. Stroke-complete mask extraction + typography & speech-bubble geometry analysis
      3. Hybrid erasure: clean solid fill for speech bubbles + full-res LaMa ONNX for scenes/memes
      4. Pluggable API translation (DeepSeek Vision / Addis AI / Google / DeepL)
      5. Pixel-exact compositing + ink-centered, balanced HarfBuzz rendering

    Returns: (translated_rgb, inpainted_rgb, block_masks, blocks)
    """
    if on_stage:
        on_stage(ItemStatus.DETECTING, "Detecting text regions...")

    ocr_blocks = ocr_engine.detect_and_group(image_rgb)
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
