import io
import zipfile
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.languages import detect_language_code, detect_script
from app.pipeline.fonts import is_raqm_available, load_font, resolve_font_path
from app.pipeline.mask import extract_block_mask_and_style
from app.pipeline.ocr import OCRBlock, OCRLine, ocr_engine
from app.pipeline.renderer import composite_and_render
from app.pipeline.translator import BaseTranslator
from app.services.pipeline import process_image_pipeline
from app.services.storage import extract_uploaded_images


class MockTranslator(BaseTranslator):
    id = "mock"
    name = "Mock"
    description = "Mock"

    def is_available(self) -> bool:
        return True

    def translate_batch(self, texts, source_lang, target_lang):
        mapping = {
            "es": "HOLA MUNDO",
            "am": "ሰላም ዓለም",
            "ar": "مرحبا بالعالم",
            "ja": "こんにちは世界",
        }
        return [mapping.get(target_lang, f"[{target_lang}] {t}") for t in texts]


def make_meme_image() -> np.ndarray:
    """Create a realistic synthetic meme: colorful gradient, untouched artwork patch, bold white outlined text."""
    w, h = 640, 480
    x = np.linspace(40, 210, w, dtype=np.uint8)
    y = np.linspace(60, 180, h, dtype=np.uint8)
    r = np.tile(x, (h, 1))
    g = np.tile(y[:, None], (1, w))
    b = np.full((h, w), 140, dtype=np.uint8)
    img = Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")

    draw = ImageDraw.Draw(img)
    # Draw an artwork shape in the center (must remain 100% untouched!)
    draw.ellipse((240, 170, 400, 310), fill=(245, 195, 65), outline=(40, 30, 20), width=4)

    font = load_font("HELLO WORLD", "en", size=44, bold=True, meme_style=True)
    draw.text(
        (165, 28),
        "HELLO WORLD",
        font=font,
        fill=(255, 255, 255),
        stroke_width=4,
        stroke_fill=(0, 0, 0),
    )
    return np.array(img, dtype=np.uint8)


def make_speech_bubble_image() -> np.ndarray:
    """Create a synthetic drawing with a white speech bubble and dark text."""
    img = Image.new("RGB", (500, 350), (110, 145, 185))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((60, 60, 440, 240), radius=35, fill=(252, 252, 252), outline=(20, 20, 20), width=4)
    font = load_font("GOOD MORNING", "en", size=34, bold=False)
    draw.text((120, 125), "GOOD MORNING", font=font, fill=(25, 25, 25))
    return np.array(img, dtype=np.uint8)


def test_script_and_language_detection():
    assert detect_script("Hello world") == "latin"
    assert detect_script("ሰላም ዓለም") == "ethiopic"
    assert detect_script("مرحبا بالعالم") == "arabic"
    assert detect_script("こんにちは世界") == "cjk_jp"

    assert detect_language_code("ሰላም ዓለም") == "am"
    assert detect_language_code("مرحبا بالعالم") == "ar"
    assert detect_language_code("Hello world", hint="fr") == "fr"


def test_multilingual_font_resolution():
    for script in ("latin", "ethiopic", "arabic", "cjk_sc", "cyrillic"):
        path = resolve_font_path(script, bold=True)
        assert path != "", f"Expected a font for script={script}"
    # Verify HarfBuzz/raqm check runs cleanly
    assert isinstance(is_raqm_available(), bool)


def test_meme_pipeline_and_pixel_exact_compositing():
    meme = make_meme_image()

    with patch("app.services.pipeline.translator_registry.get", return_value=MockTranslator()):
        translated, inpainted, block_masks, blocks = process_image_pipeline(
            image_rgb=meme,
            source_lang="en",
            target_lang="es",
            translator_id="mock",
            use_lama=False,  # Offline OpenCV fallback for fast unit tests
        )

    assert len(blocks) >= 1
    assert any("HELLO" in b.source_text.upper() for b in blocks)
    assert blocks[0].style.has_stroke is True

    # Center artwork (far from text) MUST be 100% bit-for-bit identical to original
    np.testing.assert_array_equal(translated[200:280, 270:370], meme[200:280, 270:370])

    # Toggling all blocks off MUST restore the ENTIRE image bit-for-bit!
    for b in blocks:
        b.enabled = False
    restored = composite_and_render(meme, inpainted, block_masks, blocks, target_lang="es")
    np.testing.assert_array_equal(restored, meme)


def test_speech_bubble_multilingual_rendering():
    bubble = make_speech_bubble_image()

    for target_lang in ("am", "ar", "ja", "es"):
        with patch("app.services.pipeline.translator_registry.get", return_value=MockTranslator()):
            translated, _, _, blocks = process_image_pipeline(
                image_rgb=bubble,
                source_lang="en",
                target_lang=target_lang,
                translator_id="mock",
                use_lama=False,
            )
        assert translated.shape == bubble.shape
        assert len(blocks) >= 1
        assert blocks[0].style.uniform_bg is True


def test_zip_extraction_and_filtering():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        im_buf = io.BytesIO()
        Image.new("RGB", (64, 64), (10, 20, 30)).save(im_buf, format="PNG")
        zf.writestr("page_01.png", im_buf.getvalue())
        zf.writestr("__MACOSX/._page_01.png", b"junk")
        zf.writestr("notes.txt", b"ignore me")

    extracted = extract_uploaded_images([("chapter.zip", buf.getvalue())])
    assert len(extracted) == 1
    assert extracted[0][0] == "page_01.png"


def test_fastapi_endpoints():
    from app.main import app

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        langs = client.get("/api/languages").json()
        assert any(l["code"] == "am" for l in langs)
        assert any(l["code"] == "ar" and l["rtl"] for l in langs)
        status = client.get("/api/status").json()
        assert "translators" in status


def test_fastapi_batch_job_and_instant_rerender():
    import time
    from app.main import app

    meme = make_meme_image()
    meme_buf = io.BytesIO()
    Image.fromarray(meme).save(meme_buf, format="PNG")

    blank_buf = io.BytesIO()
    Image.new("RGB", (200, 150), (80, 120, 160)).save(blank_buf, format="PNG")

    with patch("app.services.pipeline.translator_registry.get", return_value=MockTranslator()), patch(
        "app.pipeline.inpainter.inpainter._ensure_loaded", return_value=False
    ), patch("app.pipeline.comic_detector.comic_detector._ensure_loaded", return_value=False):
        with TestClient(app) as client:
            resp = client.post(
                "/api/jobs",
                data={"target_lang": "es", "source_lang": "en", "translator": "google"},
                files=[
                    ("files", ("meme.png", meme_buf.getvalue(), "image/png")),
                    ("files", ("blank.png", blank_buf.getvalue(), "image/png")),
                ],
            )
            assert resp.status_code == 200
            job_id = resp.json()["job_id"]

            # Wait for background worker to finish batch
            job = None
            for _ in range(180):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] in ("completed", "failed"):
                    break
                time.sleep(0.1)

            assert job is not None
            assert job["status"] == "completed"
            assert job["total_items"] == 2
            assert job["completed_items"] == 2

            meme_item = next(it for it in job["items"] if it["filename"] == "meme.png")
            blank_item = next(it for it in job["items"] if it["filename"] == "blank.png")
            assert meme_item["status"] == "completed"
            assert len(meme_item["blocks"]) >= 1
            assert blank_item["status"] == "no_text"

            # Instant re-render: edit block text
            blk_id = meme_item["blocks"][0]["id"]
            rr = client.post(
                f"/api/jobs/{job_id}/items/{meme_item['item_id']}/rerender",
                json={"blocks": [{"id": blk_id, "translated_text": "BUENOS DIAS", "enabled": True}]},
            )
            assert rr.status_code == 200
            assert rr.json()["blocks"][0]["translated_text"] == "BUENOS DIAS"

            # Download batch ZIP
            z_resp = client.get(f"/api/jobs/{job_id}/download")
            assert z_resp.status_code == 200
            with zipfile.ZipFile(io.BytesIO(z_resp.content)) as zf:
                names = set(zf.namelist())
                assert names == {"meme.png", "blank.png"}

            # Cleanup job
            assert client.delete(f"/api/jobs/{job_id}").status_code == 200


def test_addis_ai_translator_adapter():
    from app.config import settings
    from app.pipeline.translator import AddisAITranslator, translator_registry

    tr = translator_registry.get("addis")
    assert isinstance(tr, AddisAITranslator)

    # Unavailable when API key is blank
    with patch.object(settings, "addis_api_key", ""):
        assert tr.is_available() is False

    # Available + sends expected x-api-key header and JSON payload
    captured = {}

    class FakeResp:
        status_code = 200
        text = '{"status":"success","data":{"translation":"ሰላም ዓለም"}}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "success", "data": {"translation": "ሰላም ዓለም"}}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResp()

    with patch.object(settings, "addis_api_key", "test-addis-key"), patch(
        "app.pipeline.translator.httpx.Client", FakeClient
    ):
        assert tr.is_available() is True
        result = tr.translate_batch(["Hello world"], "en", "am")
        assert result == ["ሰላም ዓለም"]
        assert captured["url"] == settings.addis_api_url
        assert captured["headers"]["x-api-key"] == "test-addis-key"
        assert captured["json"] == {
            "text": "Hello world",
            "source_language": "en",
            "target_language": "am",
        }

        # Misdetected Latin source language (e.g. 'de') falls back to 'en' instead of 400
        result_fallback = tr.translate_batch(["KYA HOH!!!"], "de", "am")
        assert result_fallback == ["ሰላም ዓለም"]
        assert captured["json"]["source_language"] == "en"

        # Unsupported stray non-Latin block (e.g. 'zh') is preserved cleanly without crashing
        result_stray = tr.translate_batch(["我"], "zh", "am")
        assert result_stray == ["我"]

    # Verify fallback response shapes
    assert AddisAITranslator._extract_translation({"translated_text": "Selam"}, "x") == "Selam"
    assert AddisAITranslator._extract_translation({"response_text": "Selam"}, "x") == "Selam"


def test_ethiopic_punctuation_fallback_renders_question_and_exclamation():
    from app.pipeline.fonts import get_font_codepoints, load_font_chain, resolve_font_path, split_font_runs
    from app.pipeline.renderer import _draw_runs

    ethiopic_path = resolve_font_path("ethiopic", bold=True)
    ethiopic_cps = get_font_codepoints(ethiopic_path)
    assert ord("ሀ") in ethiopic_cps

    chain = load_font_chain("ምን እያደረክ ነው?!!", "am", size=32, bold=True)
    runs = split_font_runs("ምን እያደረክ ነው?!!", chain)
    # Must split Ethiopic characters and ASCII punctuation (`?!!`) into separate font runs
    assert len(runs) >= 2
    assert runs[-1][0] == "?!!"
    assert runs[-1][1] is not runs[0][1]

    # Render with and without `?!!` and verify pixels were actually drawn for `?!!`
    img_with = Image.new("RGBA", (420, 80), (255, 255, 255, 255))
    img_without = Image.new("RGBA", (420, 80), (255, 255, 255, 255))
    _draw_runs(ImageDraw.Draw(img_with), (10, 10), "ምን እያደረክ ነው?!!", chain, (0, 0, 0, 255))
    _draw_runs(ImageDraw.Draw(img_without), (10, 10), "ምን እያደረክ ነው", chain, (0, 0, 0, 255))

    diff = np.abs(np.array(img_with, dtype=np.int16) - np.array(img_without, dtype=np.int16))
    # Substantial dark pixels drawn for `?!!`
    assert int(np.count_nonzero(diff.sum(axis=-1) > 100)) > 80


def test_deepseek_translator_batch_adapter():
    import json
    from app.config import settings
    from app.pipeline.translator import DeepSeekTranslator, translator_registry

    tr = translator_registry.get("deepseek")
    assert isinstance(tr, DeepSeekTranslator)

    with patch.object(settings, "deepseek_api_key", ""):
        assert tr.is_available() is False

    captured = {}

    class FakeResp:
        status_code = 200
        text = "ok"

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "translations": [
                                        {"id": 0, "text": "What are you doing?!"},
                                        {"id": 1, "text": "Look out below!!"},
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResp()

    with patch.object(settings, "deepseek_api_key", "sk-test-deepseek"), patch(
        "app.pipeline.translator.httpx.post", fake_post
    ):
        assert tr.is_available() is True
        out = tr.translate_batch(["何してるの？！", "下を見ろ！！"], "ja", "en")
        assert out == ["What are you doing?!", "Look out below!!"]
        assert captured["url"] == settings.deepseek_api_url
        assert captured["headers"]["Authorization"] == "Bearer sk-test-deepseek"
        assert captured["json"]["model"] == settings.deepseek_model
        assert captured["json"]["response_format"] == {"type": "json_object"}


def test_vertical_cjk_column_grouping():
    from app.pipeline.ocr import OCRLine, group_ocr_lines

    # Two vertical Japanese columns inside one speech bubble (rightmost column reads first!)
    col_left = OCRLine(
        polygon=[[100, 50], [130, 50], [130, 210], [100, 210]],
        text="世界だ",
        confidence=0.95,
        bbox=(100, 50, 130, 210),
        height=30.0,
        width=30.0,
        angle=0.0,
        is_vertical=True,
    )
    col_right = OCRLine(
        polygon=[[142, 48], [172, 48], [172, 208], [142, 208]],
        text="すごい",
        confidence=0.96,
        bbox=(142, 48, 172, 208),
        height=30.0,
        width=30.0,
        angle=0.0,
        is_vertical=True,
    )
    blocks = group_ocr_lines([col_left, col_right])
    assert len(blocks) == 1
    assert blocks[0].is_vertical is True
    assert blocks[0].text == "すごい世界だ"


def test_balanced_line_wrapping():
    from app.pipeline.fonts import load_font_chain
    from app.pipeline.renderer import _wrap_balanced_words

    img = Image.new("RGBA", (400, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    chain = load_font_chain("WHAT ARE YOU DOING", "en", size=24, bold=True)
    # Width fits ~2 words per line; balanced wrapping should produce 2 balanced lines (not 3+1)
    lines = _wrap_balanced_words(draw, ["WHAT", "ARE", "YOU", "DOING?"], chain, max_width=190, stroke_width=0, direction=None)
    assert len(lines) == 2
    assert lines == ["WHAT ARE", "YOU DOING?"]


def test_comic_detector_postprocessing_and_manga_reading_order():
    from app.pipeline.comic_detector import postprocess_comic_outputs

    # Synthetic 800x600 image letterboxed at scale=1.28 into 1024x1024 (new_w=1024, new_h=768)
    scale = 1024.0 / 800.0
    new_w, new_h = 1024, 768

    blk_raw = np.zeros((1, 4, 7), dtype=np.float32)
    # Box 0: right-side vertical Japanese bubble (orig ~ 560..680, 100..340)
    blk_raw[0, 0] = [620 * scale, 220 * scale, 120 * scale, 240 * scale, 0.92, 0.05, 0.95]
    # Box 1: nested duplicate inside Box 0 (should be absorbed!)
    blk_raw[0, 1] = [630 * scale, 210 * scale, 70 * scale, 180 * scale, 0.80, 0.05, 0.90]
    # Box 2: left-side vertical bubble (orig ~ 100..220, 110..330)
    blk_raw[0, 2] = [160 * scale, 220 * scale, 120 * scale, 220 * scale, 0.88, 0.05, 0.92]
    # Box 3: false positive with no seg/det probability (should be filtered out!)
    blk_raw[0, 3] = [400 * scale, 500 * scale, 100 * scale, 80 * scale, 0.70, 0.90, 0.05]

    seg_raw = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
    det_raw = np.zeros((1, 2, 1024, 1024), dtype=np.float32)

    # Paint vertical column probability inside both real bubbles
    seg_raw[0, 0, int(120 * scale):int(320 * scale), int(580 * scale):int(660 * scale)] = 0.9
    det_raw[0, 0, int(130 * scale):int(310 * scale), int(625 * scale):int(655 * scale)] = 0.9
    det_raw[0, 0, int(140 * scale):int(290 * scale), int(585 * scale):int(610 * scale)] = 0.9

    seg_raw[0, 0, int(130 * scale):int(310 * scale), int(120 * scale):int(200 * scale)] = 0.9
    det_raw[0, 0, int(135 * scale):int(305 * scale), int(145 * scale):int(175 * scale)] = 0.9

    regions = postprocess_comic_outputs(
        blk_raw=blk_raw,
        seg_raw=seg_raw,
        det_raw=det_raw,
        orig_shape=(600, 800),
        scale=scale,
        new_w=new_w,
        new_h=new_h,
        conf_thresh=0.25,
        source_lang="ja",
    )

    assert len(regions) == 2
    # Manga right-to-left reading order: right bubble MUST come first!
    assert regions[0].bbox[0] > regions[1].bbox[0]
    assert regions[0].is_vertical is True
    assert regions[0].line_count == 2
    assert regions[0].seg_mask is not None
    assert int(np.count_nonzero(regions[0].seg_mask)) > 1000
    # Left bubble's mask must NOT bleed into right bubble's mask
    assert int(np.count_nonzero(regions[0].seg_mask[:, :300])) == 0


def test_manga_ocr_preprocessing_and_token_decoding():
    from app.pipeline.manga_ocr import decode_manga_tokens, preprocess_manga_crop

    crop = np.full((80, 50, 3), 255, dtype=np.uint8)
    tensor = preprocess_manga_crop(crop)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= -1.01 and float(tensor.max()) <= 1.01

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "何", "し", "##て", "る", "の", "?", "!", "."]
    decoded = decode_manga_tokens([2, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 3], vocab)
    assert decoded == "何してるの?!…"


def test_end_to_end_japanese_manga_page_onnx():
    from app.pipeline.comic_detector import comic_detector
    from app.pipeline.manga_ocr import manga_ocr

    if not (comic_detector.is_downloaded() and manga_ocr.is_downloaded()):
        return

    # Build a synthetic manga page with vertical Japanese bubble, horizontal English bubble & untouched art
    img = Image.new("RGB", (800, 600), (225, 225, 225))
    draw = ImageDraw.Draw(img)
    # Untouched panel artwork in bottom center
    draw.rectangle((310, 380, 490, 560), fill=(65, 95, 140), outline=(20, 20, 20), width=4)

    # Vertical Japanese speech bubble (right)
    draw.ellipse((490, 55, 750, 430), fill=(255, 255, 255), outline=(15, 15, 15), width=5)
    jp_font = load_font("何してるの", "ja", size=38, bold=True)
    for i, ch in enumerate("何してる"):
        draw.text((645, 110 + i * 52), ch, font=jp_font, fill=(12, 12, 12))
    for i, ch in enumerate("の？！"):
        draw.text((580, 125 + i * 52), ch, font=jp_font, fill=(12, 12, 12))

    # Horizontal English speech bubble (left)
    draw.rounded_rectangle((55, 115, 365, 315), radius=45, fill=(255, 255, 255), outline=(15, 15, 15), width=5)
    en_font = load_font("WAIT A SECOND!", "en", size=34, bold=True)
    draw.text((100, 170), "WAIT A", font=en_font, fill=(12, 12, 12))
    draw.text((100, 220), "SECOND!", font=en_font, fill=(12, 12, 12))

    manga_rgb = np.array(img, dtype=np.uint8)

    with patch("app.services.pipeline.translator_registry.get", return_value=MockTranslator()):
        translated, _, _, blocks = process_image_pipeline(
            image_rgb=manga_rgb,
            source_lang="auto",
            target_lang="en",
            translator_id="mock",
            use_lama=False,
            use_comic_detector=True,
        )

    assert len(blocks) == 2
    jp_blk = next((b for b in blocks if b.is_vertical), None)
    assert jp_blk is not None
    assert "何してるの" in jp_blk.source_text
    assert jp_blk.source_lang == "ja"
    assert jp_blk.style.comic_style is True
    assert jp_blk.style.uniform_bg is True

    # Untouched panel artwork must remain 100% bit-for-bit identical!
    np.testing.assert_array_equal(translated[400:540, 330:470], manga_rgb[400:540, 330:470])


def test_parallel_batch_item_execution():
    import threading
    import time
    from app.main import app

    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def slow_fake_pipeline(image_rgb, source_lang, target_lang, translator_id="google", use_lama=True, on_stage=None):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        try:
            time.sleep(0.25)
            return image_rgb.copy(), image_rgb.copy(), {}, []
        finally:
            with lock:
                in_flight -= 1

    img_buf = io.BytesIO()
    Image.new("RGB", (64, 64), (100, 120, 140)).save(img_buf, format="PNG")
    png_bytes = img_buf.getvalue()

    with patch("app.services.jobs.process_image_pipeline", side_effect=slow_fake_pipeline):
        with TestClient(app) as client:
            t0 = time.perf_counter()
            resp = client.post(
                "/api/jobs",
                data={"target_lang": "en", "source_lang": "auto", "translator": "google"},
                files=[("files", (f"page_{i}.png", png_bytes, "image/png")) for i in range(4)],
            )
            assert resp.status_code == 200
            job_id = resp.json()["job_id"]

            job = None
            for _ in range(50):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            elapsed = time.perf_counter() - t0

            assert job is not None and job["status"] == "completed"
            assert job["completed_items"] == 4
            # 4 items x 0.25s must execute concurrently (peak >= 2, total well below sequential 1.0s)
            assert max_in_flight >= 2
            assert elapsed < 0.85
            client.delete(f"/api/jobs/{job_id}")


def test_western_comic_does_not_hallucinate_japanese_manga_ocr():
    """On an English comic page with source_lang='auto', empty artwork crops must NOT invoke MangaOCR."""
    from app.pipeline.comic_detector import ComicRegion

    img = Image.new("RGB", (600, 400), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    f = load_font("HELLO", "en", size=28, bold=True)
    draw.rectangle((40, 40, 260, 120), fill=(255, 255, 255), outline=(10, 10, 10), width=3)
    draw.text((60, 62), "WAIT HERE!", font=f, fill=(10, 10, 10))
    draw.rectangle((320, 40, 560, 120), fill=(255, 255, 255), outline=(10, 10, 10), width=3)
    draw.text((340, 62), "LET US GO!", font=f, fill=(10, 10, 10))
    # Untouched artwork crop with no real text
    draw.rectangle((180, 200, 420, 340), fill=(90, 90, 90), outline=(10, 10, 10), width=3)
    rgb = np.array(img, dtype=np.uint8)

    fake_regions = [
        ComicRegion(bbox=(45, 45, 255, 115), polygon=[[45, 45], [255, 45], [255, 115], [45, 115]], confidence=0.9),
        ComicRegion(bbox=(325, 45, 555, 115), polygon=[[325, 45], [555, 45], [555, 115], [325, 115]], confidence=0.9),
        ComicRegion(bbox=(190, 210, 280, 330), polygon=[[190, 210], [280, 210], [280, 330], [190, 330]], confidence=0.6, is_vertical=True),
    ]

    with patch("app.services.pipeline.comic_detector.detect", return_value=fake_regions), patch(
        "app.services.pipeline.manga_ocr.recognize", side_effect=AssertionError("MangaOCR must not run on English comic!")
    ), patch("app.services.pipeline.translator_registry.get", return_value=MockTranslator()):
        translated, _, _, blocks = process_image_pipeline(
            image_rgb=rgb,
            source_lang="auto",
            target_lang="am",
            translator_id="mock",
            use_lama=False,
            use_comic_detector=True,
        )

    assert len(blocks) == 2
    assert all(detect_script(b.source_text) == "latin" for b in blocks)
    np.testing.assert_array_equal(translated[215:325, 195:275], rgb[215:325, 195:275])
