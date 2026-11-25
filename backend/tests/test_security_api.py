import asyncio
import io
import re
import stat
import struct
import time
import zipfile
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr, ValidationError

from app.config import Settings, settings
from app.main import app, rate_limiter
from app.schemas import ItemStatus, JobStatus, TextBlock, TextStyle
from app.services.jobs import MAX_SSE_SUBSCRIBERS_PER_JOB, job_manager, sanitize_error_message
from app.services.storage import (
    MAX_BATCH_PIXELS,
    MAX_IMAGE_DIMENSION,
    MAX_ZIP_ENTRIES,
    JobStorage,
    build_content_disposition,
    extract_uploaded_images,
    sanitize_filename,
    storage,
)


def _make_png_bytes(width: int = 32, height: int = 32, color=(60, 120, 180)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _make_gif_bytes(width: int = 32, height: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("P", (width, height), 1).save(buf, format="GIF")
    return buf.getvalue()


def _make_spoofed_header_zip(real_payload: bytes, filename: str = "bomb.png") -> bytes:
    """Create a valid deflated ZIP whose local & central headers lie about uncompressed size (set to 1)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr(filename, real_payload)
    raw = bytearray(buf.getvalue())
    real_size_le = struct.pack("<I", len(real_payload))
    spoof_size_le = struct.pack("<I", 1)

    # Local file header: signature PK\x03\x04, uncompressed size at offset +22
    lfh_idx = raw.find(b"PK\x03\x04")
    assert lfh_idx != -1
    assert raw[lfh_idx + 22 : lfh_idx + 26] == real_size_le
    raw[lfh_idx + 22 : lfh_idx + 26] = spoof_size_le

    # Central directory header: signature PK\x01\x02, uncompressed size at offset +24
    cdh_idx = raw.find(b"PK\x01\x02")
    assert cdh_idx != -1
    assert raw[cdh_idx + 24 : cdh_idx + 28] == real_size_le
    raw[cdh_idx + 24 : cdh_idx + 28] = spoof_size_le
    return bytes(raw)


def _make_fake_huge_png_header(width: int, height: int) -> bytes:
    """Craft a tiny PNG file whose IHDR header advertises huge dimensions (pixel bomb)."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return sig + ihdr + iend


def _noop_pipeline(image_rgb, source_lang, target_lang, translator_id="google", use_lama=True, on_stage=None):
    return image_rgb.copy(), image_rgb.copy(), {}, []


@pytest.fixture(autouse=True)
def _reset_rate_limiter_and_auth():
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def test_path_traversal_rejected_and_storage_containment(tmp_path: Path):
    isolated_storage = JobStorage(root=tmp_path / "jobs")
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("keep-me", encoding="utf-8")

    # Direct storage validation rejects traversal & non-32-hex IDs
    for bad_jid in ("..", "../sentinel.txt", ".", "abc123", "A" * 32, "../jobs", "a" * 31):
        with pytest.raises(ValueError):
            isolated_storage.job_dir(bad_jid)
        with pytest.raises(ValueError):
            isolated_storage.delete_job(bad_jid)

    valid_jid = "a" * 32
    for bad_item in ("..", "../other", "img_1", "img_001/../../x", "bad"):
        with pytest.raises(ValueError):
            isolated_storage.item_dir(valid_jid, bad_item)

    # Read-only item_dir(create=False) must NOT create directories on disk
    ghost_dir = isolated_storage.item_dir(valid_jid, "img_001", create=False)
    assert not ghost_dir.exists()
    assert not isolated_storage.job_dir(valid_jid).exists()

    # HTTP routes reject traversal payloads (%2e%2e, %2e, .., short IDs) and unknown valid IDs return 404
    with TestClient(app) as client:
        for bad_path in (
            "/api/jobs/%2e%2e",
            "/api/jobs/%2e",
            "/api/jobs/not-a-valid-id",
            "/api/jobs/0123456789ab",  # old 12-char format rejected
        ):
            assert client.delete(bad_path).status_code in (400, 404, 422)
            assert client.get(bad_path).status_code in (400, 404, 422)

        unknown_valid = "0" * 32
        with patch.object(storage, "delete_job", wraps=storage.delete_job) as spy_del:
            resp = client.delete(f"/api/jobs/{unknown_valid}")
            assert resp.status_code == 404
            spy_del.assert_not_called()

    assert sentinel.read_text(encoding="utf-8") == "keep-me"


def test_csprng_32_char_job_ids_and_security_headers():
    png = _make_png_bytes()
    with patch("app.services.jobs.process_image_pipeline", side_effect=_noop_pipeline):
        with TestClient(app) as client:
            r1 = client.post(
                "/api/jobs",
                data={"target_lang": "es", "source_lang": "en", "translator": "google"},
                files=[("files", ("a.png", png, "image/png"))],
            )
            r2 = client.post(
                "/api/jobs",
                data={"target_lang": "es", "source_lang": "en", "translator": "google"},
                files=[("files", ("b.png", png, "image/png"))],
            )
            assert r1.status_code == 200
            assert r2.status_code == 200

            jid1 = r1.json()["job_id"]
            jid2 = r2.json()["job_id"]
            assert re.fullmatch(r"^[a-f0-9]{32}$", jid1)
            assert re.fullmatch(r"^[a-f0-9]{32}$", jid2)
            assert jid1 != jid2

            # Security headers + Cache-Control: no-store on /api/jobs*
            for resp in (r1, client.get(f"/api/jobs/{jid1}"), client.get(f"/api/jobs/{jid1}/items/img_001/original")):
                assert resp.headers.get("x-content-type-options") == "nosniff"
                assert resp.headers.get("x-frame-options") == "DENY"
                assert resp.headers.get("referrer-policy") == "no-referrer"
                assert "camera=()" in resp.headers.get("permissions-policy", "")
                assert "frame-ancestors 'none'" in resp.headers.get("content-security-policy", "")
                assert resp.headers.get("cache-control") == "no-store"

            # OpenAPI docs disabled by default
            assert client.get("/docs").status_code == 404
            assert client.get("/openapi.json").status_code == 404

            # CORS must not set Access-Control-Allow-Credentials: true
            cors_resp = client.options(
                "/api/jobs",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert cors_resp.headers.get("access-control-allow-credentials") != "true"

            # /api/status must not leak device or model file details
            status_json = client.get("/api/status").json()
            assert "translators" in status_json
            assert "max_upload_mb" in status_json
            assert "max_batch_images" in status_json
            assert "device" not in status_json

            client.delete(f"/api/jobs/{jid1}")
            client.delete(f"/api/jobs/{jid2}")


def test_optional_api_key_authentication():
    png = _make_png_bytes()
    with patch.object(settings, "api_key", SecretStr("secret-api-key-123")), patch(
        "app.services.jobs.process_image_pipeline", side_effect=_noop_pipeline
    ):
        with TestClient(app) as client:
            # Public endpoints stay accessible
            assert client.get("/api/health").status_code == 200
            assert client.get("/api/languages").status_code == 200
            assert client.get("/api/status").status_code == 200

            # Missing or wrong key -> 401
            unauth = client.post(
                "/api/jobs",
                data={"target_lang": "es", "source_lang": "en", "translator": "google"},
                files=[("files", ("a.png", png, "image/png"))],
            )
            assert unauth.status_code == 401

            wrong = client.post(
                "/api/jobs",
                headers={"Authorization": "Bearer wrong-key"},
                data={"target_lang": "es", "source_lang": "en", "translator": "google"},
                files=[("files", ("a.png", png, "image/png"))],
            )
            assert wrong.status_code == 401

            # Valid Bearer token -> 200
            ok = client.post(
                "/api/jobs",
                headers={"Authorization": "Bearer secret-api-key-123"},
                data={"target_lang": "es", "source_lang": "en", "translator": "google"},
                files=[("files", ("a.png", png, "image/png"))],
            )
            assert ok.status_code == 200
            jid = ok.json()["job_id"]

            # Query param ?api_key= rejected on non-SSE routes, allowed on /events
            assert client.get(f"/api/jobs/{jid}?api_key=secret-api-key-123").status_code == 401
            assert client.get(f"/api/jobs/{jid}", headers={"X-API-Key": "secret-api-key-123"}).status_code == 200

            # Wait briefly for completion so SSE closes immediately with done
            for _ in range(30):
                st = client.get(f"/api/jobs/{jid}", headers={"X-API-Key": "secret-api-key-123"}).json()["status"]
                if st in ("completed", "failed"):
                    break
                time.sleep(0.05)

            sse_ok = client.get(f"/api/jobs/{jid}/events?api_key=secret-api-key-123")
            assert sse_ok.status_code == 200
            assert "event: done" in sse_ok.text

            assert client.delete(f"/api/jobs/{jid}", headers={"X-API-Key": "secret-api-key-123"}).status_code == 200


def test_rate_limiting_returns_429_with_retry_after():
    png = _make_png_bytes()
    with patch.object(settings, "rate_limit_jobs_per_minute", 2), patch(
        "app.services.jobs.process_image_pipeline", side_effect=_noop_pipeline
    ):
        with TestClient(app) as client:
            created = []
            for i in range(2):
                r = client.post(
                    "/api/jobs",
                    data={"target_lang": "en", "source_lang": "auto", "translator": "google"},
                    files=[("files", (f"p{i}.png", png, "image/png"))],
                )
                assert r.status_code == 200
                created.append(r.json()["job_id"])

            r3 = client.post(
                "/api/jobs",
                data={"target_lang": "en", "source_lang": "auto", "translator": "google"},
                files=[("files", ("p2.png", png, "image/png"))],
            )
            assert r3.status_code == 429
            assert int(r3.headers.get("retry-after", "0")) >= 1

            for jid in created:
                client.delete(f"/api/jobs/{jid}")


def test_chunked_upload_size_limit_and_file_count_cap():
    png = _make_png_bytes()
    with patch.object(settings, "max_upload_mb", 1), patch.object(settings, "max_batch_images", 2):
        with TestClient(app) as client:
            # Exceeds max_batch_images (3 > 2) -> 413
            r_count = client.post(
                "/api/jobs",
                data={"target_lang": "en", "source_lang": "auto", "translator": "google"},
                files=[("files", (f"img_{i}.png", png, "image/png")) for i in range(3)],
            )
            assert r_count.status_code == 413

            # Exceeds max_upload_mb (1.2 MB > 1 MB) -> 413 via chunked reader
            oversized = b"\x89PNG\r\n\x1a\n" + (b"\x00" * (1024 * 1024 + 200_000))
            r_size = client.post(
                "/api/jobs",
                data={"target_lang": "en", "source_lang": "auto", "translator": "google"},
                files=[("files", ("big.png", oversized, "image/png"))],
            )
            assert r_size.status_code == 413


def test_zip_bomb_spoofed_header_cumulative_multi_zip_and_corrupt_zip():
    # 1a. High-compression ZIP bomb (2 KB compressed -> 2 MB uncompressed > 512 KB cap)
    bomb_buf = io.BytesIO()
    with zipfile.ZipFile(bomb_buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("bomb.png", b"A" * (2 * 1024 * 1024))
    with pytest.raises(ValueError, match="uncompressed"):
        extract_uploaded_images([("bomb.zip", bomb_buf.getvalue())], max_uncompressed_bytes=512 * 1024)

    # 1b. Header-spoofed ZIP bomb (header lies about uncompressed size = 1 byte) must be rejected
    spoofed_zip = _make_spoofed_header_zip(b"A" * (2 * 1024 * 1024), filename="bomb.png")
    with pytest.raises(ValueError):
        extract_uploaded_images([("bomb.zip", spoofed_zip)], max_uncompressed_bytes=512 * 1024)

    # 1c. Streaming chunk guard catches entries whose actual stream exceeds declared file_size
    class _LyingStream(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    small_buf = io.BytesIO()
    with zipfile.ZipFile(small_buf, "w") as zf:
        zf.writestr("lying.png", b"tiny")
    with patch.object(zipfile.ZipFile, "open", return_value=_LyingStream(b"Z" * (600 * 1024))):
        with pytest.raises(ValueError, match="uncompressed"):
            extract_uploaded_images([("lying.zip", small_buf.getvalue())], max_uncompressed_bytes=512 * 1024)

    # 2. Cumulative multi-ZIP bomb: each ZIP is 300 KB (< 500 KB cap), but together 600 KB (> 500 KB cap)
    def _make_zip(payload: bytes, name: str) -> bytes:
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(name, payload)
        return b.getvalue()

    z1 = _make_zip(b"B" * (300 * 1024), "a.png")
    z2 = _make_zip(b"C" * (300 * 1024), "b.png")
    with pytest.raises(ValueError, match="uncompressed"):
        extract_uploaded_images(
            [("part1.zip", z1), ("part2.zip", z2)],
            max_uncompressed_bytes=500 * 1024,
        )

    # 3. Symlink entry inside ZIP must be skipped
    sym_buf = io.BytesIO()
    with zipfile.ZipFile(sym_buf, "w") as zf:
        zi = zipfile.ZipInfo("link.png")
        zi.create_system = 3  # Unix
        zi.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(zi, "/etc/passwd")
        zf.writestr("real.png", _make_png_bytes())
    extracted = extract_uploaded_images([("sym.zip", sym_buf.getvalue())])
    assert [name for name, _ in extracted] == ["real.png"]

    # 4. Entry-count bomb (> MAX_ZIP_ENTRIES) rejected before sorting
    many_buf = io.BytesIO()
    with zipfile.ZipFile(many_buf, "w") as zf:
        for i in range(MAX_ZIP_ENTRIES + 5):
            zf.writestr(f"e_{i}.txt", b"x")
    with pytest.raises(ValueError, match="too many entries"):
        extract_uploaded_images([("many.zip", many_buf.getvalue())])

    # 5. Corrupt/bad ZIP over HTTP -> 400 (NOT 500!)
    with TestClient(app) as client:
        r_bad = client.post(
            "/api/jobs",
            data={"target_lang": "en", "source_lang": "auto", "translator": "google"},
            files=[("files", ("corrupt.zip", b"not-a-valid-zip-archive", "application/zip"))],
        )
        assert r_bad.status_code == 400


def test_pixel_bomb_and_disallowed_pillow_format_rejected():
    # 1. Dimension bomb (> MAX_IMAGE_DIMENSION)
    dim_bomb = _make_fake_huge_png_header(MAX_IMAGE_DIMENSION + 100, 100)
    # 2. Total pixel bomb (6000 x 6000 = 36 MP > 25 MP)
    px_bomb = _make_fake_huge_png_header(6000, 6000)
    # 3. Disallowed format (GIF disguised as .png and as .gif)
    gif_bytes = _make_gif_bytes()

    with TestClient(app) as client:
        for fname, payload in (
            ("dim_bomb.png", dim_bomb),
            ("px_bomb.png", px_bomb),
            ("anim.gif", gif_bytes),
            ("disguised.png", gif_bytes),
        ):
            r = client.post(
                "/api/jobs",
                data={"target_lang": "en", "source_lang": "auto", "translator": "google"},
                files=[("files", (fname, payload, "image/png"))],
            )
            assert r.status_code == 400, f"Expected 400 for {fname}, got {r.status_code}"


def test_rerender_bounds_422_and_precompletion_conflict_409():
    png = _make_png_bytes()
    gate = asyncio.Event()

    def slow_pipeline(image_rgb, source_lang, target_lang, translator_id="google", use_lama=True, on_stage=None):
        time.sleep(0.5)
        return image_rgb.copy(), image_rgb.copy(), {}, []

    with patch("app.services.jobs.process_image_pipeline", side_effect=slow_pipeline):
        with TestClient(app) as client:
            resp = client.post(
                "/api/jobs",
                data={"target_lang": "en", "source_lang": "auto", "translator": "google"},
                files=[("files", ("a.png", png, "image/png"))],
            )
            assert resp.status_code == 200
            jid = resp.json()["job_id"]

            # Pre-completion rerender -> 409 Conflict
            conflict = client.post(
                f"/api/jobs/{jid}/items/img_001/rerender",
                json={"blocks": [{"id": "b1", "translated_text": "hi"}]},
            )
            assert conflict.status_code == 409

            # Bounds: >250 blocks -> 422
            too_many = client.post(
                f"/api/jobs/{jid}/items/img_001/rerender",
                json={"blocks": [{"id": f"b{i}", "translated_text": "x"} for i in range(251)]},
            )
            assert too_many.status_code == 422

            # Bounds: >2000-char translated_text -> 422
            too_long = client.post(
                f"/api/jobs/{jid}/items/img_001/rerender",
                json={"blocks": [{"id": "b1", "translated_text": "x" * 2001}]},
            )
            assert too_long.status_code == 422

            # Bounds: >64-char block id -> 422
            long_id = client.post(
                f"/api/jobs/{jid}/items/img_001/rerender",
                json={"blocks": [{"id": "b" * 65, "translated_text": "ok"}]},
            )
            assert long_id.status_code == 422

            client.delete(f"/api/jobs/{jid}")


def test_non_ascii_content_disposition_and_filename_sanitization():
    # BiDi override + control chars + excessive length stripped safely
    bidi_name = f"\u202e\x00..hidden_{'a' * 150}.png"
    sanitized = sanitize_filename(bidi_name)
    assert "\u202e" not in sanitized
    assert "\x00" not in sanitized
    assert not sanitized.startswith(".")
    assert len(sanitized) <= 120
    assert sanitized.endswith(".png")

    for unicode_name in ("translated_漫画ページ.png", "translated_ሰላም_ምስል.png"):
        cd = build_content_disposition(unicode_name)
        # Must be pure ASCII / latin-1 encodable (prevents Starlette UnicodeEncodeError)
        cd.encode("latin-1")
        assert "filename=" in cd
        assert "filename*=UTF-8''" in cd

    # End-to-end download with Japanese and Amharic filenames returns 200 with RFC 5987 header
    png = _make_png_bytes()
    with patch("app.services.jobs.process_image_pipeline", side_effect=_noop_pipeline):
        with TestClient(app) as client:
            resp = client.post(
                "/api/jobs",
                data={"target_lang": "en", "source_lang": "auto", "translator": "google"},
                files=[
                    ("files", ("漫画ページ.png", png, "image/png")),
                    ("files", ("ሰላም.png", png, "image/png")),
                ],
            )
            assert resp.status_code == 200
            jid = resp.json()["job_id"]

            for _ in range(50):
                j = client.get(f"/api/jobs/{jid}").json()
                if j["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)

            for item_id in ("img_001", "img_002"):
                dl = client.get(f"/api/jobs/{jid}/items/{item_id}/translated?download=true")
                assert dl.status_code == 200
                cd = dl.headers.get("content-disposition", "")
                assert "filename*=UTF-8''" in cd
                assert dl.headers.get("cache-control") == "no-store"

            zip_resp = client.get(f"/api/jobs/{jid}/download")
            assert zip_resp.status_code == 200
            assert f"photo-translate-{jid}.zip" in zip_resp.headers.get("content-disposition", "")
            assert zip_resp.headers.get("cache-control") == "no-store"

            client.delete(f"/api/jobs/{jid}")


def test_invalid_form_params_rejected_synchronously():
    png = _make_png_bytes()
    with TestClient(app) as client:
        # Invalid target_lang
        r1 = client.post(
            "/api/jobs",
            data={"target_lang": "xx_invalid", "source_lang": "auto", "translator": "google"},
            files=[("files", ("a.png", png, "image/png"))],
        )
        assert r1.status_code == 400

        # Invalid source_lang
        r2 = client.post(
            "/api/jobs",
            data={"target_lang": "en", "source_lang": "xx_invalid", "translator": "google"},
            files=[("files", ("a.png", png, "image/png"))],
        )
        assert r2.status_code == 400

        # Unknown translator
        r3 = client.post(
            "/api/jobs",
            data={"target_lang": "en", "source_lang": "auto", "translator": "nonexistent_engine"},
            files=[("files", ("a.png", png, "image/png"))],
        )
        assert r3.status_code == 400

        # Unavailable translator (deepl with blank key)
        with patch.object(settings, "deepl_api_key", SecretStr("")):
            r4 = client.post(
                "/api/jobs",
                data={"target_lang": "en", "source_lang": "auto", "translator": "deepl"},
                files=[("files", ("a.png", png, "image/png"))],
            )
            assert r4.status_code == 400


def test_sse_cleanup_on_delete_job_and_subscriber_cap():
    async def _run():
        from app.schemas import JobInfo

        jid = "b" * 32
        now = time.time()
        job = JobInfo(
            job_id=jid,
            status=JobStatus.PROCESSING,
            created_at=now,
            expires_at=now + 300,
            total_items=1,
            completed_items=0,
            items=[],
        )
        job_manager._jobs[jid] = job
        try:
            # Verify per-job subscriber cap
            queues = [job_manager.reserve_subscriber(jid) for _ in range(MAX_SSE_SUBSCRIBERS_PER_JOB)]
            with pytest.raises(Exception) as exc_info:
                job_manager.reserve_subscriber(jid)
            assert getattr(exc_info.value, "status_code", None) == 429

            # Consume first subscriber asynchronously and trigger delete_job
            events = []

            async def _consume(q):
                async for chunk in job_manager.subscribe(jid, queue=q):
                    events.append(chunk)

            consumer = asyncio.create_task(_consume(queues[0]))
            await asyncio.sleep(0.05)

            assert job_manager.delete_job(jid) is True
            # Generator must receive done sentinel and exit cleanly (no hang!)
            await asyncio.wait_for(consumer, timeout=2.0)
            assert any("event: done" in e for e in events)
            assert jid not in job_manager._subscribers
        finally:
            job_manager._jobs.pop(jid, None)
            job_manager._subscribers.pop(jid, None)

    asyncio.run(_run())


def test_settings_url_validation_and_secret_masking():
    # Reject non-HTTPS, loopback, private, and cloud metadata URLs
    for bad_url in (
        "http://api.deepseek.com/chat/completions",
        "https://127.0.0.1/v1",
        "https://169.254.169.254/latest/meta-data/",
        "https://10.0.0.1/api",
        "https://localhost/api",
        "https://user:pass@api.deepseek.com/v1",
    ):
        with pytest.raises(ValidationError):
            Settings(deepseek_api_url=bad_url)

    # SecretStr masks secrets in repr/str
    s = Settings(api_key=SecretStr("top-secret-token-xyz"))
    assert "top-secret-token-xyz" not in repr(s)
    assert s.get_secret("api_key") == "top-secret-token-xyz"

    # Error sanitizer strips URLs and filesystem paths
    msg = sanitize_error_message("Error at /home/user/secret/file.py calling https://internal.example/key?k=1\nboom")
    assert "/home/user" not in msg
    assert "https://internal.example" not in msg
    assert "\n" not in msg


def test_sanitized_pipeline_error_and_active_job_capacity_cap():
    png = _make_png_bytes()

    def crashing_pipeline(*args, **kwargs):
        raise RuntimeError("Secret internal failure at /home/biniyam/secret.onnx https://api.internal/v1?key=abc")

    with patch("app.services.jobs.process_image_pipeline", side_effect=crashing_pipeline):
        with TestClient(app) as client:
            r = client.post(
                "/api/jobs",
                data={"target_lang": "en", "source_lang": "auto", "translator": "google"},
                files=[("files", ("crash.png", png, "image/png"))],
            )
            assert r.status_code == 200
            jid = r.json()["job_id"]

            job = None
            for _ in range(50):
                job = client.get(f"/api/jobs/{jid}").json()
                if job["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)

            assert job is not None and job["status"] == "failed"
            err_msg = job["items"][0]["error"] or ""
            assert err_msg == "Failed to process image."
            assert "/home/biniyam" not in err_msg
            assert "https://api.internal" not in err_msg

            # Active job capacity cap (max_active_jobs=1 while 1 job exists) -> 503
            with patch.object(settings, "max_active_jobs", 1):
                r_cap = client.post(
                    "/api/jobs",
                    data={"target_lang": "en", "source_lang": "auto", "translator": "google"},
                    files=[("files", ("second.png", png, "image/png"))],
                )
                assert r_cap.status_code == 503

            client.delete(f"/api/jobs/{jid}")

