import functools
import logging
import struct
import subprocess
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

from PIL import ImageFont, features

from app.config import settings
from app.languages import detect_script, get_language

logger = logging.getLogger(__name__)

COMIC_FONT_URLS = {
    "ComicNeue-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Bold.ttf",
    "ComicNeue-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Regular.ttf",
}
_COMIC_FONT_DOWNLOAD_ATTEMPTED: set[str] = set()

# Preferred font families per Unicode script (regular, bold)
SCRIPT_FONT_FAMILIES: Dict[str, List[str]] = {
    "latin": ["Noto Sans", "Liberation Sans", "DejaVu Sans"],
    "latin_meme": ["Impact", "Liberation Sans Narrow", "Noto Sans", "DejaVu Sans"],
    "cyrillic": ["Noto Sans", "DejaVu Sans", "Liberation Sans"],
    "greek": ["Noto Sans", "DejaVu Sans", "Liberation Sans"],
    "ethiopic": ["Noto Sans Ethiopic", "Abyssinica SIL", "FreeSerif"],
    "arabic": ["Noto Sans Arabic", "Noto Kufi Arabic", "Noto Naskh Arabic", "DejaVu Sans"],
    "hebrew": ["Noto Sans Hebrew", "Noto Rashi Hebrew", "DejaVu Sans"],
    "cjk_sc": ["Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Zen Hei"],
    "cjk_tc": ["Noto Sans CJK TC", "Noto Sans TC", "Noto Sans CJK SC"],
    "cjk_jp": ["Noto Sans CJK JP", "Noto Sans JP", "Noto Sans CJK SC"],
    "cjk_kr": ["Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK SC"],
    "devanagari": ["Noto Sans Devanagari", "Lohit Devanagari", "FreeSans"],
    "thai": ["Noto Sans Thai", "Noto Looped Thai", "Garuda"],
}

FALLBACK_FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

FontEntry = Tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, FrozenSet[int]]


def is_raqm_available() -> bool:
    try:
        return bool(features.check("raqm"))
    except Exception:
        return False


def _parse_cmap_from_sfnt(data: bytes, sfnt_offset: int) -> FrozenSet[int]:
    """Extract all Unicode codepoints mapped in a TTF/OTF sfnt `cmap` table (formats 4 and 12)."""
    if sfnt_offset + 12 > len(data):
        return frozenset()
    num_tables = struct.unpack_from(">H", data, sfnt_offset + 4)[0]
    cmap_offset = None
    for i in range(num_tables):
        entry_off = sfnt_offset + 12 + i * 16
        if entry_off + 16 > len(data):
            break
        tag = data[entry_off : entry_off + 4]
        if tag == b"cmap":
            cmap_offset = struct.unpack_from(">I", data, entry_off + 8)[0]
            break

    if cmap_offset is None or cmap_offset + 4 > len(data):
        return frozenset()

    num_subtables = struct.unpack_from(">H", data, cmap_offset + 2)[0]
    codepoints: set[int] = set()

    for i in range(num_subtables):
        rec_off = cmap_offset + 4 + i * 8
        if rec_off + 8 > len(data):
            break
        platform_id, encoding_id, sub_rel = struct.unpack_from(">HHI", data, rec_off)
        # Unicode (0) or Windows Unicode BMP/UCS-4 (3, 1 or 3, 10)
        if not (platform_id == 0 or (platform_id == 3 and encoding_id in (1, 10))):
            continue
        sub_off = cmap_offset + sub_rel
        if sub_off + 2 > len(data):
            continue
        fmt = struct.unpack_from(">H", data, sub_off)[0]

        if fmt == 4 and sub_off + 14 <= len(data):
            seg_count_x2 = struct.unpack_from(">H", data, sub_off + 6)[0]
            seg_count = seg_count_x2 // 2
            end_codes_off = sub_off + 14
            start_codes_off = end_codes_off + seg_count_x2 + 2
            id_delta_off = start_codes_off + seg_count_x2
            id_range_off = id_delta_off + seg_count_x2
            if id_range_off + seg_count_x2 > len(data):
                continue
            for s in range(seg_count):
                end_code = struct.unpack_from(">H", data, end_codes_off + s * 2)[0]
                start_code = struct.unpack_from(">H", data, start_codes_off + s * 2)[0]
                if start_code == 0xFFFF and end_code == 0xFFFF:
                    continue
                id_delta = struct.unpack_from(">h", data, id_delta_off + s * 2)[0]
                range_val = struct.unpack_from(">H", data, id_range_off + s * 2)[0]
                for cp in range(start_code, end_code + 1):
                    if range_val == 0:
                        glyph_id = (cp + id_delta) & 0xFFFF
                    else:
                        idx_off = (id_range_off + s * 2) + range_val + (cp - start_code) * 2
                        if idx_off + 2 > len(data):
                            continue
                        g = struct.unpack_from(">H", data, idx_off)[0]
                        glyph_id = ((g + id_delta) & 0xFFFF) if g != 0 else 0
                    if glyph_id != 0:
                        codepoints.add(cp)

        elif fmt == 12 and sub_off + 16 <= len(data):
            num_groups = struct.unpack_from(">I", data, sub_off + 12)[0]
            grp_off = sub_off + 16
            for g in range(num_groups):
                off = grp_off + g * 12
                if off + 12 > len(data):
                    break
                start_cp, end_cp, start_gid = struct.unpack_from(">III", data, off)
                if start_gid == 0 and start_cp == end_cp:
                    continue
                # Guard against corrupt huge ranges
                if end_cp - start_cp <= 65536:
                    codepoints.update(range(start_cp, end_cp + 1))

    return frozenset(codepoints)


@functools.lru_cache(maxsize=64)
def get_font_codepoints(font_path: str) -> FrozenSet[int]:
    """Return all Unicode codepoints genuinely supported by `font_path` (excluding .notdef)."""
    if not font_path:
        return frozenset()
    try:
        data = Path(font_path).read_bytes()
        if len(data) < 12:
            return frozenset()
        if data[:4] == b"ttcf":
            # TrueType Collection (.ttc): inspect first font face (index 0, matching Pillow default)
            num_fonts = struct.unpack_from(">I", data, 8)[0]
            if num_fonts > 0 and len(data) >= 16:
                first_offset = struct.unpack_from(">I", data, 12)[0]
                return _parse_cmap_from_sfnt(data, first_offset)
            return frozenset()
        return _parse_cmap_from_sfnt(data, 0)
    except Exception as exc:
        logger.debug("Could not parse cmap for %s: %s", font_path, exc)
        return frozenset()


@functools.lru_cache(maxsize=64)
def _fc_match(pattern: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["fc-match", "-f", "%{file}", pattern],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        if out and Path(out).is_file():
            return out
    except Exception:
        pass
    return None


def _resolve_comic_font_path(bold: bool = True) -> Optional[str]:
    """
    Resolve cached open-source comic lettering font (`ComicNeue-Bold.ttf` / `ComicNeue-Regular.ttf`).
    Attempts a one-time silent download into `settings.fonts_dir` if missing; falls back cleanly if offline.
    """
    fname = "ComicNeue-Bold.ttf" if bold else "ComicNeue-Regular.ttf"
    target = settings.fonts_dir / fname
    if target.is_file() and target.stat().st_size > 10_000:
        return str(target)

    if fname not in _COMIC_FONT_DOWNLOAD_ATTEMPTED:
        _COMIC_FONT_DOWNLOAD_ATTEMPTED.add(fname)
        url = COMIC_FONT_URLS.get(fname)
        if url:
            try:
                import httpx

                settings.fonts_dir.mkdir(parents=True, exist_ok=True)
                resp = httpx.get(url, follow_redirects=True, timeout=8.0)
                resp.raise_for_status()
                if len(resp.content) > 10_000:
                    target.write_bytes(resp.content)
                    return str(target)
            except Exception as exc:
                logger.debug("Could not auto-download comic font %s (%s); using Noto Sans.", fname, exc)
    return None


@functools.lru_cache(maxsize=64)
def resolve_font_path(
    script: str,
    bold: bool = False,
    meme_style: bool = False,
    comic_style: bool = False,
) -> str:
    """Resolve the best local TrueType/OpenType font file for a given script and weight."""
    if comic_style and not meme_style and script == "latin":
        comic_path = _resolve_comic_font_path(bold=True)
        if comic_path:
            return comic_path

    key = "latin_meme" if (meme_style and script == "latin") else script
    families = SCRIPT_FONT_FAMILIES.get(key, SCRIPT_FONT_FAMILIES["latin"])
    weight = "Bold" if bold else "Regular"

    for family in families:
        pattern = f"{family}:style={weight}" if bold else f"{family}:style=Regular"
        path = _fc_match(pattern)
        if path:
            # Confirm fc-match didn't silently fall back to an unrelated Latin font for non-Latin scripts
            fname = Path(path).name.lower()
            if script == "ethiopic" and not any(k in fname for k in ("ethiopic", "abyssinica", "nyala", "free")):
                continue
            if script == "arabic" and not any(k in fname for k in ("arabic", "naskh", "kufi", "dejavu", "amiri")):
                continue
            if script.startswith("cjk") and not any(k in fname for k in ("cjk", "noto", "wqy", "han", "source")):
                continue
            if script == "hebrew" and not any(k in fname for k in ("hebrew", "dejavu")):
                continue
            if script == "thai" and not any(k in fname for k in ("thai", "garuda", "loma")):
                continue
            if script == "devanagari" and not any(k in fname for k in ("devanagari", "lohit", "free")):
                continue
            return path

    # Script-agnostic fallback via fc-match language/script query
    if script == "ethiopic":
        p = _fc_match(":lang=am")
        if p:
            return p
    elif script == "arabic":
        p = _fc_match(":lang=ar")
        if p:
            return p
    elif script == "hebrew":
        p = _fc_match(":lang=he")
        if p:
            return p
    elif script == "devanagari":
        p = _fc_match(":lang=hi")
        if p:
            return p
    elif script == "thai":
        p = _fc_match(":lang=th")
        if p:
            return p
    elif script == "cjk_jp":
        p = _fc_match(":lang=ja")
        if p:
            return p
    elif script == "cjk_kr":
        p = _fc_match(":lang=ko")
        if p:
            return p
    elif script.startswith("cjk"):
        p = _fc_match(":lang=zh")
        if p:
            return p

    for candidate in FALLBACK_FONT_PATHS:
        if Path(candidate).is_file():
            return candidate

    return ""


def _open_font(font_path: str, size: int) -> Optional[ImageFont.FreeTypeFont]:
    if not font_path:
        return None
    layout_engine = ImageFont.Layout.RAQM if is_raqm_available() else ImageFont.Layout.BASIC
    try:
        return ImageFont.truetype(font_path, size=max(size, 6), layout_engine=layout_engine)
    except Exception as exc:
        logger.warning("Failed loading font %s: %s", font_path, exc)
        return None


def load_font(
    text: str,
    target_lang: str,
    size: int,
    bold: bool = False,
    meme_style: bool = False,
    comic_style: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    lang_info = get_language(target_lang)
    detected_script = detect_script(text)
    script = detected_script if detected_script != "latin" else lang_info.script
    font_path = resolve_font_path(script, bold=bold, meme_style=meme_style, comic_style=comic_style)
    loaded = _open_font(font_path, size)
    return loaded if loaded is not None else ImageFont.load_default()


def load_font_chain(
    text: str,
    target_lang: str,
    size: int,
    bold: bool = False,
    meme_style: bool = False,
    comic_style: bool = False,
) -> List[FontEntry]:
    """
    Build an ordered font fallback chain (primary script font -> matching-weight Latin/punctuation
    Noto Sans -> broad DejaVu fallback) so ASCII punctuation (`?`, `!`, digits, Latin names)
    never renders as missing glyphs when using script-subset fonts like NotoSansEthiopic.
    """
    lang_info = get_language(target_lang)
    detected_script = detect_script(text)
    script = detected_script if detected_script != "latin" else lang_info.script

    candidate_paths: List[str] = []
    primary = resolve_font_path(script, bold=bold, meme_style=meme_style, comic_style=comic_style)
    if primary:
        candidate_paths.append(primary)

    # Always append matching-weight Latin/punctuation Noto Sans fallback if primary is a non-Latin subset
    latin_fallback = resolve_font_path("latin", bold=bold, meme_style=meme_style, comic_style=False)
    if latin_fallback and latin_fallback not in candidate_paths:
        candidate_paths.append(latin_fallback)

    # Broad Unicode symbol fallback (DejaVu Sans)
    dejavu = _fc_match("DejaVu Sans:style=Bold" if bold else "DejaVu Sans:style=Regular")
    if dejavu and dejavu not in candidate_paths:
        candidate_paths.append(dejavu)

    chain: List[FontEntry] = []
    for path in candidate_paths:
        f = _open_font(path, size)
        if f is not None:
            chain.append((f, get_font_codepoints(path)))

    if not chain:
        chain.append((ImageFont.load_default(), frozenset()))
    return chain


def split_font_runs(
    text: str,
    font_chain: List[FontEntry],
) -> List[Tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]]:
    """
    Split `text` into minimal contiguous runs where each run uses the first font in
    `font_chain` that actually contains its glyphs.
    """
    if not text or len(font_chain) <= 1:
        return [(text, font_chain[0][0])] if text else []

    runs: List[Tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]] = []
    cur_font: Optional[ImageFont.FreeTypeFont | ImageFont.ImageFont] = None
    cur_chars: List[str] = []

    for ch in text:
        cp = ord(ch)
        # Keep spaces attached to the current run when supported so kerning/word spacing stays natural
        chosen = None
        if ch.isspace() and cur_font is not None:
            for f, cps in font_chain:
                if f is cur_font and (not cps or cp in cps):
                    chosen = cur_font
                    break

        if chosen is None:
            for f, cps in font_chain:
                if not cps or cp in cps:
                    chosen = f
                    break
            if chosen is None:
                chosen = font_chain[0][0]

        if cur_font is None:
            cur_font = chosen
            cur_chars = [ch]
        elif chosen is cur_font:
            cur_chars.append(ch)
        else:
            runs.append(("".join(cur_chars), cur_font))
            cur_font = chosen
            cur_chars = [ch]

    if cur_chars and cur_font is not None:
        runs.append(("".join(cur_chars), cur_font))
    return runs
