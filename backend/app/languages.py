import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

_LANGDETECT_LOCK = threading.Lock()


@dataclass(frozen=True)
class LanguageInfo:
    code: str           # Short ISO-like code used by API/UI (e.g. "en", "am", "ar", "zh")
    name: str           # English display name
    native_name: str    # Native name
    nllb_code: str      # FLORES-200 / NLLB code
    script: str         # Primary script identifier for font selection
    rtl: bool = False
    deepl_code: Optional[str] = None
    google_code: Optional[str] = None


LANGUAGES: List[LanguageInfo] = [
    LanguageInfo("en", "English", "English", "eng_Latn", "latin", False, "EN", "en"),
    LanguageInfo("am", "Amharic", "አማርኛ", "amh_Ethi", "ethiopic", False, None, "am"),
    LanguageInfo("ti", "Tigrinya", "ትግርኛ", "tir_Ethi", "ethiopic", False, None, "ti"),
    LanguageInfo("om", "Oromo", "Afaan Oromoo", "gaz_Latn", "latin", False, None, "om"),
    LanguageInfo("ar", "Arabic", "العربية", "arb_Arab", "arabic", True, "AR", "ar"),
    LanguageInfo("es", "Spanish", "Español", "spa_Latn", "latin", False, "ES", "es"),
    LanguageInfo("fr", "French", "Français", "fra_Latn", "latin", False, "FR", "fr"),
    LanguageInfo("de", "German", "Deutsch", "deu_Latn", "latin", False, "DE", "de"),
    LanguageInfo("it", "Italian", "Italiano", "ita_Latn", "latin", False, "IT", "it"),
    LanguageInfo("pt", "Portuguese", "Português", "por_Latn", "latin", False, "PT-BR", "pt"),
    LanguageInfo("ru", "Russian", "Русский", "rus_Cyrl", "cyrillic", False, "RU", "ru"),
    LanguageInfo("uk", "Ukrainian", "Українська", "ukr_Cyrl", "cyrillic", False, "UK", "uk"),
    LanguageInfo("zh", "Chinese (Simplified)", "简体中文", "zho_Hans", "cjk_sc", False, "ZH-HANS", "zh-CN"),
    LanguageInfo("zh-TW", "Chinese (Traditional)", "繁體中文", "zho_Hant", "cjk_tc", False, "ZH-HANT", "zh-TW"),
    LanguageInfo("ja", "Japanese", "日本語", "jpn_Jpan", "cjk_jp", False, "JA", "ja"),
    LanguageInfo("ko", "Korean", "한국어", "kor_Hang", "cjk_kr", False, "KO", "ko"),
    LanguageInfo("hi", "Hindi", "हिन्दी", "hin_Deva", "devanagari", False, None, "hi"),
    LanguageInfo("he", "Hebrew", "עברית", "heb_Hebr", "hebrew", True, None, "iw"),
    LanguageInfo("fa", "Persian", "فارسی", "pes_Arab", "arabic", True, None, "fa"),
    LanguageInfo("tr", "Turkish", "Türkçe", "tur_Latn", "latin", False, "TR", "tr"),
    LanguageInfo("nl", "Dutch", "Nederlands", "nld_Latn", "latin", False, "NL", "nl"),
    LanguageInfo("pl", "Polish", "Polski", "pol_Latn", "latin", False, "PL", "pl"),
    LanguageInfo("sv", "Swedish", "Svenska", "swe_Latn", "latin", False, "SV", "sv"),
    LanguageInfo("el", "Greek", "Ελληνικά", "ell_Grek", "greek", False, "EL", "el"),
    LanguageInfo("th", "Thai", "ไทย", "tha_Thai", "thai", False, None, "th"),
    LanguageInfo("vi", "Vietnamese", "Tiếng Việt", "vie_Latn", "latin", False, None, "vi"),
    LanguageInfo("id", "Indonesian", "Bahasa Indonesia", "ind_Latn", "latin", False, "ID", "id"),
    LanguageInfo("sw", "Swahili", "Kiswahili", "swh_Latn", "latin", False, None, "sw"),
]

LANGUAGE_MAP: Dict[str, LanguageInfo] = {lang.code: lang for lang in LANGUAGES}


def get_language(code: str) -> LanguageInfo:
    if code in LANGUAGE_MAP:
        return LANGUAGE_MAP[code]
    normalized = code.split("-")[0].lower()
    if normalized in LANGUAGE_MAP:
        return LANGUAGE_MAP[normalized]
    return LANGUAGE_MAP["en"]


def detect_script(text: str) -> str:
    """Detect dominant Unicode script in text for font selection & OCR/lang hints."""
    counts: Dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if 0x1200 <= cp <= 0x137F or 0x1380 <= cp <= 0x139F or 0x2D80 <= cp <= 0x2DDF:
            script = "ethiopic"
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
            script = "arabic"
        elif 0x0590 <= cp <= 0x05FF:
            script = "hebrew"
        elif 0x0400 <= cp <= 0x04FF:
            script = "cyrillic"
        elif 0x0370 <= cp <= 0x03FF:
            script = "greek"
        elif 0x0900 <= cp <= 0x097F:
            script = "devanagari"
        elif 0x0E00 <= cp <= 0x0E7F:
            script = "thai"
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            script = "cjk_kr"
        elif 0x3040 <= cp <= 0x30FF:
            script = "cjk_jp"
        elif 0x4E00 <= cp <= 0x9FFF:
            script = "cjk_sc"
        elif ch.isalpha():
            script = "latin"
        else:
            continue
        counts[script] = counts.get(script, 0) + 1

    if not counts:
        return "latin"
    # Japanese often mixes Kanji (cjk_sc) with Hiragana/Katakana (cjk_jp)
    if counts.get("cjk_jp", 0) > 0:
        return "cjk_jp"
    if counts.get("cjk_kr", 0) > 0:
        return "cjk_kr"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def detect_language_code(text: str, hint: Optional[str] = None) -> str:
    """Detect ISO language code using script heuristics + langdetect."""
    if hint and hint != "auto" and hint in LANGUAGE_MAP:
        return hint

    cleaned = text.strip()
    if not cleaned:
        return "en"

    script = detect_script(cleaned)
    script_defaults = {
        "ethiopic": "am",
        "arabic": "ar",
        "hebrew": "he",
        "devanagari": "hi",
        "thai": "th",
        "greek": "el",
        "cjk_jp": "ja",
        "cjk_kr": "ko",
        "cjk_sc": "zh",
    }

    try:
        with _LANGDETECT_LOCK:
            from langdetect import detect

            detected = detect(cleaned)
        if detected == "zh-cn":
            return "zh"
        if detected == "zh-tw":
            return "zh-TW"
        if detected in LANGUAGE_MAP:
            # Guard against langdetect misclassifying short Ethiopic/CJK/Arabic text
            if script in script_defaults and LANGUAGE_MAP[detected].script != script:
                return script_defaults[script]
            return detected
    except Exception:
        pass

    return script_defaults.get(script, "en")
