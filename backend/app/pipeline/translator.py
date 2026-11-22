import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List

import httpx

from app.config import settings
from app.languages import detect_script, get_language
from app.schemas import TranslatorInfo

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    id: str
    name: str
    description: str
    is_local: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        ...


class GoogleWebTranslator(BaseTranslator):
    """Lightweight Google Translate API adapter (no API key required)."""

    id = "google"
    name = "Google Translate"
    description = "Fast online translation with broad language coverage (no API key needed)"
    is_local = False

    def is_available(self) -> bool:
        return True

    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        if not texts or source_lang == target_lang:
            return list(texts)

        src_code = get_language(source_lang).google_code or source_lang
        tgt_code = get_language(target_lang).google_code or target_lang
        outputs: List[str] = []

        with httpx.Client(timeout=15.0) as client:
            for txt in texts:
                resp = client.get(
                    "https://translate.googleapis.com/translate_a/single",
                    params={
                        "client": "gtx",
                        "sl": src_code,
                        "tl": tgt_code,
                        "dt": "t",
                        "q": txt,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                translated = "".join(seg[0] for seg in data[0] if seg and seg[0])
                outputs.append(translated.strip() or txt)
        return outputs


class AddisAITranslator(BaseTranslator):
    """
    Addis AI adapter (https://api.addisassistant.com/api/v1/translate).
    Specialized for Ethiopian & East African languages (Amharic, Afaan Oromo, Tigrinya, English, etc.).
    Enabled when PHOTO_TRANSLATE_ADDIS_API_KEY is configured.
    """

    id = "addis"
    name = "Addis AI (Amharic / Oromo / EN)"
    description = "Specialized Ethiopian language translation (requires PHOTO_TRANSLATE_ADDIS_API_KEY)"
    is_local = False

    SUPPORTED_TARGET_LANGS = {"am", "om", "ti", "en", "sw", "fr"}
    SUPPORTED_SOURCE_LANGS = {"am", "om", "ti", "en", "sw", "fr"}

    def is_available(self) -> bool:
        return bool(settings.addis_api_key.strip())

    @staticmethod
    def _extract_translation(payload: object, fallback: str) -> str:
        if isinstance(payload, str):
            return payload.strip() or fallback
        if not isinstance(payload, dict):
            return fallback

        candidates = [payload]
        if isinstance(payload.get("data"), dict):
            candidates.insert(0, payload["data"])
        elif isinstance(payload.get("data"), str) and payload["data"].strip():
            return payload["data"].strip()

        for obj in candidates:
            for key in ("translated_text", "translation", "response_text", "translatedText", "text", "result"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return fallback

    @staticmethod
    def _format_api_error(resp: httpx.Response) -> str:
        try:
            body = resp.json()
            err = body.get("error") if isinstance(body, dict) else None
            if isinstance(err, dict) and err.get("message"):
                return f"Addis AI error ({resp.status_code}): {err['message']}"
            if isinstance(body, dict) and body.get("message"):
                return f"Addis AI error ({resp.status_code}): {body['message']}"
        except Exception:
            pass
        return f"Addis AI error ({resp.status_code}): {resp.text[:200]}"

    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        if not texts or source_lang == target_lang:
            return list(texts)
        if not self.is_available():
            raise RuntimeError("Addis AI API key is not configured (set PHOTO_TRANSLATE_ADDIS_API_KEY).")

        src_code = source_lang.split("-")[0].lower()
        tgt_code = target_lang.split("-")[0].lower()
        if tgt_code not in self.SUPPORTED_TARGET_LANGS:
            raise ValueError(
                f"Addis AI supports target languages: {', '.join(sorted(self.SUPPORTED_TARGET_LANGS))} (got '{target_lang}')."
            )

        headers = {
            "Content-Type": "application/json",
            "x-api-key": settings.addis_api_key.strip(),
        }
        outputs: List[str] = []

        with httpx.Client(timeout=30.0) as client:
            for txt in texts:
                effective_src = src_code
                if effective_src not in self.SUPPORTED_SOURCE_LANGS:
                    if detect_script(txt) == "latin":
                        effective_src = "en"
                    else:
                        logger.warning(
                            "Addis AI does not support source_language='%s'; preserving original text '%s'.",
                            effective_src,
                            txt[:30],
                        )
                        outputs.append(txt)
                        continue

                if effective_src == tgt_code:
                    outputs.append(txt)
                    continue

                resp = client.post(
                    settings.addis_api_url,
                    headers=headers,
                    json={
                        "text": txt,
                        "source_language": effective_src,
                        "target_language": tgt_code,
                    },
                )
                if resp.status_code == 400 and effective_src != "en" and detect_script(txt) == "latin":
                    resp = client.post(
                        settings.addis_api_url,
                        headers=headers,
                        json={
                            "text": txt,
                            "source_language": "en",
                            "target_language": tgt_code,
                        },
                    )
                if resp.status_code >= 400:
                    raise RuntimeError(self._format_api_error(resp))
                outputs.append(self._extract_translation(resp.json(), txt))
        return outputs


class DeepLTranslator(BaseTranslator):
    """DeepL API adapter (enabled when PHOTO_TRANSLATE_DEEPL_API_KEY is configured)."""

    id = "deepl"
    name = "DeepL API"
    description = "High-nuance cloud translation for Japanese, Chinese & European languages"
    is_local = False

    def is_available(self) -> bool:
        return bool(settings.deepl_api_key.strip())

    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        if not texts or source_lang == target_lang:
            return list(texts)
        if not self.is_available():
            raise RuntimeError("DeepL API key is not configured.")

        tgt_code = get_language(target_lang).deepl_code
        if not tgt_code:
            raise ValueError(f"DeepL does not support target language '{target_lang}'.")

        resp = httpx.post(
            settings.deepl_api_url,
            headers={"Authorization": f"DeepL-Auth-Key {settings.deepl_api_key.strip()}"},
            json={"text": texts, "target_lang": tgt_code},
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["text"].strip() for item in data.get("translations", [])]


class DeepSeekTranslator(BaseTranslator):
    """
    DeepSeek API adapter (OpenAI-compatible chat completions).
    Translates all text blocks on a page in a single context-aware JSON batch request —
    ideal for Japanese/Chinese manga & manhua, memes, and colloquial dialogue.
    Enabled when PHOTO_TRANSLATE_DEEPSEEK_API_KEY is configured.
    """

    id = "deepseek"
    name = "DeepSeek (CJK / Comics / All)"
    description = "Context-aware batch translation — best for Chinese/Japanese to English and idioms"
    is_local = False

    def is_available(self) -> bool:
        return bool(settings.deepseek_api_key.strip())

    @staticmethod
    def _format_api_error(resp: httpx.Response) -> str:
        try:
            body = resp.json()
            err = body.get("error") if isinstance(body, dict) else None
            if isinstance(err, dict) and err.get("message"):
                return f"DeepSeek error ({resp.status_code}): {err['message']}"
            if isinstance(body, dict) and body.get("message"):
                return f"DeepSeek error ({resp.status_code}): {body['message']}"
        except Exception:
            pass
        return f"DeepSeek error ({resp.status_code}): {resp.text[:200]}"

    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        if not texts or source_lang == target_lang:
            return list(texts)
        if not self.is_available():
            raise RuntimeError("DeepSeek API key is not configured (set PHOTO_TRANSLATE_DEEPSEEK_API_KEY).")

        src_name = get_language(source_lang).name if source_lang and source_lang != "auto" else "the detected source language"
        tgt_info = get_language(target_lang)
        tgt_name = f"{tgt_info.name} ({tgt_info.native_name})"

        items_payload = [{"id": idx, "text": txt} for idx, txt in enumerate(texts)]
        system_prompt = (
            f"You are an expert translator for comics, manga, manhua, memes, and images. "
            f"Translate every item from {src_name} into natural, idiomatic {tgt_name}. "
            f"Keep translations concise so they fit back inside speech bubbles and captions, "
            f"preserve tone, humor, and punctuation (such as ?, !, ...), and gracefully fix minor OCR spacing errors. "
            f"Return ONLY a valid JSON object of the exact shape: "
            f'{{"translations": [{{"id": 0, "text": "..."}}, ...]}} with the exact same ids and count.'
        )

        resp = httpx.post(
            settings.deepseek_api_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.deepseek_api_key.strip()}",
            },
            json={
                "model": settings.deepseek_model,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps({"items": items_payload}, ensure_ascii=False)},
                ],
            },
            timeout=45.0,
        )
        if resp.status_code >= 400:
            raise RuntimeError(self._format_api_error(resp))

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        translated_map: Dict[int, str] = {}
        rows = parsed.get("translations") if isinstance(parsed, dict) else None
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and "id" in row and isinstance(row.get("text"), str):
                    try:
                        translated_map[int(row["id"])] = row["text"].strip()
                    except (TypeError, ValueError):
                        continue

        return [translated_map.get(idx) or txt for idx, txt in enumerate(texts)]


class TranslatorRegistry:
    def __init__(self) -> None:
        self._translators: Dict[str, BaseTranslator] = {
            "deepseek": DeepSeekTranslator(),
            "addis": AddisAITranslator(),
            "google": GoogleWebTranslator(),
            "deepl": DeepLTranslator(),
        }

    def get(self, translator_id: str) -> BaseTranslator:
        return self._translators.get(translator_id, self._translators["google"])

    def list_info(self) -> List[TranslatorInfo]:
        return [
            TranslatorInfo(
                id=t.id,
                name=t.name,
                description=t.description,
                available=t.is_available(),
                is_local=t.is_local,
            )
            for t in self._translators.values()
        ]


translator_registry = TranslatorRegistry()
