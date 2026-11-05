"""Round-robin Gemini API key rotation and resilient async LLM cascade router module.

Provides a thread-safe, async-safe KeyManager with automated client connection pooling,
instant key rotation upon 429/503 errors, model cascade fallback, and automated failover
to Groq ('llama-3.3-70b-versatile') when all Gemini keys are exhausted.
"""

import asyncio
import json
import logging
import os
import re
import threading
import urllib.request
from typing import Any, Iterator, Optional, Sequence

import aiohttp
from google import genai
from google.genai import errors, types

from config import DEFAULT_GEMINI_MODEL, DEFAULT_GROQ_MODEL, GROQ_API_KEY, parse_gemini_api_keys

logger = logging.getLogger(__name__)

# Standard browser User-Agent for resilient external HTTP requests
BROWSER_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Target language mapping and strict directive builder
TARGET_LANGUAGE_MAP: dict[str, str] = {
    "en": "English",
    "uk": "Ukrainian",
    "ru": "Russian",
}


def get_target_language_name(target_lang: str = "en") -> str:
    """Resolve language code to human-readable target language name.

    Args:
        target_lang: Language code ('en', 'uk', 'ru').

    Returns:
        str: Mapped language name ('English', 'Ukrainian', 'Russian').
    """
    clean = (target_lang or "en").lower().strip()
    return TARGET_LANGUAGE_MAP.get(clean, "English")


def get_target_language_directive(target_lang: str = "en") -> str:
    """Generate strict target language directive for LLM inference.

    Args:
        target_lang: Language code ('en', 'uk', 'ru').

    Returns:
        str: Strict instruction enforcing complete translation-summarization.
    """
    target_language_name = get_target_language_name(target_lang)
    return (
        f"CRITICAL INSTRUCTION: Analyze the provided content (text, web article, document, or audio) "
        f"and generate the complete structured executive summary strictly in {target_language_name} "
        f'(English for "en", Ukrainian for "uk", Russian for "ru"), fully translating and adapting all headings, '
        f"takeaways, and body text while preserving factual accuracy, proper nouns, and technical metrics."
    )


# Backward-compatible default directive pointing to English
CRITICAL_LANGUAGE_DIRECTIVE: str = get_target_language_directive("en")

# Unified high-density executive summary system prompt
UNIFIED_EXECUTIVE_SYSTEM_PROMPT: str = (
    "Ты — ведущий корпоративный аналитик и бизнес-ассистент.\n"
    "Твоя задача — формировать предельно ёмкие, структурированные выжимки (Executive Summary) "
    "из любых предоставленных материалов БЕЗ ВОДЫ, вводных фраз и шаблонных вступлений.\n\n"
    f"{CRITICAL_LANGUAGE_DIRECTIVE}\n\n"
    "СТРОГО СОБЛЮДАЙ СТРУКТУРУ ВЫВОДА:\n"
    "1. 🎯 Главная суть (1–2 ёмких предложения, передающих ключевой смысл материала).\n"
    "2. 📊 Ключевые факты и цифры (маркированный список важнейших фактов, метрик, аргументов или данных).\n"
    "3. 💡 Практические выводы / Действия (конкретные шаги, инсайты или рекомендации к действию).\n\n"
    "Требования:\n"
    "- Используй профессиональный тон, лаконичный Markdown (жирный шрифт для акцентов, списки).\n"
    "- Никаких общих фраз, клише и воды — только факты и практическая ценность."
)

FORMAT_LABELS: dict[str, str] = {
    "SHORT": "⚡ Шпаргалка",
    "MEDIUM": "📋 Конспект",
    "LONG": "📚 Пересказ",
}

FORMAT_INSTRUCTIONS: dict[str, str] = {
    "SHORT": (
        "ФОРМАТ СЖАТИЯ: ⚡ ШПАРГАЛКА (МАКСИМАЛЬНО КРАТКО).\n"
        "- Длина: строго 5–7 самых важных фактов или цифр на один экран.\n"
        "- 1–2 ключевых практических вывода."
    ),
    "MEDIUM": (
        "ФОРМАТ СЖАТИЯ: 📋 СТРУКТУРИРОВАННЫЙ КОНСПЕКТ (СРЕДНИЙ ОБЪЁМ).\n"
        "- Длина: 8–12 ключевых фактов, логически сгруппированных.\n"
        "- 2–4 практических вывода."
    ),
    "LONG": (
        "ФОРМАТ СЖАТИЯ: 📚 ДЕТАЛЬНЫЙ РАЗБОР (МАКСИМАЛЬНАЯ ГЛУБИНА).\n"
        "- Подробно разбери все разделы с контекстом, нюансами и метриками.\n"
        "- Развёрнутые практические выводы и рекомендации."
    ),
}


class LLMRotationError(RuntimeError):
    """Base exception for LLM routing and key rotation failures."""


class AllKeysExhaustedError(LLMRotationError):
    """Raised when all Gemini API keys and Groq failover paths are exhausted."""


def is_resource_exhausted(exc: Exception) -> bool:
    """Determine whether an exception corresponds to HTTP 429, 503, or service exhaustion.

    Args:
        exc: The caught exception instance.

    Returns:
        bool: True if the exception indicates rate limiting, quota exhaustion, or service unavailability.
    """
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        status = str(getattr(exc, "status", "") or "").upper()
        if code in (429, 503):
            return True
        if status in ("RESOURCE_EXHAUSTED", "UNAVAILABLE"):
            return True

    exc_str = str(exc).upper()
    return any(
        marker in exc_str
        for marker in (
            "429",
            "503",
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "QUOTA EXCEEDED",
            "RATE LIMIT",
            "HIGH DEMAND",
            "OVERLOADED",
            "SERVICE UNAVAILABLE",
        )
    )


def mask_api_key(key: str) -> str:
    """Mask sensitive API key strings for secure diagnostic logging.

    Args:
        key: The raw API key string.

    Returns:
        str: Partially masked key (e.g., 'AIza...3f8x').
    """
    cleaned = key.strip()
    if len(cleaned) <= 8:
        return "***"
    return f"{cleaned[:4]}...{cleaned[-4:]}"


def get_model_cascade(primary_model: Optional[str] = None) -> list[str]:
    """Construct an ordered, deduplicated list of Gemini models for the cascade.

    Starts with the working model from .env (GEMINI_MODEL), alongside proven free-tier
    models ('gemini-2.5-flash', 'gemini-2.0-flash'), eliminating duplicates without
    including retired 404 models.

    Args:
        primary_model: Optional explicit model override.

    Returns:
        list[str]: Deduplicated cascade list of model identifiers.
    """
    primary = (primary_model or os.getenv("GEMINI_MODEL", "")).strip() or DEFAULT_GEMINI_MODEL
    candidates = [primary, "gemini-2.5-flash", "gemini-2.0-flash"]
    cascade: list[str] = []
    seen: set[str] = set()
    for m in candidates:
        clean = m.strip()
        if clean and clean not in seen:
            seen.add(clean)
            cascade.append(clean)
    return cascade


def extract_text_from_contents(contents: Any) -> str:
    """Extract plain text string representation from diverse GenAI content payloads.

    Args:
        contents: String, list of parts, or structured content objects.

    Returns:
        str: Clean plain text extracted from the payload.
    """
    if isinstance(contents, str):
        return contents
    if isinstance(contents, (list, tuple)):
        parts: list[str] = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "\n\n".join(parts)
    return str(contents)


def _smart_compress_text(text: str, max_chars: int = 20000) -> str:
    """Safely truncate and compress input text to prevent TPM limit spikes.

    Args:
        text: Input raw text string.
        max_chars: Maximum allowable character count (defaults to 20,000).

    Returns:
        str: Safely bounded text string.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    # Truncate safely at max_chars limit to protect TPM quotas
    return text[:max_chars]


def _local_extractive_fallback(raw_text: str, source_title: str) -> str:
    """Extractive paragraph fallback when both Gemini and Groq completely fail.

    Args:
        raw_text: Source text to extract paragraphs from.
        source_title: Headline or source title.

    Returns:
        str: Locally generated structural summary.
    """
    compressed = _smart_compress_text(raw_text, max_chars=6000)
    paragraphs = [p.strip() for p in compressed.split("\n") if len(p.strip()) > 60]
    selected = paragraphs[:14]
    bullets = "\n\n".join(f"• {p[:320]}..." if len(p) > 320 else f"• {p}" for p in selected)
    return (
        f"⚡ **Автономная структурная выжимка (Аварийный режим)**\n"
        f"📌 **Источник:** {source_title} (Объём: {len(raw_text)} симв.)\n\n"
        f"🎯 **Главная суть:**\nМатериал обработан локальным парсером в режиме изоляции.\n\n"
        f"📊 **Ключевые извлечённые фрагменты:**\n{bullets}\n\n"
        f"💡 **Практические выводы:**\nОбратитесь к исходному материалу для полного контекста."
    )


class LLMResponse:
    """Uniform container wrapping generation output across Gemini and Groq providers."""

    def __init__(
        self,
        text: str,
        provider: str = "gemini",
        model: str = "",
        candidates: Optional[list[Any]] = None,
    ) -> None:
        """Initialize LLMResponse instance.

        Args:
            text: Generated content text string.
            provider: Name of the responding provider ('gemini' or 'groq').
            model: Specific model identifier used for generation.
            candidates: Optional candidate structures for compatibility.
        """
        self.text: str = text
        self.provider: str = provider
        self.model: str = model
        self.candidates: list[Any] = candidates or []

    def __str__(self) -> str:
        """Return text representation for string coercion."""
        return self.text


class KeyManager:
    """Thread-safe and async-safe round-robin API key manager and client connection pool.

    Maintains cached genai.Client instances to avoid connection setup overhead while
    guaranteeing atomic pointer advancement during error rotation.
    """

    def __init__(self, keys: Optional[Sequence[str]] = None) -> None:
        """Initialize the KeyManager with a sequence of API keys or environment discovery.

        Args:
            keys: Optional sequence of API keys. If omitted or None, parsed from
                the GEMINI_API_KEYS and GEMINI_API_KEY environment variables.
        """
        self._lock = threading.Lock()
        self._index: int = 0
        self._clients: dict[str, genai.Client] = {}

        if keys is not None:
            self._keys: list[str] = [k.strip() for k in keys if k.strip()]
        else:
            self._keys = parse_gemini_api_keys()

    def _ensure_keys(self) -> list[str]:
        """Verify key pool is non-empty, attempting environment refresh if empty.

        Returns:
            list[str]: The active list of API keys.

        Raises:
            ValueError: If no valid keys are available in the pool.
        """
        if not self._keys:
            refreshed = parse_gemini_api_keys()
            if refreshed:
                self._keys = refreshed
        if not self._keys:
            raise ValueError(
                "No Gemini API keys configured. Set GEMINI_API_KEYS in your environment or .env."
            )
        return self._keys

    @property
    def total_keys(self) -> int:
        """Return the total number of configured API keys in the pool."""
        with self._lock:
            return len(self._keys)

    def __len__(self) -> int:
        """Return the total number of configured API keys in the pool."""
        with self._lock:
            return len(self._keys)

    def get_all_keys(self) -> list[str]:
        """Return a copy of the list of API keys in the rotation pool."""
        with self._lock:
            return list(self._keys)

    def get_current_index(self) -> int:
        """Return the 0-based index of the currently active key.

        Returns:
            int: 0-based key index.
        """
        with self._lock:
            keys = self._ensure_keys()
            return self._index % len(keys)

    def get_current_key(self) -> str:
        """Retrieve the currently active API key without advancing the pointer.

        Returns:
            str: Currently active API key.
        """
        with self._lock:
            keys = self._ensure_keys()
            return keys[self._index % len(keys)]

    def get_next_key(self) -> str:
        """Advance pointer and return the next API key in round-robin sequence.

        Returns:
            str: Next API key.
        """
        with self._lock:
            keys = self._ensure_keys()
            key = keys[self._index % len(keys)]
            self._index = (self._index + 1) % len(keys)
            return key

    def rotate_key(self) -> str:
        """Advance pointer to next key and return the newly active key.

        Returns:
            str: Newly active API key after rotation.
        """
        with self._lock:
            keys = self._ensure_keys()
            self._index = (self._index + 1) % len(keys)
            return keys[self._index % len(keys)]

    def get_client(self, api_key: Optional[str] = None) -> genai.Client:
        """Retrieve or instantiate a cached genai.Client for the specified or current key.

        Args:
            api_key: Target key to look up. If None, uses the currently active key.

        Returns:
            genai.Client: Cached Google GenAI client instance.
        """
        with self._lock:
            keys = self._ensure_keys()
            target_key = api_key if api_key else keys[self._index % len(keys)]
            if target_key not in self._clients:
                self._clients[target_key] = genai.Client(api_key=target_key)
            return self._clients[target_key]

    def key_generator(self) -> Iterator[str]:
        """Generate an infinite stream of API keys in round-robin rotation.

        Yields:
            str: API key string.
        """
        while True:
            yield self.get_next_key()

    def set_keys(self, new_keys: Sequence[str]) -> None:
        """Replace the current key pool with a new sequence of API keys.

        Args:
            new_keys: New sequence of non-empty API keys.

        Raises:
            ValueError: If new_keys is empty.
        """
        with self._lock:
            cleaned = [k.strip() for k in new_keys if k.strip()]
            if not cleaned:
                raise ValueError("Cannot set an empty key list in KeyManager.")
            self._keys = cleaned
            self._index = 0
            self._clients = {k: client for k, client in self._clients.items() if k in cleaned}


async def call_groq_api(
    prompt: str,
    system_prompt: Optional[str] = None,
    groq_key: Optional[str] = None,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.25,
    timeout: int = 35,
) -> str:
    """Execute asynchronous chat completion call to Groq API.

    Tries the requested primary model ('llama-3.3-70b-versatile') and gracefully
    falls back to secondary available models if the primary model is unavailable.

    Args:
        prompt: User message content.
        system_prompt: Optional system instruction prompt.
        groq_key: Optional Groq API key (defaults to environment).
        model: Target Groq model identifier.
        temperature: Sampling temperature.
        timeout: Network timeout in seconds.

    Returns:
        str: Clean text generated by Groq.

    Raises:
        RuntimeError: If Groq API invocation fails across all models.
    """
    key = (groq_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")).strip().strip('"').strip("'")
    if not key:
        raise ValueError("GROQ_API_KEY is not configured.")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    candidate_models = [model, "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": BROWSER_USER_AGENT,
    }

    timeout_cfg = aiohttp.ClientTimeout(total=timeout)
    last_exc: Optional[Exception] = None

    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        for target_model in candidate_models:
            payload = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
            }
            try:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            answer = choices[0].get("message", {}).get("content", "").strip()
                            if answer:
                                logger.info(
                                    "[Provider: Groq, Model: %s] Fallback generation succeeded.",
                                    target_model,
                                )
                                return answer
                    error_text = await resp.text()
                    logger.warning(
                        "[Provider: Groq, Model: %s] HTTP %d: %s",
                        target_model,
                        resp.status,
                        error_text[:200],
                    )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[Provider: Groq, Model: %s] Connection error: %s",
                    target_model,
                    exc,
                )

    raise RuntimeError(f"All Groq fallback models failed. Last error: {last_exc}")


def call_groq_api_sync(
    prompt: str,
    system_prompt: Optional[str] = None,
    groq_key: Optional[str] = None,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.25,
    timeout: int = 35,
) -> str:
    """Execute synchronous chat completion call to Groq API using standard HTTP library.

    Args:
        prompt: User message content.
        system_prompt: Optional system instruction prompt.
        groq_key: Optional Groq API key.
        model: Target Groq model identifier.
        temperature: Sampling temperature.
        timeout: Network timeout in seconds.

    Returns:
        str: Clean text generated by Groq.

    Raises:
        RuntimeError: If Groq API invocation fails.
    """
    key = (groq_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")).strip().strip('"').strip("'")
    if not key:
        raise ValueError("GROQ_API_KEY is not configured.")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    candidate_models = [model, "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": BROWSER_USER_AGENT,
    }

    last_exc: Optional[Exception] = None
    for target_model in candidate_models:
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=data_bytes,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                choices = data.get("choices", [])
                if choices:
                    ans = choices[0].get("message", {}).get("content", "").strip()
                    if ans:
                        logger.info(
                            "[Provider: Groq, Model: %s] Sync fallback generation succeeded.",
                            target_model,
                        )
                        return ans
        except Exception as exc:
            last_exc = exc
            logger.warning("[Provider: Groq, Model: %s] Sync error: %s", target_model, exc)

    raise RuntimeError(f"All Groq sync models failed. Last error: {last_exc}")


class LLMRouter:
    """Asynchronous router executing Google GenAI calls with automated key rotation and retries."""

    def __init__(
        self,
        key_manager: Optional[KeyManager] = None,
        groq_api_key: Optional[str] = None,
    ) -> None:
        """Initialize the router with an assigned KeyManager and Groq configuration.

        Args:
            key_manager: Optional KeyManager instance. If None, instantiates a default.
            groq_api_key: Optional Groq API key for failover routing.
        """
        self.key_manager: KeyManager = key_manager or KeyManager()
        self.groq_api_key: str = (groq_api_key or GROQ_API_KEY).strip()

    async def generate_content(
        self,
        model: Optional[str] = None,
        contents: Any = None,
        config: Optional[types.GenerateContentConfig] = None,
        client: Optional[genai.Client] = None,
        models: Optional[Sequence[str]] = None,
        fallback_to_groq: bool = True,
        temperature: float = 0.25,
        **kwargs: Any,
    ) -> Any:
        """Execute async generate_content with automated cascade rotation and Groq failover.

        Tries the specified model alongside fallback models ('gemini-2.5-flash',
        'gemini-2.0-flash'). If rate limits (429) or high-load (503) errors occur, rotates
        through all available Gemini API keys. If all Gemini keys fail, automatically
        falls back to Groq ('llama-3.3-70b-versatile') using the exact same extracted text.

        Args:
            model: Primary Gemini model identifier (or None to use cascade default).
            contents: Generation content payload (string or parts).
            config: Optional generation configuration.
            client: Optional static client to attempt first.
            models: Optional explicit sequence of models for the cascade.
            fallback_to_groq: Whether to trigger Groq failover when Gemini keys fail.
            temperature: Sampling temperature.
            **kwargs: Extra arguments passed to GenAI client.

        Returns:
            Any: The response object (Google GenAI response or LLMResponse).

        Raises:
            AllKeysExhaustedError: If all Gemini keys and Groq failover fail.
            Exception: If an unhandled non-retriable error occurs.
        """
        # If a static client is passed explicitly (e.g. In unit tests), attempt it first
        if client is not None:
            try:
                target_model = model or DEFAULT_GEMINI_MODEL
                if hasattr(client, "aio") and hasattr(client.aio, "models"):
                    return await client.aio.models.generate_content(
                        model=target_model,
                        contents=contents,
                        config=config,
                        **kwargs,
                    )
                return await asyncio.to_thread(
                    client.models.generate_content,
                    model=target_model,
                    contents=contents,
                    config=config,
                    **kwargs,
                )
            except Exception as exc:
                if not is_resource_exhausted(exc):
                    raise
                logger.warning(
                    "[Provider: Gemini, Model: %s] Static client encountered 429/503. "
                    "Falling back to KeyManager rotation pool...",
                    model,
                )

        pool_size = len(self.key_manager)
        cascade = list(models) if models else get_model_cascade(model)
        plain_text = extract_text_from_contents(contents)
        last_error: Optional[Exception] = None

        if pool_size > 0:
            for target_model in cascade:
                model_unavailable = False
                for attempt in range(pool_size):
                    key = self.key_manager.get_current_key()
                    key_idx = self.key_manager.get_current_index()
                    active_client = self.key_manager.get_client(key)
                    masked = mask_api_key(key)

                    try:
                        if hasattr(active_client, "aio") and hasattr(active_client.aio, "models"):
                            response = await active_client.aio.models.generate_content(
                                model=target_model,
                                contents=contents,
                                config=config,
                                **kwargs,
                            )
                        else:
                            response = await asyncio.to_thread(
                                active_client.models.generate_content,
                                model=target_model,
                                contents=contents,
                                config=config,
                                **kwargs,
                            )
                        logger.info(
                            "[Provider: Gemini, Model: %s, Key Index: %d] Execution succeeded.",
                            target_model,
                            key_idx,
                        )
                        return response
                    except Exception as exc:
                        last_error = exc
                        logger.warning(
                            "[Provider: Gemini, Model: %s, Key Index: %d (%s)] Error: %s",
                            target_model,
                            key_idx,
                            masked,
                            exc,
                        )

                        if is_resource_exhausted(exc):
                            logger.info(
                                "[Provider: Gemini, Model: %s, Key Index: %d] Rate-limited (429/503). "
                                "Rotating key (attempt %d/%d)...",
                                target_model,
                                key_idx,
                                attempt + 1,
                                pool_size,
                            )
                            self.key_manager.rotate_key()
                            continue

                        # If model is retired or not found (404), advance to next cascade model immediately
                        exc_str = str(exc).upper()
                        if "404" in exc_str or "NOT_FOUND" in exc_str or "NOT SUPPORTED" in exc_str:
                            logger.warning(
                                "[Provider: Gemini, Model: %s, Key Index: %d] Model 404/Not Found. "
                                "Advancing to next model in cascade...",
                                target_model,
                                key_idx,
                            )
                            model_unavailable = True
                            break

                        # Non-quota client validation error
                        if not isinstance(exc, (errors.APIError, RuntimeError)):
                            raise

                if model_unavailable:
                    continue

        logger.critical(
            "[Provider: Gemini] All %d Gemini API keys exhausted across models %s. "
            "Initiating fallback to Groq...",
            pool_size,
            cascade,
        )

        # Fallback to Groq if configured and text payload is present
        groq_key = (self.groq_api_key or os.getenv("GROQ_API_KEY", "")).strip().strip('"').strip("'")
        if fallback_to_groq and groq_key and plain_text:
            sys_prompt: Optional[str] = None
            if config and getattr(config, "system_instruction", None):
                sys_prompt = str(config.system_instruction)

            try:
                logger.info(
                    "[Provider: Groq, Model: %s] Calling Groq fallback with extracted text...",
                    DEFAULT_GROQ_MODEL,
                )
                groq_answer = await call_groq_api(
                    prompt=plain_text,
                    system_prompt=sys_prompt,
                    groq_key=groq_key,
                    model=DEFAULT_GROQ_MODEL,
                    temperature=temperature,
                )
                return LLMResponse(
                    text=groq_answer,
                    provider="groq",
                    model=DEFAULT_GROQ_MODEL,
                )
            except Exception as groq_exc:
                logger.error(
                    "[Provider: Groq, Model: %s] Groq failover error: %s",
                    DEFAULT_GROQ_MODEL,
                    groq_exc,
                )

        raise AllKeysExhaustedError(
            f"All {pool_size} Gemini API key(s) exhausted across cascade {cascade} "
            f"and Groq failover failed. Last error: {last_error}"
        ) from last_error

    def generate_content_sync(
        self,
        model: Optional[str] = None,
        contents: Any = None,
        config: Optional[types.GenerateContentConfig] = None,
        client: Optional[genai.Client] = None,
        models: Optional[Sequence[str]] = None,
        fallback_to_groq: bool = True,
        temperature: float = 0.25,
        **kwargs: Any,
    ) -> Any:
        """Execute synchronous generate_content with cascade rotation and Groq failover.

        Args:
            model: Primary Gemini model identifier.
            contents: Generation content payload.
            config: Optional generation configuration.
            client: Optional static client to attempt first.
            models: Optional sequence of models for the cascade.
            fallback_to_groq: Whether to trigger Groq failover when Gemini keys fail.
            temperature: Sampling temperature.
            **kwargs: Extra parameters passed to the underlying client call.

        Returns:
            Any: The response object (GenAI response or LLMResponse).

        Raises:
            AllKeysExhaustedError: If all keys and failovers fail.
            Exception: If an unhandled non-retriable error occurs.
        """
        if client is not None:
            try:
                target_model = model or DEFAULT_GEMINI_MODEL
                return client.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config=config,
                    **kwargs,
                )
            except Exception as exc:
                if not is_resource_exhausted(exc):
                    raise
                logger.warning(
                    "[Provider: Gemini, Model: %s] Static client encountered 429/503. "
                    "Falling back to KeyManager rotation pool...",
                    model,
                )

        pool_size = len(self.key_manager)
        cascade = list(models) if models else get_model_cascade(model)
        plain_text = extract_text_from_contents(contents)
        last_error: Optional[Exception] = None

        if pool_size > 0:
            for target_model in cascade:
                model_unavailable = False
                for attempt in range(pool_size):
                    key = self.key_manager.get_current_key()
                    key_idx = self.key_manager.get_current_index()
                    active_client = self.key_manager.get_client(key)
                    masked = mask_api_key(key)

                    try:
                        response = active_client.models.generate_content(
                            model=target_model,
                            contents=contents,
                            config=config,
                            **kwargs,
                        )
                        logger.info(
                            "[Provider: Gemini, Model: %s, Key Index: %d] Sync execution succeeded.",
                            target_model,
                            key_idx,
                        )
                        return response
                    except Exception as exc:
                        last_error = exc
                        logger.warning(
                            "[Provider: Gemini, Model: %s, Key Index: %d (%s)] Sync error: %s",
                            target_model,
                            key_idx,
                            masked,
                            exc,
                        )

                        if is_resource_exhausted(exc):
                            self.key_manager.rotate_key()
                            continue

                        exc_str = str(exc).upper()
                        if "404" in exc_str or "NOT_FOUND" in exc_str or "NOT SUPPORTED" in exc_str:
                            model_unavailable = True
                            break

                        if not isinstance(exc, (errors.APIError, RuntimeError)):
                            raise

                if model_unavailable:
                    continue

        groq_key = (self.groq_api_key or os.getenv("GROQ_API_KEY", "")).strip().strip('"').strip("'")
        if fallback_to_groq and groq_key and plain_text:
            sys_prompt: Optional[str] = None
            if config and getattr(config, "system_instruction", None):
                sys_prompt = str(config.system_instruction)

            try:
                ans = call_groq_api_sync(
                    prompt=plain_text,
                    system_prompt=sys_prompt,
                    groq_key=groq_key,
                    model=DEFAULT_GROQ_MODEL,
                    temperature=temperature,
                )
                return LLMResponse(text=ans, provider="groq", model=DEFAULT_GROQ_MODEL)
            except Exception as groq_exc:
                logger.error("[Provider: Groq, Model: %s] Sync failover error: %s", DEFAULT_GROQ_MODEL, groq_exc)

        raise AllKeysExhaustedError(
            f"All {pool_size} Gemini API key(s) exhausted across cascade {cascade} "
            f"and Groq failover failed. Last error: {last_error}"
        ) from last_error


def build_executive_system_prompt(target_lang: str = "en", format_size: str = "MEDIUM") -> str:
    """Construct full executive system prompt with strict target language directive and compression tier guide.

    Args:
        target_lang: Target language code ('en', 'uk', 'ru').
        format_size: Compression tier ('SHORT', 'MEDIUM', 'LONG').

    Returns:
        str: Fully composed system instruction.
    """
    directive = get_target_language_directive(target_lang)
    target_name = get_target_language_name(target_lang)
    fmt_key = format_size.upper() if format_size.upper() in FORMAT_INSTRUCTIONS else "MEDIUM"
    size_guide = FORMAT_INSTRUCTIONS[fmt_key]

    return (
        f"You are a leading corporate analyst and executive business assistant.\n"
        f"Your task is to form exceptionally dense, structured executive summaries (Executive Summary) "
        f"from any provided materials without filler, introductory chatter, or boilerplate phrases.\n\n"
        f"{directive}\n\n"
        f"COMPRESSION AND FORMAT GUIDELINES:\n"
        f"{size_guide}\n\n"
        f"OUTPUT STRUCTURE REQUIREMENTS:\n"
        f"1. 🎯 Main Essence / Core Thesis (1–2 dense sentences in {target_name}).\n"
        f"2. 📊 Key Facts, Figures & Metrics (bulleted list of vital facts, metrics, and arguments in {target_name}).\n"
        f"3. 💡 Actionable Conclusions / Practical Steps (concrete actions and strategic insights in {target_name}).\n\n"
        f"Requirements:\n"
        f"- Use a professional executive tone and concise Markdown (bold accents, clean bullet lists).\n"
        f"- Fully translate and adapt all headings, takeaways, and body text strictly into {target_name}.\n"
        f"- No generic fluff, introductory preamble, or empty clichés — only factual accuracy and high practical value."
    )


# Default singleton instances for process-wide sharing
default_key_manager = KeyManager()
default_router = LLMRouter(key_manager=default_key_manager)

# Convenience aliases
generate_content = default_router.generate_content
generate_content_sync = default_router.generate_content_sync
get_client = default_key_manager.get_client
rotate = default_key_manager.rotate_key


async def generate_executive_summary(
    raw_text: str,
    source_title: str = "",
    format_size: str = "MEDIUM",
    retry_variation: bool = False,
    target_lang: str = "en",
    router: Optional[LLMRouter] = None,
) -> str:
    """Generate high-density Executive Summary translated and adapted into target_lang.

    Requires:
    (1) Core Essence in 1-2 sentences,
    (2) Key Facts/Metrics as bullet points,
    (3) Actionable Takeaways,
    strictly generated in target_lang ('en', 'uk', 'ru').

    Never falls back to dumb character slicing unless both Gemini and Groq completely fail.

    Args:
        raw_text: Extracted plain text content.
        source_title: Source article or document title.
        format_size: Compression tier ('SHORT', 'MEDIUM', 'LONG').
        retry_variation: Whether this is a regenerated variation request.
        target_lang: Target interface language ('en', 'uk', 'ru').
        router: Optional custom LLMRouter instance.

    Returns:
        str: Formatted markdown Executive Summary in target_lang.
    """
    clean_target_lang = (target_lang or "en").lower().strip()
    if clean_target_lang not in TARGET_LANGUAGE_MAP:
        clean_target_lang = "en"

    not_enough_text_map = {
        "en": "⚠️ Could not extract sufficient text from the source.",
        "uk": "⚠️ Не вдалося вилучити достатній обсяг тексту з джерела.",
        "ru": "⚠️ Не удалось извлечь достаточный объём текста из источника.",
    }
    if not raw_text or len(raw_text.strip()) < 10:
        return not_enough_text_map.get(clean_target_lang, not_enough_text_map["en"])

    # Truncate text safely at 20,000 characters to prevent TPM limit spikes
    prepared_text = _smart_compress_text(raw_text, max_chars=20000)

    variation_notes = {
        "en": "\nATTENTION: This is a re-analysis! Focus on alternative details, nuanced perspectives, and fresh insights.\n",
        "uk": "\nУВАГА: Це повторний аналіз! Зроби акцент на альтернативних деталях та неочевидних інсайтах.\n",
        "ru": "\nВНИМАНИЕ: Это повторный анализ! Сделай упор на альтернативных деталях и неочевидных инсайтах.\n",
    }
    variation_note = variation_notes.get(clean_target_lang, variation_notes["en"]) if retry_variation else ""

    target_language_name = get_target_language_name(clean_target_lang)
    full_system_instruction = build_executive_system_prompt(target_lang=clean_target_lang, format_size=format_size)

    user_prompt = (
        f"SOURCE: {source_title or 'Document'}\n"
        f"TARGET LANGUAGE: {target_language_name} ({clean_target_lang})\n"
        f"{variation_note}\n"
        f"--- START CONTENT ---\n{prepared_text}\n--- END CONTENT ---"
    )

    config = types.GenerateContentConfig(
        system_instruction=full_system_instruction,
        temperature=0.55 if retry_variation else 0.25,
    )

    active_router = router or default_router

    try:
        response = await active_router.generate_content(
            contents=user_prompt,
            config=config,
            fallback_to_groq=True,
            temperature=0.55 if retry_variation else 0.25,
        )
        response_text = getattr(response, "text", "") or ""
        if response_text.strip():
            return response_text.strip()
    except Exception as exc:
        logger.warning(
            "[Executive Summary Cascade] Both Gemini and Groq failover failed for '%s': %s. "
            "Engaging emergency local extractor...",
            source_title,
            exc,
        )

    # Emergency fallback only when both Gemini and Groq completely fail
    return _local_extractive_fallback(raw_text, source_title or "Document")


async def summarize_text(
    raw_text: str,
    source_title: str = "",
    format_size: str = "MEDIUM",
    retry_variation: bool = False,
    target_lang: str = "en",
    router: Optional[LLMRouter] = None,
) -> str:
    """Asynchronous text summarization entry point honoring target_lang.

    Args:
        raw_text: Extracted plain text content.
        source_title: Source article or document title.
        format_size: Compression tier ('SHORT', 'MEDIUM', 'LONG').
        retry_variation: Whether this is a regenerated variation request.
        target_lang: Target interface language ('en', 'uk', 'ru').
        router: Optional custom LLMRouter instance.

    Returns:
        str: Formatted markdown summary in target_lang.
    """
    return await generate_executive_summary(
        raw_text=raw_text,
        source_title=source_title,
        format_size=format_size,
        retry_variation=retry_variation,
        target_lang=target_lang,
        router=router,
    )


def process_audio_with_gemini(
    file_path: str,
    mime_type: str,
    source_title: str,
    format_size: str = "MEDIUM",
    target_lang: str = "en",
    lang: Optional[str] = None,
    client: Optional[genai.Client] = None,
    router: Optional[LLMRouter] = None,
) -> str:
    """Upload audio stream via Google GenAI File API, generate summary in target_lang, and cleanup.

    Args:
        file_path: Local filesystem path to audio recording.
        mime_type: Audio MIME type ('audio/ogg', 'audio/mpeg', etc.).
        source_title: Source label or filename.
        format_size: Compression tier ('SHORT', 'MEDIUM', 'LONG').
        target_lang: Target language code ('en', 'uk', 'ru').
        lang: Backward-compatible alias for target_lang.
        client: Optional genai.Client instance.
        router: Optional LLMRouter instance for fallback.

    Returns:
        str: Formatted summary in target_lang.
    """
    effective_lang = (target_lang or lang or "en").lower().strip()
    if effective_lang not in TARGET_LANGUAGE_MAP:
        effective_lang = "en"

    active_client = client or default_key_manager.get_client()
    uploaded_file = None
    try:
        fmt_key = format_size.upper() if format_size.upper() in FORMAT_INSTRUCTIONS else "MEDIUM"
        uploaded_file = active_client.files.upload(
            file=file_path,
            config=types.UploadFileConfig(mime_type=mime_type),
        )

        lang_directive = get_target_language_directive(effective_lang)
        target_name = get_target_language_name(effective_lang)

        if effective_lang == "ru":
            task_instruction = f"Внимательно прослушай эту аудиозапись ({source_title}) и сформируй структурированную выжимку строго на русском языке."
        elif effective_lang == "uk":
            task_instruction = f"Уважно прослухай цей аудіозапис ({source_title}) та сформуй структуровану вижимку строго українською мовою."
        else:
            task_instruction = f"Listen carefully to this audio recording ({source_title}) and generate a structured executive summary strictly in English."

        prompt = (
            f"You are a leading corporate analyst and executive business assistant.\n\n"
            f"{lang_directive}\n\n"
            f"{FORMAT_INSTRUCTIONS[fmt_key]}\n\n"
            f"STRICTLY ADHERE TO OUTPUT STRUCTURE:\n"
            f"1. 🎯 Main Essence / Core Thesis (1-2 dense sentences in {target_name}).\n"
            f"2. 📊 Key Facts, Figures & Metrics (bulleted list in {target_name}).\n"
            f"3. 💡 Actionable Conclusions / Practical Steps (in {target_name}).\n\n"
            f"{task_instruction}"
        )

        active_router = router or default_router
        try:
            response = active_router.generate_content_sync(
                contents=[uploaded_file, prompt],
                temperature=0.25,
            )
            return str(response).strip()
        except Exception as exc:
            logger.error("[Audio Pipeline] Generation failed: %s", exc)
            raise RuntimeError(f"Audio summarization failed across cascade: {exc}") from exc
    finally:
        if uploaded_file:
            try:
                active_client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


def summarize_with_gemini(
    raw_text: str,
    source_title: str,
    format_size: str = "MEDIUM",
    retry_variation: bool = False,
    target_lang: str = "en",
) -> str:
    """Synchronous backward-compatible bridge for executive summary generation.

    Args:
        raw_text: Extracted plain text content.
        source_title: Source article or document title.
        format_size: Compression tier ('SHORT', 'MEDIUM', 'LONG').
        retry_variation: Whether this is a regenerated variation request.
        target_lang: Target interface language ('en', 'uk', 'ru').

    Returns:
        str: Formatted markdown summary.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    generate_executive_summary(
                        raw_text,
                        source_title,
                        format_size,
                        retry_variation,
                        target_lang=target_lang,
                    ),
                ).result()
        return loop.run_until_complete(
            generate_executive_summary(
                raw_text,
                source_title,
                format_size,
                retry_variation,
                target_lang=target_lang,
            )
        )
    except Exception:
        return _local_extractive_fallback(raw_text, source_title)
    except Exception:
        return _local_extractive_fallback(raw_text, source_title)
