"""Round-robin Gemini API key rotation and resilient async LLM router module.

Provides a 10x-scalable, thread-safe, and async-safe KeyManager with automated
client connection pooling, instant key rotation upon 429 RESOURCE_EXHAUSTED errors,
and retry guarantees across all configured API keys.
"""

import asyncio
import itertools
import logging
import threading
from typing import Any, Iterator, Optional, Sequence

from google import genai
from google.genai import errors, types

from config import parse_gemini_api_keys

logger = logging.getLogger(__name__)


class LLMRotationError(RuntimeError):
    """Base exception for LLM routing and key rotation failures."""


class AllKeysExhaustedError(LLMRotationError):
    """Raised when all Gemini API keys in the rotation pool have been exhausted by rate limits."""


def is_resource_exhausted(exc: Exception) -> bool:
    """Determine whether an exception corresponds to HTTP 429 or RESOURCE_EXHAUSTED.

    Args:
        exc: The caught exception instance.

    Returns:
        bool: True if the exception indicates rate limiting or quota exhaustion.
    """
    if isinstance(exc, errors.APIError):
        if getattr(exc, "code", None) == 429:
            return True
        if getattr(exc, "status", None) == "RESOURCE_EXHAUSTED":
            return True

    exc_str = str(exc).upper()
    return "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str or "QUOTA EXCEEDED" in exc_str


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


class KeyManager:
    """Thread-safe and async-safe round-robin API key manager and client connection pool.

    Supports dynamic scale up to dozens of keys, maintaining cached genai.Client
    instances to avoid connection setup overhead while guaranteeing safe concurrent access.
    """

    def __init__(self, keys: Optional[Sequence[str]] = None) -> None:
        """Initialize the KeyManager with a sequence of API keys or environment discovery.

        Args:
            keys: Optional sequence of API keys. If omitted or None, parsed from
                the GEMINI_API_KEYS environment variable.
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
            return keys[self._index]

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


class LLMRouter:
    """Asynchronous router executing Google GenAI calls with automated key rotation and retries."""

    def __init__(self, key_manager: Optional[KeyManager] = None) -> None:
        """Initialize the router with an assigned or default KeyManager.

        Args:
            key_manager: Optional KeyManager instance. If None, instantiates a default.
        """
        self.key_manager: KeyManager = key_manager or KeyManager()

    async def generate_content(
        self,
        model: str,
        contents: Any,
        config: Optional[types.GenerateContentConfig] = None,
        client: Optional[genai.Client] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute async generate_content with automated round-robin retry upon 429 errors.

        If a specific client is provided and encounters a 429 error, execution falls back
        to rotating through the KeyManager pool. If all available keys in the pool fail
        with 429 RESOURCE_EXHAUSTED, raises AllKeysExhaustedError.

        Args:
            model: Gemini model name or path string.
            contents: Generation content payload (string, list, or Part objects).
            config: Optional generation configuration.
            client: Optional static client to attempt first before pool rotation.
            **kwargs: Extra parameters passed to the underlying client call.

        Returns:
            Any: The GenerateContentResponse returned by Google GenAI.

        Raises:
            AllKeysExhaustedError: If all keys in the rotation pool return 429.
            Exception: If an unhandled non-retriable error occurs.
        """
        # If a static client is passed explicitly (e.g., in legacy mocks), attempt it first
        if client is not None:
            try:
                if hasattr(client, "aio") and hasattr(client.aio, "models"):
                    return await client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                        **kwargs,
                    )
                return await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config,
                    **kwargs,
                )
            except Exception as exc:
                if not is_resource_exhausted(exc):
                    raise
                logger.warning(
                    "Provided static client encountered 429 RESOURCE_EXHAUSTED. "
                    "Falling back to KeyManager rotation pool..."
                )

        pool_size = len(self.key_manager)
        if pool_size == 0:
            raise AllKeysExhaustedError(
                "No Gemini API keys configured in the rotation pool to execute request."
            )

        last_error: Optional[Exception] = None

        for attempt in range(pool_size):
            key = self.key_manager.get_current_key()
            active_client = self.key_manager.get_client(key)
            masked = mask_api_key(key)

            try:
                if hasattr(active_client, "aio") and hasattr(active_client.aio, "models"):
                    response = await active_client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                        **kwargs,
                    )
                else:
                    response = await asyncio.to_thread(
                        active_client.models.generate_content,
                        model=model,
                        contents=contents,
                        config=config,
                        **kwargs,
                    )
                return response
            except Exception as exc:
                if is_resource_exhausted(exc):
                    last_error = exc
                    logger.warning(
                        "Gemini API key [%s] hit 429 RESOURCE_EXHAUSTED (attempt %d/%d). "
                        "Rotating key and retrying instantly...",
                        masked,
                        attempt + 1,
                        pool_size,
                    )
                    self.key_manager.rotate_key()
                    continue

                logger.error("Non-retriable Gemini error on key [%s]: %s", masked, exc)
                raise

        logger.critical(
            "All %d Gemini API keys in rotation pool failed with 429 RESOURCE_EXHAUSTED.",
            pool_size,
        )
        raise AllKeysExhaustedError(
            f"All {pool_size} Gemini API key(s) exhausted by rate limits "
            f"(HTTP 429 RESOURCE_EXHAUSTED). Last error: {last_error}"
        ) from last_error

    def generate_content_sync(
        self,
        model: str,
        contents: Any,
        config: Optional[types.GenerateContentConfig] = None,
        client: Optional[genai.Client] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute synchronous generate_content with round-robin retry on 429 errors.

        Args:
            model: Gemini model name or path string.
            contents: Generation content payload.
            config: Optional generation configuration.
            client: Optional static client to attempt first.
            **kwargs: Extra parameters passed to the underlying client call.

        Returns:
            Any: The GenerateContentResponse returned by Google GenAI.

        Raises:
            AllKeysExhaustedError: If all keys in the rotation pool return 429.
            Exception: If an unhandled non-retriable error occurs.
        """
        if client is not None:
            try:
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                    **kwargs,
                )
            except Exception as exc:
                if not is_resource_exhausted(exc):
                    raise
                logger.warning(
                    "Provided static client encountered 429 RESOURCE_EXHAUSTED. "
                    "Falling back to KeyManager rotation pool..."
                )

        pool_size = len(self.key_manager)
        if pool_size == 0:
            raise AllKeysExhaustedError(
                "No Gemini API keys configured in the rotation pool to execute request."
            )

        last_error: Optional[Exception] = None

        for attempt in range(pool_size):
            key = self.key_manager.get_current_key()
            active_client = self.key_manager.get_client(key)
            masked = mask_api_key(key)

            try:
                return active_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                    **kwargs,
                )
            except Exception as exc:
                if is_resource_exhausted(exc):
                    last_error = exc
                    logger.warning(
                        "Gemini API key [%s] hit 429 RESOURCE_EXHAUSTED (attempt %d/%d). "
                        "Rotating key and retrying instantly...",
                        masked,
                        attempt + 1,
                        pool_size,
                    )
                    self.key_manager.rotate_key()
                    continue

                logger.error("Non-retriable Gemini error on key [%s]: %s", masked, exc)
                raise

        logger.critical(
            "All %d Gemini API keys in rotation pool failed with 429 RESOURCE_EXHAUSTED.",
            pool_size,
        )
        raise AllKeysExhaustedError(
            f"All {pool_size} Gemini API key(s) exhausted by rate limits "
            f"(HTTP 429 RESOURCE_EXHAUSTED). Last error: {last_error}"
        ) from last_error


# Default singleton instances for process-wide sharing
default_key_manager = KeyManager()
default_router = LLMRouter(key_manager=default_key_manager)

# Convenience aliases
generate_content = default_router.generate_content
generate_content_sync = default_router.generate_content_sync
get_client = default_key_manager.get_client
rotate = default_key_manager.rotate_key
