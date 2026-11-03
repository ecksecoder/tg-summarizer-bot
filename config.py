"""Application configuration and environment variable parsing module.

Provides centralized, validated configuration parameters parsed from environment
variables with robust multi-key support for Google Gemini API rotation.
"""

import os
import sys
import tempfile
from typing import Optional

from dotenv import load_dotenv

# Ensure local .env file values are loaded into the process environment
load_dotenv()


def parse_gemini_api_keys(raw_val: Optional[str] = None) -> list[str]:
    """Parse comma-separated Gemini API keys into a list of clean strings.

    Extracts keys from GEMINI_API_KEYS with a backward-compatible fallback to
    the legacy GEMINI_API_KEY variable if the plural variable is absent.

    Args:
        raw_val: Optional raw key string to parse. If None, retrieves from
            environment variables.

    Returns:
        list[str]: Cleaned, non-empty Gemini API keys.
    """
    if raw_val is None:
        raw_val = os.getenv("GEMINI_API_KEYS", "").strip()
        if not raw_val:
            raw_val = os.getenv("GEMINI_API_KEY", "").strip()

    if not raw_val:
        return []

    keys: list[str] = []
    for item in raw_val.split(","):
        cleaned = item.strip().strip('"').strip("'")
        if cleaned:
            keys.append(cleaned)

    return keys


def parse_allowed_user_ids(raw_val: Optional[str] = None) -> set[int]:
    """Parse comma-separated Telegram user IDs into a set of integers.

    Args:
        raw_val: Optional raw user ID string to parse. If None, retrieves from
            ALLOWED_USER_IDS environment variable.

    Returns:
        set[int]: Set of permitted numeric user IDs.
    """
    if raw_val is None:
        raw_val = os.getenv("ALLOWED_USER_IDS", "").strip()

    if not raw_val:
        return set()

    user_ids: set[int] = set()
    for token in raw_val.split(","):
        cleaned = token.strip().strip('"').strip("'")
        if cleaned.isdigit():
            user_ids.add(int(cleaned))

    return user_ids


# Core authentication and integration credentials
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEYS: list[str] = parse_gemini_api_keys()
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
ADMIN_ID: str = os.getenv("ADMIN_ID", "").strip()

# Access control allowlist
ALLOWED_USER_IDS: set[int] = parse_allowed_user_ids()

# Model preferences
DEFAULT_GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

# System operational limits and paths
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
SLIDING_WINDOW_RAW_LIMIT: int = int(os.getenv("SLIDING_WINDOW_RAW_LIMIT", "10"))
DB_PATH: str = os.getenv("DB_PATH", "context_hub.db")
TEMP_DIR: str = os.path.join(tempfile.gettempdir(), "ai_context_hub_temp")
os.makedirs(TEMP_DIR, exist_ok=True)


def validate_config(raise_on_error: bool = False) -> bool:
    """Validate presence and integrity of essential configuration parameters.

    Args:
        raise_on_error: Whether to raise ValueError instead of returning False.

    Returns:
        bool: True if configuration is valid.

    Raises:
        ValueError: If essential parameters are missing and raise_on_error is True.
    """
    missing: list[str] = []
    if not BOT_TOKEN:
        missing.append("BOT_TOKEN")
    if not GEMINI_API_KEYS:
        missing.append("GEMINI_API_KEYS")

    if missing:
        msg = f"Missing required configuration variables: {', '.join(missing)}"
        if raise_on_error:
            raise ValueError(msg)
        return False

    return True
