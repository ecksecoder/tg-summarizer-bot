"""AI Context Hub — Enterprise Multi-Modal Telegram Assistant.

Asynchronous content ingestion, native Google AI web summarization,
document/audio parsing, and resilient multi-model LLM summarization powered
by aiogram 3.x and SQLite WAL.
"""

import asyncio
import ipaddress
import logging
import os
import re
import sqlite3
import tempfile
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader

from config import (
    ALLOWED_USER_IDS,
    BOT_TOKEN,
    DB_PATH,
    DEFAULT_GEMINI_MODEL,
    GEMINI_API_KEYS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    SLIDING_WINDOW_RAW_LIMIT,
    TEMP_DIR,
)
from llm_router import (
    FORMAT_INSTRUCTIONS,
    UNIFIED_EXECUTIVE_SYSTEM_PROMPT,
    _local_extractive_fallback,
    _smart_compress_text,
    default_router,
    generate_executive_summary,
    get_client,
    is_resource_exhausted,
    process_audio_with_gemini as router_process_audio,
    rotate,
    summarize_text,
    summarize_with_gemini,
)
from locales import (
    FORMAT_BADGES,
    detect_locale,
    get_button_variants,
    get_format_badge,
    get_language_inline_kb,
    get_text,
)
from parser import LinkSummaryResponse, parse_and_summarize_url

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("bot_audit.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("AIContextHub")

PRIMARY_PRIMARY_GEMINI_MODEL = DEFAULT_GEMINI_MODEL

if not BOT_TOKEN or not GEMINI_API_KEYS:
    logger.critical("CONFIGURATION ERROR: BOT_TOKEN or GEMINI_API_KEYS not set in environment or .env!")
    raise SystemExit("Fill in BOT_TOKEN and GEMINI_API_KEYS in .env before running.")

ai_client = get_client()

SAFETY_OFF = [
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
]


def is_user_authorized(user_id: int) -> bool:
    """Verify whether the Telegram user ID is permitted by the RBAC allowlist."""
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def sanitize_filename(filename: str) -> str:
    """Strip unsafe path traversal characters and normalize uploaded filenames."""
    clean = re.sub(r'[\\/*?:"<>|]', "_", os.path.basename(filename))
    return clean[:120] or "document.bin"


def is_safe_url(url: str) -> tuple[bool, str]:
    """Validate target URL against SSRF vectors (loopback, private, and reserved IPs)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, "Only HTTP and HTTPS protocols are allowed."

        hostname = (parsed.hostname or "").lower()
        if not hostname or hostname in ("localhost", "0.0.0.0", "127.0.0.1", "::1"):
            return False, "Access to localhost addresses is restricted by security policy."

        path_lower = parsed.path.lower()
        dangerous_exts = (".exe", ".bat", ".cmd", ".sh", ".scr", ".pif", ".msi", ".jar", ".vbs")
        if any(path_lower.endswith(ext) for ext in dangerous_exts):
            return False, "Downloading executable files via links is forbidden."

        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, "Internal and private network addresses are restricted (Anti-SSRF)."
        except ValueError:
            if hostname.endswith((".local", ".internal", ".lan", ".localhost")):
                return False, "Access to internal local domain names is prohibited."

        return True, "OK"
    except Exception as e:
        return False, f"Malformed URL format: {e}"


from database import (
    _CursorContext,
    _DBConnectionContext,
    _SQLiteCompat,
    aiosqlite,
    get_user_language,
    init_db,
    is_user_onboarded,
    set_user_language,
)


async def resolve_user_locale(user_id: int, telegram_lang: Optional[str] = None) -> tuple[str, bool]:
    """Resolve active language and auto-detection flag for user.

    Checks DB:
    - If DB language is set (e.g. 'en', 'uk', 'ru'), returns (lang, False).
    - If DB language is NULL, returns (detect_locale(telegram_lang), True).

    Args:
        user_id: Telegram user ID.
        telegram_lang: Language code from Telegram client profile.

    Returns:
        tuple[str, bool]: (resolved_language_code, is_auto)
    """
    db_lang = await get_user_language(user_id)
    if db_lang:
        return db_lang, False
    return detect_locale(telegram_lang), True


async def resolve_user_lang(user_id: int, telegram_lang: Optional[str] = None) -> str:
    """Resolve active language for user, checking database first with fallback to Telegram locale."""
    lang, _ = await resolve_user_locale(user_id, telegram_lang)
    return lang


async def ensure_user_settings(user_id: int, lang: Optional[str] = None) -> tuple[int, str]:
    """Retrieve or initialize default active folder, summary format, and language for a user."""
    resolved_lang = lang or "en"
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT active_folder_id, default_format, language FROM user_settings WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()

        default_format = row[1] if row and row[1] else "MEDIUM"
        active_folder_id = row[0] if row else None
        current_lang = row[2] if row and len(row) > 2 and row[2] else None

        if active_folder_id:
            async with db.execute(
                "SELECT id FROM folders WHERE id = ? AND user_id = ?",
                (active_folder_id, user_id),
            ) as cur:
                if not await cur.fetchone():
                    active_folder_id = None

        if not active_folder_id:
            async with db.execute(
                "SELECT id FROM folders WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                (user_id,),
            ) as cur:
                f_row = await cur.fetchone()
            if f_row:
                active_folder_id = f_row[0]
            else:
                now = datetime.now().strftime("%Y-%m-%d %H:%M")
                effective_lang = current_lang or resolved_lang
                inbox_name = get_text("default_inbox_name", effective_lang)
                async with db.execute(
                    "INSERT INTO folders (user_id, name, created_at) VALUES (?, ?, ?)",
                    (user_id, inbox_name, now),
                ) as cur:
                    active_folder_id = cur.lastrowid

        if row:
            await db.execute(
                """
                UPDATE user_settings
                SET active_folder_id = ?, default_format = ?
                WHERE user_id = ?
                """,
                (active_folder_id, default_format, user_id),
            )
        else:
            await db.execute(
                """
                INSERT INTO user_settings (user_id, active_folder_id, default_format, language)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, active_folder_id, default_format, current_lang),
            )
        await db.commit()
        return active_folder_id, default_format


async def set_user_active_folder(user_id: int, folder_id: int):
    await ensure_user_settings(user_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE user_settings SET active_folder_id = ? WHERE user_id = ?",
            (folder_id, user_id),
        )
        await db.commit()


async def set_user_default_format(user_id: int, fmt: str):
    await ensure_user_settings(user_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE user_settings SET default_format = ? WHERE user_id = ?",
            (fmt, user_id),
        )
        await db.commit()


async def prune_old_raw_texts(user_id: int):
    """Enforce sliding-window storage hygiene by pruning old raw_text payloads."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"""
            UPDATE summaries
            SET raw_text = NULL
            WHERE id IN (
                SELECT s.id
                FROM summaries s
                JOIN folders f ON s.folder_id = f.id
                WHERE f.user_id = ? AND s.raw_text IS NOT NULL
                ORDER BY s.id DESC
                LIMIT -1 OFFSET {SLIDING_WINDOW_RAW_LIMIT}
            )
            """,
            (user_id,),
        )
        await db.commit()


def clean_extracted_text(text: str) -> str:
    """Normalize whitespace and strip redundant blank lines from extracted text."""
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_extracted_text("\n".join(pages))


def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_data:
                paragraphs.append(" | ".join(row_data))
    return clean_extracted_text("\n".join(paragraphs))


def extract_text_from_txt(file_path: str) -> str:
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                return clean_extracted_text(f.read())
        except UnicodeDecodeError:
            continue
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_extracted_text(f.read())


async def fetch_url_text(url: str, format_size: str = "MEDIUM") -> str:
    """Delegate URL extraction and summarization to local parser and LLM cascade."""
    result = await parse_and_summarize_url(
        url=url,
        format_size=format_size,
        model=PRIMARY_PRIMARY_GEMINI_MODEL,
    )
    return result.summary


FORMAT_LABELS = {
    "SHORT": "Cheat Sheet",
    "MEDIUM": "Structured Summary",
    "LONG": "In-Depth Retelling",
}

BASE_SYSTEM_PROMPT = UNIFIED_EXECUTIVE_SYSTEM_PROMPT


def _call_gemini_with_fallback(contents: Any, temperature: float = 0.25) -> str:
    """Execute LLM inference with automated key rotation, model cascade, and Groq fallback.

    Args:
        contents: Generation payload (string, list, or Part objects).
        temperature: Sampling temperature for generation.

    Returns:
        str: Generated content string.

    Raises:
        RuntimeError: If all cascade options and failovers fail.
    """
    try:
        response = default_router.generate_content_sync(
            contents=contents,
            temperature=temperature,
        )
        return str(response).strip()
    except Exception as exc:
        logger.error("[LLM Cascade] Synchronous generation error: %s", exc)
        raise RuntimeError(f"All LLM cascade tiers failed: {exc}") from exc


def process_audio_with_gemini(
    file_path: str,
    mime_type: str,
    source_title: str,
    format_size: str = "MEDIUM",
    target_lang: str = "en",
    lang: Optional[str] = None,
) -> str:
    """Upload audio stream via File API, generate summary in target_lang, and clean up remote state."""
    effective_lang = target_lang or lang or "en"
    return router_process_audio(
        file_path=file_path,
        mime_type=mime_type,
        source_title=source_title,
        format_size=format_size,
        target_lang=effective_lang,
        client=ai_client,
        router=default_router,
    )


USER_QUEUES: dict[int, list[dict]] = {}
ACTIVE_WORKERS: dict[int, bool] = {}


def review_inline_kb(
    summary_id: int,
    current_format: str = "MEDIUM",
    lang: str = "en",
) -> InlineKeyboardMarkup:
    """Build inline action and format-switching keyboard for completed summaries."""
    fmt_buttons = []
    for fmt_code in ("SHORT", "MEDIUM", "LONG"):
        label = get_format_badge(fmt_code, lang)
        prefix = "• " if fmt_code == current_format else ""
        fmt_buttons.append(
            InlineKeyboardButton(
                text=f"{prefix}{label}",
                callback_data=f"rev_fmt_{summary_id}_{fmt_code}",
            )
        )

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=get_text("btn_keep", lang), callback_data=f"rev_ok_{summary_id}"),
                InlineKeyboardButton(text=get_text("btn_regenerate", lang), callback_data=f"rev_regen_{summary_id}"),
                InlineKeyboardButton(text=get_text("btn_delete", lang), callback_data=f"rev_del_{summary_id}"),
            ],
            fmt_buttons,
        ]
    )


async def send_smart_summary_with_review(
    bot: Bot,
    chat_id: int,
    header: str,
    summary_text: str,
    summary_id: int,
    format_size: str = "MEDIUM",
    lang: str = "en",
):
    """Deliver long summaries in paginated chunks and attach review controls."""
    full_msg = f"{header}\n\n{summary_text}"
    max_len = 3800
    kb = review_inline_kb(summary_id, format_size, lang)

    if len(full_msg) <= max_len:
        await bot.send_message(chat_id, full_msg, reply_markup=kb)
    else:
        chunks = [full_msg[i : i + max_len] for i in range(0, len(full_msg), max_len)]
        for idx, chunk in enumerate(chunks):
            is_last = idx == len(chunks) - 1
            chunk_msg = get_text("msg_part", lang, current=idx + 1, total=len(chunks), chunk=chunk)
            await bot.send_message(
                chat_id,
                chunk_msg,
                reply_markup=kb if is_last else None,
            )
            await asyncio.sleep(0.3)


async def process_user_queue(user_id: int, bot: Bot):
    """Background priority queue worker handling per-user ingestion tasks sequentially."""
    if ACTIVE_WORKERS.get(user_id):
        return

    ACTIVE_WORKERS[user_id] = True
    loop = asyncio.get_running_loop()

    try:
        while USER_QUEUES.get(user_id):
            USER_QUEUES[user_id].sort(key=lambda x: x["priority"])
            task = USER_QUEUES[user_id].pop(0)
            remaining = len(USER_QUEUES[user_id])

            chat_id = task["chat_id"]
            folder_id = task["folder_id"]
            source_type = task["source_type"]
            source_title = task["source_title"]
            format_size = task.get("format_size", "MEDIUM")
            # Worker locale isolation: fetch user language and auto flag from DB
            lang, _ = await resolve_user_locale(user_id, task.get("telegram_lang") or task.get("lang"))
            fmt_badge = get_format_badge(format_size, lang)

            status_msg = await bot.send_message(
                chat_id,
                get_text(
                    "status_processing",
                    lang,
                    source_title=source_title,
                    format_badge=fmt_badge,
                    remaining=remaining,
                ),
            )

            raw_text = None
            try:
                if source_type == "text":
                    raw_text = clean_extracted_text(task["raw_text"])
                    summary = await generate_executive_summary(
                        raw_text=raw_text,
                        source_title=source_title,
                        format_size=format_size,
                        retry_variation=False,
                        target_lang=lang,
                    )
                elif source_type == "url":
                    link_res = await parse_and_summarize_url(
                        url=task["url"],
                        format_size=format_size,
                        model=PRIMARY_PRIMARY_GEMINI_MODEL,
                        target_lang=lang,
                    )
                    summary = link_res.summary
                    raw_text = link_res.raw_text or link_res.summary
                    if link_res.title:
                        source_title = link_res.title
                elif source_type in ("pdf", "docx", "txt"):
                    file_path = task["file_path"]
                    if source_type == "pdf":
                        raw_text = await loop.run_in_executor(None, extract_text_from_pdf, file_path)
                    elif source_type == "docx":
                        raw_text = await loop.run_in_executor(None, extract_text_from_docx, file_path)
                    else:
                        raw_text = await loop.run_in_executor(None, extract_text_from_txt, file_path)

                    summary = await generate_executive_summary(
                        raw_text=raw_text,
                        source_title=source_title,
                        format_size=format_size,
                        retry_variation=False,
                        target_lang=lang,
                    )
                elif source_type == "audio":
                    summary = await loop.run_in_executor(
                        None,
                        process_audio_with_gemini,
                        task["file_path"],
                        task["mime_type"],
                        source_title,
                        format_size,
                        lang,
                    )
                    raw_text = summary
                else:
                    summary = get_text("error_unknown_format", lang)

                now = datetime.now().strftime("%Y-%m-%d %H:%M")
                async with aiosqlite.connect(DB_PATH) as db:
                    async with db.execute(
                        """
                        INSERT INTO summaries
                        (folder_id, source_type, source_title, summary_text, raw_text, format_size, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (folder_id, source_type, source_title, summary, raw_text, format_size, now),
                    ) as cursor:
                        summary_id = cursor.lastrowid
                    await db.commit()

                await prune_old_raw_texts(user_id)

                try:
                    await status_msg.delete()
                except Exception:
                    pass

                header = get_text("header_summary_saved", lang, format_badge=fmt_badge, source_title=source_title)
                await send_smart_summary_with_review(
                    bot, chat_id, header, summary, summary_id, format_size, lang
                )

            except Exception as e:
                logger.error(f"Error processing '{source_title}': {e}")
                await status_msg.edit_text(get_text("error_processing", lang, source_title=source_title))

            finally:
                file_path = task.get("file_path")
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass

            await asyncio.sleep(0.3)

    finally:
        ACTIVE_WORKERS[user_id] = False


router = Router()


class HubStates(StatesGroup):
    waiting_for_folder_name = State()
    waiting_for_input = State()


def main_menu_kb(lang: str = "en") -> ReplyKeyboardMarkup:
    """Generate dynamic persistent menu keyboard based on user language."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text=get_text("btn_my_folders", lang)),
                KeyboardButton(text=get_text("btn_create_folder", lang)),
            ],
            [
                KeyboardButton(text=get_text("btn_upload_content", lang)),
                KeyboardButton(text=get_text("btn_summary_format", lang)),
            ],
            [
                KeyboardButton(text=get_text("btn_lang", lang)),
            ],
        ],
        resize_keyboard=True,
    )


# Recognized button sets across languages for reliable filter matching
ALL_MENU_TEXTS = set()
for btn_k in (
    "btn_my_folders",
    "btn_create_folder",
    "btn_upload_content",
    "btn_summary_format",
    "btn_lang",
    "btn_language",
):
    ALL_MENU_TEXTS.update(get_button_variants(btn_k))


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    user_id = message.from_user.id
    if not is_user_authorized(user_id):
        lang, _ = await resolve_user_locale(user_id, message.from_user.language_code)
        await message.answer(get_text("unauthorized", lang))
        return

    saved_lang = await get_user_language(user_id)
    onboarded = await is_user_onboarded(user_id)
    if saved_lang is None and not onboarded:
        await state.clear()
        detected_lang = detect_locale(message.from_user.language_code)
        kb = get_language_inline_kb(
            active_lang=detected_lang,
            is_auto=True,
            prefix="lang_init",
        )
        await message.answer(
            "🌐 Choose interface language / Оберіть мову інтерфейсу / Выберите язык интерфейса:",
            reply_markup=kb,
        )
        return

    lang, _ = await resolve_user_locale(user_id, message.from_user.language_code)
    await state.clear()
    folder_id, fmt = await ensure_user_settings(user_id, lang=lang)
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT name FROM folders WHERE id = ?", (folder_id,)) as cur:
            row = await cur.fetchone()
    folder_name = row[0] if row else get_text("default_inbox_name", lang)
    fmt_badge = get_format_badge(fmt, lang)

    await message.answer(
        get_text("start_greeting", lang, folder_name=folder_name, format_badge=fmt_badge),
        reply_markup=main_menu_kb(lang),
    )


@router.callback_query(F.data.startswith("lang_init:"))
async def cb_lang_init(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    if not is_user_authorized(user_id):
        await callback.answer(get_text("unauthorized", "en"), show_alert=True)
        return

    choice = callback.data.split(":", 1)[1]
    is_auto = (choice == "auto")
    if is_auto:
        persisted_lang = None
        resolved_lang = detect_locale(callback.from_user.language_code)
    else:
        persisted_lang = choice if choice in ("en", "uk", "ru") else "en"
        resolved_lang = persisted_lang

    await state.clear()
    await set_user_language(user_id, persisted_lang)

    # Create initial default folder using localized title via get_text('default_inbox_name', resolved_lang)
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, name FROM folders WHERE user_id = ? ORDER BY id ASC LIMIT 1",
            (user_id,),
        ) as cur:
            existing = await cur.fetchone()

        if existing:
            folder_id, folder_name = existing[0], existing[1]
        else:
            inbox_name = get_text("default_inbox_name", resolved_lang)
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            async with db.execute(
                "INSERT INTO folders (user_id, name, created_at) VALUES (?, ?, ?)",
                (user_id, inbox_name, now),
            ) as cur:
                folder_id = cur.lastrowid
            await db.commit()
            folder_name = inbox_name

    await set_user_active_folder(user_id, folder_id)
    _, fmt = await ensure_user_settings(user_id, lang=resolved_lang)
    fmt_badge = get_format_badge(fmt, resolved_lang)

    try:
        await callback.message.delete()
    except Exception:
        pass

    greeting = get_text(
        "start_greeting",
        resolved_lang,
        folder_name=folder_name,
        format_badge=fmt_badge,
    )
    await callback.message.answer(
        greeting,
        reply_markup=main_menu_kb(resolved_lang),
    )
    await callback.answer()


@router.message(Command("lang"))
@router.message(F.text.in_(get_button_variants("btn_lang")))
async def cmd_language(message: Message):
    user_id = message.from_user.id
    if not is_user_authorized(user_id):
        lang, _ = await resolve_user_locale(user_id, message.from_user.language_code)
        await message.answer(get_text("unauthorized", lang))
        return

    active_lang, is_auto = await resolve_user_locale(user_id, message.from_user.language_code)
    kb = get_language_inline_kb(
        active_lang=active_lang,
        is_auto=is_auto,
        prefix="lang_set",
    )
    await message.answer(
        "🌐 Choose interface language / Оберіть мову інтерфейсу / Выберите язык интерфейса:",
        reply_markup=kb,
    )


@router.callback_query(F.data.startswith("lang_set:") | F.data.startswith("setlang:"))
async def cb_set_language(callback: CallbackQuery):
    user_id = callback.from_user.id
    if not is_user_authorized(user_id):
        await callback.answer(get_text("unauthorized", "en"), show_alert=True)
        return

    choice = callback.data.split(":", 1)[1]
    if choice == "auto":
        await set_user_language(user_id, None)
        active_lang = detect_locale(callback.from_user.language_code)
        is_auto = True
        confirmation = get_text("lang_switched_auto", active_lang)
    elif choice in ("en", "uk", "ru"):
        await set_user_language(user_id, choice)
        active_lang = choice
        is_auto = False
        confirmation = get_text("lang_switched", choice)
    else:
        await callback.answer()
        return

    # In-place inline markup update with updated checkmark and auto-button text
    new_kb = get_language_inline_kb(
        active_lang=active_lang,
        is_auto=is_auto,
        prefix="lang_set",
    )
    try:
        await callback.message.edit_reply_markup(reply_markup=new_kb)
    except Exception:
        pass

    # Refresh persistent ReplyKeyboardMarkup
    await callback.message.answer(
        confirmation,
        reply_markup=main_menu_kb(active_lang),
    )
    await callback.answer()


@router.message(F.text.in_(get_button_variants("btn_summary_format")))
async def menu_format_settings(message: Message):
    if not is_user_authorized(message.from_user.id):
        return
    lang = await resolve_user_lang(message.from_user.id, message.from_user.language_code)
    _, current_fmt = await ensure_user_settings(message.from_user.id, lang=lang)
    buttons = []
    for code in ("SHORT", "MEDIUM", "LONG"):
        label = get_format_badge(code, lang)
        mark = "• " if code == current_fmt else ""
        buttons.append([InlineKeyboardButton(text=f"{mark}{label}", callback_data=f"setfmt_{code}")])

    await message.answer(
        get_text("format_settings_prompt", lang),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("setfmt_"))
async def cb_set_default_format(callback: CallbackQuery):
    lang = await resolve_user_lang(callback.from_user.id, callback.from_user.language_code)
    fmt = callback.data.replace("setfmt_", "")
    if fmt in ("SHORT", "MEDIUM", "LONG"):
        await set_user_default_format(callback.from_user.id, fmt)
        badge = get_format_badge(fmt, lang)
        await callback.message.edit_text(
            get_text("format_switched", lang, format_badge=badge)
        )
    await callback.answer()


@router.message(F.text.in_(get_button_variants("btn_create_folder")))
async def start_new_folder(message: Message, state: FSMContext):
    if not is_user_authorized(message.from_user.id):
        return
    lang = await resolve_user_lang(message.from_user.id, message.from_user.language_code)
    await state.set_state(HubStates.waiting_for_folder_name)
    await message.answer(get_text("enter_folder_name", lang))


@router.message(HubStates.waiting_for_folder_name)
async def process_folder_name(message: Message, state: FSMContext):
    lang = await resolve_user_lang(message.from_user.id, message.from_user.language_code)
    folder_name = (message.text or "").strip()
    if not folder_name or folder_name in ALL_MENU_TEXTS:
        await message.answer(get_text("invalid_folder_name", lang))
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "INSERT INTO folders (user_id, name, created_at) VALUES (?, ?, ?)",
            (message.from_user.id, folder_name, now),
        ) as cur:
            new_folder_id = cur.lastrowid
        await db.commit()

    await set_user_active_folder(message.from_user.id, new_folder_id)
    await state.clear()
    await message.answer(
        get_text("folder_created", lang, folder_name=folder_name),
        reply_markup=main_menu_kb(lang),
    )


@router.message(F.text.in_(get_button_variants("btn_my_folders")))
async def open_folders_list(message: Message, state: FSMContext):
    if not is_user_authorized(message.from_user.id):
        return
    lang = await resolve_user_lang(message.from_user.id, message.from_user.language_code)
    await state.clear()
    active_id, _ = await ensure_user_settings(message.from_user.id, lang=lang)

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, name FROM folders WHERE user_id = ? ORDER BY id DESC",
            (message.from_user.id,),
        ) as cursor:
            folders = await cursor.fetchall()

    if not folders:
        await message.answer(get_text("no_folders", lang))
        return

    buttons = []
    for f_id, f_name in folders:
        pin = "[Active] " if f_id == active_id else ""
        buttons.append([InlineKeyboardButton(text=f"{pin}{f_name}", callback_data=f"folder_view_{f_id}")])

    await message.answer(
        get_text("folders_list_title", lang),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("folder_view_"))
async def view_folder_details(callback: CallbackQuery):
    lang = await resolve_user_lang(callback.from_user.id, callback.from_user.language_code)
    folder_id = int(callback.data.split("_")[-1])

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT name, created_at FROM folders WHERE id = ? AND user_id = ?",
            (folder_id, callback.from_user.id),
        ) as cursor:
            folder = await cursor.fetchone()

        if not folder:
            await callback.answer(get_text("folder_not_found", lang), show_alert=True)
            return

        async with db.execute(
            "SELECT COUNT(*) FROM summaries WHERE folder_id = ?", (folder_id,)
        ) as cursor:
            count = (await cursor.fetchone())[0]

    await set_user_active_folder(callback.from_user.id, folder_id)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=get_text("btn_upload_to_folder", lang), callback_data=f"upload_to_{folder_id}")],
            [InlineKeyboardButton(text=get_text("btn_export_folder", lang), callback_data=f"export_txt_{folder_id}")],
            [InlineKeyboardButton(text=get_text("btn_delete_folder", lang), callback_data=f"delete_folder_{folder_id}")],
        ]
    )

    await callback.message.edit_text(
        get_text(
            "folder_details",
            lang,
            folder_name=folder[0],
            created_at=folder[1],
            count=count,
        ),
        reply_markup=kb,
    )
    await callback.answer(get_text("folder_set_active", lang))


@router.callback_query(F.data.startswith("delete_folder_"))
async def confirm_delete_folder(callback: CallbackQuery):
    lang = await resolve_user_lang(callback.from_user.id, callback.from_user.language_code)
    folder_id = int(callback.data.split("_")[-1])
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=get_text("btn_confirm_delete", lang), callback_data=f"confirm_del_{folder_id}"),
                InlineKeyboardButton(text=get_text("btn_cancel", lang), callback_data=f"folder_view_{folder_id}"),
            ]
        ]
    )
    await callback.message.edit_text(
        get_text("confirm_delete_title", lang),
        reply_markup=kb,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("confirm_del_"))
async def execute_delete_folder(callback: CallbackQuery):
    lang = await resolve_user_lang(callback.from_user.id, callback.from_user.language_code)
    folder_id = int(callback.data.split("_")[-1])
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM summaries WHERE folder_id = ?", (folder_id,))
        await db.execute(
            "DELETE FROM folders WHERE id = ? AND user_id = ?",
            (folder_id, callback.from_user.id),
        )
        await db.commit()

    await ensure_user_settings(callback.from_user.id, lang=lang)
    await callback.message.edit_text(get_text("folder_deleted", lang))
    await callback.answer()


@router.callback_query(F.data.startswith("export_txt_"))
async def download_folder_archive(callback: CallbackQuery):
    lang = await resolve_user_lang(callback.from_user.id, callback.from_user.language_code)
    folder_id = int(callback.data.split("_")[-1])

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT name FROM folders WHERE id = ? AND user_id = ?",
            (folder_id, callback.from_user.id),
        ) as cursor:
            folder = await cursor.fetchone()

        if not folder:
            await callback.answer(get_text("folder_not_found", lang), show_alert=True)
            return

        async with db.execute(
            "SELECT source_type, source_title, summary_text, format_size, created_at "
            "FROM summaries WHERE folder_id = ? ORDER BY id ASC",
            (folder_id,),
        ) as cursor:
            items = await cursor.fetchall()

    if not items:
        await callback.answer(get_text("folder_archive_empty", lang), show_alert=True)
        return

    folder_name = folder[0]
    lines = [
        "==================================================",
        get_text("archive_header_project", lang, folder_name=folder_name),
        get_text("archive_header_exported", lang, date=datetime.now().strftime("%Y-%m-%d %H:%M")),
        get_text("archive_header_total", lang, count=len(items)),
        "==================================================\n",
    ]

    for idx, (s_type, s_title, s_text, s_fmt, s_date) in enumerate(items, 1):
        fmt_label = get_format_badge(s_fmt, lang)
        lines.append(
            get_text("archive_item_header", lang, idx=idx, type=s_type.upper(), format_badge=fmt_label, date=s_date)
        )
        lines.append(get_text("archive_source_label", lang, source=s_title) + "\n")
        lines.append(s_text)
        lines.append("\n" + "=" * 50 + "\n")

    safe_name = sanitize_filename(folder_name)
    export_path = os.path.join(TEMP_DIR, f"Archive_{safe_name}_{folder_id}.txt")

    with open(export_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    try:
        doc_file = FSInputFile(export_path, filename=f"{safe_name}_context.txt")
        await callback.message.answer_document(
            doc_file,
            caption=get_text("archive_caption", lang, folder_name=folder_name, count=len(items)),
        )
    finally:
        if os.path.exists(export_path):
            os.remove(export_path)

    await callback.answer()


@router.message(F.text.in_(get_button_variants("btn_upload_content")))
async def start_upload_flow(message: Message):
    if not is_user_authorized(message.from_user.id):
        return
    lang = await resolve_user_lang(message.from_user.id, message.from_user.language_code)
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, name FROM folders WHERE user_id = ? ORDER BY id DESC",
            (message.from_user.id,),
        ) as cursor:
            folders = await cursor.fetchall()

    if not folders:
        await message.answer(get_text("need_create_folder_first", lang))
        return

    buttons = [
        [InlineKeyboardButton(text=f_name, callback_data=f"upload_to_{f_id}")]
        for f_id, f_name in folders
    ]
    await message.answer(
        get_text("select_upload_folder", lang),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("upload_to_"))
async def select_folder_for_upload(callback: CallbackQuery, state: FSMContext):
    lang = await resolve_user_lang(callback.from_user.id, callback.from_user.language_code)
    folder_id = int(callback.data.split("_")[-1])
    await set_user_active_folder(callback.from_user.id, folder_id)
    await state.set_state(HubStates.waiting_for_input)
    await state.update_data(active_folder_id=folder_id)

    await callback.message.edit_text(get_text("upload_folder_activated", lang))
    await callback.answer()


@router.callback_query(F.data.startswith("rev_ok_"))
async def review_ok_handler(callback: CallbackQuery):
    lang = await resolve_user_lang(callback.from_user.id, callback.from_user.language_code)
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer(get_text("review_keep_alert", lang))


@router.callback_query(F.data.startswith("rev_del_"))
async def review_delete_handler(callback: CallbackQuery):
    lang = await resolve_user_lang(callback.from_user.id, callback.from_user.language_code)
    summary_id = int(callback.data.split("_")[-1])
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM summaries WHERE id = ?", (summary_id,))
        await db.commit()

    try:
        await callback.message.delete()
    except Exception:
        await callback.message.edit_text(get_text("review_del_msg", lang))
    await callback.answer(get_text("review_del_alert", lang))


@router.callback_query(F.data.startswith("rev_regen_") | F.data.startswith("rev_fmt_"))
async def review_regenerate_or_change_format(callback: CallbackQuery):
    """Re-synthesize summary in a new format or variation using cached source text."""
    lang = await resolve_user_lang(callback.from_user.id, callback.from_user.language_code)
    data = callback.data
    if data.startswith("rev_regen_"):
        summary_id = int(data.replace("rev_regen_", ""))
        target_format = None
        is_variation = True
    else:
        parts = data.split("_")
        summary_id = int(parts[2])
        target_format = parts[3]
        is_variation = False

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT source_title, summary_text, raw_text, format_size FROM summaries WHERE id = ?",
            (summary_id,),
        ) as cur:
            row = await cur.fetchone()

    if not row:
        await callback.answer(get_text("review_not_found", lang), show_alert=True)
        return

    source_title, current_summary, raw_text, current_fmt = row
    current_fmt = current_fmt or "MEDIUM"
    new_fmt = target_format or current_fmt

    size_rank = {"SHORT": 1, "MEDIUM": 2, "LONG": 3}

    if not raw_text:
        if size_rank.get(new_fmt, 2) > size_rank.get(current_fmt, 2):
            await callback.answer(
                get_text("review_pruned_warning", lang),
                show_alert=True,
            )
            return
        base_material = current_summary
    else:
        base_material = raw_text

    fmt_badge = get_format_badge(new_fmt, lang)
    await callback.answer(get_text("review_rebuilding_alert", lang, format_badge=fmt_badge))
    await callback.message.edit_text(
        get_text("review_rebuilding_msg", lang, source_title=source_title, format_badge=fmt_badge)
    )

    try:
        new_summary = await generate_executive_summary(
            raw_text=base_material,
            source_title=source_title,
            format_size=new_fmt,
            retry_variation=is_variation,
            target_lang=lang,
        )

        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE summaries SET summary_text = ?, format_size = ? WHERE id = ?",
                (new_summary, new_fmt, summary_id),
            )
            await db.commit()

        header = get_text("header_summary_updated", lang, format_badge=fmt_badge, source_title=source_title)
        full_msg = f"{header}\n\n{new_summary}"
        await callback.message.edit_text(
            full_msg[:3900],
            reply_markup=review_inline_kb(summary_id, new_fmt, lang),
        )
    except Exception as e:
        logger.error(f"Error regenerating record #{summary_id}: {e}")
        await callback.message.edit_text(get_text("review_rebuild_error", lang))


@router.message()
async def enqueue_incoming_material(message: Message, state: FSMContext, bot: Bot):
    """Zero-click universal handler for documents, voice/audio, URLs, and raw notes."""
    user_id = message.from_user.id
    lang, _ = await resolve_user_locale(user_id, message.from_user.language_code)
    if not is_user_authorized(user_id):
        await message.answer(get_text("unauthorized", lang))
        return

    data = await state.get_data()
    default_folder_id, default_format = await ensure_user_settings(user_id, lang=lang)
    folder_id = data.get("active_folder_id") or default_folder_id

    if user_id not in USER_QUEUES:
        USER_QUEUES[user_id] = []

    new_tasks = []
    telegram_lang = message.from_user.language_code

    if message.document:
        doc = message.document
        if doc.file_size and doc.file_size > MAX_FILE_SIZE_BYTES:
            await message.answer(
                get_text("file_limit_exceeded", lang, file_name=doc.file_name or "file", max_mb=MAX_FILE_SIZE_MB)
            )
            return

        file_name = sanitize_filename(doc.file_name or "document")
        ext = os.path.splitext(file_name)[1].lower()
        supported_txt = (".txt", ".md", ".py", ".csv", ".log", ".json", ".ini", ".cfg", ".gsc")

        if ext not in (".pdf", ".docx") + supported_txt:
            await message.answer(get_text("unsupported_format", lang, ext=ext))
            return

        local_path = os.path.join(TEMP_DIR, f"{user_id}_{doc.file_unique_id}_{file_name}")
        await bot.download(doc, destination=local_path)

        s_type = "pdf" if ext == ".pdf" else ("docx" if ext == ".docx" else "txt")
        priority = 1 if s_type == "txt" else 2
        new_tasks.append({
            "priority": priority,
            "chat_id": message.chat.id,
            "folder_id": folder_id,
            "source_type": s_type,
            "source_title": get_text("title_file", lang, file_name=file_name),
            "file_path": local_path,
            "format_size": default_format,
            "lang": lang,
            "telegram_lang": telegram_lang,
        })

    elif message.voice or message.audio:
        media = message.voice or message.audio
        if media.file_size and media.file_size > MAX_FILE_SIZE_BYTES:
            await message.answer(get_text("audio_limit_exceeded", lang, max_mb=MAX_FILE_SIZE_MB))
            return

        ext = ".ogg" if message.voice else ".mp3"
        mime = media.mime_type or ("audio/ogg" if message.voice else "audio/mpeg")
        if message.voice:
            title = get_text("title_voice", lang, duration=media.duration)
        else:
            audio_name = sanitize_filename(getattr(media, "file_name", None) or "audio.mp3")
            title = get_text("title_audio", lang, file_name=audio_name)

        local_path = os.path.join(TEMP_DIR, f"{user_id}_{media.file_unique_id}{ext}")
        await bot.download(media, destination=local_path)

        new_tasks.append({
            "priority": 4,
            "chat_id": message.chat.id,
            "folder_id": folder_id,
            "source_type": "audio",
            "source_title": title,
            "file_path": local_path,
            "mime_type": mime,
            "format_size": default_format,
            "lang": lang,
            "telegram_lang": telegram_lang,
        })

    elif message.text or message.caption:
        text = (message.text or message.caption or "").strip()
        urls = re.findall(r"https?://[^\s<>\"']+", text)

        if urls:
            for url in urls:
                new_tasks.append({
                    "priority": 3,
                    "chat_id": message.chat.id,
                    "folder_id": folder_id,
                    "source_type": "url",
                    "source_title": url,
                    "url": url,
                    "format_size": default_format,
                    "lang": lang,
                    "telegram_lang": telegram_lang,
                })
        elif len(text) >= 5:
            short_title = text[:40].replace("\n", " ") + ("..." if len(text) > 40 else "")
            new_tasks.append({
                "priority": 1,
                "chat_id": message.chat.id,
                "folder_id": folder_id,
                "source_type": "text",
                "source_title": get_text("title_note", lang, short_title=short_title),
                "raw_text": text,
                "format_size": default_format,
                "lang": lang,
                "telegram_lang": telegram_lang,
            })

    if not new_tasks:
        return

    USER_QUEUES[user_id].extend(new_tasks)
    asyncio.create_task(process_user_queue(user_id, bot))


async def main():
    await init_db()
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    logger.info("AI Context Hub Enterprise engine running.")
    # Flush backlog of stale Telegram updates immediately before polling
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot process terminated gracefully.")
