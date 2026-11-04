"""
AI Context Hub — Enterprise Multi-Modal Telegram Assistant.

Asynchronous content ingestion, native Google AI web summarization,
document/audio parsing, and resilient multi-model LLM summarization powered
by aiogram 3.x and SQLite WAL.
"""

import os
import re
import asyncio
import logging
import sqlite3
import tempfile
import ipaddress
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import (
    Message,
    CallbackQuery,
    FSInputFile,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

from pypdf import PdfReader
from docx import Document

from google import genai
from google.genai import types

from parser import parse_and_summarize_url, LinkSummaryResponse


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
    FORMAT_LABELS,
    UNIFIED_EXECUTIVE_SYSTEM_PROMPT,
    _local_extractive_fallback,
    _smart_compress_text,
    default_router,
    generate_executive_summary,
    get_client,
    is_resource_exhausted,
    rotate,
    summarize_with_gemini,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("bot_audit.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AIContextHub")

PRIMARY_PRIMARY_GEMINI_MODEL = DEFAULT_GEMINI_MODEL

if not BOT_TOKEN or not GEMINI_API_KEYS:
    logger.critical("ОШИБКА КОНФИГУРАЦИИ: Не заданы BOT_TOKEN или GEMINI_API_KEYS в файле .env!")
    raise SystemExit("Заполните BOT_TOKEN и GEMINI_API_KEYS в файле .env перед запуском.")

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
            return False, "Разрешены только защищенные протоколы HTTP и HTTPS."

        hostname = (parsed.hostname or "").lower()
        if not hostname or hostname in ("localhost", "0.0.0.0", "127.0.0.1", "::1"):
            return False, "Доступ к локальным адресам заблокирован политикой безопасности."

        path_lower = parsed.path.lower()
        dangerous_exts = (".exe", ".bat", ".cmd", ".sh", ".scr", ".pif", ".msi", ".jar", ".vbs")
        if any(path_lower.endswith(ext) for ext in dangerous_exts):
            return False, "Скачивание исполняемых файлов по ссылкам запрещено."

        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, "Внутренние и приватные IP-адреса заблокированы (Anti-SSRF)."
        except ValueError:
            if hostname.endswith((".local", ".internal", ".lan", ".localhost")):
                return False, "Запрещены обращения к внутренним локальным доменам."

        return True, "OK"
    except Exception as e:
        return False, f"Некорректный формат URL: {e}"


class _CursorContext:
    def __init__(self, conn, sql, params):
        self._conn, self._sql, self._params, self._cursor = conn, sql, params, None
    async def _run(self):
        self._cursor = await asyncio.to_thread(self._conn.execute, self._sql, self._params)
        return self
    def __await__(self):
        return self._run().__await__()
    async def __aenter__(self):
        return await self._run()
    async def __aexit__(self, t, v, tb):
        if self._cursor: await asyncio.to_thread(self._cursor.close)
    async def fetchone(self):
        return await asyncio.to_thread(self._cursor.fetchone)
    async def fetchall(self):
        return await asyncio.to_thread(self._cursor.fetchall)
    @property
    def lastrowid(self):
        return self._cursor.lastrowid if self._cursor else None


class _DBConnectionContext:
    """Async SQLite connection manager enforcing WAL mode and foreign keys."""
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn = None

    def _open_and_configure(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    async def __aenter__(self):
        self._conn = await asyncio.to_thread(self._open_and_configure)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._conn:
            await asyncio.to_thread(self._conn.close)

    def execute(self, sql: str, params: tuple = ()):
        return _CursorContext(self._conn, sql, params)

    async def commit(self):
        await asyncio.to_thread(self._conn.commit)


class _SQLiteCompat:
    def connect(self, db_path: str):
        return _DBConnectionContext(db_path)


aiosqlite = _SQLiteCompat()


async def init_db():
    """Initialize SQLite WAL schema and run idempotent column migrations."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_id INTEGER NOT NULL,
                source_type TEXT NOT NULL,
                source_title TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                raw_text TEXT DEFAULT NULL,
                format_size TEXT DEFAULT 'MEDIUM',
                created_at TEXT NOT NULL,
                FOREIGN KEY (folder_id) REFERENCES folders (id) ON DELETE CASCADE
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                active_folder_id INTEGER DEFAULT NULL,
                default_format TEXT DEFAULT 'MEDIUM'
            )
        """)

        async with db.execute("PRAGMA table_info(summaries)") as cursor:
            cols = [row[1] for row in await cursor.fetchall()]
        for col_name, col_def in (
            ("folder_id", "INTEGER NOT NULL DEFAULT 1"),
            ("source_type", "TEXT NOT NULL DEFAULT 'text'"),
            ("source_title", "TEXT NOT NULL DEFAULT 'Материал'"),
            ("summary_text", "TEXT NOT NULL DEFAULT ''"),
            ("raw_text", "TEXT DEFAULT NULL"),
            ("format_size", "TEXT DEFAULT 'MEDIUM'"),
            ("created_at", "TEXT NOT NULL DEFAULT ''"),
        ):
            if col_name not in cols:
                await db.execute(f"ALTER TABLE summaries ADD COLUMN {col_name} {col_def}")
                logger.info(f"Миграция БД: добавлена колонка {col_name} в таблицу summaries.")

        await db.commit()
    logger.info("База данных SQLite (WAL) успешно инициализирована.")


async def ensure_user_settings(user_id: int) -> tuple[int, str]:
    """Retrieve or initialize default active folder and summary format for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT active_folder_id, default_format FROM user_settings WHERE user_id = ?",
            (user_id,)
        ) as cur:
            row = await cur.fetchone()

        default_format = row[1] if row and row[1] else "MEDIUM"
        active_folder_id = row[0] if row else None

        if active_folder_id:
            async with db.execute(
                "SELECT id FROM folders WHERE id = ? AND user_id = ?",
                (active_folder_id, user_id)
            ) as cur:
                if not await cur.fetchone():
                    active_folder_id = None

        if not active_folder_id:
            async with db.execute(
                "SELECT id FROM folders WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                (user_id,)
            ) as cur:
                f_row = await cur.fetchone()
            if f_row:
                active_folder_id = f_row[0]
            else:
                now = datetime.now().strftime("%Y-%m-%d %H:%M")
                async with db.execute(
                    "INSERT INTO folders (user_id, name, created_at) VALUES (?, ?, ?)",
                    (user_id, "📥 Быстрый входящий", now)
                ) as cur:
                    active_folder_id = cur.lastrowid

        await db.execute("""
            INSERT INTO user_settings (user_id, active_folder_id, default_format)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                active_folder_id = excluded.active_folder_id,
                default_format = excluded.default_format
        """, (user_id, active_folder_id, default_format))
        await db.commit()
        return active_folder_id, default_format


async def set_user_active_folder(user_id: int, folder_id: int):
    await ensure_user_settings(user_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE user_settings SET active_folder_id = ? WHERE user_id = ?",
            (folder_id, user_id)
        )
        await db.commit()


async def set_user_default_format(user_id: int, fmt: str):
    await ensure_user_settings(user_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE user_settings SET default_format = ? WHERE user_id = ?",
            (fmt, user_id)
        )
        await db.commit()


async def prune_old_raw_texts(user_id: int):
    """Enforce sliding-window storage hygiene by pruning old raw_text payloads."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"""
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
        """, (user_id,))
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
    "SHORT": "⚡ Шпаргалка",
    "MEDIUM": "📋 Конспект",
    "LONG": "📚 Пересказ",
}

FORMAT_INSTRUCTIONS = {
    "SHORT": (
        "ФОРМАТ ВЫВОДА: ⚡ ШПАРГАЛКА (МАКСИМАЛЬНО СЖАТО).\n"
        "- Объём: строго 5–10 самых главных пунктов или цифр на один экран.\n"
        "- Только сухая выжимка: ключевые события, выводы и факты без воды."
    ),
    "MEDIUM": (
        "ФОРМАТ ВЫВОДА: 📋 СТРУКТУРИРОВАННЫЙ КОНСПЕКТ (СРЕДНИЙ ОБЪЁМ).\n"
        "- Раздели материал на логические блоки с подзаголовками.\n"
        "- Сохрани все важные факты, сюжетные линии, параметры и причинно-следственные связи."
    ),
    "LONG": (
        "ФОРМАТ ВЫВОДА: 📚 ПОЛНЫЙ ДЕТАЛЬНЫЙ ПЕРЕСКАЗ (МАКСИМАЛЬНАЯ ГЛУБИНА).\n"
        "- Подробно разбери каждый раздел или главу в хронологическом порядке.\n"
        "- Сохрани все примеры, тонкости, героев и нюансы так, чтобы чтение пересказа "
        "полностью заменяло чтение оригинала."
    ),
}

BASE_SYSTEM_PROMPT = UNIFIED_EXECUTIVE_SYSTEM_PROMPT


def _smart_compress_text(text: str, max_chars: int = 22000) -> str:
    """Execute _smart_compress_text workflow."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    num_chunks = 10
    chunk_size = max_chars // num_chunks
    step = max(1, (len(text) - chunk_size) // (num_chunks - 1))

    parts = []
    for i in range(num_chunks):
        start = i * step
        snippet = text[start : start + chunk_size].strip()
        pct = int((start / len(text)) * 100)
        parts.append(f"--- [Часть {i+1}/{num_chunks} (~{pct}% книги)] ---\n{snippet}")

    return (
        f"[Масштабный документ ({len(text)} симв.). Равномерная выборка ключевых глав:]\n\n"
        + "\n\n".join(parts)
    )


def _local_extractive_fallback(raw_text: str, source_title: str) -> str:
    """Execute _local_extractive_fallback workflow."""
    compressed = _smart_compress_text(raw_text, max_chars=6000)
    paragraphs = [p.strip() for p in compressed.split("\n") if len(p.strip()) > 60]
    selected = paragraphs[:18]
    bullets = "\n\n".join(f"• {p[:350]}..." if len(p) > 350 else f"• {p}" for p in selected)
    return (
        f"⚡ **Автономная структурная выжимка (Режим высокой нагрузки облака)**\n"
        f"📌 **Документ:** {source_title} (Объём оригинала: {len(raw_text)} симв.)\n\n"
        f"{bullets}"
    )


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
    format_size: str = "MEDIUM"
) -> str:
    """Upload audio stream via File API, generate summary, and clean up remote state."""
    uploaded_file = None
    try:
        fmt_key = format_size if format_size in FORMAT_INSTRUCTIONS else "MEDIUM"
        uploaded_file = ai_client.files.upload(
            file=file_path,
            config=types.UploadFileConfig(mime_type=mime_type)
        )
        prompt = (
            f"{BASE_SYSTEM_PROMPT}\n\n"
            f"{FORMAT_INSTRUCTIONS[fmt_key]}\n"
            f"Прослушай эту аудиозапись ({source_title}) и сделай точную выжимку."
        )
        return _call_gemini_with_fallback([uploaded_file, prompt], temperature=0.25)
    finally:
        if uploaded_file:
            try:
                ai_client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


USER_QUEUES: dict[int, list[dict]] = {}
ACTIVE_WORKERS: dict[int, bool] = {}


def review_inline_kb(summary_id: int, current_format: str = "MEDIUM") -> InlineKeyboardMarkup:
    """Build inline action and format-switching keyboard for completed summaries."""
    fmt_buttons = []
    for fmt_code, label in FORMAT_LABELS.items():
        prefix = "🔘 " if fmt_code == current_format else ""
        fmt_buttons.append(
            InlineKeyboardButton(
                text=f"{prefix}{label}",
                callback_data=f"rev_fmt_{summary_id}_{fmt_code}"
            )
        )

    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Ок (Оставить)", callback_data=f"rev_ok_{summary_id}"),
            InlineKeyboardButton(text="🔄 Переделать", callback_data=f"rev_regen_{summary_id}"),
            InlineKeyboardButton(text="❌ Удалить", callback_data=f"rev_del_{summary_id}"),
        ],
        fmt_buttons
    ])


async def send_smart_summary_with_review(
    bot: Bot,
    chat_id: int,
    header: str,
    summary_text: str,
    summary_id: int,
    format_size: str = "MEDIUM"
):
    """Deliver long summaries in paginated chunks and attach review controls."""
    full_msg = f"{header}\n\n{summary_text}"
    max_len = 3800
    kb = review_inline_kb(summary_id, format_size)

    if len(full_msg) <= max_len:
        await bot.send_message(chat_id, full_msg, reply_markup=kb)
    else:
        chunks = [full_msg[i:i + max_len] for i in range(0, len(full_msg), max_len)]
        for idx, chunk in enumerate(chunks):
            is_last = (idx == len(chunks) - 1)
            await bot.send_message(
                chat_id,
                f"📄 Часть {idx + 1}/{len(chunks)}:\n\n{chunk}",
                reply_markup=kb if is_last else None
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
            fmt_badge = FORMAT_LABELS.get(format_size, "📋 Конспект")

            status_msg = await bot.send_message(
                chat_id,
                f"⚙️ Обрабатываю: {source_title} [{fmt_badge}]...\n"
                f"(В очереди ещё: {remaining})"
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
                    )
                elif source_type == "url":
                    link_res = await parse_and_summarize_url(
                        url=task["url"],
                        format_size=format_size,
                        model=PRIMARY_PRIMARY_GEMINI_MODEL,
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
                    )
                elif source_type == "audio":
                    summary = await loop.run_in_executor(
                        None,
                        process_audio_with_gemini,
                        task["file_path"],
                        task["mime_type"],
                        source_title,
                        format_size
                    )
                    raw_text = summary
                else:
                    summary = "⚠️ Неизвестный формат данных."

                now = datetime.now().strftime("%Y-%m-%d %H:%M")
                async with aiosqlite.connect(DB_PATH) as db:
                    async with db.execute(
                        """
                        INSERT INTO summaries
                        (folder_id, source_type, source_title, summary_text, raw_text, format_size, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (folder_id, source_type, source_title, summary, raw_text, format_size, now)
                    ) as cursor:
                        summary_id = cursor.lastrowid
                    await db.commit()

                await prune_old_raw_texts(user_id)

                try:
                    await status_msg.delete()
                except Exception:
                    pass

                header = f"✅ Выжимка сохранена [{fmt_badge}]\n📌 Источник: {source_title}"
                await send_smart_summary_with_review(
                    bot, chat_id, header, summary, summary_id, format_size
                )

            except Exception as e:
                logger.error(f"Ошибка при обработке '{source_title}': {e}")
                await status_msg.edit_text(f"❌ Не удалось обработать «{source_title}». Детали записаны в журнал аудита.")

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


def main_menu_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📁 Мои папки"), KeyboardButton(text="➕ Создать папку")],
            [KeyboardButton(text="📥 Загрузить материалы"), KeyboardButton(text="📐 Формат выжимки")],
        ],
        resize_keyboard=True,
    )


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    if not is_user_authorized(message.from_user.id):
        await message.answer("⛔ Доступ ограничен корпоративной политикой безопасности.")
        return

    await state.clear()
    folder_id, fmt = await ensure_user_settings(message.from_user.id)
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT name FROM folders WHERE id = ?", (folder_id,)) as cur:
            row = await cur.fetchone()
    folder_name = row[0] if row else "📥 Быстрый входящий"

    await message.answer(
        "🚀 AI Context Hub v2.3 Enterprise готов к работе!\n\n"
        f"📂 Активная папка по умолчанию: «{folder_name}»\n"
        f"📐 Текущий размер выжимки: {FORMAT_LABELS.get(fmt, '📋 Конспект')}\n\n"
        "💡 Быстрый режим (Zero-Click): ты можешь просто кидать сюда файлы, ссылки, "
        "голосовые или текст в любой момент без нажатия кнопок — они автоматически "
        "обработаются и лягут в активную папку!",
        reply_markup=main_menu_kb(),
    )


@router.message(F.text == "📐 Формат выжимки")
async def menu_format_settings(message: Message):
    if not is_user_authorized(message.from_user.id):
        return
    _, current_fmt = await ensure_user_settings(message.from_user.id)
    buttons = []
    for code, label in FORMAT_LABELS.items():
        mark = "✅ " if code == current_fmt else ""
        buttons.append([InlineKeyboardButton(text=f"{mark}{label}", callback_data=f"setfmt_{code}")])

    await message.answer(
        "📐 Выбери размер выжимки по умолчанию для новых материалов:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons)
    )


@router.callback_query(F.data.startswith("setfmt_"))
async def cb_set_default_format(callback: CallbackQuery):
    fmt = callback.data.replace("setfmt_", "")
    if fmt in FORMAT_LABELS:
        await set_user_default_format(callback.from_user.id, fmt)
        await callback.message.edit_text(
            f"✅ Формат по умолчанию переключён на: {FORMAT_LABELS[fmt]}"
        )
    await callback.answer()


@router.message(F.text == "➕ Создать папку")
async def start_new_folder(message: Message, state: FSMContext):
    if not is_user_authorized(message.from_user.id):
        return
    await state.set_state(HubStates.waiting_for_folder_name)
    await message.answer("✏️ Введи название для новой папки (проекта):")


@router.message(HubStates.waiting_for_folder_name)
async def process_folder_name(message: Message, state: FSMContext):
    folder_name = (message.text or "").strip()
    if not folder_name or folder_name in (
        "📁 Мои папки", "➕ Создать папку", "📥 Загрузить материалы", "📐 Формат выжимки"
    ):
        await message.answer("⚠️ Введи корректное текстовое название папки.")
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
        f"✅ Папка «{folder_name}» создана и назначена активной по умолчанию!\n"
        f"Можешь сразу закидывать в бота материалы.",
        reply_markup=main_menu_kb(),
    )


@router.message(F.text == "📁 Мои папки")
async def open_folders_list(message: Message, state: FSMContext):
    if not is_user_authorized(message.from_user.id):
        return
    await state.clear()
    active_id, _ = await ensure_user_settings(message.from_user.id)

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, name FROM folders WHERE user_id = ? ORDER BY id DESC",
            (message.from_user.id,),
        ) as cursor:
            folders = await cursor.fetchall()

    if not folders:
        await message.answer("У тебя пока нет папок. Нажми «➕ Создать папку».")
        return

    buttons = []
    for f_id, f_name in folders:
        pin = "📌 " if f_id == active_id else "📁 "
        buttons.append([InlineKeyboardButton(text=f"{pin}{f_name}", callback_data=f"folder_view_{f_id}")])

    await message.answer(
        "📂 Твои папки (📌 отмечена активная папка для быстрой загрузки):",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons)
    )


@router.callback_query(F.data.startswith("folder_view_"))
async def view_folder_details(callback: CallbackQuery):
    folder_id = int(callback.data.split("_")[-1])

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT name, created_at FROM folders WHERE id = ? AND user_id = ?",
            (folder_id, callback.from_user.id),
        ) as cursor:
            folder = await cursor.fetchone()

        if not folder:
            await callback.answer("Папка не найдена.", show_alert=True)
            return

        async with db.execute(
            "SELECT COUNT(*) FROM summaries WHERE folder_id = ?", (folder_id,)
        ) as cursor:
            count = (await cursor.fetchone())[0]

    await set_user_active_folder(callback.from_user.id, folder_id)

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📥 Загрузить сюда материалы", callback_data=f"upload_to_{folder_id}")],
        [InlineKeyboardButton(text="📄 Скачать весь архив (.txt)", callback_data=f"export_txt_{folder_id}")],
        [InlineKeyboardButton(text="🗑 Удалить папку", callback_data=f"delete_folder_{folder_id}")],
    ])

    await callback.message.edit_text(
        f"📁 Папка: {folder[0]} (назначена активной 📌)\n"
        f"📅 Создана: {folder[1]}\n"
        f"📊 Сохранено выжимок: {count}",
        reply_markup=kb,
    )
    await callback.answer("Папка назначена активной!")


@router.callback_query(F.data.startswith("delete_folder_"))
async def confirm_delete_folder(callback: CallbackQuery):
    folder_id = int(callback.data.split("_")[-1])
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="❗️ Да, удалить навсегда", callback_data=f"confirm_del_{folder_id}"),
            InlineKeyboardButton(text="🔙 Отмена", callback_data=f"folder_view_{folder_id}"),
        ]
    ])
    await callback.message.edit_text(
        "⚠️ Точно удалить эту папку и все сохранённые в ней выжимки?",
        reply_markup=kb,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("confirm_del_"))
async def execute_delete_folder(callback: CallbackQuery):
    folder_id = int(callback.data.split("_")[-1])
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM summaries WHERE folder_id = ?", (folder_id,))
        await db.execute(
            "DELETE FROM folders WHERE id = ? AND user_id = ?",
            (folder_id, callback.from_user.id),
        )
        await db.commit()

    await ensure_user_settings(callback.from_user.id)
    await callback.message.edit_text("🗑 Папка и все её материалы удалены.")
    await callback.answer()


@router.callback_query(F.data.startswith("export_txt_"))
async def download_folder_archive(callback: CallbackQuery):
    folder_id = int(callback.data.split("_")[-1])

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT name FROM folders WHERE id = ? AND user_id = ?",
            (folder_id, callback.from_user.id),
        ) as cursor:
            folder = await cursor.fetchone()

        if not folder:
            await callback.answer("Папка не найдена.", show_alert=True)
            return

        async with db.execute(
            "SELECT source_type, source_title, summary_text, format_size, created_at "
            "FROM summaries WHERE folder_id = ? ORDER BY id ASC",
            (folder_id,),
        ) as cursor:
            items = await cursor.fetchall()

    if not items:
        await callback.answer("В этой папке пока нет выжимок!", show_alert=True)
        return

    folder_name = folder[0]
    lines = [
        f"==================================================",
        f"АРХИВ ПРОЕКТА: {folder_name}",
        f"Экспортировано: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Всего материалов: {len(items)}",
        f"==================================================\n",
    ]

    for idx, (s_type, s_title, s_text, s_fmt, s_date) in enumerate(items, 1):
        fmt_label = FORMAT_LABELS.get(s_fmt, "📋 Конспект")
        lines.append(f"--- МАТЕРИАЛ #{idx} [{s_type.upper()} | {fmt_label}] ({s_date}) ---")
        lines.append(f"ИСТОЧНИК: {s_title}\n")
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
            caption=f"📦 Полный архив выжимок папки «{folder_name}» ({len(items)} шт.)",
        )
    finally:
        if os.path.exists(export_path):
            os.remove(export_path)

    await callback.answer()


@router.message(F.text == "📥 Загрузить материалы")
async def start_upload_flow(message: Message):
    if not is_user_authorized(message.from_user.id):
        return
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, name FROM folders WHERE user_id = ? ORDER BY id DESC",
            (message.from_user.id,),
        ) as cursor:
            folders = await cursor.fetchall()

    if not folders:
        await message.answer("Сначала создай хотя бы одну папку через «➕ Создать папку».")
        return

    buttons = [
        [InlineKeyboardButton(text=f"📁 {f_name}", callback_data=f"upload_to_{f_id}")]
        for f_id, f_name in folders
    ]
    await message.answer(
        "Выбери активную папку для приёма материалов:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons)
    )


@router.callback_query(F.data.startswith("upload_to_"))
async def select_folder_for_upload(callback: CallbackQuery, state: FSMContext):
    folder_id = int(callback.data.split("_")[-1])
    await set_user_active_folder(callback.from_user.id, folder_id)
    await state.set_state(HubStates.waiting_for_input)
    await state.update_data(active_folder_id=folder_id)

    await callback.message.edit_text(
        "📥 Папка активирована!\n\n"
        "Закидывай материалы в любом количестве:\n"
        "• Текстовые сообщения и заметки\n"
        "• Ссылки на статьи (3-уровневый обход блокировок)\n"
        "• Документы (.pdf, .docx, .txt, .py, .md, .log, .json)\n"
        "• Голосовые сообщения и аудиофайлы"
    )
    await callback.answer()


@router.callback_query(F.data.startswith("rev_ok_"))
async def review_ok_handler(callback: CallbackQuery):
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer("Отлично! Оставляем в базе 👍")


@router.callback_query(F.data.startswith("rev_del_"))
async def review_delete_handler(callback: CallbackQuery):
    summary_id = int(callback.data.split("_")[-1])
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM summaries WHERE id = ?", (summary_id,))
        await db.commit()

    try:
        await callback.message.delete()
    except Exception:
        await callback.message.edit_text("🗑 Выжимка удалена из базы данных.")
    await callback.answer("Удалено из базы!")


@router.callback_query(F.data.startswith("rev_regen_") | F.data.startswith("rev_fmt_"))
async def review_regenerate_or_change_format(callback: CallbackQuery):
    """Re-synthesize summary in a new format or variation using cached source text."""
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
            (summary_id,)
        ) as cur:
            row = await cur.fetchone()

    if not row:
        await callback.answer("⚠️ Запись не найдена в базе.", show_alert=True)
        return

    source_title, current_summary, raw_text, current_fmt = row
    current_fmt = current_fmt or "MEDIUM"
    new_fmt = target_format or current_fmt

    size_rank = {"SHORT": 1, "MEDIUM": 2, "LONG": 3}

    if not raw_text:
        if size_rank.get(new_fmt, 2) > size_rank.get(current_fmt, 2):
            await callback.answer(
                "⚠️ Исходник этого документа уже выгружен из буфера (хранятся последние 10). "
                "Из меньшего формата нельзя развернуть больший, но можно переделать текущий или сжать!",
                show_alert=True
            )
            return
        base_material = current_summary
    else:
        base_material = raw_text

    fmt_badge = FORMAT_LABELS.get(new_fmt, "📋 Конспект")
    await callback.answer(f"Перестраиваю в формат: {fmt_badge}...")
    await callback.message.edit_text(
        f"🔄 Перестраиваю выжимку для «{source_title}» -> [{fmt_badge}]..."
    )

    try:
        new_summary = await generate_executive_summary(
            raw_text=base_material,
            source_title=source_title,
            format_size=new_fmt,
            retry_variation=is_variation,
        )

        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE summaries SET summary_text = ?, format_size = ? WHERE id = ?",
                (new_summary, new_fmt, summary_id)
            )
            await db.commit()

        header = f"♻️ Обновлённая выжимка [{fmt_badge}]\n📌 Источник: {source_title}"
        full_msg = f"{header}\n\n{new_summary}"
        await callback.message.edit_text(
            full_msg[:3900],
            reply_markup=review_inline_kb(summary_id, new_fmt)
        )
    except Exception as e:
        logger.error(f"Ошибка при переделке записи #{summary_id}: {e}")
        await callback.message.edit_text("❌ Ошибка при перестройке текста. Попробуйте позже.")


@router.message()
async def enqueue_incoming_material(message: Message, state: FSMContext, bot: Bot):
    """Zero-click universal handler for documents, voice/audio, URLs, and raw notes."""
    user_id = message.from_user.id
    if not is_user_authorized(user_id):
        await message.answer("⛔ Доступ запрещён.")
        return

    data = await state.get_data()
    default_folder_id, default_format = await ensure_user_settings(user_id)
    folder_id = data.get("active_folder_id") or default_folder_id

    if user_id not in USER_QUEUES:
        USER_QUEUES[user_id] = []

    new_tasks = []

    if message.document:
        doc = message.document
        if doc.file_size and doc.file_size > MAX_FILE_SIZE_BYTES:
            await message.answer(
                f"⚠️ Файл «{doc.file_name}» превышает безопасный лимит ({MAX_FILE_SIZE_MB} МБ)."
            )
            return

        file_name = sanitize_filename(doc.file_name or "document")
        ext = os.path.splitext(file_name)[1].lower()
        supported_txt = (".txt", ".md", ".py", ".csv", ".log", ".json", ".ini", ".cfg", ".gsc")

        if ext not in (".pdf", ".docx") + supported_txt:
            await message.answer(f"⚠️ Формат {ext} пока не поддерживается.")
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
            "source_title": f"Файл {file_name}",
            "file_path": local_path,
            "format_size": default_format,
        })

    elif message.voice or message.audio:
        media = message.voice or message.audio
        if media.file_size and media.file_size > MAX_FILE_SIZE_BYTES:
            await message.answer(f"⚠️ Аудиофайл превышает лимит {MAX_FILE_SIZE_MB} МБ.")
            return

        ext = ".ogg" if message.voice else ".mp3"
        mime = media.mime_type or ("audio/ogg" if message.voice else "audio/mpeg")
        title = (
            f"Голосовое ({media.duration} сек)"
            if message.voice
            else f"Аудио ({ sanitize_filename(getattr(media, 'file_name', None) or 'audio.mp3') })"
        )
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
                })
        elif len(text) >= 5:
            short_title = text[:40].replace("\n", " ") + ("..." if len(text) > 40 else "")
            new_tasks.append({
                "priority": 1,
                "chat_id": message.chat.id,
                "folder_id": folder_id,
                "source_type": "text",
                "source_title": f"Заметка: {short_title}",
                "raw_text": text,
                "format_size": default_format,
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

    logger.info("🚀 Бот AI Context Hub v2.3 Enterprise успешно запущен!")
    # Flush backlog of stale Telegram updates immediately before polling
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот штатно остановлен.")