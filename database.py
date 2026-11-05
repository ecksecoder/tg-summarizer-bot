"""Asynchronous SQLite storage engine with WAL mode and schema migration layer.

Provides connection context management, thread-safe asynchronous querying,
automatic idempotent schema migrations, and user locale persistence helpers.
"""

import asyncio
import logging
import sqlite3
from typing import Optional

from config import DB_PATH

logger = logging.getLogger(__name__)


class _CursorContext:
    """Async cursor wrapper executing SQLite statements on worker threads."""

    def __init__(self, conn: sqlite3.Connection, sql: str, params: tuple):
        self._conn = conn
        self._sql = sql
        self._params = params
        self._cursor: Optional[sqlite3.Cursor] = None

    async def _run(self) -> "_CursorContext":
        self._cursor = await asyncio.to_thread(self._conn.execute, self._sql, self._params)
        return self

    def __await__(self):
        return self._run().__await__()

    async def __aenter__(self) -> "_CursorContext":
        return await self._run()

    async def __aexit__(self, exc_type, exc, tb):
        if self._cursor:
            await asyncio.to_thread(self._cursor.close)

    async def fetchone(self):
        return await asyncio.to_thread(self._cursor.fetchone)

    async def fetchall(self):
        return await asyncio.to_thread(self._cursor.fetchall)

    @property
    def lastrowid(self) -> Optional[int]:
        return self._cursor.lastrowid if self._cursor else None


class _DBConnectionContext:
    """Async SQLite connection manager enforcing WAL mode, busy timeouts, and foreign keys."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def _open_and_configure(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    async def __aenter__(self) -> "_DBConnectionContext":
        self._conn = await asyncio.to_thread(self._open_and_configure)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._conn:
            await asyncio.to_thread(self._conn.close)

    def execute(self, sql: str, params: tuple = ()) -> _CursorContext:
        return _CursorContext(self._conn, sql, params)

    async def commit(self):
        await asyncio.to_thread(self._conn.commit)


class _SQLiteCompat:
    """Convenience connection factory matching aiosqlite syntax."""

    def connect(self, db_path: str) -> _DBConnectionContext:
        return _DBConnectionContext(db_path)


aiosqlite = _SQLiteCompat()


async def init_db() -> None:
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
                default_format TEXT DEFAULT 'MEDIUM',
                language TEXT DEFAULT NULL
            )
        """)

        # Verify and migrate missing columns in summaries
        async with db.execute("PRAGMA table_info(summaries)") as cursor:
            summary_cols = [row[1] for row in await cursor.fetchall()]
        for col_name, col_def in (
            ("folder_id", "INTEGER NOT NULL DEFAULT 1"),
            ("source_type", "TEXT NOT NULL DEFAULT 'text'"),
            ("source_title", "TEXT NOT NULL DEFAULT 'Material'"),
            ("summary_text", "TEXT NOT NULL DEFAULT ''"),
            ("raw_text", "TEXT DEFAULT NULL"),
            ("format_size", "TEXT DEFAULT 'MEDIUM'"),
            ("created_at", "TEXT NOT NULL DEFAULT ''"),
        ):
            if col_name not in summary_cols:
                await db.execute(f"ALTER TABLE summaries ADD COLUMN {col_name} {col_def}")
                logger.info("Database migration: added column '%s' to table summaries.", col_name)

        # Verify and migrate missing columns in user_settings, specifically language
        async with db.execute("PRAGMA table_info(user_settings)") as cursor:
            us_cols = [row[1] for row in await cursor.fetchall()]
        if "language" not in us_cols:
            await db.execute("ALTER TABLE user_settings ADD COLUMN language TEXT DEFAULT NULL")
            logger.info("Database migration: added column 'language' to table user_settings.")

        await db.commit()
    logger.info("SQLite database (WAL mode) successfully initialized.")


async def get_user_language(user_id: int) -> str | None:
    """Retrieve persisted user language from SQLite database.

    Args:
        user_id: Telegram user ID.

    Returns:
        str | None: 'en', 'uk', 'ru', or None if unset/auto.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT language FROM user_settings WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            if row and row[0]:
                val = str(row[0]).strip().lower()
                if val in ("en", "uk", "ru"):
                    return val
    return None


async def is_user_onboarded(user_id: int) -> bool:
    """Check whether a user has already completed initial onboarding.

    Args:
        user_id: Telegram user ID.

    Returns:
        bool: True if user record exists in user_settings.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT 1 FROM user_settings WHERE user_id = ?",
            (user_id,),
        ) as cur:
            return bool(await cur.fetchone())


async def set_user_language(user_id: int, lang: str | None) -> None:
    """Persist user language preference into SQLite database.

    Args:
        user_id: Telegram user ID.
        lang: Target language code ('en', 'uk', 'ru') or None/'auto' to enable dynamic auto-detection.
    """
    clean_lang: str | None = None
    if lang is not None:
        candidate = str(lang).strip().lower()
        if candidate in ("en", "uk", "ru"):
            clean_lang = candidate

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT user_id FROM user_settings WHERE user_id = ?",
            (user_id,),
        ) as cur:
            exists = await cur.fetchone()
        if exists:
            await db.execute(
                "UPDATE user_settings SET language = ? WHERE user_id = ?",
                (clean_lang, user_id),
            )
        else:
            await db.execute(
                "INSERT INTO user_settings (user_id, language, default_format) VALUES (?, ?, 'MEDIUM')",
                (user_id, clean_lang),
            )
        await db.commit()
