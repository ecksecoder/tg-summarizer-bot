"""Trilingual localization module for AI Context Hub Enterprise.

Provides comprehensive translation dictionaries covering 100% of UI strings,
language detection, inline language keyboards with strict auto-button formatting,
and text retrieval utilities supporting English ('en', GB), Ukrainian ('uk', UA),
and Russian ('ru', RU) with zero-bifurcation safety and fallback to English.
"""

import re
from typing import Any, Optional
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # Core & Access
        "unauthorized": "Access restricted by corporate security policy.",
        "onboarding_header": "🌐 Choose interface language / Оберіть мову інтерфейсу / Выберите язык интерфейса:",
        "onboarding_banner": "🌐 Choose interface language / Оберіть мову інтерфейсу / Выберите язык интерфейса:",
        "start_greeting": (
            "AI Context Hub Enterprise is ready.\n\n"
            "Active default folder: «{folder_name}»\n"
            "Current summary format: {format_badge}\n\n"
            "Zero-Click Mode: send files, web links, audio/voice notes, or raw text "
            "at any time without navigating menus. Materials are automatically queued, "
            "summarized, and organized into your active folder."
        ),
        "default_folder_name": "📥 Quick Inbox",
        "default_inbox_name": "📥 Quick Inbox",

        # Menu Keyboard Buttons
        "btn_my_folders": "📁 My Folders",
        "btn_create_folder": "➕ Create Folder",
        "btn_upload_content": "📥 Upload Content",
        "btn_summary_format": "⚙️ Summary Format",
        "btn_lang": "🌐 Change Language",
        "btn_language": "🌐 Change Language",

        # Review Actions
        "btn_keep": "Keep",
        "btn_regenerate": "Regenerate",
        "btn_delete": "Delete",

        # Language Selection
        "lang_select_prompt": "🌐 Choose interface language / Оберіть мову інтерфейсу / Выберите язык интерфейса:",
        "lang_switched": "Interface language switched to English.",
        "lang_switched_auto": "Interface language set to Auto (Telegram locale).",

        # Compression Tiers
        "tier_short": "SHORT",
        "tier_medium": "MEDIUM",
        "tier_long": "LONG",
        "format_short": "Cheat Sheet",
        "format_medium": "Structured Summary",
        "format_long": "In-Depth Retelling",

        # Summary Format Selection
        "format_settings_prompt": "Select default summary format for incoming materials:",
        "format_switched": "Default summary format switched to: {format_badge}",

        # Folder Management & Actions
        "enter_folder_name": "Enter a name for the new folder (project):",
        "invalid_folder_name": "Please enter a valid text name for the folder.",
        "folder_created": (
            "Folder «{folder_name}» created and set as active default.\n"
            "You can start forwarding or uploading materials directly."
        ),
        "no_folders": "You have no active folders. Use «Create Folder» to initialize one.",
        "folders_list_title": "Your folders (pinned marks active destination for quick ingestion):",
        "folder_not_found": "Folder not found.",
        "folder_set_active": "Folder assigned as active.",
        "folder_details": (
            "Folder: {folder_name} (Active)\n"
            "Created: {created_at}\n"
            "Saved summaries: {count}"
        ),
        "btn_upload_to_folder": "Upload materials here",
        "btn_export_folder": "Export archive (.txt)",
        "btn_delete_folder": "Delete folder",
        "confirm_delete_title": "Confirm deletion of this folder and all stored summaries?",
        "btn_confirm_delete": "Confirm Delete",
        "btn_cancel": "Cancel",
        "folder_deleted": "Folder and all associated summaries have been deleted.",
        "folder_archive_empty": "This folder does not contain any summaries yet.",
        "archive_caption": "Full export of summaries for folder «{folder_name}» ({count} items).",
        "select_upload_folder": "Select target folder for content ingestion:",
        "need_create_folder_first": "Create at least one folder before uploading materials.",
        "upload_folder_activated": (
            "Folder activated.\n\n"
            "Accepting content:\n"
            "- Text notes and code snippets\n"
            "- Web links (anti-SSRF protected)\n"
            "- Documents (.pdf, .docx, .txt, .md, .py, .csv, .log, .json, .ini, .cfg, .gsc)\n"
            "- Voice and audio recordings"
        ),

        # Queue Statuses & Notifications
        "status_processing": (
            "Processing: {source_title} [{format_badge}]...\n"
            "(Pending in queue: {remaining})"
        ),
        "status_saved": "Summary saved [{format_badge}]\nSource: {source_title}",
        "status_error": "Failed to process «{source_title}». Technical details logged in audit trail.",
        "header_summary_saved": "Summary saved [{format_badge}]\nSource: {source_title}",
        "header_summary_updated": "Updated summary [{format_badge}]\nSource: {source_title}",
        "error_processing": "Failed to process «{source_title}». Technical details logged in audit trail.",
        "error_unknown_format": "Unknown data format.",
        "msg_part": "Part {current}/{total}:\n\n{chunk}",

        # Review Callbacks
        "review_keep_alert": "Confirmed. Retained in database.",
        "review_del_msg": "Summary removed from database.",
        "review_del_alert": "Deleted from database.",
        "review_not_found": "Record not found in database.",
        "review_pruned_warning": (
            "Original document buffer was pruned (sliding window limit reached). "
            "Expanding to a larger format is unavailable, but you can compress or regenerate."
        ),
        "review_rebuilding_alert": "Regenerating in format: {format_badge}...",
        "review_rebuilding_msg": "Regenerating summary for «{source_title}» -> [{format_badge}]...",
        "review_rebuild_error": "Inference error during text restructuring. Please retry shortly.",

        # Ingestion Limits & Titles
        "file_limit_exceeded": "File «{file_name}» exceeds operational limit of {max_mb} MB.",
        "audio_limit_exceeded": "Audio payload exceeds operational limit of {max_mb} MB.",
        "unsupported_format": "Extension {ext} is not supported by available parsers.",
        "title_voice": "Voice recording ({duration}s)",
        "title_audio": "Audio file ({file_name})",
        "title_file": "Document: {file_name}",
        "title_note": "Note: {short_title}",

        # Export Archive Formats
        "archive_header_project": "PROJECT ARCHIVE: {folder_name}",
        "archive_header_exported": "Export Date: {date}",
        "archive_header_total": "Total Items: {count}",
        "archive_item_header": "--- ITEM #{idx} [{type} | {format_badge}] ({date}) ---",
        "archive_source_label": "SOURCE: {source}",
    },
    "uk": {
        # Core & Access
        "unauthorized": "Доступ обмежено корпоративною політикою безпеки.",
        "onboarding_header": "🌐 Choose interface language / Оберіть мову інтерфейсу / Выберите язык интерфейса:",
        "onboarding_banner": "🌐 Choose interface language / Оберіть мову інтерфейсу / Выберите язык интерфейса:",
        "start_greeting": (
            "AI Context Hub Enterprise готовий до роботи!\n\n"
            "Активна папка за замовчуванням: «{folder_name}»\n"
            "Поточний формат вижимки: {format_badge}\n\n"
            "Швидкий режим (Zero-Click): ви можете надсилати файли, посилання, "
            "голосові або текст у будь-який момент без навігації по меню — вони автоматично "
            "обробляться та збережуться в активну папку!"
        ),
        "default_folder_name": "📥 Швидкі вхідні",
        "default_inbox_name": "📥 Швидкі вхідні",

        # Menu Keyboard Buttons
        "btn_my_folders": "📁 Мої папки",
        "btn_create_folder": "➕ Створити папку",
        "btn_upload_content": "📥 Завантажити матеріали",
        "btn_summary_format": "⚙️ Формат вижимки",
        "btn_lang": "🌐 Змінити мову",
        "btn_language": "🌐 Змінити мову",

        # Review Actions
        "btn_keep": "Зберегти",
        "btn_regenerate": "Переробити",
        "btn_delete": "Видалити",

        # Language Selection
        "lang_select_prompt": "🌐 Choose interface language / Оберіть мову інтерфейсу / Выберите язык интерфейса:",
        "lang_switched": "Мову інтерфейсу змінено на українську.",
        "lang_switched_auto": "Мову інтерфейсу встановлено в режим Авто (Telegram).",

        # Compression Tiers
        "tier_short": "SHORT",
        "tier_medium": "MEDIUM",
        "tier_long": "LONG",
        "format_short": "Шпаргалка",
        "format_medium": "Конспект",
        "format_long": "Переказ",

        # Summary Format Selection
        "format_settings_prompt": "Оберіть формат вижимки за замовчуванням для нових матеріалів:",
        "format_switched": "Формат за замовчуванням змінено на: {format_badge}",

        # Folder Management & Actions
        "enter_folder_name": "Введіть назву для нової папки (проєкту):",
        "invalid_folder_name": "Введіть коректну текстову назву папки.",
        "folder_created": (
            "Папку «{folder_name}» створено та призначено активною за замовчуванням!\n"
            "Можете одразу надсилати матеріали в бота."
        ),
        "no_folders": "У вас ще немає папок. Натисніть «Створити папку».",
        "folders_list_title": "Ваші папки (позначено активну папку для швидкого завантаження):",
        "folder_not_found": "Папку не знайдено.",
        "folder_set_active": "Папку призначено активною!",
        "folder_details": (
            "Папка: {folder_name} (Активна)\n"
            "Створена: {created_at}\n"
            "Збережено вижимок: {count}"
        ),
        "btn_upload_to_folder": "Завантажити сюди матеріали",
        "btn_export_folder": "Завантажити весь архів (.txt)",
        "btn_delete_folder": "Видалити папку",
        "confirm_delete_title": "Точно видалити цю папку та всі збережені в ній вижимки?",
        "btn_confirm_delete": "Так, видалити назавжди",
        "btn_cancel": "Скасувати",
        "folder_deleted": "Папку та всі її матеріали видалено.",
        "folder_archive_empty": "У цій папці ще немає вижимок!",
        "archive_caption": "Повний архів вижимок папки «{folder_name}» ({count} шт.).",
        "select_upload_folder": "Оберіть активну папку для прийому матеріалів:",
        "need_create_folder_first": "Спочатку створіть хоча б одну папку через «Створити папку».",
        "upload_folder_activated": (
            "Папку активовано!\n\n"
            "Надсилайте матеріали в будь-якій кількості:\n"
            "- Текстові повідомлення та нотатки\n"
            "- Посилання на статті (захист Anti-SSRF)\n"
            "- Документи (.pdf, .docx, .txt, .md, .py, .csv, .log, .json, .ini, .cfg, .gsc)\n"
            "- Голосові повідомлення та аудіофайли"
        ),

        # Queue Statuses & Notifications
        "status_processing": (
            "Обробляю: {source_title} [{format_badge}]...\n"
            "(У черзі ще: {remaining})"
        ),
        "status_saved": "Вижимку збережено [{format_badge}]\nДжерело: {source_title}",
        "status_error": "Не вдалося обробити «{source_title}». Деталі записано в журнал аудиту.",
        "header_summary_saved": "Вижимку збережено [{format_badge}]\nДжерело: {source_title}",
        "header_summary_updated": "Оновлена вижимка [{format_badge}]\nДжерело: {source_title}",
        "error_processing": "Не вдалося обробити «{source_title}». Деталі записано в журнал аудиту.",
        "error_unknown_format": "Невідомий формат даних.",
        "msg_part": "Частина {current}/{total}:\n\n{chunk}",

        # Review Callbacks
        "review_keep_alert": "Чудово! Залишаємо в базі.",
        "review_del_msg": "Вижимку видалено з бази даних.",
        "review_del_alert": "Видалено з бази!",
        "review_not_found": "Запис не знайдено в базі.",
        "review_pruned_warning": (
            "Вихідний текст цього документа вже вивантажено з буфера (зберігаються останні 10). "
            "З меншого формату не можна розгорнути більший, але можна переробити поточний або стиснути!"
        ),
        "review_rebuilding_alert": "Перебудовую у формат: {format_badge}...",
        "review_rebuilding_msg": "Перебудовую вижимку для «{source_title}» -> [{format_badge}]...",
        "review_rebuild_error": "Помилка при перебудові тексту. Спробуйте пізніше.",

        # Ingestion Limits & Titles
        "file_limit_exceeded": "Файл «{file_name}» перевищує безпечний ліміт ({max_mb} МБ).",
        "audio_limit_exceeded": "Аудіофайл перевищує ліміт {max_mb} МБ.",
        "unsupported_format": "Формат {ext} наразі не підтримується.",
        "title_voice": "Голосове ({duration} сек)",
        "title_audio": "Аудіо ({file_name})",
        "title_file": "Файл {file_name}",
        "title_note": "Нотатка: {short_title}",

        # Export Archive Formats
        "archive_header_project": "АРХІВ ПРОЄКТУ: {folder_name}",
        "archive_header_exported": "Експортовано: {date}",
        "archive_header_total": "Всього матеріалів: {count}",
        "archive_item_header": "--- МАТЕРІАЛ #{idx} [{type} | {format_badge}] ({date}) ---",
        "archive_source_label": "ДЖЕРЕЛО: {source}",
    },
    "ru": {
        # Core & Access
        "unauthorized": "Доступ ограничен корпоративной политикой безопасности.",
        "onboarding_header": "🌐 Choose interface language / Оберіть мову інтерфейсу / Выберите язык интерфейса:",
        "onboarding_banner": "🌐 Choose interface language / Оберіть мову інтерфейсу / Выберите язык интерфейса:",
        "start_greeting": (
            "AI Context Hub Enterprise готов к работе!\n\n"
            "Активная папка по умолчанию: «{folder_name}»\n"
            "Текущий размер выжимки: {format_badge}\n\n"
            "Быстрый режим (Zero-Click): ты можешь просто отправлять сюда файлы, ссылки, "
            "голосовые или текст в любой момент без нажатия кнопок — они автоматически "
            "обработаются и сохранятся в активную папку!"
        ),
        "default_folder_name": "📥 Быстрый входящий",
        "default_inbox_name": "📥 Быстрый входящий",

        # Menu Keyboard Buttons
        "btn_my_folders": "📁 Мои папки",
        "btn_create_folder": "➕ Создать папку",
        "btn_upload_content": "📥 Загрузить материалы",
        "btn_summary_format": "⚙️ Формат выжимки",
        "btn_lang": "🌐 Сменить язык",
        "btn_language": "🌐 Сменить язык",

        # Review Actions
        "btn_keep": "Ок (Оставить)",
        "btn_regenerate": "Переделать",
        "btn_delete": "Удалить",

        # Language Selection
        "lang_select_prompt": "🌐 Choose interface language / Оберіть мову інтерфейсу / Выберите язык интерфейса:",
        "lang_switched": "Язык интерфейса переключён на русский.",
        "lang_switched_auto": "Язык интерфейса переключён в режим Авто (Telegram).",

        # Compression Tiers
        "tier_short": "SHORT",
        "tier_medium": "MEDIUM",
        "tier_long": "LONG",
        "format_short": "Шпаргалка",
        "format_medium": "Конспект",
        "format_long": "Пересказ",

        # Summary Format Selection
        "format_settings_prompt": "Выбери размер выжимки по умолчанию для новых материалов:",
        "format_switched": "Формат по умолчанию переключён на: {format_badge}",

        # Folder Management & Actions
        "enter_folder_name": "Введи название для новой папки (проекта):",
        "invalid_folder_name": "Введи корректное текстовое название папки.",
        "folder_created": (
            "Папка «{folder_name}» создана и назначена активной по умолчанию!\n"
            "Можешь сразу отправлять материалы в бота."
        ),
        "no_folders": "У тебя пока нет папок. Нажми «Создать папку».",
        "folders_list_title": "Твои папки (отмечена активная папка для быстрой загрузки):",
        "folder_not_found": "Папка не найдена.",
        "folder_set_active": "Папка назначена активной!",
        "folder_details": (
            "Папка: {folder_name} (Активная)\n"
            "Создана: {created_at}\n"
            "Сохранено выжимок: {count}"
        ),
        "btn_upload_to_folder": "Загрузить сюда материалы",
        "btn_export_folder": "Скачать весь архив (.txt)",
        "btn_delete_folder": "Удалить папку",
        "confirm_delete_title": "Точно удалить эту папку и все сохранённые в ней выжимки?",
        "btn_confirm_delete": "Да, удалить навсегда",
        "btn_cancel": "Отмена",
        "folder_deleted": "Папка и все её материалы удалены.",
        "folder_archive_empty": "В этой папке пока нет выжимок!",
        "archive_caption": "Полный архив выжимок папки «{folder_name}» ({count} шт.).",
        "select_upload_folder": "Выбери активную папку для приёма материалов:",
        "need_create_folder_first": "Сначала создай хотя бы одну папку через «Создать папку».",
        "upload_folder_activated": (
            "Папка активирована!\n\n"
            "Закидывай материалы в любом количестве:\n"
            "- Текстовые сообщения и заметки\n"
            "- Ссылки на статьи (защита Anti-SSRF)\n"
            "- Документы (.pdf, .docx, .txt, .md, .py, .csv, .log, .json, .ini, .cfg, .gsc)\n"
            "- Голосовые сообщения и аудиофайлы"
        ),

        # Queue Statuses & Notifications
        "status_processing": (
            "Обрабатываю: {source_title} [{format_badge}]...\n"
            "(В очереди ещё: {remaining})"
        ),
        "status_saved": "Выжимка сохранена [{format_badge}]\nИсточник: {source_title}",
        "status_error": "Не удалось обработать «{source_title}». Детали записаны в журнал аудита.",
        "header_summary_saved": "Выжимка сохранена [{format_badge}]\nИсточник: {source_title}",
        "header_summary_updated": "Обновлённая выжимка [{format_badge}]\nИсточник: {source_title}",
        "error_processing": "Не удалось обработать «{source_title}». Детали записаны в журнал аудита.",
        "error_unknown_format": "Неизвестный формат данных.",
        "msg_part": "Часть {current}/{total}:\n\n{chunk}",

        # Review Callbacks
        "review_keep_alert": "Отлично! Оставляем в базе.",
        "review_del_msg": "Выжимка удалена из базы данных.",
        "review_del_alert": "Удалено из базы!",
        "review_not_found": "Запись не найдена в базе.",
        "review_pruned_warning": (
            "Исходник этого документа уже выгружен из буфера (хранятся последние 10). "
            "Из меньшего формата нельзя развернуть больший, но можно переделать текущий или сжать!"
        ),
        "review_rebuilding_alert": "Перестраиваю в формат: {format_badge}...",
        "review_rebuilding_msg": "Перестраиваю выжимку для «{source_title}» -> [{format_badge}]...",
        "review_rebuild_error": "Ошибка при перестройке текста. Попробуйте позже.",

        # Ingestion Limits & Titles
        "file_limit_exceeded": "Файл «{file_name}» превышает безопасный лимит ({max_mb} МБ).",
        "audio_limit_exceeded": "Аудиофайл превышает лимит {max_mb} МБ.",
        "unsupported_format": "Формат {ext} пока не поддерживается.",
        "title_voice": "Голосовое ({duration} сек)",
        "title_audio": "Аудио ({file_name})",
        "title_file": "Файл {file_name}",
        "title_note": "Заметка: {short_title}",

        # Export Archive Formats
        "archive_header_project": "АРХИВ ПРОЕКТА: {folder_name}",
        "archive_header_exported": "Экспортировано: {date}",
        "archive_header_total": "Всего материалов: {count}",
        "archive_item_header": "--- МАТЕРИАЛ #{idx} [{type} | {format_badge}] ({date}) ---",
        "archive_source_label": "ИСТОЧНИК: {source}",
    },
}

FORMAT_BADGES: dict[str, dict[str, str]] = {
    "SHORT": {
        "en": "Cheat Sheet",
        "uk": "Шпаргалка",
        "ru": "Шпаргалка",
    },
    "MEDIUM": {
        "en": "Structured Summary",
        "uk": "Конспект",
        "ru": "Конспект",
    },
    "LONG": {
        "en": "In-Depth Retelling",
        "uk": "Переказ",
        "ru": "Пересказ",
    },
}

FORMAT_LABELS: dict[str, dict[str, str]] = FORMAT_BADGES


def detect_locale(telegram_lang_code: Optional[str]) -> str:
    """Detect interface language code from Telegram client language.

    Args:
        telegram_lang_code: Raw language code from Telegram user profile.

    Returns:
        str: 'uk' for Ukrainian, 'ru' for Russian/Belarusian, fallback 'en'.
    """
    if not telegram_lang_code:
        return "en"
    code = str(telegram_lang_code).lower().strip()
    if code in ("uk", "ua") or code.startswith(("uk-", "uk_", "ua-", "ua_")):
        return "uk"
    if code in ("ru", "be") or code.startswith(("ru-", "ru_", "be-", "be_")):
        return "ru"
    return "en"


def get_text(key: str, lang: str = "en", **kwargs: Any) -> str:
    """Retrieve translated and safely formatted string with fallback to 'en'.

    Args:
        key: Translation key identifier.
        lang: Target language code ('en', 'uk', or 'ru').
        **kwargs: Dynamic arguments to format into the localized string template.

    Returns:
        str: Translated and formatted string.
    """
    target_lang = lang if lang in TRANSLATIONS else "en"
    locale_dict = TRANSLATIONS.get(target_lang) or TRANSLATIONS["en"]
    template = locale_dict.get(key)
    if template is None:
        template = TRANSLATIONS["en"].get(key, key)

    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError, IndexError):
            return template

    return template


def get_format_badge(format_size: str, lang: str = "en") -> str:
    """Retrieve localized summary format badge.

    Args:
        format_size: Compression tier code ('SHORT', 'MEDIUM', 'LONG').
        lang: Target language code ('en', 'uk', or 'ru').

    Returns:
        str: Localized badge string.
    """
    size = (format_size or "MEDIUM").upper()
    badges = FORMAT_BADGES.get(size) or FORMAT_BADGES["MEDIUM"]
    target_lang = lang if lang in ("en", "uk", "ru") else "en"
    return badges.get(target_lang) or badges.get("en", size)


def get_language_inline_kb(
    active_lang: str = "en",
    is_auto: bool = False,
    prefix: str = "lang_set",
    *,
    current_lang: Optional[str] = None,
    telegram_lang: Optional[str] = None,
) -> InlineKeyboardMarkup:
    """Build inline keyboard for language selection.

    Rules:
    - Language buttons are '🇬🇧 English', '🇺🇦 Українська', '🇷🇺 Русский'
      (with trailing ' ✅' on active selection when not is_auto).
    - Auto button label is derived strictly from active_lang with (Telegram) always in Latin:
      '🔄 Auto (Telegram)' for en,
      '🔄 Автоматично (Telegram)' for uk,
      '🔄 Авто (Telegram)' for ru.
    - If is_auto is True, prepend '✅ ' to the auto button text.

    Args:
        active_lang: Currently active language code ('en', 'uk', 'ru').
        is_auto: Whether the user currently has automatic Telegram locale detection enabled.
        prefix: Callback data prefix ('lang_set' or 'lang_init').
        current_lang: Backward-compatible alias for active_lang.
        telegram_lang: Backward-compatible argument for client language detection.

    Returns:
        InlineKeyboardMarkup: Inline keyboard with localized language buttons.
    """
    clean_prefix = prefix.rstrip(":_")

    target_lang = (current_lang or active_lang or "en").lower().strip()
    if target_lang not in ("en", "uk", "ru"):
        target_lang = "en"

    en_text = "🇬🇧 English"
    uk_text = "🇺🇦 Українська"
    ru_text = "🇷🇺 Русский"

    # Trailing ' ✅' on active selection when not is_auto
    if not is_auto:
        if target_lang == "en":
            en_text = f"{en_text} ✅"
        elif target_lang == "uk":
            uk_text = f"{uk_text} ✅"
        elif target_lang == "ru":
            ru_text = f"{ru_text} ✅"

    # Auto button text derived strictly from active_lang with (Telegram) in Latin
    if target_lang == "uk":
        auto_text = "🔄 Автоматично (Telegram)"
    elif target_lang == "ru":
        auto_text = "🔄 Авто (Telegram)"
    else:
        auto_text = "🔄 Auto (Telegram)"

    # Prepend '✅ ' if is_auto is True
    if is_auto:
        auto_text = f"✅ {auto_text}"

    row1 = [
        InlineKeyboardButton(text=en_text, callback_data=f"{clean_prefix}:en"),
        InlineKeyboardButton(text=uk_text, callback_data=f"{clean_prefix}:uk"),
        InlineKeyboardButton(text=ru_text, callback_data=f"{clean_prefix}:ru"),
    ]
    row2 = [
        InlineKeyboardButton(text=auto_text, callback_data=f"{clean_prefix}:auto"),
    ]

    return InlineKeyboardMarkup(inline_keyboard=[row1, row2])


def get_button_variants(key: str) -> tuple[str, ...]:
    """Return all localized values of a button across ('en', 'uk', 'ru') for aiogram message filtering.

    Args:
        key: Button key in TRANSLATIONS (e.g. 'btn_lang', 'btn_my_folders').

    Returns:
        tuple[str, ...]: Unique button texts across all supported languages.
    """
    keys_to_check = [key]
    if key in ("btn_lang", "btn_language"):
        keys_to_check = ["btn_lang", "btn_language"]

    variants: list[str] = []
    for k in keys_to_check:
        for lang in ("en", "uk", "ru"):
            val = TRANSLATIONS.get(lang, {}).get(k)
            if val and val not in variants:
                variants.append(val)
                # Register variant without leading emoji / symbols if present
                stripped = re.sub(r"^[^\w\s]+\s*", "", val).strip()
                if stripped and stripped not in variants:
                    variants.append(stripped)

    # Legacy fallback variants for menu buttons to prevent filter misses
    legacy_map: dict[str, list[str]] = {
        "btn_lang": ["🌐 Language", "🌐 Мова", "🌐 Язык", "Language", "Мова", "Язык"],
        "btn_language": ["🌐 Language", "🌐 Мова", "🌐 Язык", "Language", "Мова", "Язык"],
    }
    for leg in legacy_map.get(key, []):
        if leg not in variants:
            variants.append(leg)

    return tuple(variants)
