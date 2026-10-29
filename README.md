# 🤖 AI Context Hub — Telegram AI Summarizer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/aiogram-3.x-blue?logo=telegram" alt="aiogram 3.x">
  <img src="https://img.shields.io/badge/Google%20Gemini%20API-SDK%20v2-orange?logo=google" alt="Google Gemini API">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p>

<p align="center">
  <a href="#-ai-context-hub--умный-ai-суммаризатор-текста-pdf-и-документов">🇷🇺 Русский</a> &nbsp;|&nbsp;
  <a href="#-ai-context-hub--smart-ai-summarizer-for-text-pdf-and-documents">🇬🇧 English</a>
</p>

---

## 🇷🇺 AI Context Hub — Умный AI-суммаризатор текста, PDF и документов

**AI Context Hub** — современный асинхронный Telegram-бот на базе искусственного интеллекта для автоматического анализа, извлечения сути и интеллектуальной суммаризации любого входящего контента: заметок, веб-статей, книг и документов (.pdf, .docx, .txt), а также голосовых сообщений и аудиозаписей.

---

### 📑 Поддерживаемые форматы выжимки

Бот формирует структурированные материалы под любую задачу с возможностью переключения «на лету» прямо под сообщением:

1. ⚡ **Шпаргалка / Заметка (`SHORT`)**
   - Ультракомпактная выжимка: строго 5–10 ключевых тезисов, цифр и выводов на один экран смартфона.
   - Максимум фактов без «воды» — идеально для быстрого сканирования сути.
2. 📋 **Конспект (`MEDIUM`)**
   - Сбалансированный структурированный конспект, разделенный на логические блоки с заголовками.
   - Сохраняет важные параметры, термины, сюжетные линии и причинно-следственные связи.
3. 📚 **Глубокий пересказ (`LONG`)**
   - Подробный хронологический разбор каждой главы или раздела с сохранением примеров, цитат и нюансов.
   - Полностью заменяет чтение объемного оригинала.

> 💡 **Интерактивное управление**: под каждой готовой выжимкой доступны inline-кнопки для мгновенной смены формата (`⚡ Шпаргалка` ↔ `📋 Конспект` ↔ `📚 Пересказ`), кнопка «🔄 Переделать» (генерация альтернативного варианта) и «❌ Удалить».

---

### 🌟 Ключевые возможности

* **Zero-Click Ingestion (Приём в один клик):** Отправляйте текст, файлы, аудио или ссылки прямо в чат без блуждания по меню. Бот автоматически определит тип контента, поставит в очередь с учетом приоритетов и сохранит в активную папку.
* **Поддержка документов и книг:** Обработка файлов `.pdf`, `.docx`, `.txt`, `.md`, `.py`, `.json`, `.csv`, `.log` объемом до 20 МБ.
* **Голосовые и аудиоматериалы:** Распознавание и суммаризация голосовых сообщений и файлов `.mp3` / `.ogg` через Google Gemini File API.
* **5-эшелонный парсинг веб-страниц:** Извлечение чистых статей (Jina Reader API, прямое подключение Chrome TLS, Markdown.new, CORS-прокси) со встроенной защитой от SSRF-атак (`Anti-SSRF`).
* **Отказоустойчивый роутер AI:** Автоматическое обнаружение доступных моделей Google Gemini (`google-genai` SDK), резервная маршрутизация к Groq API при пиковых нагрузках (429/503) и автономный локальный анализатор.
* **Проекты и папки:** Организация материалов по тематическим папкам с возможностью экспорта всех выжимок проекта в единый файл `.txt`.
* **База данных SQLite (WAL):** Высокоскоростное асинхронное хранение с автоматическими миграциями и скользящим буфером сырых данных (`N=10`) для экономии места на диске.

---

### 🛠 Стек технологий

* **Язык программирования:** Python 3.10+ (`asyncio`)
* **Telegram-фреймворк:** [aiogram 3.x](https://github.com/aiogram/aiogram) (FSM, Routers, Inline Callbacks, MemoryStorage)
* **Нейросетевой стек:** [Google GenAI SDK](https://github.com/google-gemini/google-genai) (Gemini Flash), резервный шлюз Groq API
* **База данных:** SQLite 3 в режиме Write-Ahead Logging (`PRAGMA journal_mode=WAL`)
* **Парсинг и обработка данных:** `aiohttp`, `beautifulsoup4`, `pypdf`, `python-docx`
* **Конфигурация:** `python-dotenv`

---

### 🚀 Пошаговая инструкция по установке и запуску

#### 1. Клонирование репозитория
```bash
git clone https://github.com/your-username/ai-context-hub.git
cd ai-context-hub
```

#### 2. Создание и активация виртуального окружения

* **Windows:**
  ```powershell
  python -m venv venv
  venv\Scripts\activate
  ```
* **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### 3. Установка зависимостей
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Настройка переменных окружения
Создайте файл `.env` на основе шаблона:
* **Windows:**
  ```cmd
  copy .env.example .env
  ```
* **Linux / macOS:**
  ```bash
  cp .env.example .env
  ```

Откройте файл `.env` и вставьте ваши ключи:
```ini
# Токен бота от @BotFather в Telegram
BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ

# Ключ Google Gemini (бесплатно в https://aistudio.google.com/)
GEMINI_API_KEY=AIzaSy...

# (Опционально) Резервный ключ Groq (https://console.groq.com/)
GROQ_API_KEY=gsk_...

# (Опционально) Telegram ID администраторов через запятую (оставьте пустым для общего доступа)
ALLOWED_USER_IDS=
```

#### 5. Запуск бота
* Напрямую через Python:
  ```bash
  python bot.py
  ```
* На Windows через готовый батник:
  ```cmd
  run_bot.bat
  ```

---

<br>

## 🇬🇧 AI Context Hub — Smart AI Summarizer for Text, PDF, and Documents

**AI Context Hub** is an asynchronous Telegram assistant powered by artificial intelligence. It automatically ingests, extracts, and summarizes raw notes, web articles, documents (`.pdf`, `.docx`, `.txt`), and voice/audio files into structured summaries.

---

### 📑 Supported Summary Formats

The bot provides three distinct compression levels, switchable on-the-fly via inline buttons:

1. ⚡ **Cheat Sheet / Quick Note (`SHORT`)**
   - High-density distillation: strictly 5–10 key takeaways, metrics, and bullet points.
   - Designed for instant reading on a single phone screen with zero filler.
2. 📋 **Structured Notes (`MEDIUM`)**
   - Well-balanced notes categorized into logical sections with descriptive headings.
   - Retains essential parameters, key terminology, arguments, and cause-and-effect relationships.
3. 📚 **In-Depth Retelling (`LONG`)**
   - Comprehensive chronological breakdown of every chapter or section.
   - Preserves examples, context, and nuances, completely eliminating the need to read the full source text.

> 💡 **Inline Controls**: Every generated summary includes buttons to re-synthesize into another format (`⚡ Cheat Sheet` ↔ `📋 Structured Notes` ↔ `📚 Deep Retelling`), request an alternative perspective (`🔄 Regenerate`), or delete the entry (`❌ Delete`).

---

### 🌟 Key Features

* **Zero-Click Ingestion Pipeline:** Drop raw notes, documents, voice notes, or links directly into chat—tasks are automatically prioritized, queued, and mapped to your active project folder.
* **Multi-Format Document Support:** Processes `.pdf`, `.docx`, `.txt`, `.md`, `.py`, `.json`, `.csv`, `.log` files up to 20 MB.
* **Audio & Voice Transcription:** Ingests voice messages and `.mp3` / `.ogg` files via the Google Gemini File API.
* **5-Tier Web Article Extractor:** Robust multi-tier scraper (Jina Reader API, direct Chrome TLS, Markdown.new, CORS fallback) equipped with anti-SSRF IP validation.
* **Resilient Multi-Model Routing:** Dynamic model discovery with the `google-genai` SDK, automatic failover to Groq API endpoints on rate limits (429/503), and local extractive fallback.
* **Project Folders & Archives:** Organize items into custom project folders and export the entire collection as a single `.txt` archive.
* **SQLite WAL Storage:** Thread-pooled database running in `WAL` (Write-Ahead Logging) mode with auto-migrations and sliding-window cleanup (`N=10`) to keep storage lightweight.

---

### 🛠 Tech Stack

* **Language:** Python 3.10+ (`asyncio`)
* **Telegram Framework:** [aiogram 3.x](https://github.com/aiogram/aiogram) (FSM, Routers, Inline Callbacks, MemoryStorage)
* **LLM Engine:** [Google GenAI SDK](https://github.com/google-gemini/google-genai) (Gemini Flash) with Groq API failover
* **Database:** SQLite 3 (`PRAGMA journal_mode=WAL`)
* **Parsing & Extraction:** `aiohttp`, `beautifulsoup4`, `pypdf`, `python-docx`
* **Configuration:** `python-dotenv`

---

### 🚀 Step-by-Step Installation & Local Setup

#### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-context-hub.git
cd ai-context-hub
```

#### 2. Create and activate a virtual environment

* **Windows:**
  ```powershell
  python -m venv venv
  venv\Scripts\activate
  ```
* **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Configure environment variables
Copy `.env.example` to `.env`:
* **Windows:**
  ```cmd
  copy .env.example .env
  ```
* **Linux / macOS:**
  ```bash
  cp .env.example .env
  ```

Fill in your API credentials inside `.env`:
```ini
# Bot token from @BotFather
BOT_TOKEN=your_telegram_bot_token_here

# Google Gemini API key (https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_api_key_here

# (Optional) Groq API key for fallback routing (https://console.groq.com/)
GROQ_API_KEY=your_groq_api_key_here

# (Optional) Comma-separated Telegram User IDs for RBAC whitelist (leave empty for public)
ALLOWED_USER_IDS=
```

#### 5. Launch the bot
* Using Python:
  ```bash
  python bot.py
  ```
* On Windows using the batch launcher:
  ```cmd
  run_bot.bat
  ```

---

## 📄 License
This project is open-source and distributed under the [MIT License](LICENSE).
