# AI Context Hub

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![aiogram 3.x](https://img.shields.io/badge/aiogram-3.x-blue)](https://github.com/aiogram/aiogram)
[![SQLite WAL](https://img.shields.io/badge/SQLite-WAL-lightgrey)](https://www.sqlite.org/)
[![Google GenAI SDK](https://img.shields.io/badge/Google%20GenAI-SDK-orange)](https://github.com/google-gemini/google-genai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[English] | [Документація українською](README_UA.md) | [Документация на русском](README_RU.md)

---

## Architectural Overview

```
+--------------------+
|  Telegram Client   |
+---------+----------+
          |
          v
+-------------------------------+
|   aiogram 3.x Router / FSM    |
+---------------+---------------+
                |
                v
+-------------------------------+
|     Priority Task Queue       |
+---------------+---------------+
                |
                v
+-----------------------------------------------------------------------------------------------+
|                                    Local Extraction Layer                                     |
|  - Documents and Plaintext: Local extractors (.pdf, .docx, .txt, .md, .py, .csv, .log,        |
|    .json, .ini, .cfg, .gsc via pypdf and python-docx)                                         |
|  - Audio and Voice: Google GenAI File API (.mp3, .ogg with ephemeral lifecycle)               |
|  - Web Articles: aiohttp + BeautifulSoup DOM cleaning (anti-SSRF validation, 20,000 char max) |
+-----------------------------------------------+-----------------------------------------------+
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
|                                          LLM Router                                           |
|  - Gemini Flash Cascade: Thread-safe round-robin API key rotation and model fallback         |
|  - Failover Gateway: Groq API (llama-3.3-70b-versatile) on HTTP 429/503 quota exhaustion     |
|  - Cross-Lingual Intelligence: Automatic end-to-end translation into user UI language         |
|  - Emergency Extraction: Deterministic paragraph extraction on total upstream outage          |
+-----------------------------------------------+-----------------------------------------------+
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
|                                      SQLite WAL Storage                                       |
|  - Automated idempotent schema migrations (language in user_settings, format_size/raw_text)   |
|  - Sliding-window raw text pruning (configurable SLIDING_WINDOW_RAW_LIMIT)                    |
|  - Project folder grouping and consolidated TXT archive export                                |
+-----------------------------------------------+-----------------------------------------------+
```

---

## Core Features

### Cross-Lingual Intelligence (Universal Translation & Summarization)
AI Context Hub acts as a universal cross-lingual synthesis engine. Any incoming material—whether web articles and links, PDF documents, DOCX files, plaintext source code, or spoken voice recordings—arriving in any foreign language (e.g. French, German, Spanish, Chinese, Japanese, etc.) is automatically analyzed, fully translated, and summarized directly into the user's active interface language (English, Ukrainian, or Russian). All structured headings, bullet points, executive takeaways, and action items are natively adapted into the destination language while preserving factual accuracy, technical metrics, and proper nouns.

### Multi-Source Ingestion Pipeline
- Document and Code Extraction: Local asynchronous extraction for binary documents (PDF via `pypdf`, DOCX via `python-docx`) and structured plaintext files (`.txt`, `.md`, `.py`, `.csv`, `.log`, `.json`, `.ini`, `.cfg`, `.gsc`) up to 20 MB without external cloud parsing dependencies.
- Audio and Voice Processing: Ephemeral audio ingestion (`.mp3`, `.ogg`, and Telegram voice notes) via the Google GenAI File API, generating structured executive summaries followed by immediate remote file deletion.
- Web Content Parsing (`parser.py`): Local asynchronous web page fetching using `aiohttp` with standard browser User-Agent headers, anti-SSRF verification (blocking private, link-local, loopback, and reserved IP ranges), DOM cleaning (`script`, `style`, `nav`, `header`, `footer`) via `BeautifulSoup`, and safe text truncation at 20,000 characters to protect model context quotas.

### Multi-Tier Summaries
Materials are synthesized into one of three distinct compression tiers, configured by default or toggled interactively via inline review buttons:
- SHORT (Cheat Sheet): High-density distillation of 5 to 7 key facts, numbers, and parameters fitted for single-screen review.
- MEDIUM (Structured Summary): Balanced brief containing 8 to 12 logically grouped points preserving context, arguments, and action items.
- LONG (In-Depth Retelling): Comprehensive section-by-section breakdown detailing methodology, technical nuances, and background context.

### Background Queue Worker & Locale Isolation
All user submissions are scheduled in an in-memory priority FIFO queue:
- Priority 1: Raw text notes and plaintext files.
- Priority 2: Structured documents (PDF, DOCX).
- Priority 3: Web links.
- Priority 4: Audio recordings and voice notes.

The queue worker processes items sequentially per user to prevent RPM/TPM rate limit spikes. Before processing each task, the worker resolves the user's active language from the database, guaranteeing absolute locale isolation across background operations.

### Trilingual Interface with Auto-Detection & Runtime Override
- Onboarding Gate: First-time users without a configured language in SQLite are presented with a trilingual onboarding interface upon `/start` offering British English (`en`), Ukrainian (`uk`), Russian (`ru`), and dynamic Telegram client matching (`auto`).
- Dynamic Auto Resolution: The Auto option dynamically detects the Telegram client language while persisting NULL in the database, allowing cross-device locale responsiveness.
- Runtime Override: Users can switch languages at any time via `/lang` or the persistent menu button (`btn_lang`), updating SQLite settings, inline button markers, and the persistent ReplyKeyboardMarkup in-place.
- Resilient Routing: aiogram handlers use `get_button_variants` to match incoming messages across all supported language permutations.

### Automated SQLite Schema Migrations
- WAL Mode Concurrency: Multi-threaded SQLite database operations executed via an asynchronous wrapper with WAL journal mode and busy timeouts.
- Idempotent Column Additions: Automatic startup detection and migration for missing columns (`language` in `user_settings`, `raw_text` and `format_size` in `summaries`).
- Sliding-Window Storage Hygiene: Automated pruning enforced by `SLIDING_WINDOW_RAW_LIMIT`, clearing uncompressed raw text payloads for older records while retaining the generated executive summaries.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Runtime Environment | Python 3.10+ | Asynchronous execution runtime |
| Framework | aiogram 3.x | Telegram Bot API router, FSM context, inline and reply keyboards |
| Primary Inference | Google GenAI SDK | Multi-key round-robin rotation across Gemini Flash models |
| Failover Inference | Groq Cloud SDK / REST | Automatic fallback to llama-3.3-70b-versatile upon quota exhaustion |
| Database Engine | SQLite 3 (WAL mode) | Concurrency-optimized persistent storage with foreign keys |
| Web Scraping | aiohttp + BeautifulSoup4 | Asynchronous HTTP fetching, anti-SSRF validation, DOM cleaning |
| Document Parsing | pypdf, python-docx | Text extraction from PDF and DOCX binary files |
| Configuration | python-dotenv | Environment variable parsing and type casting |

---

## Setup and Environment Configuration

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-context-hub.git
cd ai-context-hub
```

### 2. Configure Virtual Environment

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:

On Linux / macOS:
```bash
cp .env.example .env
```

On Windows (CMD):
```cmd
copy .env.example .env
```

Define the following environment variables in `.env`:

| Variable | Required | Description |
|---|---|---|
| `BOT_TOKEN` | Yes | Telegram Bot API authentication token obtained from BotFather. |
| `GEMINI_API_KEYS` | Yes | Comma-separated list of Google Gemini API keys used for round-robin rotation. |
| `GROQ_API_KEY` | Optional | Groq Cloud API key utilized as a failover backend when Gemini quotas are exhausted. |
| `ALLOWED_USER_IDS` | Optional | Comma-separated Telegram user IDs permitted access. Leave empty for unrestricted access. |
| `SLIDING_WINDOW_RAW_LIMIT` | Optional | Maximum number of uncompressed raw text payloads retained per user (default: 10). |

Example `.env` configuration:
```ini
BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
GEMINI_API_KEYS=AIzaSyA1...,AIzaSyB2...,AIzaSyC3...
GROQ_API_KEY=gsk_your_groq_key_here
ALLOWED_USER_IDS=123456789,987654321
SLIDING_WINDOW_RAW_LIMIT=10
```

### 5. Start the Application
```bash
python bot.py
```

---

## Testing and Verification

To verify syntax integrity and validate compilation across all modules, run:

```bash
python -m py_compile bot.py parser.py llm_router.py locales.py database.py config.py
```

To execute unit tests for the extraction and routing layers:

```bash
python test_parser.py
python test_llm_router.py
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
