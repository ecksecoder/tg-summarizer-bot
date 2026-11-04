"""Web content parsing and local text extraction module.

Restores reliable asynchronous local webpage fetching using aiohttp with browser User-Agent
and timeout, strips HTML tags, scripts, and styles using BeautifulSoup to extract clean plain
text truncated safely at 20,000 characters, and passes the clean text to the unified LLM router
without using Google Search Grounding tools or passing raw URLs directly to Google GenAI.
"""

import asyncio
import html as html_lib
import ipaddress
import logging
import re
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from config import DEFAULT_GEMINI_MODEL
from llm_router import (
    BROWSER_USER_AGENT,
    FORMAT_INSTRUCTIONS,
    LLMRouter,
    UNIFIED_EXECUTIVE_SYSTEM_PROMPT,
    default_router,
    generate_executive_summary,
)

logger = logging.getLogger(__name__)

# Maximum characters extracted from webpage to protect LLM TPM limits
MAX_PAGE_CHARS: int = 20000


def is_safe_url(url: str) -> tuple[bool, str]:
    """Validate target URL against SSRF vectors (loopback, private, and reserved IPs).

    Args:
        url: The web URL string to validate.

    Returns:
        tuple[bool, str]: (is_safe, error_reason)
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        return False, f"Malformed URL: {exc}"

    if parsed.scheme not in ("http", "https"):
        return False, "Only HTTP and HTTPS protocols are allowed."

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False, "Missing hostname in target URL."

    if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return False, "Access to localhost addresses is restricted."

    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False, "Access to private or local network resources is restricted."
    except ValueError:
        # Hostname is a domain name rather than an IP literal
        pass

    return True, ""


def clean_extracted_text(text: str) -> str:
    """Normalize whitespace and strip redundant blank lines from extracted text.

    Args:
        text: Raw text string.

    Returns:
        str: Normalized text string.
    """
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_clean_text_from_html(html_text: str) -> tuple[str, str]:
    """Strip markup, scripts, and styles from HTML to produce clean plain text.

    Extracts page title and primary textual content using BeautifulSoup with regex
    fallback, safely truncating plain text at MAX_PAGE_CHARS (20,000 characters).

    Args:
        html_text: Raw HTML markup string.

    Returns:
        tuple[str, str]: (extracted_title, clean_plain_text)
    """
    if not html_text:
        return "", ""

    title = ""
    clean_text = ""

    try:
        soup = BeautifulSoup(html_text, "html.parser")

        # Extract title from title tag, meta tags, or primary header
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title:
            og_title = soup.find("meta", property="og:title") or soup.find(
                "meta", attrs={"name": "og:title"}
            )
            if og_title and og_title.get("content"):
                title = str(og_title["content"]).strip()
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        # Decompose non-content, interactive, and structural chrome elements
        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "nav",
                "aside",
                "form",
                "iframe",
                "svg",
                "button",
                "dialog",
                "template",
            ]
        ):
            tag.decompose()

        # Locate primary content container
        main_node = soup.find("article") or soup.find("main") or soup.body or soup
        body_text = main_node.get_text(separator="\n")
        clean_text = clean_extracted_text(body_text)
    except Exception as exc:
        logger.debug("BeautifulSoup parsing encountered issue: %s. Using regex fallback.", exc)
        # Regex fallback
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = html_lib.unescape(title_match.group(1)).strip()

        stripped = re.sub(
            r"<(script|style|noscript|header|footer|nav|aside)[^>]*>.*?</\1>",
            "",
            html_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        clean_text = clean_extracted_text(html_lib.unescape(stripped))

    # Safe truncation at 20,000 characters to prevent TPM limit spikes
    if len(clean_text) > MAX_PAGE_CHARS:
        clean_text = clean_text[:MAX_PAGE_CHARS]

    return title[:150], clean_text


async def fetch_webpage_content(
    url: str,
    timeout: int = 20,
    session: Optional[aiohttp.ClientSession] = None,
) -> tuple[str, str]:
    """Asynchronously fetch webpage and extract clean plain text using aiohttp.

    Uses browser User-Agent headers, follows redirects, respects timeouts,
    and handles SSL/charset decoding gracefully.

    Args:
        url: Target web URL string.
        timeout: Total network timeout in seconds.
        session: Optional external aiohttp ClientSession.

    Returns:
        tuple[str, str]: (page_title, clean_plain_text)

    Raises:
        ValueError: If the URL fails SSRF security validation.
        RuntimeError: If HTTP retrieval fails or content is unreadable.
    """
    clean_url = url.strip()
    safe, reason = is_safe_url(clean_url)
    if not safe:
        raise ValueError(f"URL failed security validation: {reason}")

    # Standardize social/code hosting preview URLs for crawler accessibility
    if "t.me/" in clean_url and "/s/" not in clean_url:
        clean_url = re.sub(r"https?://t\.me/([^/]+)/(\d+)", r"https://t.me/s/\1/\2", clean_url)
    elif "github.com/" in clean_url and "/blob/" in clean_url:
        clean_url = clean_url.replace("github.com/", "raw.githubusercontent.com/").replace("/blob/", "/")

    headers = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
    }

    timeout_cfg = aiohttp.ClientTimeout(total=timeout, connect=8)
    connector = aiohttp.TCPConnector(ssl=False)

    async def _fetch(sess: aiohttp.ClientSession) -> tuple[str, str]:
        async with sess.get(clean_url, headers=headers, allow_redirects=True) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} when fetching URL: {clean_url}")
            html_text = await resp.text(errors="ignore")
            title, text = extract_clean_text_from_html(html_text)
            if not title:
                title = urlparse(clean_url).netloc or clean_url
            if len(text.strip()) < 15:
                raise RuntimeError(f"Insufficient readable content extracted from URL: {clean_url}")
            return title, text

    if session is not None:
        return await _fetch(session)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_cfg) as new_session:
        return await _fetch(new_session)


class LinkSummaryResponse(BaseModel):
    """Structured response container for processed and summarized web links."""

    url: str = Field(description="The source URL analyzed.")
    title: str = Field(default="", description="Extracted or inferred title of the webpage.")
    summary: str = Field(description="Structured markdown summary generated by Google AI / Groq.")
    key_points: list[str] = Field(
        default_factory=list,
        description="Key takeaways or bullet points from the content.",
    )
    format_size: str = Field(
        default="MEDIUM",
        description="Compression tier requested (SHORT, MEDIUM, or LONG).",
    )
    model: str = Field(
        default=DEFAULT_GEMINI_MODEL,
        description="The LLM model used for analysis.",
    )
    grounding_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Empty grounding metadata preserved for schema backward-compatibility.",
    )
    raw_text: str = Field(
        default="",
        description="Clean extracted plain text from local HTML stripping.",
    )

    def to_display_text(self) -> str:
        """Return formatted markdown text suitable for chat interface presentation.

        Returns:
            str: Markdown formatted summary string.
        """
        return self.summary

    def __str__(self) -> str:
        """Return the summary as string for backward-compatible string coercion.

        Returns:
            str: Markdown formatted summary string.
        """
        return self.summary


def _extract_title(text: str, fallback_url: str) -> str:
    """Extract a representative title from model output or fallback to hostname.

    Args:
        text: The generated summary text.
        fallback_url: URL to extract hostname from if no title is present.

    Returns:
        str: Cleaned title string.
    """
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            cleaned = line.lstrip("#").strip()
            if cleaned:
                return cleaned[:120]
        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            return line.strip("*").strip()[:120]

    parsed = urlparse(fallback_url)
    return parsed.netloc or fallback_url


def _extract_key_points(text: str) -> list[str]:
    """Parse bullet points from generated markdown text.

    Args:
        text: The generated summary text.

    Returns:
        list[str]: Extracted bullet point takeaways.
    """
    bullets: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ", "• ")) and len(line) > 5:
            item = line[2:].strip()
            if item:
                bullets.append(item)
    return bullets[:12]


def _extract_grounding_metadata(response: Any) -> dict[str, Any]:
    """Extract grounding citations and web queries safely from candidate response.

    Args:
        response: The Google GenAI GenerateContentResponse object.

    Returns:
        dict[str, Any]: Extracted search queries and source references.
    """
    metadata: dict[str, Any] = {}
    try:
        candidates = getattr(response, "candidates", None)
        if candidates and len(candidates) > 0:
            candidate = candidates[0]
            g_meta = getattr(candidate, "grounding_metadata", None)
            if g_meta:
                if hasattr(g_meta, "web_search_queries") and g_meta.web_search_queries:
                    metadata["search_queries"] = list(g_meta.web_search_queries)
                if hasattr(g_meta, "grounding_chunks") and g_meta.grounding_chunks:
                    chunks = []
                    for c in g_meta.grounding_chunks:
                        web = getattr(c, "web", None)
                        if web:
                            chunks.append({
                                "uri": getattr(web, "uri", ""),
                                "title": getattr(web, "title", ""),
                            })
                    if chunks:
                        metadata["sources"] = chunks
    except Exception as exc:
        logger.debug("Failed to extract grounding metadata: %s", exc)
    return metadata


async def parse_and_summarize_url(
    url: str,
    format_size: str = "MEDIUM",
    client: Optional[Any] = None,
    model: Optional[str] = None,
    system_instruction: Optional[str] = None,
    temperature: float = 0.25,
    use_search_grounding: bool = False,
    router: Optional[LLMRouter] = None,
    **kwargs: Any,
) -> LinkSummaryResponse:
    """Analyze and summarize web content from a URL via local text extraction and LLM cascade.

    1. Restores asynchronous local webpage fetching via aiohttp with browser User-Agent.
    2. Strips HTML tags, scripts, and styles via BeautifulSoup, truncating safely at 20,000 chars.
    3. Passes extracted plain text to the unified LLM router and Executive Summary pipeline
       without Google Search Grounding tools or raw URL passing.

    Args:
        url: Target web URL to fetch and summarize.
        format_size: Target summary length tier ('SHORT', 'MEDIUM', 'LONG').
        client: Optional pre-configured Google GenAI client instance.
        model: Target Gemini model identifier (or None for cascade).
        system_instruction: Optional custom system instructions.
        temperature: Sampling temperature for generation.
        use_search_grounding: Disabled; kept only for backward signature compatibility.
        router: Optional custom LLMRouter instance for key rotation.
        **kwargs: Unused backward-compatibility kwargs.

    Returns:
        LinkSummaryResponse: Structured object containing URL, title, summary,
        key points, and clean plain text.

    Raises:
        ValueError: If the URL fails SSRF safety validation.
        RuntimeError: If local extraction or LLM inference fails.
    """
    clean_url = url.strip()

    # 1 & 2: Local asynchronous webpage fetching and HTML stripping
    title, clean_text = await fetch_webpage_content(clean_url)

    # 3: Pass extracted clean text to the unified LLM router
    active_router = router or default_router
    active_model = model or DEFAULT_GEMINI_MODEL

    summary_text = await generate_executive_summary(
        raw_text=clean_text,
        source_title=title,
        format_size=format_size,
        router=active_router,
    )

    if not summary_text.strip():
        raise RuntimeError(f"LLM cascade returned empty response for URL: {clean_url}")

    key_points = _extract_key_points(summary_text)

    return LinkSummaryResponse(
        url=clean_url,
        title=title,
        summary=summary_text.strip(),
        key_points=key_points,
        format_size=format_size,
        model=active_model,
        grounding_metadata={},
        raw_text=clean_text,
    )


# Alias for intuitive interface naming
process_link = parse_and_summarize_url


async def fetch_url_text(url: str, format_size: str = "MEDIUM") -> str:
    """Legacy compatibility bridge returning raw summary text for a URL.

    Args:
        url: Web URL to process.
        format_size: Compression tier for summary.

    Returns:
        str: Formatted summary text.
    """
    result = await parse_and_summarize_url(url, format_size=format_size)
    return result.summary
