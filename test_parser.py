"""Unit tests for the local web parser, HTML text extractor, and URL summarizer."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from parser import (
    FORMAT_INSTRUCTIONS,
    LinkSummaryResponse,
    _extract_grounding_metadata,
    _extract_key_points,
    _extract_title,
    clean_extracted_text,
    extract_clean_text_from_html,
    fetch_url_text,
    fetch_webpage_content,
    is_safe_url,
    parse_and_summarize_url,
)


class TestUrlSafetyValidation(unittest.TestCase):
    """Test URL validation and SSRF prevention logic."""

    def test_valid_http_urls(self):
        safe, _ = is_safe_url("https://example.com/article/123")
        self.assertTrue(safe)
        safe, _ = is_safe_url("http://sub.domain.org/path?q=test")
        self.assertTrue(safe)

    def test_invalid_schemes(self):
        safe, reason = is_safe_url("ftp://example.com/file")
        self.assertFalse(safe)
        self.assertIn("HTTP", reason)

        safe, _ = is_safe_url("file:///etc/passwd")
        self.assertFalse(safe)

    def test_localhost_and_loopback_restriction(self):
        safe, _ = is_safe_url("http://localhost/admin")
        self.assertFalse(safe)

        safe, _ = is_safe_url("http://127.0.0.1:8080/")
        self.assertFalse(safe)

        safe, _ = is_safe_url("http://0.0.0.0/")
        self.assertFalse(safe)

    def test_private_ip_restriction(self):
        safe, _ = is_safe_url("http://192.168.1.1/")
        self.assertFalse(safe)

        safe, _ = is_safe_url("http://10.0.0.5:3000/")
        self.assertFalse(safe)


class TestHtmlTextExtraction(unittest.TestCase):
    """Test local HTML tag stripping, script/style removal, and truncation."""

    def test_strip_scripts_styles_and_extract_title(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enterprise AI Architecture 2026</title>
            <script>alert('malicious tracker');</script>
            <style>body { background: #fff; }</style>
        </head>
        <body>
            <header><nav><a href="/">Home</a></nav></header>
            <main>
                <h1>Main Heading</h1>
                <p>This is the primary substantive article paragraph detailing system metrics.</p>
                <aside><p>Sidebar fluff and related ads</p></aside>
            </main>
            <footer><p>Copyright 2026</p></footer>
        </body>
        </html>
        """
        title, clean_text = extract_clean_text_from_html(html)
        self.assertEqual(title, "Enterprise AI Architecture 2026")
        self.assertIn("This is the primary substantive article paragraph", clean_text)
        self.assertNotIn("alert('malicious tracker')", clean_text)
        self.assertNotIn("background: #fff", clean_text)
        self.assertNotIn("Sidebar fluff", clean_text)
        self.assertNotIn("Copyright 2026", clean_text)

    def test_truncation_safely_at_20000_chars(self):
        massive_body = "<p>" + ("A" * 30000) + "</p>"
        html = f"<html><head><title>Huge Page</title></head><body>{massive_body}</body></html>"
        title, clean_text = extract_clean_text_from_html(html)
        self.assertEqual(title, "Huge Page")
        self.assertLessEqual(len(clean_text), 20000)

    def test_clean_extracted_text_normalizes_whitespace(self):
        raw = "Line 1   with   spaces\n\n\n\n\nLine 2\n\n\nLine 3"
        cleaned = clean_extracted_text(raw)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("Line 1 with spaces", cleaned)


class TestLinkSummaryResponseModel(unittest.TestCase):
    """Test LinkSummaryResponse container behavior."""

    def test_response_fields_and_string_coercion(self):
        res = LinkSummaryResponse(
            url="https://example.com",
            title="Sample Article",
            summary="# Sample Article\n\n- Key insight 1\n- Key insight 2",
            key_points=["Key insight 1", "Key insight 2"],
            format_size="MEDIUM",
            model="gemini-3.8-flash",
            raw_text="Extracted plain text sample",
        )
        self.assertEqual(res.url, "https://example.com")
        self.assertEqual(res.title, "Sample Article")
        self.assertEqual(str(res), res.summary)
        self.assertEqual(res.to_display_text(), res.summary)
        self.assertEqual(len(res.key_points), 2)
        self.assertEqual(res.raw_text, "Extracted plain text sample")


class TestHelperExtractors(unittest.TestCase):
    """Test title, bullet points, and grounding metadata extraction."""

    def test_title_extraction_from_markdown(self):
        markdown_text = "# Breakthrough in Quantum Computing\n\nSome body text..."
        title = _extract_title(markdown_text, "https://tech-news.com/quantum")
        self.assertEqual(title, "Breakthrough in Quantum Computing")

    def test_title_extraction_fallback(self):
        plain_text = "Just some text without headers."
        title = _extract_title(plain_text, "https://example.org/news/today")
        self.assertEqual(title, "example.org")

    def test_key_points_extraction(self):
        markdown_text = (
            "Here is the summary:\n"
            "- Point alpha with sufficient length\n"
            "* Point beta with sufficient length\n"
            "• Point gamma with sufficient length\n"
            "Regular sentence."
        )
        points = _extract_key_points(markdown_text)
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0], "Point alpha with sufficient length")
        self.assertEqual(points[1], "Point beta with sufficient length")
        self.assertEqual(points[2], "Point gamma with sufficient length")

    def test_grounding_metadata_extraction(self):
        mock_resp = MagicMock()
        mock_candidate = MagicMock()
        mock_g_meta = MagicMock()
        mock_g_meta.web_search_queries = ["quantum computing news"]

        mock_chunk = MagicMock()
        mock_chunk.web.uri = "https://source.com/article"
        mock_chunk.web.title = "Source Article"
        mock_g_meta.grounding_chunks = [mock_chunk]

        mock_candidate.grounding_metadata = mock_g_meta
        mock_resp.candidates = [mock_candidate]

        extracted = _extract_grounding_metadata(mock_resp)
        self.assertEqual(extracted["search_queries"], ["quantum computing news"])
        self.assertEqual(extracted["sources"][0]["uri"], "https://source.com/article")


class TestParseAndSummarizeUrl(unittest.IsolatedAsyncioTestCase):
    """Test asynchronous URL local fetching and LLM executive summary pipeline."""

    async def test_unsafe_url_rejection(self):
        with self.assertRaises(ValueError):
            await parse_and_summarize_url("http://127.0.0.1:8000/secret")

    @patch("parser.fetch_webpage_content")
    async def test_local_fetch_and_llm_pipeline(self, mock_fetch):
        mock_fetch.return_value = (
            "Test Article Title",
            "This is the clean plain text extracted from local HTML stripping.",
        )

        mock_summary = (
            "🎯 **Главная суть:**\nСтатья описывает внедрение AI.\n\n"
            "📊 **Ключевые факты:**\n- Флагманская модель 3.8\n- Скорость выросла на 40%\n\n"
            "💡 **Практические выводы:**\nОбновить шлюзы маршрутизации."
        )

        with patch("parser.generate_executive_summary", AsyncMock(return_value=mock_summary)):
            result = await parse_and_summarize_url(
                url="https://example.com/ai-article",
                format_size="MEDIUM",
            )

        mock_fetch.assert_awaited_once_with("https://example.com/ai-article")
        self.assertIsInstance(result, LinkSummaryResponse)
        self.assertEqual(result.title, "Test Article Title")
        self.assertEqual(result.summary, mock_summary)
        self.assertIn("Флагманская модель 3.8", result.key_points)
        self.assertEqual(result.raw_text, "This is the clean plain text extracted from local HTML stripping.")

    @patch("parser.fetch_webpage_content")
    async def test_empty_response_raises_runtime_error(self, mock_fetch):
        mock_fetch.return_value = ("Sample Title", "Some article text")

        with patch("parser.generate_executive_summary", AsyncMock(return_value="   ")):
            with self.assertRaises(RuntimeError):
                await parse_and_summarize_url("https://example.com/empty-summary")

    @patch("parser.fetch_webpage_content")
    async def test_fetch_url_text_bridge(self, mock_fetch):
        mock_fetch.return_value = ("Bridge Title", "Bridge extracted content")

        mock_summary = "🎯 **Главная суть:** Результат моста."
        with patch("parser.generate_executive_summary", AsyncMock(return_value=mock_summary)):
            text = await fetch_url_text("https://example.com/bridge")
            self.assertEqual(text, mock_summary)


if __name__ == "__main__":
    unittest.main()
