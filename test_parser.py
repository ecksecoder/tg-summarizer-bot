"""Unit tests for the Google GenAI web parser and summarizer module."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from google.genai import types

from parser import (
    FORMAT_INSTRUCTIONS,
    LinkSummaryResponse,
    _extract_grounding_metadata,
    _extract_key_points,
    _extract_title,
    fetch_url_text,
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
        )
        self.assertEqual(res.url, "https://example.com")
        self.assertEqual(res.title, "Sample Article")
        self.assertEqual(str(res), res.summary)
        self.assertEqual(res.to_display_text(), res.summary)
        self.assertEqual(len(res.key_points), 2)


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
    """Test asynchronous URL summarization with native Google GenAI client."""

    async def test_unsafe_url_rejection(self):
        with self.assertRaises(ValueError):
            await parse_and_summarize_url("http://127.0.0.1:8000/secret")

    async def test_native_async_client_flow(self):
        mock_genai_response = MagicMock()
        mock_genai_response.text = (
            "# Artificial Intelligence Updates\n\n"
            "This is a high quality summary.\n\n"
            "- First core point of substance\n"
            "- Second core point of substance"
        )
        mock_genai_response.candidates = []

        mock_client = MagicMock()
        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_genai_response)

        result = await parse_and_summarize_url(
            url="https://example.com/ai-update",
            format_size="SHORT",
            client=mock_client,
            model="gemini-3.8-flash",
            use_search_grounding=True,
        )

        mock_client.aio.models.generate_content.assert_awaited_once()
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-3.8-flash")
        config = call_kwargs["config"]
        self.assertIsInstance(config, types.GenerateContentConfig)
        self.assertIsNotNone(config.tools)
        self.assertIsInstance(config.tools[0].google_search, types.GoogleSearch)

        self.assertIsInstance(result, LinkSummaryResponse)
        self.assertEqual(result.title, "Artificial Intelligence Updates")
        self.assertEqual(result.format_size, "SHORT")
        self.assertEqual(len(result.key_points), 2)

    async def test_sync_client_thread_fallback(self):
        mock_genai_response = MagicMock()
        mock_genai_response.text = "# Sync Fallback Article\n\n- Key takeaway one"
        mock_genai_response.candidates = []

        mock_client = MagicMock(spec=["models"])
        mock_client.models.generate_content.return_value = mock_genai_response

        result = await parse_and_summarize_url(
            url="https://example.com/fallback-test",
            client=mock_client,
        )

        mock_client.models.generate_content.assert_called_once()
        self.assertEqual(result.title, "Sync Fallback Article")

    async def test_empty_response_raises_runtime_error(self):
        mock_genai_response = MagicMock()
        mock_genai_response.text = "   "
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_genai_response)

        with self.assertRaises(RuntimeError):
            await parse_and_summarize_url(
                url="https://example.com/empty",
                client=mock_client,
            )

    async def test_fetch_url_text_bridge(self):
        mock_genai_response = MagicMock()
        mock_genai_response.text = "Clean summary text output"
        mock_genai_response.candidates = []

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_genai_response)

        with patch("parser.genai.Client", return_value=mock_client):
            with patch.dict("os.environ", {"GEMINI_API_KEY": "fake_test_key"}):
                text = await fetch_url_text("https://example.com/bridge")
                self.assertEqual(text, "Clean summary text output")


if __name__ == "__main__":
    unittest.main()
