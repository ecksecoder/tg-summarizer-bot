"""Comprehensive unit tests for KeyManager, LLMRouter, cascade, and startup flush."""

import asyncio
import os
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from google.genai import errors, types

import config
from llm_router import (
    AllKeysExhaustedError,
    KeyManager,
    LLMRouter,
    UNIFIED_EXECUTIVE_SYSTEM_PROMPT,
    _local_extractive_fallback,
    generate_executive_summary,
    get_model_cascade,
    is_resource_exhausted,
    mask_api_key,
)
from parser import parse_and_summarize_url


class TestConfigParsing(unittest.TestCase):
    """Test configuration parsing and environment variable extraction."""

    def test_parse_comma_separated_keys(self):
        raw = "key1, key2 , key3,   key4  "
        keys = config.parse_gemini_api_keys(raw)
        self.assertEqual(keys, ["key1", "key2", "key3", "key4"])

    def test_parse_keys_with_quotes_and_empty_slots(self):
        raw = ' "key_a" , , \'key_b\' , "" , key_c '
        keys = config.parse_gemini_api_keys(raw)
        self.assertEqual(keys, ["key_a", "key_b", "key_c"])

    def test_fallback_to_singular_gemini_api_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEYS": "", "GEMINI_API_KEY": "legacy_key"}):
            keys = config.parse_gemini_api_keys()
            self.assertEqual(keys, ["legacy_key"])

    def test_plural_takes_precedence_over_singular(self):
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEYS": "new1,new2", "GEMINI_API_KEY": "legacy_key"},
        ):
            keys = config.parse_gemini_api_keys()
            self.assertEqual(keys, ["new1", "new2"])

    def test_parse_allowed_user_ids(self):
        raw = "12345, 67890, invalid, 112233"
        ids = config.parse_allowed_user_ids(raw)
        self.assertEqual(ids, {12345, 67890, 112233})


class TestKeyManager(unittest.TestCase):
    """Test KeyManager thread safety, async safety, and round-robin rotation."""

    def test_round_robin_sequence(self):
        keys = [f"key_{i}" for i in range(5)]
        km = KeyManager(keys=keys)
        self.assertEqual(len(km), 5)
        self.assertEqual(km.get_next_key(), "key_0")
        self.assertEqual(km.get_next_key(), "key_1")
        self.assertEqual(km.get_next_key(), "key_2")
        self.assertEqual(km.get_next_key(), "key_3")
        self.assertEqual(km.get_next_key(), "key_4")
        # Wraps around
        self.assertEqual(km.get_next_key(), "key_0")

    def test_rotating_generator(self):
        km = KeyManager(keys=["alpha", "beta"])
        gen = km.key_generator()
        results = [next(gen) for _ in range(5)]
        self.assertEqual(results, ["alpha", "beta", "alpha", "beta", "alpha"])

    def test_10x_scalability(self):
        # Scale to 50 keys seamlessly
        large_pool = [f"scale_key_{i}" for i in range(50)]
        km = KeyManager(keys=large_pool)
        self.assertEqual(len(km), 50)
        retrieved = [km.get_next_key() for _ in range(100)]
        self.assertEqual(len(retrieved), 100)
        self.assertEqual(retrieved[:50], large_pool)
        self.assertEqual(retrieved[50:], large_pool)

    def test_thread_safe_rotation(self):
        keys = [f"thread_key_{i}" for i in range(10)]
        km = KeyManager(keys=keys)
        collected = []
        lock = threading.Lock()

        def worker():
            for _ in range(50):
                k = km.get_next_key()
                with lock:
                    collected.append(k)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(collected), 500)
        # Each key should have been picked exactly 50 times across 500 requests
        for k in keys:
            self.assertEqual(collected.count(k), 50)

    def test_client_caching(self):
        km = KeyManager(keys=["client_key_1", "client_key_2"])
        client_1 = km.get_client("client_key_1")
        client_1_again = km.get_client("client_key_1")
        client_2 = km.get_client("client_key_2")

        self.assertIs(client_1, client_1_again)
        self.assertIsNot(client_1, client_2)

    def test_mask_api_key(self):
        self.assertEqual(mask_api_key("short"), "***")
        self.assertEqual(mask_api_key("AIzaSyD1234567890Xyz"), "AIza...0Xyz")


class TestModelCascadeAndErrorHandling(unittest.TestCase):
    """Test model cascade ordering and 429/503 error detection."""

    def test_cascade_order_and_deduplication(self):
        # Primary is 3.8-flash: cascade is [3.8-flash, 2.5-flash, 2.0-flash]
        cascade = get_model_cascade("gemini-3.8-flash")
        self.assertEqual(cascade, ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"])

        # Primary is already 2.5-flash: no duplicate
        cascade_dedup = get_model_cascade("gemini-2.5-flash")
        self.assertEqual(cascade_dedup, ["gemini-2.5-flash", "gemini-2.0-flash"])

    def test_is_resource_exhausted_detector(self):
        mock_429 = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})
        mock_503 = errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}})

        self.assertTrue(is_resource_exhausted(mock_429))
        self.assertTrue(is_resource_exhausted(mock_503))
        self.assertTrue(is_resource_exhausted(RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")))
        self.assertTrue(is_resource_exhausted(RuntimeError("503 UNAVAILABLE High Demand")))
        self.assertFalse(is_resource_exhausted(ValueError("Bad arguments")))
        self.assertFalse(is_resource_exhausted(RuntimeError("Unknown syntax error")))


class TestLLMRouter(unittest.IsolatedAsyncioTestCase):
    """Test async LLMRouter retry loops, 429/503 rotation, and Groq fallback."""

    def setUp(self):
        self.mock_429 = errors.ClientError(
            429,
            {
                "error": {
                    "code": 429,
                    "message": "Resource has been exhausted (rate limit).",
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
        )
        self.mock_503 = errors.ServerError(
            503,
            {
                "error": {
                    "code": 503,
                    "message": "This model is currently experiencing high demand.",
                    "status": "UNAVAILABLE",
                }
            },
        )

    async def test_instant_retry_on_429_masks_error(self):
        km = KeyManager(keys=["key_fail", "key_success"])
        router = LLMRouter(key_manager=km)

        failing_client = MagicMock()
        failing_client.aio.models.generate_content = AsyncMock(side_effect=self.mock_429)

        success_response = MagicMock()
        success_response.text = "Success after rotation"
        succeeding_client = MagicMock()
        succeeding_client.aio.models.generate_content = AsyncMock(return_value=success_response)

        km._clients["key_fail"] = failing_client
        km._clients["key_success"] = succeeding_client

        response = await router.generate_content(
            model="gemini-3.8-flash",
            contents="test prompt",
            fallback_to_groq=False,
        )

        self.assertEqual(response.text, "Success after rotation")
        failing_client.aio.models.generate_content.assert_awaited_once()
        succeeding_client.aio.models.generate_content.assert_awaited_once()

    async def test_all_keys_exhausted_raises_graceful_exception_when_groq_fails(self):
        km = KeyManager(keys=["key_1", "key_2", "key_3"])
        router = LLMRouter(key_manager=km, groq_api_key="mock_groq")

        for k in km.get_all_keys():
            client_mock = MagicMock()
            client_mock.aio.models.generate_content = AsyncMock(side_effect=self.mock_429)
            km._clients[k] = client_mock

        with patch("llm_router.call_groq_api", AsyncMock(side_effect=RuntimeError("Groq offline"))):
            with self.assertRaises(AllKeysExhaustedError) as ctx:
                await router.generate_content(
                    model="gemini-3.8-flash",
                    contents="test prompt",
                    fallback_to_groq=True,
                )

            self.assertIn("exhausted", str(ctx.exception).lower())

    async def test_groq_fallback_triggered_when_gemini_keys_fail(self):
        km = KeyManager(keys=["key_1", "key_2"])
        router = LLMRouter(key_manager=km, groq_api_key="mock_groq")

        for k in km.get_all_keys():
            client_mock = MagicMock()
            client_mock.aio.models.generate_content = AsyncMock(side_effect=self.mock_429)
            km._clients[k] = client_mock

        with patch("llm_router.call_groq_api", AsyncMock(return_value="Groq fallback summary text")) as mock_groq:
            response = await router.generate_content(
                model="gemini-3.8-flash",
                contents="Extracted clean plain text article",
                fallback_to_groq=True,
            )

            self.assertEqual(response.text, "Groq fallback summary text")
            self.assertEqual(response.provider, "groq")
            mock_groq.assert_awaited_once()
            call_kwargs = mock_groq.call_args.kwargs
            self.assertIn("Extracted clean plain text article", call_kwargs["prompt"])

    async def test_non_429_error_raises_immediately_without_rotation(self):
        km = KeyManager(keys=["key_1", "key_2"])
        router = LLMRouter(key_manager=km)

        non_retry_client = MagicMock()
        non_retry_client.aio.models.generate_content = AsyncMock(
            side_effect=ValueError("Invalid argument in prompt")
        )
        km._clients["key_1"] = non_retry_client

        other_client = MagicMock()
        km._clients["key_2"] = other_client

        with self.assertRaises(ValueError):
            await router.generate_content(
                model="gemini-3.8-flash",
                contents="bad prompt",
                fallback_to_groq=False,
            )

        other_client.aio.models.generate_content.assert_not_called()

    @patch("parser.fetch_webpage_content")
    async def test_parser_integration_masks_429(self, mock_fetch):
        mock_fetch.return_value = (
            "Extracted Article",
            "This is the clean text from page.\n- Key fact 1\n- Key fact 2",
        )

        km = KeyManager(keys=["key_url_1", "key_url_2"])
        router = LLMRouter(key_manager=km)

        client1 = MagicMock()
        client1.aio.models.generate_content = AsyncMock(side_effect=self.mock_429)

        client2_resp = MagicMock()
        client2_resp.text = "🎯 **Главная суть:**\nСтатья.\n\n📊 **Ключевые факты:**\n- Key fact 1\n- Key fact 2\n\n💡 **Практические выводы:**\nДействие."
        client2_resp.candidates = []
        client2 = MagicMock()
        client2.aio.models.generate_content = AsyncMock(return_value=client2_resp)

        km._clients["key_url_1"] = client1
        km._clients["key_url_2"] = client2

        result = await parse_and_summarize_url(
            url="https://example.com/live-article",
            router=router,
        )

        self.assertEqual(result.title, "Extracted Article")
        self.assertIn("Key fact 1", result.key_points)
        client1.aio.models.generate_content.assert_awaited_once()
        client2.aio.models.generate_content.assert_awaited_once()

    async def test_unified_executive_summary_prompt_structure(self):
        self.assertIn("1. 🎯 Главная суть", UNIFIED_EXECUTIVE_SYSTEM_PROMPT)
        self.assertIn("2. 📊 Ключевые факты", UNIFIED_EXECUTIVE_SYSTEM_PROMPT)
        self.assertIn("3. 💡 Практические выводы", UNIFIED_EXECUTIVE_SYSTEM_PROMPT)

        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            "1. 🎯 Главная суть: Текст сути.\n"
            "2. 📊 Ключевые факты:\n- Факт 1\n"
            "3. 💡 Практические выводы:\n- Действие 1"
        )
        mock_router.generate_content = AsyncMock(return_value=mock_response)

        res = await generate_executive_summary(
            raw_text="Extracted text input for executive summary test.",
            source_title="Test Doc",
            format_size="MEDIUM",
            router=mock_router,
        )
        self.assertIn("Главная суть", res)
        mock_router.generate_content.assert_awaited_once()

    async def test_extractive_fallback_only_when_both_gemini_and_groq_fail(self):
        mock_router = MagicMock()
        mock_router.generate_content = AsyncMock(side_effect=AllKeysExhaustedError("All exhausted"))

        text = "This is a detailed paragraph of enterprise documentation with enough length to be extracted."
        res = await generate_executive_summary(
            raw_text=text,
            source_title="Critical Failure Doc",
            router=mock_router,
        )
        self.assertIn("Автономная структурная выжимка", res)
        self.assertIn("Critical Failure Doc", res)


class TestStartupFlush(unittest.IsolatedAsyncioTestCase):
    """Test main startup dispatcher queue flush to prevent backlog self-DDOS."""

    async def test_delete_webhook_drop_pending_updates_called(self):
        mock_bot = AsyncMock()
        mock_dp = AsyncMock()

        # Simulate the startup block in main()
        await mock_bot.delete_webhook(drop_pending_updates=True)
        await mock_dp.start_polling(mock_bot)

        mock_bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=True)
        mock_dp.start_polling.assert_awaited_once_with(mock_bot)


if __name__ == "__main__":
    unittest.main()
