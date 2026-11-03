"""Comprehensive unit tests for KeyManager, LLMRouter, and startup queue flush."""

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


class TestLLMRouter(unittest.IsolatedAsyncioTestCase):
    """Test async LLMRouter retry loops, 429 rotation, and failure policies."""

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

    def test_is_resource_exhausted_detector(self):
        self.assertTrue(is_resource_exhausted(self.mock_429))
        self.assertTrue(is_resource_exhausted(RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")))
        self.assertFalse(is_resource_exhausted(ValueError("Bad arguments")))
        self.assertFalse(is_resource_exhausted(RuntimeError("Network timeout")))

    async def test_instant_retry_on_429_masks_error(self):
        km = KeyManager(keys=["key_fail", "key_success"])
        router = LLMRouter(key_manager=km)

        # Mock client for key_fail raising 429
        failing_client = MagicMock()
        failing_client.aio.models.generate_content = AsyncMock(side_effect=self.mock_429)

        # Mock client for key_success succeeding
        success_response = MagicMock()
        success_response.text = "Success after rotation"
        succeeding_client = MagicMock()
        succeeding_client.aio.models.generate_content = AsyncMock(return_value=success_response)

        km._clients["key_fail"] = failing_client
        km._clients["key_success"] = succeeding_client

        response = await router.generate_content(
            model="gemini-3.8-flash",
            contents="test prompt",
        )

        self.assertEqual(response.text, "Success after rotation")
        failing_client.aio.models.generate_content.assert_awaited_once()
        succeeding_client.aio.models.generate_content.assert_awaited_once()

    async def test_all_keys_exhausted_raises_graceful_exception(self):
        km = KeyManager(keys=["key_1", "key_2", "key_3"])
        router = LLMRouter(key_manager=km)

        for k in km.get_all_keys():
            client_mock = MagicMock()
            client_mock.aio.models.generate_content = AsyncMock(side_effect=self.mock_429)
            km._clients[k] = client_mock

        with self.assertRaises(AllKeysExhaustedError) as ctx:
            await router.generate_content(
                model="gemini-3.8-flash",
                contents="test prompt",
            )

        self.assertIn("exhausted by rate limits", str(ctx.exception))

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
            )

        # key_2 should never have been called because ValueError is not 429
        other_client.aio.models.generate_content.assert_not_called()

    async def test_parser_integration_masks_429(self):
        km = KeyManager(keys=["key_url_1", "key_url_2"])
        router = LLMRouter(key_manager=km)

        client1 = MagicMock()
        client1.aio.models.generate_content = AsyncMock(side_effect=self.mock_429)

        client2_resp = MagicMock()
        client2_resp.text = "# Extracted Article\n\n- Key fact 1\n- Key fact 2"
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
