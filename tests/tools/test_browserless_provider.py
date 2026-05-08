"""Tests for BrowserlessProvider (tools/browser_providers/browserless.py)."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

import tools.browser_providers.browserless as browserless_module
from tools.browser_providers.browserless import BrowserlessProvider


class _Response:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int = 200, payload=None, text: str = ""):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload or {}
        self.text = text or json.dumps(payload or {})

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# is_configured / provider_name
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_provider_name(self):
        provider = BrowserlessProvider()
        assert provider.provider_name() == "Browserless"

    def test_is_configured_false_when_missing(self, monkeypatch):
        monkeypatch.delenv("BROWSERLESS_API_KEY", raising=False)
        provider = BrowserlessProvider()
        assert provider.is_configured() is False

    def test_is_configured_false_when_whitespace_only(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "   ")
        provider = BrowserlessProvider()
        assert provider.is_configured() is False

    def test_is_configured_true_when_set(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        provider = BrowserlessProvider()
        assert provider.is_configured() is True


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_minimal_create_session(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.delenv("BROWSERLESS_BASE_URL", raising=False)
        monkeypatch.delenv("BROWSERLESS_SESSION_TTL_MS", raising=False)
        monkeypatch.delenv("BROWSERLESS_STEALTH", raising=False)
        monkeypatch.delenv("BROWSERLESS_BLOCK_ADS", raising=False)
        monkeypatch.delenv("BROWSERLESS_PROCESS_KEEP_ALIVE_MS", raising=False)

        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return _Response(
                200,
                payload={
                    "id": "sess_123",
                    "connect": "wss://production-sfo.browserless.io/e/xyz?token=tok_abc",
                    "stop": "https://production-sfo.browserless.io/session/sess_123?token=tok_abc",
                    "ttl": 300000,
                },
            )

        with patch.object(browserless_module.requests, "post", side_effect=fake_post):
            provider = BrowserlessProvider()
            result = provider.create_session("task-minimal")

        # URL uses the default base and embeds the token in the query string
        assert captured["url"].startswith("https://production-sfo.browserless.io/session?token=")
        assert "tok_abc" in captured["url"]
        # Default TTL is 5 minutes
        assert captured["json"]["ttl"] == 300000
        # Optional flags are omitted when not explicitly enabled
        assert "stealth" not in captured["json"]
        assert "blockAds" not in captured["json"]
        assert "processKeepAlive" not in captured["json"]

        # Returned session dict matches the CloudBrowserProvider contract
        assert result["bb_session_id"] == "sess_123"
        assert result["cdp_url"] == "wss://production-sfo.browserless.io/e/xyz?token=tok_abc"
        assert result["session_name"].startswith("hermes_task-minimal_")
        # Features dict includes new proxy/headless fields (all off by default)
        features = result["features"]
        assert features["stealth"] is False
        assert features["block_ads"] is False
        assert features["process_keep_alive"] is False
        assert features["headless"] is True
        assert features["residential_proxy"] is False
        assert features["external_proxy"] is False

    def test_stealth_and_block_ads_forwarded(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.setenv("BROWSERLESS_STEALTH", "true")
        monkeypatch.setenv("BROWSERLESS_BLOCK_ADS", "true")

        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _Response(
                200,
                payload={
                    "id": "sess_s",
                    "connect": "wss://host/e/s?token=tok_abc",
                    "stop": "https://host/session/sess_s?token=tok_abc",
                },
            )

        with patch.object(browserless_module.requests, "post", side_effect=fake_post):
            provider = BrowserlessProvider()
            result = provider.create_session("task-stealth")

        assert captured["json"]["stealth"] is True
        assert captured["json"]["blockAds"] is True
        assert result["features"]["stealth"] is True
        assert result["features"]["block_ads"] is True

    def test_process_keep_alive_capped_to_ttl(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.setenv("BROWSERLESS_SESSION_TTL_MS", "60000")
        monkeypatch.setenv("BROWSERLESS_PROCESS_KEEP_ALIVE_MS", "120000")  # > ttl

        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _Response(
                200,
                payload={"id": "s", "connect": "wss://host/e/s?token=tok_abc"},
            )

        with patch.object(browserless_module.requests, "post", side_effect=fake_post):
            provider = BrowserlessProvider()
            provider.create_session("task-keep")

        # processKeepAlive must be <= ttl per Browserless docs
        assert captured["json"]["ttl"] == 60000
        assert captured["json"]["processKeepAlive"] == 60000

    def test_invalid_ttl_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.setenv("BROWSERLESS_SESSION_TTL_MS", "not-a-number")

        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _Response(
                200,
                payload={"id": "s", "connect": "wss://host/e/s?token=tok_abc"},
            )

        with patch.object(browserless_module.requests, "post", side_effect=fake_post):
            provider = BrowserlessProvider()
            provider.create_session("task-bad-ttl")

        assert captured["json"]["ttl"] == 300000

    def test_negative_ttl_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.setenv("BROWSERLESS_SESSION_TTL_MS", "-1")

        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _Response(
                200,
                payload={"id": "s", "connect": "wss://host/e/s?token=tok_abc"},
            )

        with patch.object(browserless_module.requests, "post", side_effect=fake_post):
            provider = BrowserlessProvider()
            provider.create_session("task-neg-ttl")

        assert captured["json"]["ttl"] == 300000

    def test_custom_base_url_honored(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.setenv("BROWSERLESS_BASE_URL", "https://production-lon.browserless.io/")

        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            return _Response(
                200,
                payload={"id": "s", "connect": "wss://lon/e/s?token=tok_abc"},
            )

        with patch.object(browserless_module.requests, "post", side_effect=fake_post):
            provider = BrowserlessProvider()
            provider.create_session("task-lon")

        assert captured["url"].startswith(
            "https://production-lon.browserless.io/session?token="
        )

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("BROWSERLESS_API_KEY", raising=False)
        provider = BrowserlessProvider()

        with pytest.raises(ValueError, match="BROWSERLESS_API_KEY"):
            provider.create_session("task-no-key")

    def test_api_failure_surfaces_runtime_error(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")

        with patch.object(
            browserless_module.requests,
            "post",
            return_value=_Response(401, text="Unauthorized"),
        ):
            provider = BrowserlessProvider()
            with pytest.raises(RuntimeError, match="401"):
                provider.create_session("task-fail")

    def test_missing_connect_url_raises(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")

        with patch.object(
            browserless_module.requests,
            "post",
            return_value=_Response(200, payload={"id": "s"}),  # no 'connect'
        ):
            provider = BrowserlessProvider()
            with pytest.raises(RuntimeError, match="connect"):
                provider.create_session("task-broken")


# ---------------------------------------------------------------------------
# close_session
# ---------------------------------------------------------------------------


class TestCloseSession:
    def test_close_session_success(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        captured = {}

        def fake_delete(url, timeout=None):
            captured["url"] = url
            return _Response(204)

        with patch.object(browserless_module.requests, "delete", side_effect=fake_delete):
            provider = BrowserlessProvider()
            ok = provider.close_session("sess_123")

        assert ok is True
        assert "/session/sess_123?token=tok_abc" in captured["url"]

    def test_close_session_missing_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("BROWSERLESS_API_KEY", raising=False)
        provider = BrowserlessProvider()
        assert provider.close_session("sess_123") is False

    def test_close_session_network_error_returns_false(self, monkeypatch):
        import requests as real_requests

        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")

        with patch.object(
            browserless_module.requests,
            "delete",
            side_effect=real_requests.ConnectionError("boom"),
        ):
            provider = BrowserlessProvider()
            assert provider.close_session("sess_123") is False

    def test_close_session_non_2xx_returns_false(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")

        with patch.object(
            browserless_module.requests,
            "delete",
            return_value=_Response(500, text="server error"),
        ):
            provider = BrowserlessProvider()
            assert provider.close_session("sess_123") is False

    def test_close_session_already_gone_treated_as_success(self, monkeypatch):
        """400/404 with 'not found' means the session already ended — no-op."""
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")

        with patch.object(
            browserless_module.requests,
            "delete",
            return_value=_Response(
                400,
                text='Failed to delete session: "Session ID \\"sess_123\\" wasn\'t found."',
            ),
        ):
            provider = BrowserlessProvider()
            # Browserless returns 400 when the session ended on its own side;
            # treat that as success so the supervisor doesn't spam warnings.
            assert provider.close_session("sess_123") is True

    def test_close_session_404_treated_as_success(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")

        with patch.object(
            browserless_module.requests,
            "delete",
            return_value=_Response(404, text="Session not found"),
        ):
            provider = BrowserlessProvider()
            assert provider.close_session("sess_123") is True


# ---------------------------------------------------------------------------
# emergency_cleanup
# ---------------------------------------------------------------------------


class TestEmergencyCleanup:
    def test_emergency_cleanup_no_key_is_silent(self, monkeypatch):
        monkeypatch.delenv("BROWSERLESS_API_KEY", raising=False)
        provider = BrowserlessProvider()
        # Must not raise — called from atexit / signal handlers.
        provider.emergency_cleanup("sess_123")

    def test_emergency_cleanup_swallows_exceptions(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        with patch.object(
            browserless_module.requests,
            "delete",
            side_effect=RuntimeError("any error"),
        ):
            provider = BrowserlessProvider()
            # Must not propagate
            provider.emergency_cleanup("sess_123")


# ---------------------------------------------------------------------------
# Headless mode
# ---------------------------------------------------------------------------


class TestHeadlessMode:
    def test_headless_true_by_default_omits_param(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.delenv("BROWSERLESS_HEADLESS", raising=False)
        monkeypatch.delenv("BROWSERLESS_EXTERNAL_PROXY", raising=False)
        monkeypatch.delenv("BROWSERLESS_PROXY", raising=False)
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _Response(200, payload={"id": "s", "connect": "wss://host/e/s?token=tok_abc"})

        with patch.object(browserless_module.requests, "post", side_effect=fake_post):
            BrowserlessProvider().create_session("task-headless-default")

        # headless omitted when True — Browserless server defaults to headless
        assert "headless" not in captured["json"]

    def test_headless_false_sent_in_body(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.setenv("BROWSERLESS_HEADLESS", "false")
        monkeypatch.delenv("BROWSERLESS_EXTERNAL_PROXY", raising=False)
        monkeypatch.delenv("BROWSERLESS_PROXY", raising=False)
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _Response(200, payload={"id": "s", "connect": "wss://host/e/s?token=tok_abc"})

        with patch.object(browserless_module.requests, "post", side_effect=fake_post):
            BrowserlessProvider().create_session("task-headful")

        assert captured["json"].get("headless") is False


# ---------------------------------------------------------------------------
# Built-in residential proxy
# ---------------------------------------------------------------------------


class TestResidentialProxy:
    _BASE_CONNECT = "wss://production-sfo.browserless.io/e/xyz?token=tok_abc"

    def _fake_post(self, url, headers=None, json=None, timeout=None):
        return _Response(200, payload={"id": "sess_proxy", "connect": self._BASE_CONNECT})

    def test_residential_proxy_params_on_connect_url(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.setenv("BROWSERLESS_PROXY", "residential")
        monkeypatch.setenv("BROWSERLESS_PROXY_COUNTRY", "us")
        monkeypatch.delenv("BROWSERLESS_EXTERNAL_PROXY", raising=False)
        monkeypatch.delenv("BROWSERLESS_PROXY_CITY", raising=False)
        monkeypatch.delenv("BROWSERLESS_PROXY_STICKY", raising=False)
        monkeypatch.delenv("BROWSERLESS_PROXY_LOCALE_MATCH", raising=False)

        with patch.object(browserless_module.requests, "post", side_effect=self._fake_post):
            result = BrowserlessProvider().create_session("task-proxy")

        assert "proxy=residential" in result["cdp_url"]
        assert "proxyCountry=us" in result["cdp_url"]
        assert result["features"]["residential_proxy"] is True
        assert result["features"]["external_proxy"] is False

    def test_proxy_sticky_appended(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.setenv("BROWSERLESS_PROXY", "residential")
        monkeypatch.setenv("BROWSERLESS_PROXY_STICKY", "true")
        monkeypatch.delenv("BROWSERLESS_PROXY_COUNTRY", raising=False)
        monkeypatch.delenv("BROWSERLESS_PROXY_CITY", raising=False)
        monkeypatch.delenv("BROWSERLESS_EXTERNAL_PROXY", raising=False)
        monkeypatch.delenv("BROWSERLESS_PROXY_LOCALE_MATCH", raising=False)

        with patch.object(browserless_module.requests, "post", side_effect=self._fake_post):
            result = BrowserlessProvider().create_session("task-sticky")

        assert "proxySticky=true" in result["cdp_url"]

    def test_no_proxy_env_no_params(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.delenv("BROWSERLESS_PROXY", raising=False)
        monkeypatch.delenv("BROWSERLESS_EXTERNAL_PROXY", raising=False)

        def fake_no_proxy(url, headers=None, json=None, timeout=None):
            return _Response(200, payload={"id": "s", "connect": "wss://host/e/s?token=tok_abc"})

        with patch.object(browserless_module.requests, "post", side_effect=fake_no_proxy):
            result = BrowserlessProvider().create_session("task-no-proxy")

        assert "proxy" not in result["cdp_url"]
        assert result["features"]["residential_proxy"] is False
        assert result["features"]["external_proxy"] is False

    def test_append_query_params_preserves_existing(self):
        provider = BrowserlessProvider()
        url = "wss://host/e/abc?token=tok&foo=bar"
        result = provider._append_query_params(url, {"proxy": "residential", "proxyCountry": "de"})
        assert "token=tok" in result
        assert "foo=bar" in result
        assert "proxy=residential" in result
        assert "proxyCountry=de" in result


# ---------------------------------------------------------------------------
# External (third-party) proxy
# ---------------------------------------------------------------------------


class TestExternalProxy:
    def test_external_proxy_sent_in_session_body(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.setenv("BROWSERLESS_EXTERNAL_PROXY", "socks5://user:pass@proxy.example.com:1080")
        monkeypatch.delenv("BROWSERLESS_PROXY", raising=False)
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _Response(200, payload={"id": "s", "connect": "wss://host/e/s?token=tok_abc"})

        with patch.object(browserless_module.requests, "post", side_effect=fake_post):
            result = BrowserlessProvider().create_session("task-ext-proxy")

        assert captured["json"]["externalProxy"] == "socks5://user:pass@proxy.example.com:1080"
        assert result["features"]["external_proxy"] is True

    def test_external_proxy_401_retries_without_proxy(self, monkeypatch):
        """If external proxy returns 401 (plan limitation), retry without it."""
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.setenv("BROWSERLESS_EXTERNAL_PROXY", "http://proxy.example.com:8080")
        monkeypatch.delenv("BROWSERLESS_PROXY", raising=False)
        call_count = 0

        def fake_post(url, headers=None, json=None, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _Response(401, text="Unauthorized")
            return _Response(200, payload={"id": "s", "connect": "wss://host/e/s?token=tok_abc"})

        with patch.object(browserless_module.requests, "post", side_effect=fake_post):
            result = BrowserlessProvider().create_session("task-ext-proxy-fallback")

        assert call_count == 2
        assert result["features"]["external_proxy"] is False

    def test_external_proxy_not_sent_when_empty(self, monkeypatch):
        monkeypatch.setenv("BROWSERLESS_API_KEY", "tok_abc")
        monkeypatch.setenv("BROWSERLESS_EXTERNAL_PROXY", "")
        monkeypatch.delenv("BROWSERLESS_PROXY", raising=False)
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _Response(200, payload={"id": "s", "connect": "wss://host/e/s?token=tok_abc"})

        with patch.object(browserless_module.requests, "post", side_effect=fake_post):
            BrowserlessProvider().create_session("task-no-ext-proxy")

        assert "externalProxy" not in captured["json"]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_registered_under_browserless_key(self):
        from tools.browser_tool import _PROVIDER_REGISTRY

        assert "browserless" in _PROVIDER_REGISTRY
        assert _PROVIDER_REGISTRY["browserless"] is BrowserlessProvider

    def test_normalize_accepts_browserless(self):
        from tools.tool_backend_helpers import normalize_browser_cloud_provider

        assert normalize_browser_cloud_provider("Browserless") == "browserless"
        assert normalize_browser_cloud_provider("  browserless  ") == "browserless"
