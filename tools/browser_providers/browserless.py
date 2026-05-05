"""Browserless.io cloud browser provider."""

import logging
import os
import uuid
from typing import Dict

import requests

from tools.browser_providers.base import CloudBrowserProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://production-sfo.browserless.io"
_DEFAULT_TTL_MS = 300_000  # 5 minutes — Browserless kills the session after this.


class BrowserlessProvider(CloudBrowserProvider):
    """Browserless (https://browserless.io) cloud browser backend.

    Uses the BaaS Session API (``POST /session``) so each Hermes task runs
    against a named, disposable browser with deterministic lifecycle
    (explicit create + stop).  The token is embedded in the returned
    ``connect`` / ``stop`` URLs — there is no separate project ID.

    Environment variables:
        BROWSERLESS_API_KEY  (required)  API token from browserless.io.
        BROWSERLESS_BASE_URL (optional)  Region / self-host override
                                         (default: production-sfo).
        BROWSERLESS_SESSION_TTL_MS (optional) Per-session TTL in ms
                                         (default: 300000 = 5 min).
        BROWSERLESS_STEALTH  (optional)  ``true`` → enable stealth mode.
        BROWSERLESS_BLOCK_ADS (optional) ``true`` → enable ad blocking.
        BROWSERLESS_PROCESS_KEEP_ALIVE_MS (optional) Reconnect window in
                                         ms (default: 0 = disabled).
    """

    def provider_name(self) -> str:
        return "Browserless"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _base_url(self) -> str:
        return os.environ.get("BROWSERLESS_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")

    def _api_key_or_none(self) -> str | None:
        key = os.environ.get("BROWSERLESS_API_KEY", "").strip()
        return key or None

    def _api_key(self) -> str:
        key = self._api_key_or_none()
        if not key:
            raise ValueError(
                "BROWSERLESS_API_KEY environment variable is required. "
                "Get your key at https://browserless.io"
            )
        return key

    def is_configured(self) -> bool:
        return self._api_key_or_none() is not None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _session_ttl_ms(self) -> int:
        raw = os.environ.get("BROWSERLESS_SESSION_TTL_MS", "").strip()
        if not raw:
            return _DEFAULT_TTL_MS
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                "Invalid BROWSERLESS_SESSION_TTL_MS=%r, falling back to %d ms",
                raw,
                _DEFAULT_TTL_MS,
            )
            return _DEFAULT_TTL_MS
        if value <= 0:
            logger.warning(
                "BROWSERLESS_SESSION_TTL_MS=%r must be positive, falling back to %d ms",
                raw,
                _DEFAULT_TTL_MS,
            )
            return _DEFAULT_TTL_MS
        return value

    def _process_keep_alive_ms(self) -> int:
        raw = os.environ.get("BROWSERLESS_PROCESS_KEEP_ALIVE_MS", "").strip()
        if not raw:
            return 0
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning(
                "Invalid BROWSERLESS_PROCESS_KEEP_ALIVE_MS=%r, disabling keep-alive",
                raw,
            )
            return 0

    def create_session(self, task_id: str) -> Dict[str, object]:
        api_key = self._api_key()
        base = self._base_url()

        ttl_ms = self._session_ttl_ms()
        keep_alive_ms = self._process_keep_alive_ms()
        stealth = os.environ.get("BROWSERLESS_STEALTH", "").strip().lower() == "true"
        block_ads = os.environ.get("BROWSERLESS_BLOCK_ADS", "").strip().lower() == "true"

        body: Dict[str, object] = {"ttl": ttl_ms}
        if keep_alive_ms:
            # processKeepAlive must be <= ttl per Browserless docs.
            body["processKeepAlive"] = min(keep_alive_ms, ttl_ms)
        if stealth:
            body["stealth"] = True
        if block_ads:
            body["blockAds"] = True

        features_enabled = {
            "stealth": stealth,
            "block_ads": block_ads,
            "process_keep_alive": bool(keep_alive_ms),
        }

        try:
            response = requests.post(
                f"{base}/session?token={api_key}",
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to reach Browserless at {base}: {exc}"
            ) from exc

        if not response.ok:
            raise RuntimeError(
                f"Failed to create Browserless session: "
                f"{response.status_code} {response.text}"
            )

        data = response.json()

        try:
            connect_url = data["connect"]
            session_id = data["id"]
        except KeyError as exc:
            raise RuntimeError(
                f"Unexpected Browserless response (missing {exc.args[0]}): {data}"
            ) from exc

        session_name = f"hermes_{task_id}_{uuid.uuid4().hex[:8]}"

        feature_str = ", ".join(k for k, v in features_enabled.items() if v) or "none"
        logger.info(
            "Created Browserless session %s (id=%s, features: %s)",
            session_name,
            session_id,
            feature_str,
        )

        return {
            "session_name": session_name,
            "bb_session_id": session_id,
            "cdp_url": connect_url,
            "features": features_enabled,
        }

    # ------------------------------------------------------------------
    # Session teardown
    # ------------------------------------------------------------------

    def close_session(self, session_id: str) -> bool:
        key = self._api_key_or_none()
        if not key:
            logger.warning(
                "Cannot close Browserless session %s — BROWSERLESS_API_KEY not set",
                session_id,
            )
            return False
        base = self._base_url()

        try:
            response = requests.delete(
                f"{base}/session/{session_id}?token={key}",
                timeout=10,
            )
            if response.status_code in (200, 201, 204):
                logger.debug("Successfully closed Browserless session %s", session_id)
                return True
            # 400 / 404 with a "not found" message means the session already
            # ended (e.g. browser.close() from the client side).  Treat that
            # as success — we achieved the goal of having no live session.
            body = (response.text or "").lower()
            if response.status_code in (400, 404) and (
                "not found" in body or "wasn't found" in body or "wasn\\'t found" in body
            ):
                logger.debug(
                    "Browserless session %s already ended — treating close as no-op",
                    session_id,
                )
                return True
            logger.warning(
                "Failed to close Browserless session %s: HTTP %s - %s",
                session_id,
                response.status_code,
                response.text[:200],
            )
            return False
        except requests.RequestException as exc:
            logger.error("Exception closing Browserless session %s: %s", session_id, exc)
            return False

    def emergency_cleanup(self, session_id: str) -> None:
        key = self._api_key_or_none()
        if not key:
            logger.debug(
                "Skipping emergency cleanup for Browserless session %s — "
                "BROWSERLESS_API_KEY not set",
                session_id,
            )
            return
        base = self._base_url()
        try:
            requests.delete(
                f"{base}/session/{session_id}?token={key}",
                timeout=5,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            logger.debug(
                "Emergency cleanup failed for Browserless session %s: %s",
                session_id,
                exc,
            )
