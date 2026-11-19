"""Databricks authentication provider supporting CLI OAuth U2M and static PAT fallback."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import subprocess
import threading
import time
from typing import Callable


def _parse_iso_expiry(expiry_str: str | None) -> float | None:
    """Parse an ISO-8601 expiry timestamp from `databricks auth token` into epoch seconds."""
    if not expiry_str:
        return None
    try:
        dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


class DatabricksAuthError(RuntimeError):
    """Raised when a Databricks bearer token cannot be obtained."""


class DatabricksAuthProvider:
    """Provides fresh Databricks bearer tokens via static PAT or `databricks auth token`."""

    def __init__(
        self,
        host: str,
        *,
        profile: str = "DEFAULT",
        static_token: str | None = None,
        refresh_skew_s: float = 60.0,
        cli_runner: Callable[[list[str]], str] | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.profile = profile
        self._static_token = static_token
        self._refresh_skew_s = refresh_skew_s
        self._cli_runner = cli_runner or self._default_cli_runner
        self._lock = threading.Lock()
        self._cached_token: str | None = None
        self._expires_at: float | None = None

    @staticmethod
    def _default_cli_runner(cmd: list[str]) -> str:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise DatabricksAuthError(
                f"Failed to obtain Databricks OAuth token via `{' '.join(cmd)}`: {err}. "
                f"Run `databricks auth login --host {cmd[-1]}` or set DATABRICKS_TOKEN."
            )
        return proc.stdout

    def get_token(self, *, force_refresh: bool = False) -> str:
        """Return a valid bearer token, refreshing via Databricks CLI if expired."""
        if self._static_token:
            return self._static_token

        with self._lock:
            now = time.time()
            if (
                not force_refresh
                and self._cached_token is not None
                and (self._expires_at is None or now + self._refresh_skew_s < self._expires_at)
            ):
                return self._cached_token

            cmd = ["databricks", "auth", "token", "--host", self.host]
            if self.profile:
                cmd.extend(["--profile", self.profile])
            raw = self._cli_runner(cmd)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DatabricksAuthError(
                    f"Unexpected non-JSON output from `databricks auth token`: {raw[:200]}"
                ) from exc

            token = payload.get("access_token")
            if not isinstance(token, str) or not token.strip():
                raise DatabricksAuthError(
                    "Databricks CLI returned JSON without an `access_token` field."
                )

            self._cached_token = token.strip()
            expiry = _parse_iso_expiry(payload.get("expiry"))
            # Default 10-minute local cache if expiry is absent
            self._expires_at = expiry if expiry is not None else (now + 600.0)
            return self._cached_token

    def auth_headers(self, *, force_refresh: bool = False) -> dict[str, str]:
        """Return HTTP Authorization header dict."""
        return {"Authorization": f"Bearer {self.get_token(force_refresh=force_refresh)}"}
