"""Unified rate limiter: proactive token bucket + reactive stepped cooldown.

Two layers, one file:

1. **Proactive (cross-process):** `_acquire_rate_token(provider)` — token bucket in
   `~/.hermes/state.db` (SQLite WAL), throttles outbound LLM requests *before* they
   hit the provider ceiling. Config: `rate_limiter.<provider>.min_interval_ms` /
   `.burst` in config.yaml. Wired via `_acquire_global_rate_limit(agent)` in
   `chat_completion_helpers.py`.

2. **Reactive (per-model):** `RateLimiter` / `rate_limiter` — upstream-style stepped
   cooldown ladder (30s → 60s → 300s, reset after 10 min quiet) that kicks in when a
   429 actually arrives. Kept API-compatible with the upstream commit
   1ddb03b76f ("per-model rate limit handler with stepped cooldown").
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Dict

# ---------------------------------------------------------------------------
# Layer 1: proactive cross-process token bucket (v2, razul 2026-08-17)
# ---------------------------------------------------------------------------

_DB_LOCK = threading.Lock()


def _db_path() -> str:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return os.path.join(home, "state.db")


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS rate_limiter_state (
               provider TEXT PRIMARY KEY,
               tokens   REAL NOT NULL,
               last_refill REAL NOT NULL,
               burst    INTEGER NOT NULL DEFAULT 1
           )"""
    )


def _get_provider_config(provider: str) -> dict:
    cfg = {"min_interval_ms": 200, "burst": 1}
    try:
        import yaml  # type: ignore
        path = os.path.join(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"), "config.yaml")
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
        section = (doc.get("rate_limiter") or {}).get(provider) or (doc.get("rate_limiter") or {}).get("default") or {}
        if section.get("min_interval_ms") is not None:
            cfg["min_interval_ms"] = float(section["min_interval_ms"])
        if section.get("burst") is not None:
            cfg["burst"] = int(section["burst"])
    except Exception:
        pass
    return cfg


def _acquire_rate_token(provider: str) -> None:
    """Block until a rate-limit token is available for `provider`.

    Token bucket persisted in SQLite so gateway / CLI / cron / subagents all
    share the same throttle. Fail-open: any error returns immediately (rate
    limiting must never block a call).
    """
    cfg = _get_provider_config(provider)
    interval = max(0.0, float(cfg["min_interval_ms"])) / 1000.0
    burst = max(1, int(cfg["burst"]))
    if interval <= 0:
        return
    try:
        with _DB_LOCK:
            conn = sqlite3.connect(_db_path(), timeout=5.0)
            try:
                _ensure_table(conn)
                while True:
                    now = time.monotonic()
                    row = conn.execute(
                        "SELECT tokens, last_refill FROM rate_limiter_state WHERE provider=?",
                        (provider,),
                    ).fetchone()
                    if row is None:
                        tokens, last_refill = float(burst), now
                        conn.execute(
                            "INSERT OR REPLACE INTO rate_limiter_state(provider,tokens,last_refill,burst) VALUES(?,?,?,?)",
                            (provider, tokens, last_refill, burst),
                        )
                    else:
                        tokens, last_refill = float(row[0]), float(row[1])
                    # refill
                    tokens = min(float(burst), tokens + (now - last_refill) / interval)
                    if tokens >= 1.0:
                        conn.execute(
                            "UPDATE rate_limiter_state SET tokens=?, last_refill=? WHERE provider=?",
                            (tokens - 1.0, now, provider),
                        )
                        conn.commit()
                        return
                    # wait for next token
                    wait = (1.0 - tokens) * interval
                    conn.execute(
                        "UPDATE rate_limiter_state SET tokens=?, last_refill=? WHERE provider=?",
                        (tokens, now, provider),
                    )
                    conn.commit()
                    if wait > 5.0:
                        wait = 5.0
                    time.sleep(wait)
            finally:
                conn.close()
    except Exception:
        return


# ---------------------------------------------------------------------------
# Layer 2: reactive per-model stepped cooldown (verbatim upstream 1ddb03b76f)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stepped cooldown ladder (seconds)
_COOLDOWN_STEPS: tuple[int, ...] = (30, 60, 300)

# After this many seconds with no new rate-limit hits the step counter resets.
_RESET_WINDOW: float = 600.0  # 10 minutes


# ---------------------------------------------------------------------------
# Internal per-model state
# ---------------------------------------------------------------------------

@dataclass
class _ModelCooldownState:
    """Mutable cooldown state for a single model."""

    # How many consecutive rate-limit hits (1-indexed).
    step: int = 0

    # ``time.monotonic()`` timestamp when the current cooldown ends.
    cooldown_until: float = 0.0

    # ``time.monotonic()`` of the last hit – used for the reset window.
    last_hit: float = 0.0


# ---------------------------------------------------------------------------
# Public API – singleton ``RateLimiter``
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe, per-model rate-limit handler with stepped cooldown."""

    def __init__(
        self,
        cooldown_steps: tuple[int, ...] = _COOLDOWN_STEPS,
        reset_window: float = _RESET_WINDOW,
    ) -> None:
        self._cooldown_steps = cooldown_steps
        self._reset_window = reset_window
        self._lock = threading.Lock()
        self._models: Dict[str, _ModelCooldownState] = {}

    # -- helpers ----------------------------------------------------------

    def _get_state(self, model: str) -> _ModelCooldownState:
        """Return (or create) the state object for *model*.  Caller must hold ``_lock``."""
        if model not in self._models:
            self._models[model] = _ModelCooldownState()
        return self._models[model]

    def _maybe_reset(self, state: _ModelCooldownState, now: float) -> None:
        """Reset the step counter if the reset window has elapsed since the last hit.

        Caller must hold ``_lock``.
        """
        if state.last_hit and (now - state.last_hit) >= self._reset_window:
            state.step = 0

    # -- public interface -------------------------------------------------

    def check_rate_limit(self, model: str) -> float:
        """Return remaining cooldown seconds for *model*, or ``0`` if none."""
        now = time.monotonic()
        with self._lock:
            state = self._get_state(model)
            remaining = max(0.0, state.cooldown_until - now)
        return remaining

    def record_rate_limit(self, model: str) -> float:
        """Record a rate-limit hit for *model* and return the cooldown duration (seconds).

        The returned value is the number of seconds to wait before the next
        attempt.
        """
        now = time.monotonic()
        with self._lock:
            state = self._get_state(model)

            # Reset step counter if the reset window elapsed.
            self._maybe_reset(state, now)

            # Advance the step (clamped to the ladder length).
            state.step = min(state.step + 1, len(self._cooldown_steps))

            # Look up the cooldown for this step (1-indexed → 0-indexed).
            cooldown = self._cooldown_steps[state.step - 1]

            state.cooldown_until = now + cooldown
            state.last_hit = now

        return float(cooldown)

    def get_step(self, model: str) -> int:
        """Return the current step number for *model* (0 means no active cooldown)."""
        now = time.monotonic()
        with self._lock:
            state = self._get_state(model)
            self._maybe_reset(state, now)
            return state.step

    def get_cooldown_status(self) -> Dict[str, Dict[str, float]]:
        """Return a snapshot of all models with an active cooldown.

        Returns a dict mapping model name → ``{"remaining": <secs>, "step": <int>}``.
        Models whose cooldown has already expired are omitted.
        """
        now = time.monotonic()
        result: Dict[str, Dict[str, float]] = {}
        with self._lock:
            for model, state in self._models.items():
                remaining = max(0.0, state.cooldown_until - now)
                if remaining > 0:
                    result[model] = {
                        "remaining": round(remaining, 2),
                        "step": state.step,
                    }
        return result

    def reset(self, model: str | None = None) -> None:
        """Reset cooldown state.  If *model* is ``None``, reset everything."""
        with self._lock:
            if model is None:
                self._models.clear()
            elif model in self._models:
                del self._models[model]


# Module-level singleton for convenient import.
rate_limiter = RateLimiter()
