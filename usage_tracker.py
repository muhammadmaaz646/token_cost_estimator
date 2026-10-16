"""
Usage Tracker — persists every call's cost to SQLite, per endpoint,
and answers "are we near/over budget?" before and after each call.

An "endpoint" here means a logical name for whatever feature is
calling the model — e.g. "chatbot", "summarizer", "email-drafter".
Multiple physical API routes could even share one endpoint name if
you want them tracked as one budget.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "usage.db"

# Monthly budget ceiling per endpoint, in USD. "default" applies to
# any endpoint name not listed explicitly. Move this to a config file
# or env vars later if you want it changeable without a code deploy.
BUDGET_LIMITS_USD = {
    "default": 5.00,
    "chatbot": 20.00,
    "summarizer": 5.00,
}

# Alert before you actually hit the ceiling, not just when you cross it.
WARNING_THRESHOLD_RATIO = 0.8


@contextmanager
def _get_connection():
    """Context manager so every caller gets the connection closed
    automatically, even if an error happens mid-query."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the usage_log table if it doesn't exist yet. Safe to
    call on every app startup — CREATE TABLE IF NOT EXISTS is a no-op
    if the table's already there."""
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                estimated_cost REAL NOT NULL,
                actual_cost REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def log_usage(
    endpoint: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    estimated_cost: float,
    actual_cost: float,
) -> None:
    """Record one completed call. Called AFTER a real Gemini response
    comes back, using its real token counts."""
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO usage_log
                (endpoint, model, input_tokens, output_tokens,
                 estimated_cost, actual_cost, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                endpoint,
                model,
                input_tokens,
                output_tokens,
                estimated_cost,
                actual_cost,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_total_spent(endpoint: str, since_month_start: bool = True) -> float:
    """Sum of actual_cost logged for this endpoint. Defaults to the
    current calendar month, since budgets are almost always monthly —
    change since_month_start=False for an all-time total instead."""
    query = "SELECT COALESCE(SUM(actual_cost), 0) FROM usage_log WHERE endpoint = ?"
    params = [endpoint]

    if since_month_start:
        month_start = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        query += " AND created_at >= ?"
        params.append(month_start.isoformat())

    with _get_connection() as conn:
        result = conn.execute(query, params).fetchone()
        return result[0]


def get_budget_limit(endpoint: str) -> float:
    return BUDGET_LIMITS_USD.get(endpoint, BUDGET_LIMITS_USD["default"])


def check_budget(endpoint: str, projected_additional_cost: float) -> tuple[str, float, float]:
    """Called BEFORE the real API call. Answers: if this call goes
    through at its estimated worst-case cost, where does that put us?

    Returns (status, current_spent, limit) where status is one of:
        "ok"       — plenty of room
        "warning"  — this call would push us past WARNING_THRESHOLD_RATIO
        "exceeded" — this call would push us past the limit entirely
    """
    limit = get_budget_limit(endpoint)
    current_spent = get_total_spent(endpoint)
    projected_total = current_spent + projected_additional_cost

    if projected_total > limit:
        status = "exceeded"
    elif projected_total > limit * WARNING_THRESHOLD_RATIO:
        status = "warning"
    else:
        status = "ok"

    return status, current_spent, limit
