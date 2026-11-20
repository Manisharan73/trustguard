"""SQLite persistence for review cases (standard library only).

Responsibility: store and fetch ReviewCase records. No workflow rules live here
(those are in service.py) and nothing here knows about HTTP.

Design notes
------------
* One short-lived connection per operation: safe under FastAPI's threadpool and
  nothing is left open. Consequently in-memory databases (":memory:") are NOT
  supported; every connection would see a different empty database.
* The database file and schema are created lazily on first use, so constructing a
  repository performs no I/O (applications and tests that never touch reviews
  never create a database file).
* All SQL is parameterized. The only formatted SQL is built from this package's
  own constants (never from user input) so the CHECK lists cannot drift.
* The database enforces the invariants itself:
    - CHECK constraints on every enumerated column;
    - a PENDING row has no human fields, a COMPLETED row has all of them;
    - a trigger makes COMPLETED rows immutable;
    - a trigger makes the original model/guardrail output immutable.
* complete() is a single conditional UPDATE (... WHERE status = 'PENDING'), so two
  concurrent decisions on one review cannot both succeed.
"""

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from src.decision.engine import RISK_LEVELS, VALID_ACTIONS
from src.human_review.models import (
    REVIEW_COMPLETED,
    REVIEW_PENDING,
    VALID_DECISIONS,
    VALID_REVIEW_STATUSES,
    ReviewAlreadyCompletedError,
    ReviewCase,
    ReviewNotFoundError,
)

DEFAULT_DB_PATH = "data/trustguard.db"

_GUARDRAIL_STATUSES = ("PASSED", "REQUIRES_REVIEW")


def _in_list(values) -> str:
    """Render ('A','B') for a CHECK ... IN clause from trusted constants only."""
    return "(" + ",".join(f"'{value}'" for value in values) + ")"


_ORIGINAL_COLUMNS = (
    "review_id",
    "created_at",
    "transaction_json",
    "prediction",
    "probability",
    "risk_level",
    "recommended_action",
    "guardrail_status",
    "guardrail_reasons_json",
)

_SCHEMA = [
    f"""
    CREATE TABLE IF NOT EXISTS reviews (
        review_id              TEXT PRIMARY KEY,
        created_at             TEXT NOT NULL,
        review_status          TEXT NOT NULL CHECK (review_status IN {_in_list(VALID_REVIEW_STATUSES)}),
        transaction_json       TEXT NOT NULL,
        prediction             INTEGER NOT NULL CHECK (prediction IN (0, 1)),
        probability            REAL NOT NULL CHECK (probability >= 0 AND probability <= 1),
        risk_level             TEXT NOT NULL CHECK (risk_level IN {_in_list(RISK_LEVELS)}),
        recommended_action     TEXT NOT NULL CHECK (recommended_action IN {_in_list(VALID_ACTIONS)}),
        guardrail_status       TEXT NOT NULL CHECK (guardrail_status IN {_in_list(_GUARDRAIL_STATUSES)}),
        guardrail_reasons_json TEXT NOT NULL,
        reviewer_id            TEXT,
        human_decision         TEXT CHECK (human_decision IS NULL OR human_decision IN {_in_list(VALID_DECISIONS)}),
        review_reason          TEXT,
        reviewed_at            TEXT,
        CHECK (
            (review_status = '{REVIEW_PENDING}'
                AND reviewer_id IS NULL AND human_decision IS NULL
                AND review_reason IS NULL AND reviewed_at IS NULL)
            OR
            (review_status = '{REVIEW_COMPLETED}'
                AND reviewer_id IS NOT NULL AND human_decision IS NOT NULL
                AND review_reason IS NOT NULL AND reviewed_at IS NOT NULL)
        )
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews (review_status, created_at)",
    f"""
    CREATE TRIGGER IF NOT EXISTS reviews_completed_is_immutable
    BEFORE UPDATE ON reviews
    WHEN OLD.review_status = '{REVIEW_COMPLETED}'
    BEGIN
        SELECT RAISE(ABORT, 'completed reviews are immutable');
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS reviews_original_output_is_immutable
    BEFORE UPDATE ON reviews
    WHEN {" OR ".join(f"NEW.{col} IS NOT OLD.{col}" for col in _ORIGINAL_COLUMNS)}
    BEGIN
        SELECT RAISE(ABORT, 'original model and guardrail output is immutable');
    END
    """,
]


def _row_to_case(row: sqlite3.Row) -> ReviewCase:
    return ReviewCase(
        review_id=row["review_id"],
        created_at=row["created_at"],
        review_status=row["review_status"],
        transaction=json.loads(row["transaction_json"]),
        prediction=row["prediction"],
        probability=row["probability"],
        risk_level=row["risk_level"],
        recommended_action=row["recommended_action"],
        guardrail_status=row["guardrail_status"],
        guardrail_reasons=tuple(json.loads(row["guardrail_reasons_json"])),
        reviewer_id=row["reviewer_id"],
        human_decision=row["human_decision"],
        review_reason=row["review_reason"],
        reviewed_at=row["reviewed_at"],
    )


class ReviewRepository:
    """Persist and fetch ReviewCase records in a SQLite file."""

    def __init__(self, db_path=DEFAULT_DB_PATH):
        db_path = str(db_path)
        if db_path in ("", ":memory:") or db_path.startswith("file::memory:"):
            raise ValueError(
                "A file path is required; in-memory SQLite databases are not supported."
            )
        self._db_path = db_path
        self._schema_ready = False
        self._lock = threading.Lock()

    @property
    def db_path(self) -> str:
        return self._db_path

    # ---------- connection / schema ----------

    def _open(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._lock:
            if self._schema_ready:
                return
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            with closing(self._open()) as connection, connection:
                for statement in _SCHEMA:
                    connection.execute(statement)
            self._schema_ready = True

    def _connect(self) -> sqlite3.Connection:
        self._ensure_schema()
        return self._open()

    # ---------- operations ----------

    def create(self, case: ReviewCase) -> None:
        """Insert a new case. Raises sqlite3.IntegrityError on a duplicate ID or any
        violated constraint. Non-finite numbers in the transaction raise ValueError."""
        parameters = (
            case.review_id,
            case.created_at,
            case.review_status,
            json.dumps(case.transaction, allow_nan=False, sort_keys=True),
            case.prediction,
            case.probability,
            case.risk_level,
            case.recommended_action,
            case.guardrail_status,
            json.dumps(list(case.guardrail_reasons)),
            case.reviewer_id,
            case.human_decision,
            case.review_reason,
            case.reviewed_at,
        )
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO reviews (
                    review_id, created_at, review_status, transaction_json,
                    prediction, probability, risk_level, recommended_action,
                    guardrail_status, guardrail_reasons_json,
                    reviewer_id, human_decision, review_reason, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                parameters,
            )

    def get(self, review_id: str) -> ReviewCase | None:
        """Return the case, or None if no such review exists."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM reviews WHERE review_id = ?", (review_id,)
            ).fetchone()
        return _row_to_case(row) if row is not None else None

    def list_by_status(self, review_status: str) -> list[ReviewCase]:
        """Cases with the given status, oldest first (FIFO queue order)."""
        if review_status not in VALID_REVIEW_STATUSES:
            raise ValueError(
                f"Invalid review status: {review_status!r}. "
                f"Expected one of: {list(VALID_REVIEW_STATUSES)}."
            )
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM reviews WHERE review_status = ? "
                "ORDER BY created_at ASC, rowid ASC",
                (review_status,),
            ).fetchall()
        return [_row_to_case(row) for row in rows]

    def complete(
        self,
        review_id: str,
        reviewer_id: str,
        human_decision: str,
        review_reason: str,
        reviewed_at: str,
    ) -> ReviewCase:
        """Atomically record the human decision on a PENDING review.

        Raises ReviewNotFoundError if the ID is unknown and
        ReviewAlreadyCompletedError if a decision was already recorded.
        """
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE reviews
                   SET review_status = ?, reviewer_id = ?, human_decision = ?,
                       review_reason = ?, reviewed_at = ?
                 WHERE review_id = ? AND review_status = ?
                """,
                (
                    REVIEW_COMPLETED,
                    reviewer_id,
                    human_decision,
                    review_reason,
                    reviewed_at,
                    review_id,
                    REVIEW_PENDING,
                ),
            )
            row = connection.execute(
                "SELECT * FROM reviews WHERE review_id = ?", (review_id,)
            ).fetchone()

        if cursor.rowcount == 1:
            return _row_to_case(row)
        if row is None:
            raise ReviewNotFoundError(f"Review {review_id} was not found.")
        raise ReviewAlreadyCompletedError(
            f"Review {review_id} is already completed and cannot be changed."
        )
