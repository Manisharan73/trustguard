"""SQLite persistence for audit records and feedback (standard library only).

Preserves:
- Parameterized SQL.
- Complete separation of model prediction/recommendation from human decisions.
- Database triggers enforcing immutability of original outputs and decisions.
- Short-lived SQLite connection per operation.
"""

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any

from src.audit.models import AuditRecord, FeedbackRecord
from src.human_review.repository import DEFAULT_DB_PATH

_IMMUTABLE_AUDIT_COLS = (
    "audit_id",
    "created_at",
    "transaction_json",
    "prediction",
    "probability",
    "risk_level",
    "recommended_action",
    "phase5_guardrail_status",
    "phase5_guardrail_reasons_json",
    "phase8_guardrail_status",
    "phase8_guardrail_reasons_json",
    "guardrail_status",
    "guardrail_reasons_json",
    "confidence",
    "uncertainty",
    "ood_status",
    "ood_distance",
    "requires_human_review",
) 

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS audit_records (
        audit_id                      TEXT PRIMARY KEY,
        created_at                    TEXT NOT NULL,
        transaction_json              TEXT NOT NULL,
        prediction                    INTEGER NOT NULL CHECK (prediction IN (0, 1)),
        probability                   REAL NOT NULL CHECK (probability >= 0.0 AND probability <= 1.0),
        risk_level                    TEXT NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
        recommended_action            TEXT NOT NULL CHECK (recommended_action IN ('AUTO_APPROVE', 'HUMAN_REVIEW', 'AUTO_REJECT')),
        phase5_guardrail_status       TEXT NOT NULL CHECK (phase5_guardrail_status IN ('PASSED', 'REQUIRES_REVIEW')),
        phase5_guardrail_reasons_json TEXT NOT NULL,
        phase8_guardrail_status       TEXT CHECK (phase8_guardrail_status IS NULL OR phase8_guardrail_status IN ('PASSED', 'REQUIRES_REVIEW')),
        phase8_guardrail_reasons_json TEXT NOT NULL DEFAULT '[]',
        confidence                    REAL,
        uncertainty                   REAL,
        ood_status                    TEXT,
        ood_distance                  REAL,
        guardrail_status              TEXT NOT NULL CHECK (guardrail_status IN ('PASSED', 'REQUIRES_REVIEW')),
        guardrail_reasons_json        TEXT NOT NULL,
        requires_human_review         INTEGER NOT NULL CHECK (requires_human_review IN (0, 1)),
        review_id                     TEXT,
        reviewer_id                   TEXT,
        human_decision                TEXT CHECK (human_decision IS NULL OR human_decision IN ('APPROVE', 'REJECT')),
        review_reason                 TEXT,
        reviewed_at                   TEXT,
        ground_truth                  INTEGER CHECK (ground_truth IS NULL OR ground_truth IN (0, 1)),
        labeled_at                    TEXT,
        updated_at                    TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_records (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_review_id ON audit_records (review_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_audit_review_id_unique ON audit_records (review_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_decision ON audit_records (human_decision)",
    "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_records (recommended_action)",
    "CREATE INDEX IF NOT EXISTS idx_audit_ground_truth ON audit_records (ground_truth)",
    """
    CREATE TABLE IF NOT EXISTS feedback_records (
        feedback_id        TEXT PRIMARY KEY,
        audit_id           TEXT NOT NULL UNIQUE,
        review_id          TEXT,
        created_at         TEXT NOT NULL,
        recommended_action TEXT NOT NULL,
        human_decision     TEXT NOT NULL CHECK (human_decision IN ('APPROVE', 'REJECT')),
        reviewer_id        TEXT NOT NULL,
        review_reason      TEXT NOT NULL,
        disagreement       INTEGER NOT NULL CHECK (disagreement IN (0, 1)),
        is_override        INTEGER NOT NULL CHECK (is_override IN (0, 1)),
        FOREIGN KEY (audit_id) REFERENCES audit_records(audit_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback_records (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_disagreement ON feedback_records (disagreement)",
    f"""
    CREATE TRIGGER IF NOT EXISTS audit_original_output_immutable
    BEFORE UPDATE ON audit_records
    WHEN {" OR ".join(f"NEW.{col} IS NOT OLD.{col}" for col in _IMMUTABLE_AUDIT_COLS)}
    BEGIN
        SELECT RAISE(ABORT, 'original model output and recommendation in audit records are immutable');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS audit_completed_decision_immutable
    BEFORE UPDATE ON audit_records
    WHEN OLD.human_decision IS NOT NULL AND NEW.human_decision IS NOT OLD.human_decision
    BEGIN
        SELECT RAISE(ABORT, 'completed human decisions in audit records cannot be altered');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS audit_completed_review_immutable
    BEFORE UPDATE ON audit_records
    WHEN OLD.human_decision IS NOT NULL AND (
        NEW.reviewer_id IS NOT OLD.reviewer_id OR
        NEW.review_reason IS NOT OLD.review_reason OR
        NEW.reviewed_at IS NOT OLD.reviewed_at
    )
    BEGIN
        SELECT RAISE(ABORT, 'review metadata cannot be altered after decision is recorded');
    END
    """,
]


def _row_to_audit(row: sqlite3.Row) -> AuditRecord:
    return AuditRecord(
        audit_id=row["audit_id"],
        created_at=row["created_at"],
        transaction=json.loads(row["transaction_json"]),
        prediction=row["prediction"],
        probability=row["probability"],
        risk_level=row["risk_level"],
        recommended_action=row["recommended_action"],
        phase5_guardrail_status=row["phase5_guardrail_status"],
        phase5_guardrail_reasons=tuple(json.loads(row["phase5_guardrail_reasons_json"])),
        phase8_guardrail_status=row["phase8_guardrail_status"],
        phase8_guardrail_reasons=tuple(json.loads(row["phase8_guardrail_reasons_json"])),
        confidence=row["confidence"],
        uncertainty=row["uncertainty"],
        ood_status=row["ood_status"],
        ood_distance=row["ood_distance"],
        guardrail_status=row["guardrail_status"],
        guardrail_reasons=tuple(json.loads(row["guardrail_reasons_json"])),
        requires_human_review=bool(row["requires_human_review"]),
        review_id=row["review_id"],
        reviewer_id=row["reviewer_id"],
        human_decision=row["human_decision"],
        review_reason=row["review_reason"],
        reviewed_at=row["reviewed_at"],
        ground_truth=row["ground_truth"],
        labeled_at=row["labeled_at"],
        updated_at=row["updated_at"],
    )


def _row_to_feedback(row: sqlite3.Row) -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=row["feedback_id"],
        audit_id=row["audit_id"],
        review_id=row["review_id"],
        created_at=row["created_at"],
        recommended_action=row["recommended_action"],
        human_decision=row["human_decision"],
        reviewer_id=row["reviewer_id"],
        review_reason=row["review_reason"],
        disagreement=bool(row["disagreement"]),
        is_override=bool(row["is_override"]),
    )


class AuditRepository:
    """Persist and query transaction audit records and reviewer feedback in SQLite."""

    def __init__(self, db_path=DEFAULT_DB_PATH):
        db_path = str(db_path)
        if db_path in ("", ":memory:") or db_path.startswith("file::memory:"):
            raise ValueError("A file path is required; in-memory SQLite databases are not supported.")
        self._db_path = db_path
        self._schema_ready = False
        self._lock = threading.Lock()

    @property
    def db_path(self) -> str:
        return self._db_path

    def _open(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
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

    def create_audit(self, record: AuditRecord) -> None:
        """Insert a new audit record."""
        parameters = (
            record.audit_id,
            record.created_at,
            json.dumps(dict(record.transaction), allow_nan=False, sort_keys=True),
            record.prediction,
            record.probability,
            record.risk_level,
            record.recommended_action,
            record.phase5_guardrail_status,
            json.dumps(list(record.phase5_guardrail_reasons)),
            record.phase8_guardrail_status,
            json.dumps(list(record.phase8_guardrail_reasons)),
            record.confidence,
            record.uncertainty,
            record.ood_status,
            record.ood_distance,
            record.guardrail_status,
            json.dumps(list(record.guardrail_reasons)),
            1 if record.requires_human_review else 0,
            record.review_id,
            record.reviewer_id,
            record.human_decision,
            record.review_reason,
            record.reviewed_at,
            record.ground_truth,
            record.labeled_at,
            record.updated_at or record.created_at,
        )
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO audit_records (
                    audit_id, created_at, transaction_json, prediction, probability,
                    risk_level, recommended_action, phase5_guardrail_status,
                    phase5_guardrail_reasons_json, phase8_guardrail_status,
                    phase8_guardrail_reasons_json, confidence, uncertainty,
                    ood_status, ood_distance, guardrail_status, guardrail_reasons_json,
                    requires_human_review, review_id, reviewer_id, human_decision,
                    review_reason, reviewed_at, ground_truth, labeled_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                parameters,
            )

    def get_audit(self, audit_id: str) -> AuditRecord | None:
        """Fetch audit record by ID."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM audit_records WHERE audit_id = ?", (audit_id,)
            ).fetchone()
        return _row_to_audit(row) if row is not None else None

    def get_audit_by_review_id(self, review_id: str) -> AuditRecord | None:
        """Fetch audit record associated with a review_id."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM audit_records WHERE review_id = ?", (review_id,)
            ).fetchone()
        return _row_to_audit(row) if row is not None else None

    def link_review_id(self, audit_id: str, review_id: str, updated_at: str) -> None:
        """Link an existing audit record with its created review_id."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE audit_records
                   SET review_id = ?, updated_at = ?
                 WHERE audit_id = ?
                """,
                (review_id, updated_at, audit_id),
            )

    def update_decision(
        self,
        *,
        review_id: str,
        reviewer_id: str,
        human_decision: str,
        review_reason: str,
        reviewed_at: str,
        updated_at: str,
        feedback_id: str,
    ) -> tuple[AuditRecord | None, FeedbackRecord | None]:
        """Atomically record a human decision and create feedback record."""
        with closing(self._connect()) as connection, connection:
            # 1. Update audit_record
            cursor = connection.execute(
                """
                UPDATE audit_records
                   SET reviewer_id = ?, human_decision = ?, review_reason = ?,
                       reviewed_at = ?, updated_at = ?
                 WHERE review_id = ? AND human_decision IS NULL
                """,
                (reviewer_id, human_decision, review_reason, reviewed_at, updated_at, review_id),
            )
            if cursor.rowcount == 0:
                # Review not found or already decided
                row = connection.execute(
                    "SELECT * FROM audit_records WHERE review_id = ?", (review_id,)
                ).fetchone()
                if row is None:
                    return None, None
                audit = _row_to_audit(row)
                fb_row = connection.execute(
                    "SELECT * FROM feedback_records WHERE audit_id = ?", (audit.audit_id,)
                ).fetchone()
                return audit, _row_to_feedback(fb_row) if fb_row else None

            # Fetch updated row
            row = connection.execute(
                "SELECT * FROM audit_records WHERE review_id = ?", (review_id,)
            ).fetchone()
            audit = _row_to_audit(row)

            # Determine disagreement & override
            rec_act = audit.recommended_action
            is_override = (
                (rec_act == "AUTO_APPROVE" and human_decision == "REJECT")
                or (rec_act == "AUTO_REJECT" and human_decision == "APPROVE")
            )
            disagreement = is_override or (
                (rec_act == "AUTO_APPROVE" and human_decision == "REJECT")
                or (rec_act == "AUTO_REJECT" and human_decision == "APPROVE")
                or (rec_act == "HUMAN_REVIEW" and human_decision == "REJECT")
            )

            # 2. Insert feedback record
            connection.execute(
                """
                INSERT INTO feedback_records (

                    feedback_id, audit_id, review_id, created_at,
                    recommended_action, human_decision, reviewer_id,
                    review_reason, disagreement, is_override
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    audit.audit_id,
                    review_id,
                    reviewed_at,
                    rec_act,
                    human_decision,
                    reviewer_id,
                    review_reason,
                    1 if disagreement else 0,
                    1 if is_override else 0,
                ),
            )

            fb_row = connection.execute(
                "SELECT * FROM feedback_records WHERE audit_id = ?", (audit.audit_id,)
            ).fetchone()
            feedback = _row_to_feedback(fb_row) if fb_row else None

            return audit, feedback

    def set_ground_truth(self, audit_id: str, ground_truth: int, labeled_at: str) -> AuditRecord | None:
        """Record ground truth label for an audit record atomically."""
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE audit_records
                   SET ground_truth = ?, labeled_at = ?, updated_at = ?
                 WHERE audit_id = ? AND ground_truth IS NULL
                """,
                (ground_truth, labeled_at, labeled_at, audit_id),
            )
            if cursor.rowcount == 0:
                # Determine why no rows were updated
                row = connection.execute(
                    "SELECT ground_truth FROM audit_records WHERE audit_id = ?", (audit_id,)
                ).fetchone()
                if row is None:
                    return None  # audit does not exist
                if row["ground_truth"] is not None:
                    raise ValueError("Ground truth already assigned for this audit record.")
                return None
            row = connection.execute(
                "SELECT * FROM audit_records WHERE audit_id = ?", (audit_id,)
            ).fetchone()
        return _row_to_audit(row) if row is not None else None

    def list_audits(
        self,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        action: str | None = None,
        decision: str | None = None,
        requires_review: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AuditRecord], int]:
        """Query audit records with optional filters and pagination."""
        clauses = []
        params: list[Any] = []

        if start_time:
            clauses.append("created_at >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("created_at <= ?")
            params.append(end_time)
        if action:
            clauses.append("recommended_action = ?")
            params.append(action)
        if decision:
            clauses.append("human_decision = ?")
            params.append(decision)
        if requires_review is not None:
            clauses.append("requires_human_review = ?")
            params.append(1 if requires_review else 0)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with closing(self._connect()) as connection:
            total_count = connection.execute(
                f"SELECT COUNT(*) FROM audit_records {where}", params
            ).fetchone()[0]

            rows = connection.execute(
                f"""
                SELECT * FROM audit_records
                {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            ).fetchall()

        return [_row_to_audit(r) for r in rows], total_count

    def list_feedback(
        self,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        disagreement_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[FeedbackRecord], int, int]:
        """Query feedback records with optional filters."""
        clauses = []
        params: list[Any] = []

        if start_time:
            clauses.append("created_at >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("created_at <= ?")
            params.append(end_time)
        if disagreement_only:
            clauses.append("disagreement = 1")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with closing(self._connect()) as connection:
            total_count = connection.execute(
                f"SELECT COUNT(*) FROM feedback_records {where}", params
            ).fetchone()[0]

            disagree_count = connection.execute(
                f"SELECT COUNT(*) FROM feedback_records {where} {'AND' if where else 'WHERE'} disagreement = 1"
                if not disagreement_only
                else f"SELECT COUNT(*) FROM feedback_records {where}",
                params,
            ).fetchone()[0]

            rows = connection.execute(
                f"""
                SELECT * FROM feedback_records
                {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            ).fetchall()

        return [_row_to_feedback(r) for r in rows], total_count, disagree_count
