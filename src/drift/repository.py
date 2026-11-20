"""SQLite persistence for historical drift reports."""

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from src.drift.models import DriftReport
from src.human_review.repository import DEFAULT_DB_PATH

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS drift_reports (
        analysis_id             TEXT PRIMARY KEY,
        created_at              TEXT NOT NULL,
        window_start            TEXT,
        window_end              TEXT,
        sample_count            INTEGER NOT NULL,
        reference_sample_count  INTEGER NOT NULL,
        overall_status          TEXT NOT NULL CHECK (overall_status IN ('NO_DRIFT', 'MODERATE_DRIFT', 'SIGNIFICANT_DRIFT', 'INSUFFICIENT_DATA')),
        drift_detected          INTEGER NOT NULL CHECK (drift_detected IN (0, 1)),
        mean_feature_psi        REAL NOT NULL,
        max_feature_psi         REAL NOT NULL,
        drifted_features_count  INTEGER NOT NULL,
        feature_metrics_json    TEXT NOT NULL,
        behavioral_metrics_json TEXT NOT NULL,
        concept_drift_json      TEXT NOT NULL,
        method                  TEXT NOT NULL,
        thresholds_json         TEXT NOT NULL,
        limitations_json        TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_drift_created_at ON drift_reports (created_at)",
]


def _row_to_report(row: sqlite3.Row) -> DriftReport:
    return DriftReport(
        analysis_id=row["analysis_id"],
        created_at=row["created_at"],
        window_start=row["window_start"],
        window_end=row["window_end"],
        sample_count=row["sample_count"],
        reference_sample_count=row["reference_sample_count"],
        overall_status=row["overall_status"],
        drift_detected=bool(row["drift_detected"]),
        mean_feature_psi=row["mean_feature_psi"],
        max_feature_psi=row["max_feature_psi"],
        drifted_features_count=row["drifted_features_count"],
        feature_metrics=json.loads(row["feature_metrics_json"]),
        behavioral_metrics=json.loads(row["behavioral_metrics_json"]),
        concept_drift=json.loads(row["concept_drift_json"]),
        method=row["method"],
        thresholds=json.loads(row["thresholds_json"]),
        limitations=json.loads(row["limitations_json"]),
    )


class DriftRepository:
    """Store and fetch DriftReport records in SQLite."""

    def __init__(self, db_path=DEFAULT_DB_PATH):
        db_path = str(db_path)
        if db_path in ("", ":memory:") or db_path.startswith("file::memory:"):
            raise ValueError("A file path is required; in-memory SQLite databases are not supported.")
        self._db_path = db_path
        self._schema_ready = False
        self._lock = threading.Lock()

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

    def save_report(self, report: DriftReport) -> None:
        """Persist a new drift analysis report."""
        params = (
            report.analysis_id,
            report.created_at,
            report.window_start,
            report.window_end,
            report.sample_count,
            report.reference_sample_count,
            report.overall_status,
            1 if report.drift_detected else 0,
            report.mean_feature_psi,
            report.max_feature_psi,
            report.drifted_features_count,
            json.dumps(report.feature_metrics),
            json.dumps(report.behavioral_metrics),
            json.dumps(report.concept_drift),
            report.method,
            json.dumps(report.thresholds),
            json.dumps(report.limitations),
        )
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO drift_reports (
                    analysis_id, created_at, window_start, window_end,
                    sample_count, reference_sample_count, overall_status,
                    drift_detected, mean_feature_psi, max_feature_psi,
                    drifted_features_count, feature_metrics_json,
                    behavioral_metrics_json, concept_drift_json,
                    method, thresholds_json, limitations_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )

    def get_latest_report(self) -> DriftReport | None:
        """Fetch the most recent drift report."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM drift_reports ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return _row_to_report(row) if row is not None else None

    def get_report(self, analysis_id: str) -> DriftReport | None:
        """Fetch report by analysis_id."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM drift_reports WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()
        return _row_to_report(row) if row is not None else None

    def list_reports(self, limit: int = 50, offset: int = 0) -> tuple[list[DriftReport], int]:
        """List historical reports ordered newest first."""
        with closing(self._connect()) as conn:
            total = conn.execute("SELECT COUNT(*) FROM drift_reports").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM drift_reports ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_report(r) for r in rows], total
