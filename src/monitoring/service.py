"""Monitoring and Analytics service layer.

Aggregates operational metrics, decision distributions, guardrail behavior,
human feedback statistics, and genuine model performance from persisted SQLite tables.
"""

import json
from contextlib import closing
from typing import Any

from src.audit.repository import AuditRepository
from src.monitoring.schemas import (
    DecisionMetricsResponse,
    FeedbackMetricsResponse,
    GuardrailMetricsResponse,
    MonitoringSummaryResponse,
    PerformanceMetricsResponse,
)


class MonitoringService:
    """Derive validated monitoring metrics from persisted audit and feedback records."""

    def __init__(self, audit_repository: AuditRepository):
        self._repository = audit_repository

    def _build_where(self, start_time: str | None, end_time: str | None, time_col: str = "created_at") -> tuple[str, list[Any]]:
        clauses = []
        params = []
        if start_time:
            clauses.append(f"{time_col} >= ?")
            params.append(start_time)
        if end_time:
            clauses.append(f"{time_col} <= ?")
            params.append(end_time)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def get_summary(self, start_time: str | None = None, end_time: str | None = None) -> MonitoringSummaryResponse:
        """Compute executive monitoring summary."""
        where, params = self._build_where(start_time, end_time)

        with closing(self._repository._connect()) as conn:
            # Transaction & Prediction metrics
            t_row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as fraud_preds,
                    SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as legit_preds,
                    SUM(CASE WHEN recommended_action = 'AUTO_APPROVE' THEN 1 ELSE 0 END) as auto_app,
                    SUM(CASE WHEN recommended_action = 'HUMAN_REVIEW' THEN 1 ELSE 0 END) as human_rev,
                    SUM(CASE WHEN recommended_action = 'AUTO_REJECT' THEN 1 ELSE 0 END) as auto_rej,
                    SUM(CASE WHEN requires_human_review = 1 THEN 1 ELSE 0 END) as escalated,
                    SUM(CASE WHEN human_decision IS NOT NULL THEN 1 ELSE 0 END) as completed_reviews,
                    SUM(CASE WHEN ground_truth IS NOT NULL THEN 1 ELSE 0 END) as labeled_count,
                    SUM(CASE WHEN ground_truth IS NOT NULL AND prediction = ground_truth THEN 1 ELSE 0 END) as correct_preds
                FROM audit_records
                {where}
                """,
                params,
            ).fetchone()

            # Feedback metrics
            fb_where, fb_params = self._build_where(start_time, end_time)
            fb_row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_fb,
                    SUM(CASE WHEN human_decision = 'APPROVE' THEN 1 ELSE 0 END) as approvals,
                    SUM(CASE WHEN human_decision = 'REJECT' THEN 1 ELSE 0 END) as rejections,
                    SUM(CASE WHEN disagreement = 1 THEN 1 ELSE 0 END) as disagreements
                FROM feedback_records
                {fb_where}
                """,
                fb_params,
            ).fetchone()

        total = t_row["total"] or 0
        fraud_preds = t_row["fraud_preds"] or 0
        legit_preds = t_row["legit_preds"] or 0
        fraud_rate = float(fraud_preds / total) if total > 0 else 0.0

        auto_app = t_row["auto_app"] or 0
        human_rev = t_row["human_rev"] or 0
        completed_reviews = t_row["completed_reviews"] or 0
        auto_rej = t_row["auto_rej"] or 0
        escalated = t_row["escalated"] or 0
        escalation_rate = float(escalated / total) if total > 0 else 0.0

        total_fb = fb_row["total_fb"] or 0
        approvals = fb_row["approvals"] or 0
        rejections = fb_row["rejections"] or 0
        disagreements = fb_row["disagreements"] or 0
        disagreement_rate = float(disagreements / total_fb) if total_fb > 0 else 0.0

        labeled_count = t_row["labeled_count"] or 0
        correct_preds = t_row["correct_preds"] or 0
        model_accuracy = float(correct_preds / labeled_count) if labeled_count > 0 else None

        return MonitoringSummaryResponse(
            total_transactions=total,
            fraud_predictions=fraud_preds,
            legitimate_predictions=legit_preds,
            fraud_rate=round(fraud_rate, 4),
            auto_approve_count=auto_app,
            human_review_count=completed_reviews,
            auto_reject_count=auto_rej,
            escalation_rate=round(escalation_rate, 4),
            total_human_decisions=total_fb,
            human_approvals=approvals,
            human_rejections=rejections,
            disagreement_count=disagreements,
            disagreement_rate=round(disagreement_rate, 4),
            labeled_transaction_count=labeled_count,
            model_accuracy=round(model_accuracy, 4) if model_accuracy is not None else None,
            window_start=start_time,
            window_end=end_time,
        )

    def get_decision_metrics(self, start_time: str | None = None, end_time: str | None = None) -> DecisionMetricsResponse:
        """Breakdown of system actions and human review decisions."""
        where, params = self._build_where(start_time, end_time)

        with closing(self._repository._connect()) as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN recommended_action = 'AUTO_APPROVE' THEN 1 ELSE 0 END) as auto_app,
                    SUM(CASE WHEN recommended_action = 'HUMAN_REVIEW' THEN 1 ELSE 0 END) as human_rev,
                    SUM(CASE WHEN recommended_action = 'AUTO_REJECT' THEN 1 ELSE 0 END) as auto_rej,
                    SUM(CASE WHEN requires_human_review = 1 THEN 1 ELSE 0 END) as escalated
                FROM audit_records
                {where}
                """,
                params,
            ).fetchone()

            fb_where, fb_params = self._build_where(start_time, end_time)
            fb_row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_fb,
                    SUM(CASE WHEN human_decision = 'APPROVE' THEN 1 ELSE 0 END) as approvals,
                    SUM(CASE WHEN human_decision = 'REJECT' THEN 1 ELSE 0 END) as rejections
                FROM feedback_records
                {fb_where}
                """,
                fb_params,
            ).fetchone()

        total = row["total"] or 0
        auto_app = row["auto_app"] or 0
        human_rev = row["human_rev"] or 0
        auto_rej = row["auto_rej"] or 0
        escalated = row["escalated"] or 0

        total_fb = fb_row["total_fb"] or 0
        approvals = fb_row["approvals"] or 0
        rejections = fb_row["rejections"] or 0

        dec_counts = {
            "AUTO_APPROVE": auto_app,
            "HUMAN_REVIEW": human_rev,
            "AUTO_REJECT": auto_rej,
        }
        dec_pcts = {
            k: round(v / total * 100.0, 2) if total > 0 else 0.0
            for k, v in dec_counts.items()
        }

        human_counts = {
            "APPROVE": approvals,
            "REJECT": rejections,
        }
        human_pcts = {
            k: round(v / total_fb * 100.0, 2) if total_fb > 0 else 0.0
            for k, v in human_counts.items()
        }

        return DecisionMetricsResponse(
            total_transactions=total,
            decision_counts=dec_counts,
            decision_percentages=dec_pcts,
            human_review_rate=round(escalated / total, 4) if total > 0 else 0.0,
            human_decision_counts=human_counts,
            human_decision_percentages=human_pcts,
            window_start=start_time,
            window_end=end_time,
        )

    def get_guardrail_metrics(self, start_time: str | None = None, end_time: str | None = None) -> GuardrailMetricsResponse:
        """Breakdown of guardrail pass/review status and reason code frequencies."""
        where, params = self._build_where(start_time, end_time)

        with closing(self._repository._connect()) as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN guardrail_status = 'PASSED' THEN 1 ELSE 0 END) as passed_cnt,
                    SUM(CASE WHEN guardrail_status = 'REQUIRES_REVIEW' THEN 1 ELSE 0 END) as req_rev_cnt,
                    SUM(CASE WHEN phase5_guardrail_status = 'REQUIRES_REVIEW' THEN 1 ELSE 0 END) as p5_trig,
                    SUM(CASE WHEN phase8_guardrail_status = 'REQUIRES_REVIEW' THEN 1 ELSE 0 END) as p8_trig
                FROM audit_records
                {where}
                """,
                params,
            ).fetchone()

            reason_rows = conn.execute(
                f"SELECT guardrail_reasons_json FROM audit_records {where}", params
            ).fetchall()

        total = row["total"] or 0
        passed_cnt = row["passed_cnt"] or 0
        req_rev_cnt = row["req_rev_cnt"] or 0
        p5_trig = row["p5_trig"] or 0
        p8_trig = row["p8_trig"] or 0

        frequencies: dict[str, int] = {}
        for r_row in reason_rows:
            reasons = json.loads(r_row[0])
            for r in reasons:
                frequencies[r] = frequencies.get(r, 0) + 1

        return GuardrailMetricsResponse(
            total_evaluated=total,
            passed_count=passed_cnt,
            requires_review_count=req_rev_cnt,
            escalation_rate=round(req_rev_cnt / total, 4) if total > 0 else 0.0,
            phase5_triggers_count=p5_trig,
            phase8_triggers_count=p8_trig,
            reason_frequencies=frequencies,
            window_start=start_time,
            window_end=end_time,
        )

    def get_feedback_metrics(self, start_time: str | None = None, end_time: str | None = None) -> FeedbackMetricsResponse:
        """Human review feedback metrics, overrides, and agreement matrix."""
        where, params = self._build_where(start_time, end_time)

        with closing(self._repository._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT recommended_action, human_decision, disagreement, is_override
                FROM feedback_records
                {where}
                """,
                params,
            ).fetchall()

        total = len(rows)
        disagreements = sum(r["disagreement"] for r in rows)
        overrides = sum(r["is_override"] for r in rows)

        matrix = {
            "AUTO_APPROVE_APPROVE": 0,
            "AUTO_APPROVE_REJECT": 0,
            "HUMAN_REVIEW_APPROVE": 0,
            "HUMAN_REVIEW_REJECT": 0,
            "AUTO_REJECT_APPROVE": 0,
            "AUTO_REJECT_REJECT": 0,
        }

        for r in rows:
            key = f"{r['recommended_action']}_{r['human_decision']}"
            if key in matrix:
                matrix[key] += 1

        return FeedbackMetricsResponse(
            total_feedback_records=total,
            disagreement_count=disagreements,
            disagreement_rate=round(disagreements / total, 4) if total > 0 else 0.0,
            override_count=overrides,
            override_rate=round(overrides / total, 4) if total > 0 else 0.0,
            matrix=matrix,
            window_start=start_time,
            window_end=end_time,
        )

    def get_performance_metrics(self, start_time: str | None = None, end_time: str | None = None) -> PerformanceMetricsResponse:
        """Calculate model accuracy, precision, recall, F1, ROC-AUC when genuine ground-truth exists."""
        clauses = ["ground_truth IS NOT NULL"]
        params = []
        if start_time:
            clauses.append("created_at >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("created_at <= ?")
            params.append(end_time)
        where = "WHERE " + " AND ".join(clauses)

        with closing(self._repository._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT prediction, probability, ground_truth
                FROM audit_records
                {where}
                """,
                params,
            ).fetchall()

        if not rows:
            return PerformanceMetricsResponse(
                has_ground_truth=False,
                message="No ground-truth labels available for the specified period.",
                labeled_samples=0,
                window_start=start_time,
                window_end=end_time,
            )

        tp = fp = fn = tn = 0
        y_true = []
        y_prob = []

        for r in rows:
            p = r["prediction"]
            prob = r["probability"]
            gt = r["ground_truth"]

            y_true.append(gt)
            y_prob.append(prob)

            if p == 1 and gt == 1:
                tp += 1
            elif p == 1 and gt == 0:
                fp += 1
            elif p == 0 and gt == 1:
                fn += 1
            else:
                tn += 1

        n = len(rows)
        accuracy = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        roc_auc = None
        if len(set(y_true)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                roc_auc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                pass

        return PerformanceMetricsResponse(
            has_ground_truth=True,
            message=f"Model performance evaluated against {n} ground-truth labeled samples.",
            labeled_samples=n,
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
            confusion_matrix=[[tn, fp], [fn, tp]],
            window_start=start_time,
            window_end=end_time,
        )
