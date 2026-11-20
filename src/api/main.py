"""TrustGuard API (Phase 6: prediction, guardrails + human review).

Start from the project root:
    uvicorn src.api.main:app --reload
Docs: http://127.0.0.1:8000/docs

This service returns the model's prediction and fraud probability, the Risk
Engine's risk level, the Decision Engine's recommended action and the Guardrails
Engine's verdict on that recommendation. The action is a recommendation only;
the service never approves or rejects a transaction and makes no final decision.

Transactions that need human oversight can be registered and decided through the
/reviews endpoints (see src/api/reviews.py). A reviewer's decision is stored
separately and never overwrites the model's output.
"""

import os
from typing import Annotated
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request

from src.api.pipeline import run_pipeline
from src.api.reviews import router as reviews_router
from src.api.audit import router as audit_router
from src.api.monitoring import router as monitoring_router
from src.api.drift import router as drift_router
from src.api.schemas import PredictionResponse, TransactionRequest
from src.audit.repository import AuditRepository
from src.audit.service import AuditService
from src.decision.engine import DecisionConfig
from src.drift.detector import DriftDetector
from src.drift.repository import DriftRepository
from src.drift.service import DriftService
from src.explainability.engine import ExplainabilityEngine
from src.explainability.schemas import ExplanationResponse
from src.guardrails.advanced import AdvancedGuardrailConfig, OODDetector
from src.guardrails.engine import GuardrailConfig
from src.human_review.repository import DEFAULT_DB_PATH, ReviewRepository
from src.human_review.service import ReviewService
from src.ml.predict import load_model
from src.monitoring.service import MonitoringService
from src.risk.engine import RiskConfig

import math

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup, not per request.
    # If the model file is missing this raises and the app refuses to start.
    app.state.model = load_model()
    app.state.risk_config = RiskConfig()  # risk policy, passed explicitly to the engine
    app.state.decision_config = DecisionConfig()  # decision policy, passed explicitly
    app.state.guardrail_config = GuardrailConfig()  # guardrail settings, passed explicitly
    app.state.advanced_guardrail_config = AdvancedGuardrailConfig()  # Phase 8 advanced guardrails
    app.state.ood_detector = OODDetector.from_reference_file(
        preprocessor=app.state.model.named_steps["preprocessing"]
    )
    # Human review storage. The path is configurable via TRUSTGUARD_DB_PATH; the
    # SQLite file itself is created lazily on first use, not at startup.
    db_path = os.environ.get("TRUSTGUARD_DB_PATH", DEFAULT_DB_PATH)
    app.state.review_service = ReviewService(ReviewRepository(db_path))

    # Audit & Feedback
    audit_repo = AuditRepository(db_path)
    app.state.audit_repository = audit_repo
    app.state.audit_service = AuditService(audit_repo)
    app.state.record_audit_on_predict = not db_path.endswith("unused.db")

    # Monitoring & Analytics
    app.state.monitoring_service = MonitoringService(audit_repo)

    # Drift Detection
    drift_repo = DriftRepository(db_path)
    app.state.drift_repository = drift_repo
    app.state.drift_detector = DriftDetector.from_reference_file()
    app.state.drift_service = DriftService(
        detector=app.state.drift_detector,
        drift_repository=drift_repo,
        audit_repository=audit_repo,
    )

    # Explainability engine (Phase 7: feature attribution in log-odds space)
    app.state.explainability_engine = ExplainabilityEngine(app.state.model)
    yield

def sanitize_error_data(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)

    if isinstance(value, dict):
        return {
            key: sanitize_error_data(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [sanitize_error_data(item) for item in value]

    return value


app = FastAPI(
    title="TrustGuard",
    description="Risk-aware fraud decision system. Phase 8: prediction, guardrails, human review, explainability + advanced guardrails.",
    version="0.8.0",
    lifespan=lifespan,
)

app.include_router(reviews_router)
app.include_router(audit_router)
app.include_router(monitoring_router)
app.include_router(drift_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    errors = sanitize_error_data(exc.errors())

    return JSONResponse(
        status_code=422,
        content={"detail": errors}
    )


@app.post("/predict", response_model=PredictionResponse, response_model_exclude_none=True)
def predict(
    transaction: TransactionRequest,
    request: Request,
    include_advanced: Annotated[
        bool, Query(description="Whether to include advanced guardrail fields in the response")
    ] = False,
) -> dict:
    """Return prediction, probability, risk level, recommended action and guardrails."""
    # The multi-stage orchestration is shared with /reviews: see src/api/pipeline.py.
    features = transaction.model_dump()
    outcome = run_pipeline(features, request.app.state, include_advanced=include_advanced)

    audit_service = getattr(request.app.state, "audit_service", None)
    record_on_predict = getattr(request.app.state, "record_audit_on_predict", True)
    if audit_service is not None and record_on_predict:
        audit_meta = run_pipeline(
            features, request.app.state, include_advanced=True, include_audit_meta=True
        )
        audit_service.record_transaction(transaction=features, pipeline_outcome=audit_meta)

    return outcome


@app.post("/explain", response_model=ExplanationResponse)
def explain(
    transaction: TransactionRequest,
    request: Request,
    top_k: Annotated[int, Query(ge=1, le=30, description="Number of top features to return")] = 5,
) -> ExplanationResponse:
    """Explain why the model produced a prediction and fraud probability for a transaction."""
    engine = getattr(request.app.state, "explainability_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Explainability engine is not initialized.")
    features = transaction.model_dump()
    return engine.explain(features, top_k=top_k)
