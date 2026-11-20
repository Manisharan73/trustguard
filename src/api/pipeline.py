"""The multi-stage assessment pipeline shared by /predict and /reviews.

ML prediction -> Risk Engine -> Decision Engine -> Guardrails Engine (v1) -> Advanced Guardrails.

Pure orchestration: every decision is made by the engines. This module has no
FastAPI dependency; it takes the application state object (anything exposing
model, risk_config, decision_config and guardrail_config, and optionally
advanced_guardrail_config and ood_detector).
"""

from src.decision.engine import recommend_action
from src.guardrails.advanced import combine_guardrails, evaluate_advanced_guardrails
from src.guardrails.engine import evaluate_guardrails
from src.ml.predict import predict_transaction
from src.risk.engine import assess_risk


def run_pipeline(
    features: dict,
    state,
    include_advanced: bool = False,
    include_audit_meta: bool = False,
) -> dict:
    """Assess one transaction and return response fields.

    When include_advanced is False (default) and state does not mandate exposure,
    returns the six core fields for 100% backward compatibility. When True,
    includes advanced guardrail metrics (confidence, uncertainty, OOD status, distance).
    The guardrail_status and guardrail_reasons always reflect the combined verdict.
    When include_audit_meta is True, includes internal Phase 5 guardrail details for auditing.
    """
    result = predict_transaction(features, state.model)
    risk_level = assess_risk(result["probability"], state.risk_config)
    # The Decision Engine consumes the risk level only, not the raw probability.
    recommended_action = recommend_action(risk_level, state.decision_config)

    # Phase 5 guardrails
    phase5_guardrails = evaluate_guardrails(
        features,
        result["prediction"],
        risk_level,
        recommended_action,
        state.guardrail_config,
    )
    guardrails = phase5_guardrails

    # Phase 8 advanced guardrails
    advanced_config = getattr(state, "advanced_guardrail_config", None)
    advanced_result = None
    if advanced_config is not None:
        ood_detector = getattr(state, "ood_detector", None)
        advanced_result = evaluate_advanced_guardrails(
            transaction=features,
            prediction=result["prediction"],
            probability=result["probability"],
            risk_level=risk_level,
            recommended_action=recommended_action,
            config=advanced_config,
            ood_detector=ood_detector,
            phase5_reasons=phase5_guardrails.reasons,
            risk_config=getattr(state, "risk_config", None),
        )
        guardrails = combine_guardrails(phase5_guardrails, advanced_result)

    response = {
        **result,
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "guardrail_status": guardrails.status,
        "guardrail_reasons": list(guardrails.reasons),
    }

    if include_audit_meta:
        response.update(
            {
                "phase5_guardrail_status": phase5_guardrails.status,
                "phase5_guardrail_reasons": list(phase5_guardrails.reasons),
            }
        )

    should_expose = (
        include_advanced
        or getattr(state, "expose_advanced_guardrails", False)
        or (advanced_config is not None and getattr(advanced_config, "expose_in_predict", False))
    )

    if should_expose:
        if advanced_result is not None:
            response.update(
                {
                    "advanced_guardrail_status": advanced_result.status,
                    "confidence": advanced_result.confidence,
                    "uncertainty": advanced_result.uncertainty,
                    "ood_status": advanced_result.ood_status,
                    "ood_distance": advanced_result.ood_distance,
                    "advanced_guardrail_reasons": list(advanced_result.reasons),
                }
            )
        else:
            conf = max(result["probability"], 1.0 - result["probability"])
            response.update(
                {
                    "advanced_guardrail_status": "PASSED",
                    "confidence": conf,
                    "uncertainty": 1.0 - conf,
                    "ood_status": "IN_DISTRIBUTION",
                    "ood_distance": 0.0,
                    "advanced_guardrail_reasons": [],
                }
            )

    return response
