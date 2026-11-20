
import json
import subprocess
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.main import app


passed = 0


def check(condition, message):
    global passed

    if not condition:
        raise AssertionError(message)

    passed += 1
    print(f"[PASS] {message}")


def transaction():
    data = {
        "Time": -100,
        "Amount": 100
    }

    for i in range(1, 29):
        data[f"V{i}"] = 0

    return data


def find_review_id(data):
    if isinstance(data, dict):
        for key in ("review_id", "id"):
            value = data.get(key)
            if value is not None:
                return str(value)

        for value in data.values():
            result = find_review_id(value)
            if result:
                return result

    elif isinstance(data, list):
        for item in data:
            result = find_review_id(item)
            if result:
                return result

    return None


def find_status(data):
    if isinstance(data, dict):
        for key in ("review_status", "status"):
            if key in data:
                return data[key]

        for value in data.values():
            result = find_status(value)
            if result:
                return result

    return None


def main():
    global passed

    print("\n========== PHASE 6 VERIFICATION ==========\n")

    # STEP 1: Existing test suite
    print("STEP 1: Running existing pytest suite...\n")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v"],
        cwd=PROJECT_ROOT,
        check=False
    )

    check(result.returncode == 0, "Complete pytest suite passed")

    # STEP 2: Review API tests
    print("\nSTEP 2: Testing Human Review API...\n")

    with TestClient(app) as client:

        # Create first review
        response = client.post(
            "/reviews",
            json=transaction()
        )

        check(
            response.status_code in (200, 201),
            "Review creation accepted"
        )

        review = response.json()
        review_id = find_review_id(review)

        check(review_id is not None, "Review ID generated")

        check(
            find_status(review) == "PENDING",
            "New review has PENDING status"
        )

        print("\nCreated review:")
        print(json.dumps(review, indent=2))

        # Retrieve pending queue
        response = client.get("/reviews/pending")

        check(
            response.status_code == 200,
            "Pending review queue retrieved"
        )

        pending = response.json()

        check(
            review_id in json.dumps(pending),
            "New review appears in pending queue"
        )

        # Retrieve individual review
        response = client.get(f"/reviews/{review_id}")

        check(
            response.status_code == 200,
            "Individual review retrieved"
        )

        original = response.json()

        # Approve the review
        decision = {
            "reviewer_id": "reviewer_test_001",
            "decision": "APPROVE",
            "reason": "Transaction manually verified"
        }

        response = client.post(
            f"/reviews/{review_id}/decision",
            json=decision
        )

        check(
            response.status_code in (200, 201),
            "Human APPROVE decision accepted"
        )

        completed = response.json()

        check(
            find_status(completed) == "COMPLETED",
            "Review status changed to COMPLETED"
        )

        check(
            "APPROVE" in json.dumps(completed),
            "Human decision recorded"
        )

        check(
            "reviewer_test_001" in json.dumps(completed),
            "Reviewer identity recorded"
        )

        check(
            "Transaction manually verified" in json.dumps(completed),
            "Review reason recorded"
        )

        # Ensure original ML fields remain unchanged
        for field in (
            "prediction",
            "probability",
            "risk_level",
            "recommended_action",
            "guardrail_status",
            "guardrail_reasons"
        ):
            if field in original:
                check(
                    completed.get(field) == original.get(field),
                    f"Original {field} preserved"
                )

        # Duplicate decision must fail
        response = client.post(
            f"/reviews/{review_id}/decision",
            json={
                "reviewer_id": "reviewer_test_002",
                "decision": "REJECT",
                "reason": "Trying a second decision"
            }
        )

        check(
            response.status_code in (400, 409),
            "Completed review cannot be decided again"
        )

        # Completed review must not appear in pending queue
        response = client.get("/reviews/pending")

        check(
            response.status_code == 200,
            "Pending queue remains accessible"
        )

        check(
            review_id not in json.dumps(response.json()),
            "Completed review excluded from pending queue"
        )

        # Invalid review ID
        # Generate a valid UUID that does not exist in the database
        missing_review_id = str(uuid.uuid4())

        response = client.get(
            f"/reviews/{missing_review_id}"
        )

        check(
            response.status_code == 404,
            "Nonexistent review returns HTTP 404"
        )

        response = client.post(
            f"/reviews/{missing_review_id}/decision",
            json={
                "reviewer_id": "reviewer_test",
                "decision": "APPROVE",
                "reason": "Testing nonexistent review"
            }
        )

        check(
            response.status_code == 404,
            "Unknown review cannot receive a decision"
        )
        
        # Missing reviewer ID
        response = client.post(
            f"/reviews/{review_id}/decision",
            json={
                "decision": "APPROVE",
                "reason": "Missing reviewer test"
            }
        )

        check(
            response.status_code == 422,
            "Missing reviewer ID rejected"
        )

        # Empty reason
        response = client.post(
            f"/reviews/{review_id}/decision",
            json={
                "reviewer_id": "reviewer_test",
                "decision": "APPROVE",
                "reason": "   "
            }
        )

        check(
            response.status_code == 422,
            "Empty or whitespace-only reason rejected"
        )

    # STEP 3: Reopen application and verify persistence
    print("\nSTEP 3: Testing persistence...\n")

    with TestClient(app) as client:
        response = client.get(f"/reviews/{review_id}")

        check(
            response.status_code == 200,
            "Review survives application client restart"
        )

        persisted = response.json()

        check(
            find_status(persisted) == "COMPLETED",
            "Completed status persists"
        )

        check(
            "reviewer_test_001" in json.dumps(persisted),
            "Reviewer information persists"
        )

    print("\n==========================================")
    print(f"PHASE 6 VERIFICATION PASSED: {passed} checks")
    print("==========================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[FAIL] {error}")
        sys.exit(1)
