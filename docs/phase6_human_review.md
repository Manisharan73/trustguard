# Phase 6 — Human-in-the-Loop (HITL)

Some transactions should not be settled by the system alone. Phase 6 adds a review
queue where a person examines such a transaction and records an **independent**
decision (`APPROVE` or `REJECT`) with their identity, a mandatory reason and a
timestamp.

## The four things that stay separate

| Concept | Produced by | Field(s) |
|---|---|---|
| Model output | ML model | `prediction`, `probability` |
| System recommendation | Risk + Decision engines | `risk_level`, `recommended_action` |
| Guardrail evaluation | Guardrails engine | `guardrail_status`, `guardrail_reasons` |
| **Human decision** | A reviewer | `human_decision`, `reviewer_id`, `review_reason`, `reviewed_at` |

The first three are written once when the review case is created and are **never
changed** (enforced in the service, and again by database triggers). The human
decision is stored beside them, never over them. There is no `final_decision`
field: that belongs to a later phase.

## Architecture

```text
POST /reviews ── transaction ──► run_pipeline ──► ML → Risk → Decision → Guardrails
                                      │
                       needs review?  │ no  → 422, nothing stored
                                      ▼ yes
                              ReviewService ──► ReviewRepository ──► SQLite file
                                      ▲
POST /reviews/{id}/decision ──────────┘   (one decision per review, then COMPLETED)
```

Layers (each only talks to the one below):

| Layer | Module | Responsibility |
|---|---|---|
| HTTP | `src/api/reviews.py` | Parse requests, call the service, map domain errors to status codes. No workflow rules. |
| Schemas | `src/human_review/schemas.py` | Pydantic request/response models. |
| Service | `src/human_review/service.py` | Workflow rules: eligibility, validation, one decision per review. |
| Repository | `src/human_review/repository.py` | SQLite persistence, parameterized SQL, constraints and triggers. |
| Domain | `src/human_review/models.py` | `ReviewCase`, status/decision constants, domain errors. |
| Orchestration | `src/api/pipeline.py` | The four-stage pipeline shared by `/predict` and `/reviews`. |

`/predict` is unchanged in behaviour: its body now calls `run_pipeline`, so the
pipeline logic exists once.

## Endpoints

| Method and path | Purpose | Success | Errors |
|---|---|---|---|
| `POST /reviews` | Assess a transaction and, if it needs review, queue it | `201` | `422` invalid body, or no review needed |
| `GET /reviews/pending` | The pending queue, oldest first | `200` | |
| `GET /reviews/{review_id}` | One review, pending or completed | `200` | `404` unknown, `422` malformed ID |
| `POST /reviews/{review_id}/decision` | Record the human decision | `200` | `404` unknown, `409` already decided, `422` invalid input |

A transaction is queued when `guardrail_status` is `REQUIRES_REVIEW` **or** the
recommended action is `HUMAN_REVIEW`. Otherwise `POST /reviews` returns `422`.

`POST /reviews` takes the same body as `/predict`. The server recomputes the
prediction, risk, recommendation and guardrails itself, so the preserved fields
cannot be supplied or tampered with by the client. `/predict` itself never creates a
review.

### Example: queue a transaction

`POST /reviews` with the 30 transaction features (here `Amount` is negative, which
the request schema accepts but the guardrails escalate):

```json
{ "Time": 160760.0, "V1": -0.674466, "...": "V2 to V28", "Amount": -5.0 }
```

`201 Created`:

```json
{
  "review_id": "62fcf93b-6fef-4664-b17a-58bfca016432",
  "review_status": "PENDING",
  "created_at": "2026-11-20T05:17:53.876373+00:00",
  "transaction": { "Time": 160760.0, "V1": -0.674466, "...": "...", "Amount": -5.0 },
  "prediction": 0,
  "probability": 0.004543439915887623,
  "risk_level": "LOW",
  "recommended_action": "AUTO_APPROVE",
  "guardrail_status": "REQUIRES_REVIEW",
  "guardrail_reasons": ["NEGATIVE_AMOUNT"],
  "reviewer_id": null,
  "human_decision": null,
  "review_reason": null,
  "reviewed_at": null
}
```

### Example: record a decision

`POST /reviews/62fcf93b-6fef-4664-b17a-58bfca016432/decision`

```json
{
  "reviewer_id": "alice",
  "decision": "REJECT",
  "reason": "amount cannot be negative; contacted merchant"
}
```

`200 OK` returns the same review with the original fields untouched and:

```json
{
  "review_status": "COMPLETED",
  "reviewer_id": "alice",
  "human_decision": "REJECT",
  "review_reason": "amount cannot be negative; contacted merchant",
  "reviewed_at": "2026-11-20T05:17:53.894607+00:00"
}
```

Note that `recommended_action` is still `AUTO_APPROVE`: the human disagreed with the
recommendation, and both facts are preserved.

### Validation and errors

| Situation | Status |
|---|---|
| `reviewer_id` missing, empty or whitespace-only; `reason` missing, empty or whitespace-only; `decision` missing or not exactly `APPROVE`/`REJECT` | `422` |
| `reviewer_id` over 100 characters or `reason` over 2000 | `422` |
| Any extra field in the decision body (`prediction`, `risk_level`, `recommended_action`, ...) | `422` |
| `review_id` not a canonical lowercase UUID | `422` |
| Well-formed but unknown `review_id` | `404` |
| Deciding a review that already has a decision (same or different decision) | `409` |
| `PUT`, `PATCH`, `DELETE` on a review | `405` |

`reviewer_id` and `reason` are stored with surrounding whitespace stripped.

## Persistence

Standard-library `sqlite3`, no new dependency. Table `reviews` in one file.

* **Path:** environment variable `TRUSTGUARD_DB_PATH`, default `data/trustguard.db`.
* **Created lazily** on first use (file, parent directory and schema), so starting
  the application, or running tests that never touch reviews, creates nothing.
* **One connection per operation**, so it is safe under FastAPI's threadpool.
  Consequently `:memory:` databases are rejected.
* **All SQL is parameterized.**
* **The database enforces the rules too:** `CHECK` constraints on every enumerated
  column; a `PENDING` row cannot carry human fields and a `COMPLETED` row must carry
  all of them; a trigger makes `COMPLETED` rows immutable; a trigger blocks any edit
  to the original model/guardrail output.
* **Race-safe:** the decision is one conditional `UPDATE ... WHERE review_status =
  'PENDING'`, so two simultaneous decisions cannot both succeed.

Add `data/*.db` to `.gitignore` so the database is never committed.

## Running the tests

```fish
source .venv/bin/activate.fish
pytest -v                              # everything
pytest tests/human_review -v           # Phase 6 unit tests (real temporary SQLite files)
pytest tests/api/test_reviews_integration.py -v   # Phase 6 API tests
```

Every Phase 6 test uses a temporary database under pytest's `tmp_path`; the real
project database is never touched.

## Inspecting reviews

Run the API (optionally with a specific database):

```fish
env TRUSTGUARD_DB_PATH=data/trustguard.db uvicorn src.api.main:app --reload
```

Via the API (`http://127.0.0.1:8000/docs` has a "Try it out" UI for all of these):

```fish
# queue a transaction that needs review
python -c "import httpx; from tests.api.sample_transactions import LEGITIMATE_TRANSACTION as t; print(httpx.post('http://127.0.0.1:8000/reviews', json={**t, 'Amount': -5.0}).json())"

# pending queue
python -c "import httpx; print(httpx.get('http://127.0.0.1:8000/reviews/pending').json())"

# one review (pending or completed): replace REVIEW_ID
python -c "import httpx; print(httpx.get('http://127.0.0.1:8000/reviews/REVIEW_ID').json())"

# record a decision
python -c "import httpx; print(httpx.post('http://127.0.0.1:8000/reviews/REVIEW_ID/decision', json={'reviewer_id': 'alice', 'decision': 'REJECT', 'reason': 'checked with merchant'}).json())"
```

Directly in the database file:

```fish
sqlite3 data/trustguard.db "SELECT review_id, recommended_action, guardrail_reasons_json, created_at FROM reviews WHERE review_status = 'PENDING' ORDER BY created_at;"
sqlite3 data/trustguard.db "SELECT review_id, human_decision, reviewer_id, review_reason, reviewed_at FROM reviews WHERE review_status = 'COMPLETED' ORDER BY reviewed_at;"
```

## Limitations

* **No authentication or authorization.** `reviewer_id` is just a string supplied in
  the request. Anyone who can reach the API can submit a decision under any name.
* **No transaction identifier exists** in this dataset, so the same transaction
  submitted twice to `POST /reviews` creates two review cases. Content-based
  de-duplication was deliberately not added: the dataset contains genuinely distinct
  transactions with identical feature values. "A transaction can only be reviewed
  once" is enforced per review ID (one decision per review).
* **No reassignment, escalation or `REQUEST_MORE_INFORMATION`** (only `APPROVE` and
  `REJECT`), and no audit trail or feedback loop: those belong to later phases.
* **Completed reviews are listed only by ID or by querying the database**, since the
  spec calls for a pending-queue endpoint only.
* The model and guardrail output are stored as submitted; nothing here re-scores a
  stored case if the model or policies change later.
