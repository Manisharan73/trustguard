# TrustGuard

A modular end‑to‑end fraud detection system built with FastAPI, SQLite, and scikit‑learn.  
It implements the full TrustGuard pipeline:

* **Phases 0‑8** – data loading, model training, prediction API, risk engine, guardrails, explainability (SHAP) and advanced guardrails.
* **Phase 9** – audit & feedback persistence with immutable records and ground‑truth tracking.
* **Phase 10** – monitoring endpoints that compute metrics from persisted audits.
* **Phase 11** – population‑level drift detection with reference data generation.

The project is deliberately lightweight – it only depends on the Python standard library plus the packages listed in `requirements.txt`.  All configuration is environment‑driven via a **`.env`** file.

---

## Quick start

```bash
# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the API (Uvicorn)
uvicorn src.api.main:app --reload
```

The API will be reachable at `http://127.0.0.1:8000`.  Swagger UI is available at `/docs`.

---

## Configuration (`.env`)

| Variable               | Description                                 | Example |
|------------------------|---------------------------------------------|---------|
| `DATABASE_PATH`        | Path to the SQLite database used by audit, monitoring and drift modules. | `data/trustguard.db` |
| `ML_MODEL_PATH`        | Path to the trained model pickle (used by the prediction endpoint). | `model/model.pkl` |
| `HOST`                 | Host address for the FastAPI server.        | `0.0.0.0` |
| `PORT`                 | Port for the FastAPI server.                | `8000` |

Create a file named `.env` (or copy the provided `.env.example`) and set the values you need.

---

## Development & testing

```bash
# Run the full test suite
pytest -q

# Verify database schema & migrations
python scripts/verify.py

# Evaluate the baseline model (metrics must stay unchanged)
python -m src.ml.evaluate
```

All tests should pass (`770 passed, 1 warning`).

---

## Project structure (relevant parts)

```
TrustGuard/
├─ src/
│  ├─ api/                # FastAPI routers & app entry point
│  ├─ audit/              # Audit & feedback models, repository, service
│  ├─ monitoring/         # Monitoring service & schemas
│  ├─ drift/              # Drift detection implementation
│  └─ ml/                 # Model training / evaluation scripts
├─ tests/                  # Pytest suite (phases 0‑11)
├─ requirements.txt        # Dependency list
├─ README.md               # **This file**
├─ .gitignore             # Ignored files for git
└─ .env.example           # Example environment file
```

---

## License

MIT License – see `LICENSE` for details.
