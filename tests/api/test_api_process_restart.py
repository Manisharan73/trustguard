"""Separate integration test for API process restart (Phase 6 persistence).

This test:
1. Spawns an actual uvicorn subprocess hosting the TrustGuard API.
2. Creates review cases (one to complete, one to leave pending) via real HTTP.
3. Submits a human decision on the first review (marking it COMPLETED).
4. Terminates the running API process.
5. Launches a brand-new API subprocess pointing to the exact same database.
6. Verifies both the completed review and pending review survive the process restart.
7. Verifies completed review remains protected against duplicate decisions.
8. Decides the pending review to verify full functionality after restart.
9. Shuts down the second process cleanly.

Can be run via pytest:
    pytest tests/api/test_api_process_restart.py
Or directly as a standalone script:
    python tests/api/test_api_process_restart.py
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import pytest

from tests.api.sample_transactions import LEGITIMATE_TRANSACTION


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server_process(db_path: Path, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["TRUSTGUARD_DB_PATH"] = str(db_path)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def wait_until_ready(proc: subprocess.Popen, port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/reviews/pending"
    while time.time() < deadline:
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            raise RuntimeError(
                f"Server process terminated prematurely (exit code {proc.returncode}):\n{stderr.decode()}"
            )
        try:
            resp = httpx.get(url, timeout=1.0)
            if resp.status_code == 200:
                return
        except Exception:
            time.sleep(0.15)
    proc.terminate()
    raise TimeoutError(f"API process on port {port} failed to become ready within {timeout} seconds")


def stop_server_process(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=4.0)


def run_process_restart_verification(tmp_dir_path: Path) -> None:
    db_file = tmp_dir_path / "restart_integration.db"
    port_1 = find_free_port()

    # --- Phase A: Start First API Process ---
    proc_1 = start_server_process(db_file, port_1)
    try:
        wait_until_ready(proc_1, port_1)
        base_url_1 = f"http://127.0.0.1:{port_1}"

        # 1. Create Review 1 (will be completed)
        tx1 = dict(LEGITIMATE_TRANSACTION, Amount=-5.0)
        resp1 = httpx.post(f"{base_url_1}/reviews", json=tx1, timeout=5.0)
        assert resp1.status_code == 201, f"Create review 1 failed: {resp1.text}"
        review_1 = resp1.json()
        id_1 = review_1["review_id"]
        assert review_1["review_status"] == "PENDING"

        # 2. Complete Review 1
        decide_1 = httpx.post(
            f"{base_url_1}/reviews/{id_1}/decision",
            json={
                "reviewer_id": "reviewer_p1",
                "decision": "APPROVE",
                "reason": "approved prior to restart",
            },
            timeout=5.0,
        )
        assert decide_1.status_code == 200, f"Decide review 1 failed: {decide_1.text}"
        assert decide_1.json()["review_status"] == "COMPLETED"

        # 3. Create Review 2 (will remain pending)
        tx2 = dict(LEGITIMATE_TRANSACTION, Amount=-15.0)
        resp2 = httpx.post(f"{base_url_1}/reviews", json=tx2, timeout=5.0)
        assert resp2.status_code == 201, f"Create review 2 failed: {resp2.text}"
        review_2 = resp2.json()
        id_2 = review_2["review_id"]
        assert review_2["review_status"] == "PENDING"

        # 4. Verify queue state prior to restart
        pending_resp = httpx.get(f"{base_url_1}/reviews/pending", timeout=5.0)
        assert pending_resp.status_code == 200
        pending_ids = [r["review_id"] for r in pending_resp.json()]
        assert pending_ids == [id_2]
        assert id_1 not in pending_ids

    finally:
        # Stop First API Process
        stop_server_process(proc_1)

    assert db_file.exists(), "Database file must exist on disk after process shutdown"

    # --- Phase B: Start Second API Process with Same Database ---
    port_2 = find_free_port()
    proc_2 = start_server_process(db_file, port_2)
    try:
        wait_until_ready(proc_2, port_2)
        base_url_2 = f"http://127.0.0.1:{port_2}"

        # 5. Verify Completed Review 1 survived restart
        get_1 = httpx.get(f"{base_url_2}/reviews/{id_1}", timeout=5.0)
        assert get_1.status_code == 200, f"Fetch review 1 failed: {get_1.text}"
        body_1 = get_1.json()
        assert body_1["review_status"] == "COMPLETED"
        assert body_1["human_decision"] == "APPROVE"
        assert body_1["reviewer_id"] == "reviewer_p1"
        assert body_1["review_reason"] == "approved prior to restart"
        assert body_1["reviewed_at"] is not None
        # Model fields preserved
        assert body_1["guardrail_status"] == "REQUIRES_REVIEW"
        assert body_1["transaction"]["Amount"] == -5.0

        # 6. Verify Pending Review 2 survived restart
        get_2 = httpx.get(f"{base_url_2}/reviews/{id_2}", timeout=5.0)
        assert get_2.status_code == 200, f"Fetch review 2 failed: {get_2.text}"
        body_2 = get_2.json()
        assert body_2["review_status"] == "PENDING"
        assert body_2["human_decision"] is None
        assert body_2["reviewer_id"] is None
        assert body_2["review_reason"] is None
        assert body_2["reviewed_at"] is None
        assert body_2["transaction"]["Amount"] == -15.0

        # 7. Verify Pending Queue after restart
        pending_after = httpx.get(f"{base_url_2}/reviews/pending", timeout=5.0)
        assert pending_after.status_code == 200
        pending_after_ids = [r["review_id"] for r in pending_after.json()]
        assert pending_after_ids == [id_2]
        assert id_1 not in pending_after_ids

        # 8. Attempt duplicate decision on completed review (must be rejected)
        dup_resp = httpx.post(
            f"{base_url_2}/reviews/{id_1}/decision",
            json={
                "reviewer_id": "reviewer_attacker",
                "decision": "REJECT",
                "reason": "attempt overwrite after restart",
            },
            timeout=5.0,
        )
        assert dup_resp.status_code in (400, 409), f"Expected conflict, got: {dup_resp.status_code}"

        # 9. Decide pending review on second process to ensure full functionality
        decide_2 = httpx.post(
            f"{base_url_2}/reviews/{id_2}/decision",
            json={
                "reviewer_id": "reviewer_p2",
                "decision": "REJECT",
                "reason": "rejected on second server instance",
            },
            timeout=5.0,
        )
        assert decide_2.status_code == 200, f"Decide review 2 failed: {decide_2.text}"
        assert decide_2.json()["review_status"] == "COMPLETED"
        assert decide_2.json()["human_decision"] == "REJECT"

        # 10. Verify queue is now empty
        empty_queue = httpx.get(f"{base_url_2}/reviews/pending", timeout=5.0)
        assert empty_queue.status_code == 200
        assert empty_queue.json() == []

    finally:
        # Stop Second API Process
        stop_server_process(proc_2)


def test_api_process_restart(tmp_path):
    """Pytest entrypoint for the API process restart test."""
    run_process_restart_verification(tmp_path)


if __name__ == "__main__":
    print("\n--- Running standalone API process restart test ---")
    with tempfile.TemporaryDirectory() as td:
        run_process_restart_verification(Path(td))
    print("[PASS] Actual API process restart test completed successfully!\n")
