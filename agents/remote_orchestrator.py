"""
agents/remote_orchestrator.py
Simple orchestrator client that forwards a model package and optional docs
to a remote CrewAI-style orchestration endpoint. The endpoint URL should be
provided via the `CREWAI_BASE_URL` environment variable and is expected to
expose an `/api/orchestrate` POST route that accepts multipart form-data and
returns the full pipeline bundle JSON.
"""

import os
import requests


def run_remote_pipeline(pkl_path: str, docs_path: str, provider: str, api_key: str) -> dict:
    """Send the model package and docs to the remote orchestration service.

    Expects CREWAI_BASE_URL env var to be set, e.g. https://crew.example.com
    The remote endpoint should accept files `model_file` and `docs_file`, and
    form fields `provider` to choose which backend to use.
    """
    base = os.environ.get("CREWAI_BASE_URL")
    if not base:
        raise RuntimeError("CREWAI_BASE_URL is not set — cannot call remote orchestrator")

    url = base.rstrip("/") + "/api/orchestrate"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    files = {"model_file": open(pkl_path, "rb")}
    if docs_path:
        files["docs_file"] = open(docs_path, "rb")

    data = {"provider": provider}

    try:
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=120)
    finally:
        # Close opened files
        for fh in files.values():
            try:
                fh.close()
            except Exception:
                pass

    if resp.status_code != 200:
        try:
            payload = resp.json()
            msg = payload.get("error") or payload
        except Exception:
            msg = resp.text
        raise RuntimeError(f"Remote orchestrator error ({resp.status_code}): {msg}")

    return resp.json()
