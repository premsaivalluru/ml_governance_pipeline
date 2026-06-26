"""
backend/pipeline.py
LangGraph-style sequential pipeline orchestrator.
Runs Agent 01 → 02 → 03 → 04 and writes the full audit
report to outputs/.
"""

import os
import json
import datetime
import pathlib
import sys

# Allow running from any cwd
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import (
    run_governance_agent,
    run_performance_agent,
    run_risk_agent,
    run_summarizer_agent,
)

OUTPUTS_DIR = ROOT / "outputs"


def run_pipeline(pkl_path: str, docs_path: str = "", provider: str = "local", api_key: str = "") -> dict:
    """
    Execute the full 4-agent governance pipeline.

    Parameters
    ----------
    pkl_path  : path to model .pkl / .joblib governance package
    docs_path : optional path to documentation (PDF / MD / TXT)

    Returns
    -------
    Full audit bundle as a dict
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # If a remote provider is requested, delegate orchestration to the remote orchestrator
    if provider and provider != "local":
        try:
            from agents.remote_orchestrator import run_remote_pipeline

            return run_remote_pipeline(pkl_path, docs_path, provider, api_key)
        except Exception as exc:
            raise

    print("\n" + "=" * 60)
    print("  ML GOVERNANCE PIPELINE — STARTING")
    print("=" * 60)

    # ── Agent 01 ───────────────────────────────────────────────────────────
    print("\n▶ Agent 01 / Governance Policy Agent")
    gov_report = run_governance_agent(pkl_path, docs_path)

    # ── Agent 02 ───────────────────────────────────────────────────────────
    print("\n▶ Agent 02 / Performance Monitor Agent")
    perf_report = run_performance_agent(pkl_path)

    # ── Agent 03 ───────────────────────────────────────────────────────────
    print("\n▶ Agent 03 / Risk Analysis Agent")
    risk_report = run_risk_agent(pkl_path)

    # ── Agent 04 ───────────────────────────────────────────────────────────
    print("\n▶ Agent 04 / Governance Summarizer")
    summary_report = run_summarizer_agent(gov_report, perf_report, risk_report)

    # ── Bundle ─────────────────────────────────────────────────────────────
    bundle = {
        "pipeline_run_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_file":      os.path.basename(pkl_path),
        "governance":      gov_report,
        "performance":     perf_report,
        "risk":            risk_report,
        "summary":         summary_report,
    }

    # ── Save ───────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUTS_DIR / f"governance_report_{ts}.json"
    with open(out_path, "w") as fh:
        json.dump(bundle, fh, indent=2)

    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE")
    print(f"  Final decision : {summary_report['final_decision']}")
    print(f"  Composite score: {summary_report['composite_score']}/100")
    print(f"  Report saved   : {out_path}")
    print("=" * 60 + "\n")

    return bundle


# ── CLI entry-point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <model.pkl> [docs.pdf] [provider] [api_key]")
        sys.exit(1)

    pkl      = sys.argv[1]
    docs     = sys.argv[2] if len(sys.argv) > 2 else ""
    provider = sys.argv[3] if len(sys.argv) > 3 else "local"
    api_key  = sys.argv[4] if len(sys.argv) > 4 else ""
    run_pipeline(pkl, docs, provider, api_key)