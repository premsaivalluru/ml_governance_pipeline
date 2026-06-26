"""
agents/summarizer_agent.py
Agent 04 — Governance Summarizer
Receives the three specialist reports and synthesises a final
deployment decision.  This is the new agent not present in the
original notebooks.
"""

import json
from typing import List, Dict
from pydantic import BaseModel


# ── Summary report schema ──────────────────────────────────────────────────
class SummaryReport(BaseModel):
    agent_name:        str
    final_decision:    str       # DEPLOYMENT APPROVED | CONDITIONAL APPROVAL | DEPLOYMENT REJECTED
    decision_rationale: str
    composite_score:   int       # 0-100
    agent_scores: Dict           # {governance, performance, risk}
    critical_issues:   List[str]
    conditions:        List[str]
    executive_summary: str
    next_steps:        List[str]


# ── Decision thresholds ────────────────────────────────────────────────────
def _composite_score(gov_score: int, perf_score: float, risk_score: int) -> int:
    """
    Weighted composite:
      40 % governance · 40 % performance · 20 % risk (inverted, lower=better)
    """
    risk_inverted = max(0, 100 - risk_score)
    composite = (gov_score * 0.40) + (perf_score * 0.40) + (risk_inverted * 0.20)
    return round(composite)


def _decide(gov: dict, perf: dict, risk: dict, composite: int) -> str:
    gov_status   = gov.get("status", "FAIL")
    perf_status  = perf.get("status", "FAIL")
    risk_level   = risk.get("overall_risk_level", "HIGH")
    deployment   = perf.get("deployment_readiness", "NOT_READY")

    # Hard REJECT conditions
    if (
        gov_status == "FAIL"
        or risk_level == "HIGH"
        or deployment == "NOT_READY"
        or composite < 60
    ):
        return "DEPLOYMENT REJECTED"

    # All green
    if (
        gov_status == "PASS"
        and perf_status == "PASS"
        and risk_level == "LOW"
        and composite >= 85
    ):
        return "DEPLOYMENT APPROVED"

    # Everything else — conditional
    return "CONDITIONAL APPROVAL"


# ── Public entry-point ─────────────────────────────────────────────────────
def run_summarizer_agent(
    governance_report: dict,
    performance_report: dict,
    risk_report: dict,
) -> dict:
    """
    Synthesise the three agent reports.

    Parameters
    ----------
    governance_report   : output of run_governance_agent()
    performance_report  : output of run_performance_agent()
    risk_report         : output of run_risk_agent()

    Returns
    -------
    SummaryReport serialised as a plain dict
    """
    print("[Agent 04] Governance Summarizer starting…")

    gov_score   = int(governance_report.get("score", 0))
    perf_score  = float(performance_report.get("performance_score", 0))
    risk_score  = int(risk_report.get("risk_score", 100))

    composite = _composite_score(gov_score, perf_score, risk_score)
    decision  = _decide(governance_report, performance_report, risk_report, composite)

    # ── Critical issues (collect FAILs and HIGHs) ─────────────────────────
    critical_issues: List[str] = []
    if governance_report.get("status") == "FAIL":
        critical_issues.append(
            f"Governance audit FAILED (score {gov_score}/100) — "
            "deployment blocked until resolved."
        )
    if risk_report.get("overall_risk_level") == "HIGH":
        critical_issues.append(
            f"Overall risk level is HIGH (risk score {risk_score}/100) — "
            "mitigation required before production."
        )
    if performance_report.get("deployment_readiness") == "NOT_READY":
        critical_issues.append(
            "Performance thresholds not met — model is not deployment-ready."
        )
    # Bubble up bias-risk findings if HIGH
    if risk_report.get("bias_risk", {}).get("level") == "HIGH":
        critical_issues.append(
            "Bias risk is HIGH — fairness testing across demographic "
            "sub-groups is mandatory."
        )

    # ── Conditions (only for CONDITIONAL) ─────────────────────────────────
    conditions: List[str] = []
    if decision == "CONDITIONAL APPROVAL":
        if governance_report.get("status") == "WARNING":
            conditions.append(
                "Resolve all governance WARNING items within 30 days of deployment."
            )
        if performance_report.get("status") == "WARNING":
            conditions.append(
                "Implement live performance monitoring with weekly threshold checks."
            )
        if risk_report.get("overall_risk_level") == "MEDIUM":
            conditions.append(
                "Complete medium-risk mitigation plan before first production batch."
            )
        if risk_report.get("operational_risk", {}).get("level") in ("MEDIUM", "HIGH"):
            conditions.append(
                "Activate drift monitoring using the stored baseline statistics."
            )
        # Ensure at least one condition
        if not conditions:
            conditions.append(
                "Conduct a 30-day shadow-mode evaluation before full rollout."
            )

    # ── Executive summary ──────────────────────────────────────────────────
    gov_stat  = governance_report.get("status", "N/A")
    perf_stat = performance_report.get("status", "N/A")
    risk_lvl  = risk_report.get("overall_risk_level", "N/A")
    rec       = risk_report.get("recommendation", "N/A")
    dr        = performance_report.get("deployment_readiness", "N/A")

    exec_summary = (
        f"The ML Governance Pipeline evaluated the submitted model package across "
        f"three specialist agents. "
        f"Governance scored {gov_score}/100 ({gov_stat}), "
        f"Performance scored {perf_score:.1f}/100 ({perf_stat}), "
        f"and Risk Assessment returned {risk_lvl} risk ({rec}). "
        f"The composite score is {composite}/100. "
        f"Based on these results the Governance Summarizer issues a final "
        f"verdict of '{decision}' with deployment readiness rated as {dr}."
    )

    # ── Next steps ─────────────────────────────────────────────────────────
    next_steps: List[str] = []
    if decision == "DEPLOYMENT APPROVED":
        next_steps = [
            "Proceed with production deployment following standard change-management.",
            "Schedule quarterly governance re-review and re-run this pipeline.",
            "Enable automated drift alerts using the stored feature statistics.",
        ]
    elif decision == "CONDITIONAL APPROVAL":
        next_steps = [
            "Address all conditions listed above before full production rollout.",
            "Re-run this pipeline after remediation to confirm PASS status.",
            "Obtain sign-off from the model risk committee on resolved items.",
        ]
    else:  # REJECTED
        next_steps = [
            "Do not deploy to production until all critical issues are resolved.",
            "Schedule a model risk review meeting within 5 business days.",
            "Re-train or remediate the model and resubmit for pipeline evaluation.",
        ]

    report = SummaryReport(
        agent_name="Governance Summarizer",
        final_decision=decision,
        decision_rationale=(
            f"Composite score {composite}/100 with governance {gov_stat}, "
            f"performance {perf_stat}, risk {risk_lvl}."
        ),
        composite_score=composite,
        agent_scores={
            "governance":  gov_score,
            "performance": round(perf_score, 1),
            "risk":        risk_score,
        },
        critical_issues=critical_issues,
        conditions=conditions,
        executive_summary=exec_summary,
        next_steps=next_steps,
    )

    print(f"[Agent 04] Decision={report.final_decision}  Composite={composite}")
    return report.model_dump()


# ── CLI smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, pathlib
    # Expects three JSON file paths as arguments
    def _load(path):
        with open(path) as fh:
            return json.load(fh)

    if len(sys.argv) == 4:
        g, p, r = _load(sys.argv[1]), _load(sys.argv[2]), _load(sys.argv[3])
    else:
        # Dummy data for quick smoke-test
        g = {"score": 85, "status": "PASS", "findings": [], "recommendations": []}
        p = {"performance_score": 91.0, "status": "PASS",
             "deployment_readiness": "READY", "metrics": {}}
        r = {"risk_score": 20, "overall_risk_level": "LOW",
             "recommendation": "APPROVED", "bias_risk": {"level": "LOW", "details": ""},
             "operational_risk": {"level": "LOW", "details": ""}}

    result = run_summarizer_agent(g, p, r)
    print(json.dumps(result, indent=2))