"""
agents/risk_agent.py
Agent 03 — Risk Analysis Agent
Mirrors Untitled11.py (CreditRiskAgent) but works on a pre-trained
governance package rather than raw training data.
"""

import json
from typing import List, Dict
from pydantic import BaseModel


# ── Risk report schema ─────────────────────────────────────────────────────
class RiskLevel(BaseModel):
    level:   str   # LOW | MEDIUM | HIGH
    details: str


class RiskReport(BaseModel):
    agent_name:        str
    overall_risk_level: str
    risk_score:        int       # 0-100, higher = riskier
    recommendation:    str       # APPROVED | REVIEW | REJECT
    bias_risk:         RiskLevel
    operational_risk:  RiskLevel
    regulatory_risk:   RiskLevel
    top_risk_features: List[str]
    findings:          List[str]
    mitigations:       List[str]
    metrics:           Dict


# ── Risk thresholds (mirrors CreditRiskAgent.assess_risk) ─────────────────
def _assess_risk_level(accuracy: float) -> tuple:
    """Return (risk_level, recommendation) mirroring Untitled11.py logic."""
    if accuracy >= 0.85:
        return "LOW", "APPROVED"
    elif accuracy >= 0.75:
        return "MEDIUM", "REVIEW"
    else:
        return "HIGH", "REJECT"


# ── Package loader ─────────────────────────────────────────────────────────
def _load_package(pkl_path: str) -> dict:
    import joblib
    pkg = joblib.load(pkl_path)
    if isinstance(pkg, dict):
        return pkg
    # Raw model — wrap it
    return {"model": pkg, "metadata": {}, "performance_metrics": {},
            "explainability_report": {}, "feature_metadata": {}}


# ── Core risk evaluation ───────────────────────────────────────────────────
def _evaluate_risk(pkg: dict) -> RiskReport:
    meta    = pkg.get("metadata", {})
    pm      = pkg.get("performance_metrics", {})
    expl    = pkg.get("explainability_report", {})
    feat_md = pkg.get("feature_metadata", {})

    accuracy  = pm.get("accuracy",  0.0)
    roc_auc   = pm.get("roc_auc",   0.0)
    cr        = pm.get("classification_report", {})
    weighted  = cr.get("weighted avg", {})
    precision = weighted.get("precision", 0.0)
    recall    = weighted.get("recall",    0.0)
    f1        = weighted.get("f1-score",  0.0)

    overall_level, recommendation = _assess_risk_level(accuracy)

    # ── Bias risk ──────────────────────────────────────────────────────────
    sensitive_features = {"personal_status_sex", "age", "foreign_worker",
                          "gender", "race", "nationality"}
    feature_names = set(str(k).lower() for k in feat_md.keys())
    sensitive_present = sensitive_features & feature_names

    if sensitive_present:
        bias_level = "HIGH"
        bias_details = (
            f"Sensitive attributes detected in feature schema: "
            f"{', '.join(sensitive_present)}. Fairness testing is mandatory."
        )
    elif accuracy < 0.80:
        bias_level = "MEDIUM"
        bias_details = (
            "Low overall accuracy may conceal sub-group performance disparities."
        )
    else:
        bias_level = "LOW"
        bias_details = (
            "No obviously sensitive features detected. "
            "Periodic fairness audits are still recommended."
        )

    # ── Operational risk ───────────────────────────────────────────────────
    has_drift_baseline = bool(pkg.get("drift_baseline", {}))
    has_monitoring     = False  # assumed — not in package

    if not has_drift_baseline and not has_monitoring:
        op_level = "HIGH"
        op_details = "No drift baseline or monitoring pipeline detected."
    elif has_drift_baseline and not has_monitoring:
        op_level = "MEDIUM"
        op_details = (
            "Drift baseline is present but live monitoring is not configured. "
            "Silent model decay is a deployment risk."
        )
    else:
        op_level = "LOW"
        op_details = "Drift baseline and monitoring both in place."

    # ── Regulatory risk ────────────────────────────────────────────────────
    has_explainability = bool(expl) and expl.get("method")
    has_version        = bool(meta.get("version"))
    has_owner          = bool(meta.get("model_owner") or meta.get("model_name"))

    reg_issues = []
    if not has_explainability:
        reg_issues.append("no explainability report")
    if not has_version:
        reg_issues.append("no version tag")
    if not has_owner:
        reg_issues.append("no model owner field")

    if len(reg_issues) >= 2:
        reg_level = "HIGH"
        reg_details = f"Multiple EU AI Act / SR 11-7 gaps: {', '.join(reg_issues)}."
    elif len(reg_issues) == 1:
        reg_level = "MEDIUM"
        reg_details = f"Minor compliance gap: {reg_issues[0]}."
    else:
        reg_level = "LOW"
        reg_details = "Core regulatory artefacts (explainability, version, owner) present."

    # ── Risk score (0-100, higher = riskier) ──────────────────────────────
    risk_score = 0
    if overall_level == "HIGH":   risk_score += 40
    elif overall_level == "MEDIUM": risk_score += 20
    if bias_level == "HIGH":      risk_score += 25
    elif bias_level == "MEDIUM":  risk_score += 12
    if op_level == "HIGH":        risk_score += 20
    elif op_level == "MEDIUM":    risk_score += 10
    if reg_level == "HIGH":       risk_score += 15
    elif reg_level == "MEDIUM":   risk_score += 8
    risk_score = min(100, risk_score)

    # ── Top risk features (from SHAP or feature schema) ───────────────────
    top_features: List[str] = []
    if expl and expl.get("top_features"):
        # top_features is a list of [name, score] pairs
        for item in expl["top_features"][:5]:
            if isinstance(item, (list, tuple)):
                top_features.append(str(item[0]))
            else:
                top_features.append(str(item))
    else:
        top_features = list(feat_md.keys())[:5]

    if not top_features:
        top_features = ["cibil_score", "loan_amount", "income_annum",
                        "loan_income_ratio", "loan_asset_ratio"]

    # ── Findings & mitigations ─────────────────────────────────────────────
    findings = [
        f"Overall model risk classified as {overall_level} "
        f"(accuracy={accuracy:.4f}, roc_auc={roc_auc:.4f}).",
        f"Bias risk is {bias_level}: {bias_details[:80]}…" if len(bias_details) > 80 else f"Bias risk is {bias_level}: {bias_details}",
        f"Operational risk is {op_level} — drift monitoring status: "
        f"{'baseline present' if has_drift_baseline else 'no baseline'}.",
    ]

    mitigations = [
        "Implement fairness-aware evaluation across demographic sub-groups.",
        "Configure automated drift alerts using the stored feature statistics.",
        "Add an SR 11-7-compliant model risk rating to the governance package.",
    ]

    return RiskReport(
        agent_name="Risk Analysis Agent",
        overall_risk_level=overall_level,
        risk_score=risk_score,
        recommendation=recommendation,
        bias_risk=RiskLevel(level=bias_level, details=bias_details),
        operational_risk=RiskLevel(level=op_level, details=op_details),
        regulatory_risk=RiskLevel(level=reg_level, details=reg_details),
        top_risk_features=top_features,
        findings=findings,
        mitigations=mitigations,
        metrics={
            "accuracy":  round(accuracy, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1_score":  round(f1, 4),
            "roc_auc":   round(roc_auc, 4),
        },
    )


# ── Public entry-point ─────────────────────────────────────────────────────
def run_risk_agent(pkl_path: str) -> dict:
    """
    Run the Risk Analysis Agent.

    Parameters
    ----------
    pkl_path : path to the model governance package .pkl

    Returns
    -------
    RiskReport serialised as a plain dict
    """
    print("[Agent 03] Risk Analysis Agent starting…")
    pkg = _load_package(pkl_path)
    report = _evaluate_risk(pkg)
    print(
        f"[Agent 03] Risk={report.overall_risk_level}  "
        f"Score={report.risk_score}  Recommendation={report.recommendation}"
    )
    return report.model_dump()


# ── CLI smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    pkl = sys.argv[1] if len(sys.argv) > 1 else "loan_model_governance_package.pkl"
    result = run_risk_agent(pkl)
    print(json.dumps(result, indent=2))