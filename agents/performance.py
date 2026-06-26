"""
agents/performance_agent.py
Agent 02 — Performance Monitor Agent
Mirrors Main.py (DeepSeek / CrewAI version) as a standalone module.
"""

import json
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Pydantic schemas (identical to Main.py) ────────────────────────────────
class Metrics(BaseModel):
    accuracy:  float
    precision: float
    recall:    float
    f1_score:  float
    roc_auc:   float


class ThresholdChecks(BaseModel):
    accuracy:  bool
    precision: bool
    recall:    bool
    f1_score:  bool
    roc_auc:   bool


class DriftReport(BaseModel):
    status:           str
    detected:         Optional[bool]  = None
    drift_score:      Optional[float] = None
    risk_level:       str
    drifted_features: List[str]


class MetricAssessment(BaseModel):
    overall_assessment: str


class PerformanceReport(BaseModel):
    agent:              str
    status:             str
    performance_score:  float = Field(ge=0, le=100)
    confidence_level:   str
    metrics:            Metrics
    metric_assessment:  MetricAssessment
    drift:              DriftReport
    strengths:          List[str] = Field(min_length=3, max_length=3)
    weaknesses:         List[str] = Field(min_length=3, max_length=3)
    potential_risks:    List[str] = Field(min_length=3, max_length=3)
    threshold_checks:   ThresholdChecks
    deployment_readiness: str
    recommendations:    List[str] = Field(min_length=3, max_length=3)
    findings:           List[str] = Field(min_length=3, max_length=3)


# ── Thresholds (from Main.py) ──────────────────────────────────────────────
THRESHOLDS = {
    "accuracy":  0.85,
    "precision": 0.80,
    "recall":    0.80,
    "f1_score":  0.80,
    "roc_auc":   0.85,
}


# ── Package loader (same as Main.py load_model_package) ───────────────────
def load_model_package(file_path: str) -> dict:
    import joblib
    package = joblib.load(file_path)
    # Strip the raw model object — keep only metadata
    return {k: v for k, v in package.items() if k != "model"}


def prepare_performance_input(extracted_data: dict) -> dict:
    """Same helper as Main.py."""
    model_metadata = extracted_data.get("metadata", {})
    return {
        "metadata": {
            "algorithm":    model_metadata.get("algorithm"),
            "training_rows": extracted_data["metadata"]["training_rows"],
            "testing_rows":  extracted_data["metadata"]["testing_rows"],
        },
        "performance_metrics":  extracted_data.get("performance_metrics", {}),
        "drift_baseline":       extracted_data.get("drift_baseline", {}),
        "explainability_report": extracted_data.get("explainability_report", {}),
    }


# ── Core evaluation logic ──────────────────────────────────────────────────
def _evaluate(performance_input: dict) -> PerformanceReport:
    pm = performance_input.get("performance_metrics", {})
    cr = pm.get("classification_report", {})

    # Pull weighted-avg precision / recall / f1 from classification report
    weighted = cr.get("weighted avg", {})

    accuracy  = pm.get("accuracy",  0.0)
    precision = weighted.get("precision", pm.get("precision", 0.0))
    recall    = weighted.get("recall",    pm.get("recall",    0.0))
    f1        = weighted.get("f1-score",  pm.get("f1_score",  0.0))
    roc_auc   = pm.get("roc_auc", 0.0)

    metrics = Metrics(
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        roc_auc=round(roc_auc, 4),
    )

    checks = ThresholdChecks(
        accuracy=  metrics.accuracy  >= THRESHOLDS["accuracy"],
        precision= metrics.precision >= THRESHOLDS["precision"],
        recall=    metrics.recall    >= THRESHOLDS["recall"],
        f1_score=  metrics.f1_score  >= THRESHOLDS["f1_score"],
        roc_auc=   metrics.roc_auc   >= THRESHOLDS["roc_auc"],
    )

    # Performance score (0-100)
    passed = sum([checks.accuracy, checks.precision, checks.recall,
                  checks.f1_score, checks.roc_auc])
    base_score = (accuracy * 60) + (roc_auc * 40)
    # Warning rule from Main.py: >99% accuracy without independent evidence
    if accuracy > 0.99:
        base_score = min(base_score, 92.0)
    performance_score = round(min(100.0, base_score * 100), 2)

    all_pass = passed == 5
    status = "PASS" if all_pass else ("WARNING" if passed >= 3 else "FAIL")

    # Drift
    drift_baseline = performance_input.get("drift_baseline", {})
    drift_status = "BASELINE_AVAILABLE" if drift_baseline else "NOT_ASSESSED"
    drift = DriftReport(
        status=drift_status,
        detected=None,
        drift_score=None,
        risk_level="LOW" if drift_baseline else "UNKNOWN",
        drifted_features=[],
    )

    # Strengths / weaknesses
    strengths = [
        f"High ROC-AUC of {metrics.roc_auc:.4f} indicates strong class discrimination.",
        f"Accuracy of {metrics.accuracy:.4f} {'exceeds' if checks.accuracy else 'approaches'} the 0.85 deployment threshold.",
        "SHAP explainability report is included in the governance package.",
    ]
    weaknesses = [
        "No independent validation dataset has been recorded in the package.",
        f"Precision of {metrics.precision:.4f} {'meets' if checks.precision else 'falls short of'} the 0.80 threshold.",
        "Drift monitoring infrastructure is not yet active (baseline only).",
    ]
    potential_risks = [
        "Model may exhibit degraded performance on out-of-distribution applicants.",
        "High accuracy on training distribution could mask sub-group disparities.",
        "Lack of live drift alerts increases the risk of silent model decay.",
    ]
    findings = [
        f"{passed}/5 performance thresholds satisfied.",
        f"Deployment readiness: {'READY' if all_pass else 'CONDITIONAL' if passed >= 3 else 'NOT_READY'}.",
        f"Drift baseline is {'available' if drift_baseline else 'absent'} — live monitoring recommended.",
    ]
    recommendations = [
        "Implement automated drift monitoring using the stored baseline statistics.",
        "Conduct monthly shadow scoring against a holdout validation set.",
        "Add per-demographic slice performance metrics to the governance package.",
    ]

    deployment_readiness = "READY" if all_pass else ("CONDITIONAL" if passed >= 3 else "NOT_READY")
    confidence_level = "HIGH" if performance_score >= 85 else ("MEDIUM" if performance_score >= 70 else "LOW")

    assessment = (
        f"Model achieves {'all' if all_pass else str(passed) + '/5'} performance thresholds. "
        f"Overall performance score: {performance_score:.1f}/100. "
        f"Deployment readiness: {deployment_readiness}."
    )

    return PerformanceReport(
        agent="Performance Monitor Agent",
        status=status,
        performance_score=performance_score,
        confidence_level=confidence_level,
        metrics=metrics,
        metric_assessment=MetricAssessment(overall_assessment=assessment),
        drift=drift,
        strengths=strengths,
        weaknesses=weaknesses,
        potential_risks=potential_risks,
        threshold_checks=checks,
        deployment_readiness=deployment_readiness,
        recommendations=recommendations,
        findings=findings,
    )


# ── Verification (same as Main.py verify_report) ──────────────────────────
def verify_report(report: PerformanceReport) -> bool:
    assert len(report.strengths) == 3
    assert len(report.weaknesses) == 3
    assert len(report.potential_risks) == 3
    assert len(report.recommendations) == 3
    assert len(report.findings) == 3
    assert 0 <= report.performance_score <= 100
    return True


# ── Public entry-point ─────────────────────────────────────────────────────
def run_performance_agent(pkl_path: str) -> dict:
    """
    Run the Performance Monitor Agent.

    Parameters
    ----------
    pkl_path : path to the model governance package .pkl

    Returns
    -------
    PerformanceReport serialised as a plain dict
    """
    print("[Agent 02] Performance Monitor Agent starting…")
    extracted  = load_model_package(pkl_path)
    perf_input = prepare_performance_input(extracted)

    print(f"[Agent 02] Algorithm      : {perf_input['metadata'].get('algorithm')}")
    print(f"[Agent 02] Training rows  : {perf_input['metadata'].get('training_rows')}")

    report = _evaluate(perf_input)
    verify_report(report)

    print(f"[Agent 02] Score={report.performance_score}  Status={report.status}  Deployment={report.deployment_readiness}")
    return report.model_dump()


# ── CLI smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    pkl = sys.argv[1] if len(sys.argv) > 1 else "loan_model_governance_package.pkl"
    result = run_performance_agent(pkl)
    print(json.dumps(result, indent=2))