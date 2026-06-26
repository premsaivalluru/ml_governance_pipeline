"""
agents/governance_agent.py
Agent 01 — Governance Policy Agent
Mirrors governance.py (Colab notebook) but runs as a standalone module.
"""

import os
import json
import pickle
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
from pydantic import BaseModel
from typing import List, Dict


# ── Shared contract schema (same as governance.py slide 9) ─────────────────
class AgentReport(BaseModel):
    agent_name: str
    score: int
    status: str
    findings: List[str]
    recommendations: List[str]
    metadata: Dict = {}


# ── Artifact extraction helpers ────────────────────────────────────────────
def extract_documentation(file_path: str) -> str:
    """Read PDF / MD / TXT documentation and return plain text."""
    if not file_path or not os.path.exists(file_path):
        return "No formal documentation was provided for this model."

    _, ext = os.path.splitext(file_path)
    text = ""
    try:
        if ext.lower() == ".pdf":
            if PyPDF2 is None:
                return "PDF parsing library PyPDF2 is not installed; cannot read PDF documentation."
            with open(file_path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        elif ext.lower() in (".md", ".txt"):
            with open(file_path, "r", encoding="utf-8") as fh:
                text = fh.read()
    except Exception as exc:
        return f"Error reading documentation: {exc}"

    return text.strip() or "Documentation file was empty."


def extract_model_artifacts(pkl_path: str):
    """
    Load a .pkl / .joblib model package and return
    (metadata_dict, feature_schema_dict).
    Supports both raw sklearn/XGBoost objects and the
    structured governance package created by loan_approval_prediction.py.
    """
    try:
        with open(pkl_path, "rb") as fh:
            model_package = pickle.load(fh)
    except Exception as exc:
        return {"error": str(exc)}, {"error": str(exc)}

    # ── Structured governance package (dict) ──────────────────────────────
    if isinstance(model_package, dict):
        metadata      = model_package.get("metadata", {})
        feature_schema = model_package.get("feature_metadata",
                          model_package.get("feature_schema", {}))
        return metadata, feature_schema

    # ── Raw model object ───────────────────────────────────────────────────
    model = model_package
    metadata: Dict = {
        "model_type": type(model).__name__,
        "module":     type(model).__module__,
    }

    if hasattr(model, "get_params"):
        metadata["hyperparameters"] = model.get_params()
    elif hasattr(model, "save_config"):
        metadata["booster_config"] = json.loads(model.save_config())

    feature_schema: Dict = {}
    if hasattr(model, "feature_names_in_"):
        feature_schema = {f: "unknown" for f in model.feature_names_in_}
    elif hasattr(model, "feature_names") and model.feature_names:
        feature_schema = {f: "unknown" for f in model.feature_names}
    elif hasattr(model, "n_features_in_"):
        feature_schema = {"n_features": model.n_features_in_}
    else:
        feature_schema = {
            "warning": "Could not auto-detect feature schema — "
                       "relying on documentation only"
        }

    return metadata, feature_schema


# ── Scoring logic (no LLM — deterministic rule-based) ─────────────────────
def _score_governance(metadata: Dict, feature_schema: Dict, docs: str) -> AgentReport:
    score = 100
    findings: List[str] = []
    recommendations: List[str] = []

    # 1. Documentation completeness
    if len(docs) < 50:
        score -= 20
        findings.append("Formal model documentation is absent or too sparse.")
        recommendations.append(
            "Create a model card covering purpose, data sources, "
            "intended use, and limitations."
        )
    else:
        findings.append("Documentation artifact is present and readable.")

    # 2. Explainability artifacts
    has_explainability = (
        isinstance(metadata, dict)
        and "explainability_report" in str(metadata)
    ) or "shap" in docs.lower() or "lime" in docs.lower()

    if not has_explainability:
        score -= 30
        findings.append("No explainability artifacts (SHAP / LIME) detected.")
        recommendations.append(
            "Generate SHAP or LIME explanations and attach them "
            "to the model package before deployment."
        )
    else:
        findings.append("Explainability report (SHAP) found in package.")

    # 3. Data lineage & feature descriptions
    has_feature_desc = (
        isinstance(feature_schema, dict)
        and len(feature_schema) > 0
        and "warning" not in feature_schema
        and "error" not in feature_schema
    )
    if not has_feature_desc:
        score -= 15
        findings.append("Feature schema or data lineage information is incomplete.")
        recommendations.append(
            "Document each feature with dtype, source table, "
            "transformation applied, and business definition."
        )
    else:
        findings.append(
            f"Feature schema present with {len(feature_schema)} documented features."
        )

    # 4. Model owner & version traceability
    version_ok = (
        isinstance(metadata, dict)
        and metadata.get("version")
        and metadata.get("training_date")
    )
    owner_ok = isinstance(metadata, dict) and (
        metadata.get("model_owner") or metadata.get("model_name")
    )

    if not version_ok:
        score -= 15
        findings.append("Version history and training date metadata are missing.")
        recommendations.append(
            "Tag every model artefact with version, training date, "
            "and a responsible owner field."
        )

    if not owner_ok:
        score -= 10
        recommendations.append(
            "Add a model_owner field (team / individual) to the metadata dict."
        )

    # 5. Status
    if score >= 85:
        status = "PASS"
    elif score >= 70:
        status = "WARNING"
    else:
        status = "FAIL"

    # Ensure at least 3 recommendations
    while len(recommendations) < 3:
        recommendations.append(
            "Conduct quarterly governance reviews aligned with SR 11-7 guidelines."
        )

    return AgentReport(
        agent_name="Governance Policy Agent",
        score=max(0, score),
        status=status,
        findings=findings[:5],
        recommendations=recommendations[:5],
        metadata={
            "model_type":      metadata.get("model_type", "Unknown") if isinstance(metadata, dict) else "Unknown",
            "version":         metadata.get("version", "N/A") if isinstance(metadata, dict) else "N/A",
            "training_date":   metadata.get("training_date", "N/A") if isinstance(metadata, dict) else "N/A",
            "feature_count":   len(feature_schema) if isinstance(feature_schema, dict) else 0,
            "has_explainability": has_explainability,
        },
    )


# ── Public entry-point ─────────────────────────────────────────────────────
def run_governance_agent(pkl_path: str, docs_path: str = "") -> dict:
    """
    Run the Governance Policy Agent.

    Parameters
    ----------
    pkl_path  : path to the model .pkl file
    docs_path : optional path to documentation (PDF / MD / TXT)

    Returns
    -------
    AgentReport serialised as a plain dict
    """
    print("[Agent 01] Governance Policy Agent starting…")
    model_metadata, feature_metadata = extract_model_artifacts(pkl_path)
    model_docs = extract_documentation(docs_path)

    print(f"[Agent 01] Metadata keys  : {list(model_metadata.keys()) if isinstance(model_metadata, dict) else 'error'}")
    print(f"[Agent 01] Feature count  : {len(feature_metadata) if isinstance(feature_metadata, dict) else 'error'}")
    print(f"[Agent 01] Docs length    : {len(model_docs)} chars")

    report = _score_governance(model_metadata, feature_metadata, model_docs)

    print(f"[Agent 01] Score={report.score}  Status={report.status}")
    return report.model_dump()


# ── CLI smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    pkl  = sys.argv[1] if len(sys.argv) > 1 else "loan_model_governance_package.pkl"
    docs = sys.argv[2] if len(sys.argv) > 2 else ""
    result = run_governance_agent(pkl, docs)
    print(json.dumps(result, indent=2))