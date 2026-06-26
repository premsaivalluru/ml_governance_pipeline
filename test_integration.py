#!/usr/bin/env python3
"""
Updated Integration Test for AI Governance Portal
Tests functions in utils.py, fairness.py, drift.py, pdf_report.py, and agents.py
"""

import sys
import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import utils
import fairness
import drift
import pdf_report
import agents

def test_utils():
    print("\n" + "="*60)
    print("Testing utils.py")
    print("="*60)
    
    # Test synthetic validation data generation
    features = ["cibil_score", "income_annum", "loan_amount", "education"]
    df = utils.generate_synthetic_validation_data(features, n_samples=100)
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
    assert all(col in df.columns for col in features), "Some features were missing in synthetic df"
    print("✓ Synthetic validation dataset generation passed!")

def test_fairness():
    print("\n" + "="*60)
    print("Testing fairness.py")
    print("="*60)
    
    # Mock data
    np.random.seed(42)
    df = pd.DataFrame({
        "education": [0, 0, 1, 1] * 25, # 0: Graduate (privileged), 1: Not (unprivileged)
        "cibil_score": np.random.randint(300, 900, size=100)
    })
    
    # Mock predictions and ground truth
    predictions = np.where(df["cibil_score"] > 600, 0, 1) # 0: Approved, 1: Rejected
    ground_truth = predictions.copy() # perfect accuracy
    
    results = fairness.FairnessAuditor.audit_fairness(
        df=df,
        protected_attribute="education",
        predictions=predictions,
        ground_truth=ground_truth,
        privileged_value=0,
        positive_outcome=0
    )
    
    assert "error" not in results, f"Fairness audit returned error: {results.get('error')}"
    assert "disparate_impact_ratio" in results
    assert "demographic_parity_difference" in results
    assert "equal_opportunity_difference" in results
    print(f"✓ Fairness audit passed! Disparate Impact: {results['disparate_impact_ratio']:.4f}")
    
    # Test chart generation
    fig = fairness.FairnessAuditor.generate_fairness_charts(results, "education")
    assert fig is not None, "Fairness chart generation returned None"
    print("✓ Fairness Plotly chart generation passed!")

def test_drift():
    print("\n" + "="*60)
    print("Testing drift.py")
    print("="*60)
    
    # Statistical drift test
    ref_df = pd.DataFrame({"feat_1": np.random.normal(0, 1, 100)})
    cur_df = pd.DataFrame({"feat_1": np.random.normal(0, 1, 100)}) # no drift
    drifted_df = pd.DataFrame({"feat_1": np.random.normal(5, 1, 100)}) # drifted
    
    res_stable = drift.DriftDetector.detect_drift_statistical(ref_df, cur_df)
    assert not res_stable["detected"], "Should not detect drift for similar distributions"
    
    res_drifted = drift.DriftDetector.detect_drift_statistical(ref_df, drifted_df)
    assert res_drifted["detected"], "Should detect drift for shifted distribution"
    print("✓ Statistical drift detection passed!")
    
    # Package baseline statistics drift test
    baseline = {
        "feature_statistics": {
            "feat_1": {"mean": 0.0, "std": 1.0}
        }
    }
    res_pkg_drifted = drift.DriftDetector.detect_drift_from_package(baseline, drifted_df)
    assert res_pkg_drifted["detected"], "Package drift detection failed to detect shift"
    print("✓ Package-based drift detection passed!")
    
    # Test chart generation
    fig = drift.DriftDetector.generate_drift_chart(res_pkg_drifted)
    assert fig is not None, "Drift chart generation returned None"
    print("✓ Drift Plotly chart generation passed!")

def test_pdf_report():
    print("\n" + "="*60)
    print("Testing pdf_report.py")
    print("="*60)
    
    model_details = {
        "model_name": "Test Model",
        "version": "1.0",
        "algorithm": "Random Forest",
        "training_date": "2026-06-25"
    }
    
    risk_score_info = {
        "status": "APPROVED WITH CONDITIONS",
        "risk_level": "Medium",
        "score": 75
    }
    
    performance_metrics = {
        "accuracy": 0.9100,
        "roc_auc": 0.9300,
        "classification_report": {
            "weighted avg": {
                "precision": 0.9000,
                "recall": 0.9100,
                "f1-score": 0.9050
            }
        }
    }
    
    fairness_results = {
        "protected_attribute": "education",
        "disparate_impact_ratio": 0.92,
        "demographic_parity_difference": 0.05,
        "equal_opportunity_difference": 0.03
    }
    
    drift_results = {
        "drift_share": 0.1,
        "detected": False,
        "drifted_features": []
    }
    
    agent_reports = {
        "governance_agent": {
            "agent_name": "Governance Policy Agent",
            "score": 90,
            "status": "PASS",
            "findings": ["Complete documentation found", "Clear lineage"],
            "recommendations": ["Conduct quarterly reviews"]
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_report.pdf")
        pdf_report.generate_pdf_report(
            output_path=output_path,
            model_details=model_details,
            risk_score_info=risk_score_info,
            performance_metrics=performance_metrics,
            fairness_results=fairness_results,
            drift_results=drift_results,
            agent_reports=agent_reports
        )
        assert os.path.exists(output_path), "PDF file was not created"
        assert os.path.getsize(output_path) > 0, "PDF file is empty"
        print(f"✓ PDF Report generation passed! Saved to: {output_path} ({os.path.getsize(output_path)} bytes)")

def test_agents_compatibility():
    print("\n" + "="*60)
    print("Testing agents.py Architecture")
    print("="*60)
    
    print(f"  CrewAI Installed: {agents.HAS_CREWAI}")
    
    # Try creating mock LLM and checking instantiation
    llm = agents.get_llm("Gemini", "mock-api-key")
    assert llm is not None, "Failed to instantiate LLM wrapper"
    
    crew = agents.create_governance_crew(
        llm=llm,
        model_metadata={"model_name": "Test"},
        performance_metrics={"accuracy": 0.9},
        explainability_report={"method": "SHAP"},
        fairness_report={},
        drift_report={},
        documentation_text="This is test documentation."
    )
    assert crew is not None, "Failed to construct governance crew"
    assert len(crew.agents) == 6, f"Expected 6 agents, got {len(crew.agents)}"
    assert len(crew.tasks) == 6, f"Expected 6 tasks, got {len(crew.tasks)}"
    print("✓ Agent / Crew instantiation validation passed!")

def main():
    print("="*60)
    print("RUNNING AI GOVERNANCE INTEGRATION TESTS")
    print("="*60)
    try:
        test_utils()
        test_fairness()
        test_drift()
        test_pdf_report()
        test_agents_compatibility()
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        return 0
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
