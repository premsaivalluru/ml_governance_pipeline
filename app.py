import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Import local modules
import utils
import fairness
import drift
import pdf_report
import agents

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Governance & Validation Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stApp {
        background-color: #0F172A;
    }
    h1, h2, h3 {
        color: #38BDF8 !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #38BDF8;
    }
    .status-badge-approved {
        background-color: #064E3B;
        color: #34D399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
    }
    .status-badge-rejected {
        background-color: #7F1D1D;
        color: #FCA5A5;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
    }
    .status-badge-conditional {
        background-color: #78350F;
        color: #FCD34D;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html = True)

# Initialize Session State
if "model_package" not in st.session_state:
    st.session_state.model_package = None
if "validation_df" not in st.session_state:
    st.session_state.validation_df = None
if "docs_text" not in st.session_state:
    st.session_state.docs_text = "No formal documentation was provided for this model."
if "audit_results" not in st.session_state:
    st.session_state.audit_results = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# Sample Data Initializer
def load_sample_package():
    st.session_state.model_package = {
        "is_package": True,
        "metadata": {
            "model_name": "Loan Approval XGBoost",
            "version": "1.0",
            "algorithm": "XGBoost Classifier",
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature_names": ["cibil_score", "income_annum", "loan_amount", "loan_term", "no_of_dependents", "education", "self_employed", "total_assets", "loan_income_ratio", "loan_asset_ratio", "assets_income_ratio"],
            "target_name": "loan_status"
        },
        "performance_metrics": {
            "accuracy": 0.9850,
            "roc_auc": 0.9920,
            "precision": 0.9820,
            "recall": 0.9880,
            "f1_score": 0.9850
        },
        "drift_baseline": {
            "feature_statistics": {
                "cibil_score": {"mean": 600.0, "std": 150.0},
                "income_annum": {"mean": 5000000.0, "std": 2500000.0},
                "loan_amount": {"mean": 3000000.0, "std": 1500000.0},
                "loan_term": {"mean": 10.0, "std": 5.0},
                "education": {"mean": 0.25, "std": 0.43}, # Graduate (0) vs Not (1)
                "self_employed": {"mean": 0.50, "std": 0.50} # No (0) vs Yes (1)
            }
        },
        "explainability_report": {
            "method": "SHAP",
            "top_features": [
                ("cibil_score", 0.450),
                ("loan_income_ratio", 0.280),
                ("income_annum", 0.120),
                ("loan_amount", 0.080),
                ("total_assets", 0.050),
                ("loan_term", 0.020)
            ]
        }
    }
    st.session_state.validation_df = utils.generate_synthetic_validation_data(
        st.session_state.model_package["metadata"]["feature_names"], n_samples=1000
    )
    st.session_state.docs_text = (
        "LOAN APPROVAL XGBOOST MODEL CARD\n"
        "Model Owner: Risk Management Division, Retail Lending\n"
        "Version: 1.0\n"
        "Model Objective: Binary classification of loan approval likelihood to minimize credit defaults.\n"
        "Training Lineage: Trained on 3,415 records using XGBoost 1.7.\n"
        "Features: CIBIL Credit Score, Annual Income, Loan Amount, Asset Ratios.\n"
        "Compliance: Adheres to internal risk management thresholds. Version tracking managed via enterprise model registry."
    )
    st.success("Sample model package and validation data loaded successfully!")

# Sidebar UI
with st.sidebar:
    st.markdown("## Configuration Portal")
    st.info("Set credentials and upload model assets to begin auditing.")
    
    # LLM Settings
    st.markdown("### LLM Auditor Settings")
    provider = st.selectbox(
        "Select LLM Provider",
        ["Gemini", "DeepSeek (OpenRouter)"],
        index=0
    )
    
    # Detect default API keys
    #default_key = ""
    #if provider == "Gemini":
        #default_key = os.getenv("GEMINI_API_KEY", "")
    #elif provider == "DeepSeek (OpenRouter)":
        #default_key = os.getenv("OPENROUTER_API_KEY", "")
        
    #api_key = st.text_input(
        #f"Enter {provider} API Key",
        #value=default_key,
        #type="password"
    #)
    # If user enters a key, use it.
    # Otherwise use the server-side secret.
    user_api_key = st.text_input(
        f"Enter {provider} API Key (optional)",
        type="password",
        placeholder="Leave blank to use server key"
    )

    if provider == "Gemini":
        api_key = user_api_key or st.secrets["GEMINI_API_KEY"]
    else:
        api_key = user_api_key or st.secrets["OPENROUTER_API_KEY"]
    
    # Upload assets
    st.markdown("### Upload Model Assets")
    uploaded_model = st.file_uploader(
        "Model Package (.pkl / .joblib)",
        type=["pkl", "joblib"]
    )
    
    uploaded_docs = st.file_uploader(
        "Model Documentation (.pdf / .md / .txt)",
        type=["pdf", "md", "txt"]
    )
    
    uploaded_data = st.file_uploader(
        "Validation Dataset (.csv)",
        type=["csv"]
    )
    
    # Process uploads
    if uploaded_model:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_model:
            tmp_model.write(uploaded_model.read())
            tmp_model_path = tmp_model.name
        try:
            st.session_state.model_package = utils.load_model_package(tmp_model_path)
            st.success("Model package loaded successfully!")
            os.unlink(tmp_model_path)
        except Exception as e:
            st.error(f"Error loading model package: {str(e)}")
            
    if uploaded_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_docs.name)[1]) as tmp_doc:
            tmp_doc.write(uploaded_docs.read())
            tmp_doc_path = tmp_doc.name
        st.session_state.docs_text = utils.extract_documentation(tmp_doc_path)
        st.success("Documentation loaded successfully!")
        os.unlink(tmp_doc_path)
        
    if uploaded_data:
        st.session_state.validation_df = pd.read_csv(uploaded_data)
        st.success("Validation dataset uploaded successfully!")
        
    st.markdown("---")
    if st.button("Load Sample Model (Demo)", use_container_width=True):
        load_sample_package()
        
    if st.button("Reset App", use_container_width=True):
        st.session_state.model_package = None
        st.session_state.validation_df = None
        st.session_state.docs_text = "No formal documentation was provided for this model."
        st.session_state.audit_results = None
        st.session_state.pdf_path = None
        st.success("Application state cleared.")

# Main Screen Layout
st.markdown("# Enterprise AI Governance & Validation Platform")
st.markdown("Provide structural AI governance validation, fairness monitoring, and drift assessments for compliance.")
st.markdown("---")

if not st.session_state.model_package:
    # Welcome / Landing Screen
    st.markdown("""
    <div class="card">
        <h3>Welcome to the AI Governance Platform</h3>
        <p>This system allows risk officers, auditors, and data scientists to evaluate model compliance, risk, and deployment readiness.</p>
        <p><b>To start:</b></p>
        <ol>
            <li>Upload a pickled model package <code>.pkl</code> or <code>.joblib</code> in the sidebar.</li>
            <li>Optional: Upload PDF/MD documentation and a validation CSV dataset.</li>
            <li>Or simply click the <b>Load Sample Model (Demo)</b> button in the sidebar to test with pre-configured templates.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Globe image removed

else:
    pkg = st.session_state.model_package
    meta = pkg.get("metadata", {})
    perf = pkg.get("performance_metrics", {})
    explain = pkg.get("explainability_report", {})
    
    # ----------------------------------------------------
    # RUN PRE-CALCULATIONS (Fairness & Drift)
    # ----------------------------------------------------
    # Auto-generate validation data if missing
    if st.session_state.validation_df is None:
        feature_names = meta.get("feature_names", ["cibil_score"])
        st.session_state.validation_df = utils.generate_synthetic_validation_data(feature_names, n_samples=1000)
    
    val_df = st.session_state.validation_df
    
    # Run Fairness pre-calculation
    fairness_results = None
    protected_attr = "education"
    if "education" not in val_df.columns:
        # Fallback to the first binary/categorical-like feature
        for col in val_df.columns:
            if val_df[col].nunique() == 2:
                protected_attr = col
                break
                
    # If raw model exists, get predictions, else mock or use sample predictions
    if "predictions" not in val_df.columns:
        if pkg.get("raw_model") is not None:
            try:
                # Select features that model expects
                expected_feats = meta.get("feature_names", list(val_df.columns))
                X_val = val_df[expected_feats]
                val_df["predictions"] = pkg["raw_model"].predict(X_val)
            except Exception:
                # Mock predictions if model prediction fails (e.g. preprocessing required)
                val_df["predictions"] = np.random.choice([0, 1], p=[0.7, 0.3], size=len(val_df))
        else:
            # Mock predictions based on CIBIL score if present
            if "cibil_score" in val_df.columns:
                val_df["predictions"] = np.where(val_df["cibil_score"] > 600, 0, 1) # 0: Approved, 1: Rejected
            else:
                val_df["predictions"] = np.random.choice([0, 1], p=[0.7, 0.3], size=len(val_df))
                
    # Ground truth (if targets missing, mock based on predictions with 95% accuracy)
    if "ground_truth" not in val_df.columns:
        noise = np.random.choice([0, 1], p=[0.95, 0.05], size=len(val_df))
        val_df["ground_truth"] = np.where(noise == 0, val_df["predictions"], 1 - val_df["predictions"])
        
    fairness_results = fairness.FairnessAuditor.audit_fairness(
        df=val_df,
        protected_attribute=protected_attr,
        predictions=val_df["predictions"].values,
        ground_truth=val_df["ground_truth"].values,
        privileged_value=0, # Assume 0 is privileged
        positive_outcome=0 # Assume 0 is approved
    )
    
    # Run Drift pre-calculation
    drift_results = None
    if "drift_baseline" in pkg and "feature_statistics" in pkg["drift_baseline"]:
        drift_results = drift.DriftDetector.detect_drift_from_package(
            baseline_stats=pkg["drift_baseline"],
            current_df=val_df
        )
    else:
        # Run statistical drift using synthetic baseline
        ref_df = utils.generate_synthetic_validation_data(meta.get("feature_names", list(val_df.columns)))
        # Shift a couple of columns in val_df to trigger drift in demo
        if "cibil_score" in val_df.columns:
            val_df["cibil_score"] = val_df["cibil_score"] - 50 # shift distribution down
        drift_results = drift.DriftDetector.detect_drift_statistical(ref_df, val_df)

    # ----------------------------------------------------
    # RENDER TABS
    # ----------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["Governance & Agent Audits", "Technical Validation (Plotly)", "PDF Download & Reports"])
    
    with tab1:
        st.markdown("### Multi-Agent Validation Portal")
        st.write("Orchestrate an autonomous audit across Governance, Performance, Risk, Compliance, and Deployment gatekeepers.")
        
        # Run Audit Trigger
        if not api_key:
            st.warning("Please provide an API key in the sidebar to run the multi-agent autonomous auditor.")
        else:
            if st.button("Execute Autonomous Governance Audit", use_container_width=True):
                with st.spinner("Invoking Agents (this may take up to 1-2 minutes)..."):
                    try:
                        llm = agents.get_llm(provider, api_key)
                        crew = agents.create_governance_crew(
                            llm=llm,
                            model_metadata=meta,
                            performance_metrics=perf,
                            explainability_report=explain,
                            fairness_report=fairness_results,
                            drift_report=drift_results,
                            documentation_text=st.session_state.docs_text
                        )
                        
                        crew_result = crew.kickoff()
                        
                        # Collect results from tasks
                        governance_report = crew.tasks[0].output.pydantic
                        performance_report = crew.tasks[1].output.pydantic
                        risk_report = crew.tasks[2].output.pydantic
                        compliance_report = crew.tasks[3].output.pydantic
                        deployment_report = crew.tasks[4].output.pydantic
                        final_summary = crew.tasks[5].output.pydantic
                        
                        st.session_state.audit_results = {
                            "summary": final_summary.model_dump(),
                            "agents": {
                                "governance_agent": governance_report.model_dump(),
                                "performance_agent": performance_report.model_dump(),
                                "risk_agent": risk_report.model_dump(),
                                "compliance_agent": compliance_report.model_dump(),
                                "deployment_agent": deployment_report.model_dump()
                            }
                        }
                        
                        # Pre-generate PDF report path
                        pdf_file_path = os.path.join(tempfile.gettempdir(), f"ai_governance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                        pdf_report.generate_pdf_report(
                            output_path=pdf_file_path,
                            model_details=meta,
                            risk_score_info={
                                "status": final_summary.overall_status,
                                "risk_level": final_summary.overall_risk_level,
                                "score": final_summary.overall_risk_score
                            },
                            performance_metrics=perf,
                            fairness_results=fairness_results,
                            drift_results=drift_results,
                            agent_reports=st.session_state.audit_results["agents"]
                        )
                        st.session_state.pdf_path = pdf_file_path
                        
                        st.success("Audit complete! Check the results below.")
                    except Exception as e:
                        st.error(f"Error during agent execution: {str(e)}")
                        
        # Render Audit Results if available
        if st.session_state.audit_results:
            res = st.session_state.audit_results
            summary = res["summary"]
            
            # Risk Score and Status Layout
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <p style="font-size: 14px; color: #94A3B8;">DEPLOYMENT DECISION</p>
                    <span class="status-badge-{'approved' if 'APPROV' in summary['overall_status'].upper() else ('conditional' if 'COND' in summary['overall_status'].upper() or 'REV' in summary['overall_status'].upper() else 'rejected')}">
                        {summary['overall_status']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                risk_lvl = summary['overall_risk_level'].upper()
                risk_color = '#34D399' if 'LOW' in risk_lvl else ('#FCD34D' if 'MED' in risk_lvl else '#FCA5A5')
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <p style="font-size: 14px; color: #94A3B8;">RISK LEVEL</p>
                    <span style="font-size: 24px; font-weight: bold; color: {risk_color};">
                        {summary['overall_risk_level']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <p style="font-size: 14px; color: #94A3B8;">GOVERNANCE SCORE</p>
                    <span class="metric-value">{summary['overall_risk_score']}/100</span>
                </div>
                """, unsafe_allow_html=True)
                
            # Exec summary
            st.markdown("#### Executive Summary")
            st.info(summary["executive_summary"])
            
            # Agent details
            st.markdown("#### Detailed Agent Audit Logs")
            for agent_key, r in res["agents"].items():
                status_indicator = f"[{r['status']}]"
                with st.expander(f"{status_indicator} {r['agent_name']} (Score: {r['score']}/100)"):
                    st.markdown("**Findings:**")
                    for f in r["findings"]:
                        st.write(f"- {f}")
                    st.markdown("**Actionable Recommendations:**")
                    for rec in r["recommendations"]:
                        st.write(f"- {rec}")
                        
            # Remediation Roadmap
            st.markdown("#### Recommended Remediation Plan")
            for i, step in enumerate(summary["remediation_plan"]):
                st.write(f"{i+1}. {step}")
        else:
            st.info("Click **Execute Autonomous Governance Audit** to run the multi-agent auditor.")

    with tab2:
        st.markdown("### Technical Metrics & Dashboard")
        st.write("Audit model details, explainability reports, fairness compliance, and data drift diagnostics.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Details")
            st.write(f"**Model Name:** {meta.get('model_name', 'N/A')}")
            st.write(f"**Model Version:** {meta.get('version', 'N/A')}")
            st.write(f"**Algorithm:** {meta.get('algorithm', 'N/A')}")
            st.write(f"**Training Date:** {meta.get('training_date', 'N/A')}")
            st.write(f"**Features count:** {len(meta.get('feature_names', []))}")
            
            st.markdown("#### Performance Metrics Audit")
            
            # Normalize and extract scalar metrics
            display_perf = {}
            for k, v in perf.items():
                if isinstance(v, (int, float)):
                    display_perf[k] = float(v)
            
            class_report = perf.get("classification_report", {})
            if isinstance(class_report, dict):
                for avg_key in ["weighted avg", "macro avg", "1"]:
                    if avg_key in class_report and isinstance(class_report[avg_key], dict):
                        sub = class_report[avg_key]
                        if "precision" in sub and "precision" not in display_perf:
                            display_perf["precision"] = float(sub["precision"])
                        if "recall" in sub and "recall" not in display_perf:
                            display_perf["recall"] = float(sub["recall"])
                        if "f1-score" in sub and "f1_score" not in display_perf:
                            display_perf["f1_score"] = float(sub["f1-score"])
                        break
            
            perf_df = pd.DataFrame({
                "Metric": [k.replace("_", " ").title() for k in display_perf.keys()],
                "Value": [f"{v:.4f}" for v in display_perf.values()],
                "Threshold": [">= 0.85" if "accuracy" in k or "auc" in k else ">= 0.80" for k in display_perf.keys()],
                "Status": ["PASS" if (v >= 0.85 if ("accuracy" in k or "auc" in k) else v >= 0.80) else "FAIL" for k, v in display_perf.items()]
            })
            st.table(perf_df)
            
        with col2:
            st.markdown("#### Explainability (SHAP Top Features)")
            if explain and "top_features" in explain:
                feats = [item[0] for item in explain["top_features"]]
                scores = [item[1] for item in explain["top_features"]]
                
                fig_shap = go.Figure(go.Bar(
                    x=scores,
                    y=feats,
                    orientation='h',
                    marker_color='#38BDF8'
                ))
                fig_shap.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.warning("No explainability/SHAP report found in model package.")

        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Algorithmic Fairness Analysis")
            if fairness_results and "error" not in fairness_results:
                st.write(f"**Audited Protected Attribute:** `{protected_attr}`")
                
                # Fairness Table
                di_ratio = fairness_results["disparate_impact_ratio"]
                dp_diff = fairness_results["demographic_parity_difference"]
                
                di_status = "COMPLIANT" if 0.8 <= di_ratio <= 1.25 else "BIAS DETECTED"
                dp_status = "COMPLIANT" if dp_diff <= 0.10 else "BIAS DETECTED"
                
                st.write(f"- **Disparate Impact Ratio:** {di_ratio:.4f} ({di_status})")
                st.write(f"- **Demographic Parity Difference:** {dp_diff:.4f} ({dp_status})")
                
                if fairness_results.get("equal_opportunity_difference") is not None:
                    eo_diff = fairness_results["equal_opportunity_difference"]
                    eo_status = "COMPLIANT" if eo_diff <= 0.10 else "BIAS DETECTED"
                    st.write(f"- **Equal Opportunity Difference:** {eo_diff:.4f} ({eo_status})")
                    
                # Plotly Chart
                fig_fairness = fairness.FairnessAuditor.generate_fairness_charts(fairness_results, protected_attr)
                st.plotly_chart(fig_fairness, use_container_width=True)
            else:
                st.warning("Could not execute fairness analysis.")
                
        with col4:
            st.markdown("#### Data Drift Detection")
            if drift_results:
                drift_share = drift_results["drift_share"]
                drift_status = "DRIFT DETECTED" if drift_results["detected"] else "STABLE"
                st.write(f"**Drift Status:** {drift_status}")
                st.write(f"**Drift Share:** {drift_share * 100:.1f}% features drifted")
                
                fig_drift = drift.DriftDetector.generate_drift_chart(drift_results)
                st.plotly_chart(fig_drift, use_container_width=True)
            else:
                st.warning("No drift baseline statistics available.")

    with tab3:
        st.markdown("### Generate and Download Reports")
        st.write("Generate a formal, publication-ready PDF governance report combining all analysis tables and agent logs.")
        
        if st.session_state.pdf_path:
            with open(st.session_state.pdf_path, "rb") as f:
                pdf_data = f.read()
                
            st.download_button(
                label="Download PDF Audit Report",
                data=pdf_data,
                file_name=f"Model_Governance_Report_{meta.get('model_name', 'Model').replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            # Display PDF path
            st.success(f"PDF report successfully cached at: `{st.session_state.pdf_path}`")
            
            st.markdown("#### Document Markdown Draft View")
            if st.session_state.audit_results:
                summary = st.session_state.audit_results["summary"]
                st.markdown(f"""
                # AI Model Governance & Validation Report
                
                **Date:** {datetime.now().strftime('%Y-%m-%d')}
                
                ## Executive Summary
                * **Deployment Decision:** {summary['overall_status']}
                * **Overall Risk Level:** {summary['overall_risk_level']}
                * **Governance Score:** {summary['overall_risk_score']}/100
                
                {summary['executive_summary']}
                
                ## Key Critical Findings
                {chr(10).join([f'* {f}' for f in summary['key_critical_findings']])}
                
                ## Actionable Remediation Roadmap
                {chr(10).join([f'* {r}' for r in summary['remediation_plan']])}
                
                *Report prepared by Enterprise Autonomous AI Governance Auditor.*
                """)
        else:
            st.info("Execute the autonomous audit in the **Governance & Agent Audits** tab to compile the PDF download report.")
