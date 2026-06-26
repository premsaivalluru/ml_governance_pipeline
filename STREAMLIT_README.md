# Streamlit Governance Dashboard

A production-grade ML governance audit interface built with Streamlit, featuring:

- **Interactive Dashboard**: Risk levels, model metrics, and deployment status at a glance
- **CrewAI Integration**: Multi-agent orchestration with support for DeepSeek, Claude, OpenAI, and Gemini
- **SHAP & Fairness Charts**: Plotly visualizations for feature importance and demographic analysis
- **PDF Reports**: One-click download of governance audit reports
- **Real-time Status**: Async pipeline execution with progress tracking

## Quick Start

### 1. Install Dependencies

```bash
cd ~/ml_governance_pipeline
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2. Start the Backend API (if not already running)

In **Terminal A**:
```bash
cd ~/ml_governance_pipeline
source .venv/bin/activate
export CREWAI_BASE_URL="http://localhost:5002"
PORT=5001 python3 backend/app.py
```

### 3. (Optional) Start Mock Orchestrator

In **Terminal B** (for testing without real CrewAI):
```bash
cd ~/ml_governance_pipeline
python3 tools/mock_orchestrator.py
```

### 4. Launch Streamlit Dashboard

In **Terminal C**:
```bash
cd ~/ml_governance_pipeline
source .venv/bin/activate
streamlit run streamlit_app.py
```

This opens the dashboard at: **http://localhost:8501**

## Usage

1. **Sidebar Configuration**:
   - Select an LLM provider (local, DeepSeek, Claude, OpenAI, Gemini)
   - Paste your API key (optional for local mode)
   - Upload a `.pkl` or `.joblib` model file
   - Optionally upload documentation (PDF, MD, or TXT)

2. **Run Pipeline**:
   - Click "▶️ Run Governance Pipeline"
   - Watch real-time status updates
   - Dashboard auto-populates with results

3. **Review Results**:
   - **Summary Tab**: Executive decision and next steps
   - **Governance Tab**: Policy compliance scores
   - **Performance Tab**: Model metrics and deployment readiness
   - **Risk Tab**: Bias, operational, and regulatory risk assessments
   - **Raw JSON Tab**: Full audit report data

4. **Download Report**:
   - Click "⬇️ Download PDF Report" to save an audit record

## Architecture

The Streamlit app bridges:
- **Frontend**: Interactive Streamlit interface
- **Backend**: Flask REST API (`backend/app.py`)
- **Orchestration**: Local agents or remote CrewAI endpoint
- **Reporting**: PDF generation with audit results

## Environment Variables

- `CREWAI_BASE_URL`: Base URL for your CrewAI orchestrator (e.g., `http://localhost:5002`)
- `API_BASE_URL`: Backend API URL (default: `http://localhost:5001`)

## Customization

### Change API URL
Edit line ~18 in `streamlit_app.py`:
```python
API_BASE_URL = "http://your-backend:5001"
```

### Add New LLM Providers
Edit the provider selectbox around line ~85:
```python
provider = st.selectbox(
    "LLM Provider",
    ["local", "deepseek", "claude", "openai", "gemini", "your-provider"],
)
```

### Modify Dashboard Layout
Charts are generated in `create_shap_chart()` and `create_fairness_chart()` functions. Customize them to match your data structure.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **🔴 API Offline** | Ensure Flask backend is running on port 5001 |
| **PDF Download Fails** | Check ReportLab installation: `pip install reportlab` |
| **Charts Not Showing** | Verify Plotly is installed: `pip install plotly` |
| **Port Already in Use** | Change port in startup: `streamlit run streamlit_app.py --server.port 8502` |

## Next Steps

- Integrate with real CrewAI endpoint (set `CREWAI_BASE_URL`)
- Customize fairness and SHAP charts with your actual model data
- Add more dashboard tabs (e.g., model version history, audit log)
- Deploy to production with `streamlit-cloud` or Docker
