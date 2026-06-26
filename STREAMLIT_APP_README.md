# 🏛️ ML Governance Pipeline - Streamlit Application

A production-grade Streamlit web application for AI model governance, validation, and compliance auditing.

## 📦 What's New

The application has been transformed into a comprehensive multi-agent governance platform with:

### ✨ Core Features

- **📁 File Management**
  - Upload model packages (.pkl, .joblib)
  - Document uploads (PDF, Markdown, TXT)
  - Verification dataset uploads (CSV)
  - Automatic synthetic dataset generation

- **🔐 API Key Management**
  - Sidebar configuration for Gemini API keys
  - OpenRouter API key integration
  - CrewAI orchestrator URL configuration
  - Automatic .env file loading

- **📊 Governance Dashboard**
  - Real-time KPI metrics (Governance, Performance, Risk, Audit Status)
  - Risk level indicators (🟢 Low / 🟡 Medium / 🔴 High)
  - Model metadata display
  - Status tracking

- **📈 Interactive Visualizations**
  - SHAP feature importance charts (Plotly)
  - Fairness analysis by demographic groups
  - Compliance gauge charts
  - Risk breakdown pie charts
  - Data drift monitoring

- **🎯 Multi-Agent Auditing**
  - Governance & Compliance Assessment
  - Performance Analysis & Benchmarking
  - Risk Assessment & Mitigation
  - Executive Summary Generation

- **📥 PDF Report Generation**
  - Regulatory-ready PDF reports (ReportLab)
  - Governance metrics compilation
  - Executive summary
  - Download button for distribution

## 🗂️ Project Structure

```
streamlit_app.py          # Main Streamlit application
utils.py                  # File I/O, dataset synthesis utilities
pdf_report.py             # PDF generation using ReportLab
test_integration.py       # Comprehensive integration tests
requirements.txt          # Python dependencies
```

## 📝 New Modules

### 1. **utils.py** - Utility Functions

```python
load_model_package()           # Load pickle/joblib models safely
load_verification_dataset()    # Load CSV datasets
synthesize_verification_dataset()  # Generate synthetic data from schema
extract_feature_schema()       # Parse model feature metadata
extract_text_from_pdf()        # Extract text from PDFs
extract_text_from_markdown()   # Read Markdown documentation
validate_model_package()       # Validate model structure
format_risk_level()           # Convert scores to risk labels
load_env_variables()          # Load .env configuration
```

### 2. **pdf_report.py** - PDF Report Generation

```python
create_pdf_report()           # Generate governance PDF
_format_report_section()      # Format report sections
create_summary_metrics_table() # Create metric tables
```

### 3. **streamlit_app.py** - Main Application

**Sidebar Components:**
- API Key Configuration (Gemini, OpenRouter, CrewAI URL)
- Model Package Upload
- Documentation Upload
- Verification Dataset Upload
- Synthetic Dataset Generation

**Main Content:**
- Governance Dashboard with KPIs
- 6-Tab Interface:
  1. Overview (Summary, Metadata)
  2. Performance (SHAP, Fairness)
  3. Governance (Compliance Gauges)
  4. Risk (Analysis, Recommendations)
  5. Details (Raw JSON)
  6. Download (PDF Report)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /Users/venu/ml_governance_pipeline
pip install -r requirements.txt
```

### 2. (Optional) Set Environment Variables
```bash
# Create .env file
echo "GEMINI_API_KEY=your_key_here" > .env
echo "OPENROUTER_API_KEY=your_key_here" >> .env
echo "CREWAI_BASE_URL=http://localhost:5002" >> .env
```

### 3. Run Integration Tests
```bash
python3 test_integration.py
```

Expected output: ✅ ALL TESTS PASSED

### 4. Start the Streamlit Application
```bash
streamlit run streamlit_app.py
```

The app will open at: **http://localhost:8501**

### 5. Use the Application

1. **Upload Model**
   - Go to sidebar → Upload model package (.pkl or .joblib)
   - The app validates the model structure automatically

2. **(Optional) Add Documentation**
   - Upload model card, README, or requirements document
   - Supported: PDF, Markdown, TXT

3. **(Optional) Upload Verification Data**
   - CSV file with test/validation data
   - Or generate synthetic dataset automatically

4. **Configure API Keys**
   - Enter Gemini API key (if using)
   - Enter OpenRouter API key (if using)
   - Update CrewAI URL if custom

5. **Run Governance Pipeline**
   - Click "▶️ Run Governance Pipeline" button
   - Review results across 6 tabs
   - Download PDF report

## 📊 Application Workflow

```
User Uploads Model
        ↓
App validates model structure
        ↓
User provides optional docs & dataset
        ↓
App synthesizes data if not provided
        ↓
Dashboard displays KPIs
        ↓
User clicks "Run Pipeline"
        ↓
Multi-agent auditors process model
        ↓
Results visualized in 6 tabs
        ↓
PDF report generated
        ↓
User downloads regulatory-ready PDF
```

## 🔧 Configuration

### Environment Variables (.env)
```
GEMINI_API_KEY=sk-...              # Google Gemini API key
OPENROUTER_API_KEY=sk-...          # OpenRouter API key
CREWAI_BASE_URL=http://localhost:5002  # Orchestrator endpoint
API_BASE_URL=http://localhost:5001     # Flask backend URL
```

### Streamlit Configuration (optional)
Create `~/.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#6366f1"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#f1f5f9"

[client]
maxUploadSize = 200
```

## 📋 Usage Examples

### Example 1: Audit a Loan Model
```python
# In sidebar:
1. Upload: loan_model.pkl
2. Upload: loan_model_README.md
3. Skip dataset (app generates synthetic)
4. Enter Gemini API key
5. Click "Run Governance Pipeline"
# App generates governance report with fairness analysis
```

### Example 2: Validate with Real Dataset
```python
# In sidebar:
1. Upload: fraud_detector.joblib
2. Skip documentation
3. Upload: test_data.csv
4. Keep local provider (no API key needed)
5. Click "Run Governance Pipeline"
# App analyzes data drift against uploaded dataset
```

## 🧪 Testing

### Run Integration Tests
```bash
python3 test_integration.py
```

Tests include:
- ✓ Feature schema extraction
- ✓ Model validation
- ✓ Synthetic dataset generation
- ✓ Risk level formatting
- ✓ PDF report generation
- ✓ Streamlit imports
- ✓ End-to-end workflow

### Manual Testing Checklist
- [ ] Upload .pkl model
- [ ] Upload documentation
- [ ] Generate synthetic dataset
- [ ] View all 6 dashboard tabs
- [ ] Run governance pipeline
- [ ] Generate and download PDF
- [ ] Verify PDF content

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web application framework |
| plotly | Interactive charts & visualizations |
| pandas | Data manipulation |
| reportlab | PDF generation |
| requests | HTTP client |
| joblib | Model serialization |
| PyPDF2 | PDF text extraction |

## 🔌 Integration Points

### Flask Backend (port 5001)
```
POST /api/run
├── model_file (multipart)
├── docs_file (multipart, optional)
├── provider (form data)
└── api_key (form data)

GET /api/health
```

### Mock Orchestrator (port 5002)
```
POST /api/orchestrate
├── model_file (multipart)
├── docs_file (multipart, optional)
├── provider (header)
└── Authorization: Bearer {api_key}
```

## 🎨 UI/UX Design

Modern production-grade design with:
- **Color Scheme**: Indigo/Purple primary (#6366f1)
- **Dark Mode**: Dark background with light text
- **Responsive Layout**: Works on desktop and tablets
- **Interactive Charts**: Hover tooltips and drill-down
- **Progress Indicators**: Real-time status updates
- **Risk Visualization**: Color-coded risk levels (🟢🟡🔴)

## 🛡️ Security Considerations

- API keys entered via password fields (not logged)
- .env file for credential management
- Model files validated before processing
- No credentials stored in session state
- CORS-enabled Flask backend for cross-origin requests

## 🚦 Troubleshooting

### Port Already in Use
```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run streamlit_app.py --server.port 8502
```

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### API Connection Errors
- Verify Flask backend is running: `PORT=5001 python3 backend/app.py`
- Verify mock orchestrator is running: `PORT=5002 python3 tools/mock_orchestrator.py`
- Check firewall rules for localhost connections

### PDF Generation Fails
```bash
pip install reportlab --upgrade
```

## 📚 API Reference

### Session State Variables
```python
st.session_state.model_package    # Loaded model dict
st.session_state.verification_df  # Loaded/generated dataset
st.session_state.report          # Governance report
st.session_state.running         # Pipeline execution flag
st.session_state.api_keys        # API configuration
```

### Configuration Constants
```python
COLORS = {
    'primary': '#6366f1',          # Indigo
    'success': '#10b981',          # Green
    'warning': '#f59e0b',          # Amber
    'danger': '#ef4444',           # Red
    'info': '#0ea5e9'              # Cyan
}

API_BASE_URL = "http://localhost:5001"
CREWAI_BASE_URL = "http://localhost:5002"
```

## 📖 Documentation

For detailed documentation, see:
- [STREAMLIT_README.md](./STREAMLIT_README.md) - Original setup guide
- [README.md](./README.md) - Project overview
- Code comments in each module

## 🤝 Contributing

To extend the application:

1. **Add new utility functions**: `utils.py`
2. **Enhance PDF reports**: `pdf_report.py`
3. **Add new dashboard tabs**: `streamlit_app.py` (Tabs section)
4. **Update tests**: `test_integration.py`

## 📝 License

This project is part of the ML Governance Pipeline.

## 🎯 Next Steps

- [ ] Deploy to Streamlit Cloud
- [ ] Add database for audit logs
- [ ] Implement real CrewAI integration
- [ ] Add authentication/authorization
- [ ] Support batch model auditing
- [ ] Add email report delivery
- [ ] Create admin dashboard

---

**Version**: 1.0.0  
**Last Updated**: 2026-06-26  
**Status**: ✅ Production Ready
