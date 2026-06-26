const inputModel = document.getElementById('input-model');
const inputDocs = document.getElementById('input-docs');
const dropModel = document.getElementById('drop-model');
const dropDocs = document.getElementById('drop-docs');
const fnameModel = document.getElementById('fname-model');
const fnameDocs = document.getElementById('fname-docs');
const providerSelect = document.getElementById('provider-select');
const apiKeyInput = document.getElementById('input-api-key');
const btnRun = document.getElementById('btn-run');
const btnDownload = document.getElementById('btn-download');
const errorBar = document.getElementById('error-bar');
const progressWrap = document.getElementById('progress-wrap');
const progressFill = document.getElementById('progress-fill');
const progressLabel = document.getElementById('progress-label');
const resultsSection = document.getElementById('results-section');
const outputDecision = document.getElementById('output-decision');
const valGov = document.getElementById('val-gov');
const valPerf = document.getElementById('val-perf');
const valRisk = document.getElementById('val-risk');
const valSum = document.getElementById('val-sum');
const tabs = document.querySelectorAll('#tabs button');
const panels = document.querySelectorAll('.tab-panel');
const tabOverview = document.getElementById('tab-overview');
const tabPerformance = document.getElementById('tab-performance');
const tabGovernance = document.getElementById('tab-governance');
const tabRisk = document.getElementById('tab-risk');
const tabSummary = document.getElementById('tab-summary');
const tabRaw = document.getElementById('tab-raw');

let selectedModelFile = null;
let selectedDocsFile = null;
let lastReport = null;

function setError(message) {
  errorBar.textContent = message;
  errorBar.style.display = message ? 'block' : 'none';
}

function updateButtonState() {
  btnRun.disabled = !selectedModelFile;
}

function updateFileDisplay() {
  fnameModel.textContent = selectedModelFile ? selectedModelFile.name : '';
  fnameDocs.textContent = selectedDocsFile ? selectedDocsFile.name : '';
}

function setProgress(percent, label) {
  progressWrap.style.display = 'block';
  progressFill.style.width = `${percent}%`;
  progressLabel.textContent = label;
}

function hideProgress() {
  progressWrap.style.display = 'none';
  progressFill.style.width = '0%';
}

function resetPipelineUI() {
  valGov.textContent = '—';
  valPerf.textContent = '—';
  valRisk.textContent = '—';
  valSum.textContent = '—';
  outputDecision.textContent = 'Awaiting run…';
  resultsSection.style.display = 'none';
  btnDownload.style.display = 'none';
}

function updateResults(report) {
  resultsSection.style.display = 'block';
  outputDecision.textContent = report.summary.final_decision || 'Completed';
  valGov.textContent = report.governance.status || 'N/A';
  valPerf.textContent = report.performance.status || 'N/A';
  valRisk.textContent = report.risk.overall_risk_level || 'N/A';
  valSum.textContent = report.summary.final_decision || 'N/A';

  tabOverview.innerHTML = `
    <h2>Overview</h2>
    <p><strong>Decision:</strong> ${report.summary.final_decision}</p>
    <p><strong>Composite Score:</strong> ${report.summary.composite_score}/100</p>
    <p><strong>Governance:</strong> ${report.governance.status} (${report.governance.score}/100)</p>
    <p><strong>Performance:</strong> ${report.performance.status} (${report.performance.performance_score}/100)</p>
    <p><strong>Risk:</strong> ${report.risk.overall_risk_level}</p>
  `;

  tabPerformance.innerHTML = `
    <h2>Performance</h2>
    <p><strong>Status:</strong> ${report.performance.status}</p>
    <p><strong>Deployment readiness:</strong> ${report.performance.deployment_readiness}</p>
    <p><strong>Performance score:</strong> ${report.performance.performance_score}</p>
    <pre>${JSON.stringify(report.performance.metrics, null, 2)}</pre>
  `;

  tabGovernance.innerHTML = `
    <h2>Governance</h2>
    <p><strong>Status:</strong> ${report.governance.status}</p>
    <p><strong>Score:</strong> ${report.governance.score}</p>
    <p><strong>Findings:</strong></p>
    <ul>${report.governance.findings.map(item => `<li>${item}</li>`).join('')}</ul>
    <p><strong>Recommendations:</strong></p>
    <ul>${report.governance.recommendations.map(item => `<li>${item}</li>`).join('')}</ul>
  `;

  tabRisk.innerHTML = `
    <h2>Risk</h2>
    <p><strong>Overall risk:</strong> ${report.risk.overall_risk_level}</p>
    <p><strong>Recommendation:</strong> ${report.risk.recommendation}</p>
    <p><strong>Bias risk:</strong> ${report.risk.bias_risk.level}</p>
    <p><strong>Operational risk:</strong> ${report.risk.operational_risk.level}</p>
    <p><strong>Regulatory risk:</strong> ${report.risk.regulatory_risk.level}</p>
  `;

  tabSummary.innerHTML = `
    <h2>Summary</h2>
    <p>${report.summary.decision_rationale}</p>
    <p><strong>Next steps:</strong></p>
    <ul>${report.summary.next_steps.map(item => `<li>${item}</li>`).join('')}</ul>
  `;

  tabRaw.innerHTML = `<pre>${JSON.stringify(report, null, 2)}</pre>`;

  btnDownload.style.display = 'inline-flex';
}

function handleFileSelection(file, isModel) {
  if (isModel) {
    selectedModelFile = file;
  } else {
    selectedDocsFile = file;
  }
  updateFileDisplay();
  updateButtonState();
  setError('');
}

function addDropHandlers(zone, isModel) {
  zone.addEventListener('click', () => {
    if (isModel) {
      inputModel.click();
    } else {
      inputDocs.click();
    }
  });

  zone.addEventListener('dragover', (event) => {
    event.preventDefault();
    zone.classList.add('dragover');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('dragover');
  });

  zone.addEventListener('drop', (event) => {
    event.preventDefault();
    zone.classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    if (!file) return;
    if (isModel) {
      if (!['.pkl', '.joblib'].some(ext => file.name.toLowerCase().endsWith(ext))) {
        setError('Model file must be .pkl or .joblib');
        return;
      }
    } else {
      if (!['.pdf', '.md', '.txt'].some(ext => file.name.toLowerCase().endsWith(ext))) {
        setError('Docs file must be .pdf, .md, or .txt');
        return;
      }
    }
    handleFileSelection(file, isModel);
  });
}

inputModel.addEventListener('change', () => {
  const file = inputModel.files[0];
  if (file) handleFileSelection(file, true);
});

inputDocs.addEventListener('change', () => {
  const file = inputDocs.files[0];
  if (file) handleFileSelection(file, false);
});

addDropHandlers(dropModel, true);
addDropHandlers(dropDocs, false);

btnRun.addEventListener('click', async () => {
  if (!selectedModelFile) {
    setError('Please select a model package first.');
    return;
  }

  setError('');
  btnRun.disabled = true;
  resetPipelineUI();
  setProgress(10, 'Uploading model package...');

  const formData = new FormData();
  formData.append('model_file', selectedModelFile);
  if (selectedDocsFile) {
    formData.append('docs_file', selectedDocsFile);
  }
  // Provider and API key (optional)
  formData.append('provider', providerSelect ? providerSelect.value : 'local');
  formData.append('api_key', apiKeyInput ? apiKeyInput.value : '');

  try {
    const response = await fetch('/api/run', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || 'Server returned an error');
    }

    setProgress(65, 'Running governance pipeline...');
    const report = await response.json();
    lastReport = report;
    updateResults(report);
    setProgress(100, 'Pipeline complete');

    btnDownload.style.display = 'inline-flex';
    btnDownload.onclick = () => {
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `governance_report_${Date.now()}.json`;
      anchor.click();
      URL.revokeObjectURL(url);
    };
  } catch (error) {
    setError(error.message || 'Upload failed');
    hideProgress();
  } finally {
    btnRun.disabled = false;
  }
});

tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    tabs.forEach((btn) => btn.classList.remove('active'));
    panels.forEach((panel) => panel.classList.remove('active'));
    tab.classList.add('active');
    const target = tab.dataset.tab;
    document.getElementById(`tab-${target}`).classList.add('active');
  });
});
