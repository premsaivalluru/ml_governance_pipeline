"""
backend/app.py
Flask REST API — serves the pipeline and static frontend.
"""

import os
import sys
import json
import pathlib
import tempfile

from flask import (
    Flask, request, jsonify, send_from_directory, render_template
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline import run_pipeline

# ── App setup ──────────────────────────────────────────────────────────────
FRONTEND_TEMPLATES = ROOT / "frontend" / "templates"
FRONTEND_STATIC    = ROOT / "frontend" / "static"
OUTPUTS_DIR        = ROOT / "outputs"

app = Flask(
    __name__,
    template_folder=str(FRONTEND_TEMPLATES),
    static_folder=str(FRONTEND_STATIC),
    static_url_path="/static",
)
CORS(app)

ALLOWED_EXTENSIONS = {".pkl", ".joblib"}
ALLOWED_DOC_EXTENSIONS = {".pdf", ".md", ".txt"}


def _allowed(filename: str, allowed: set) -> bool:
    return pathlib.Path(filename).suffix.lower() in allowed


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main SPA."""
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    """
    POST /api/run
    Multipart form-data:
      model_file  : required .pkl / .joblib
      docs_file   : optional .pdf / .md / .txt
    Returns the full pipeline bundle as JSON.
    """
    if "model_file" not in request.files:
        return jsonify({"error": "model_file is required"}), 400

    model_file = request.files["model_file"]
    if not _allowed(model_file.filename, ALLOWED_EXTENSIONS):
        return jsonify({"error": "model_file must be .pkl or .joblib"}), 400

    docs_file = request.files.get("docs_file")

    with tempfile.TemporaryDirectory() as tmp:
        # Save model
        pkl_path = os.path.join(tmp, secure_filename(model_file.filename))
        model_file.save(pkl_path)

        # Save docs (optional)
        docs_path = ""
        if docs_file and docs_file.filename:
            if _allowed(docs_file.filename, ALLOWED_DOC_EXTENSIONS):
                docs_path = os.path.join(tmp, secure_filename(docs_file.filename))
                docs_file.save(docs_path)

        # Provider and API key (optional, forwarded from frontend)
        provider = request.form.get('provider', 'local')
        api_key = request.form.get('api_key', '')

        try:
            bundle = run_pipeline(pkl_path, docs_path, provider, api_key)
            return jsonify(bundle)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500


@app.route("/api/reports", methods=["GET"])
def api_reports():
    """
    GET /api/reports
    Returns a list of previously generated report filenames.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(OUTPUTS_DIR.glob("governance_report_*.json"), reverse=True)
    return jsonify([f.name for f in files])


@app.route("/api/reports/<filename>", methods=["GET"])
def api_report_detail(filename):
    """
    GET /api/reports/<filename>
    Returns a specific report JSON.
    """
    path = OUTPUTS_DIR / secure_filename(filename)
    if not path.exists():
        return jsonify({"error": "Report not found"}), 404
    with open(path) as fh:
        return jsonify(json.load(fh))


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── Dev server ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port, host="0.0.0.0")