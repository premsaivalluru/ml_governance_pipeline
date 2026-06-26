"""
Simple mock CrewAI orchestration server for local testing.
Accepts POST /api/orchestrate with multipart form-data:
  - model_file (required)
  - docs_file (optional)
  - provider (form field)
Requires no auth; responds with a dummy pipeline bundle JSON similar
to the local pipeline output so you can test end-to-end.

Run:
  python3 tools/mock_orchestrator.py

By default listens on port 5002.
"""
from flask import Flask, request, jsonify
import pathlib
import datetime
import json
import traceback

app = Flask(__name__)

@app.route('/api/orchestrate', methods=['POST'])
def orchestrate():
    try:
        if 'model_file' not in request.files:
            return jsonify({'error': 'model_file is required'}), 400

        model = request.files['model_file']
        docs = request.files.get('docs_file')
        provider = request.form.get('provider', 'unknown')
        print(f'[mock] Received provider={provider}, model={model.filename}')

        # Save files to outputs dir for inspection (optional)
        out_dir = pathlib.Path('outputs')
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            model_path = out_dir / f'mock_received_{ts}_{model.filename}'
            model.save(str(model_path))
            print(f'[mock] Saved model to {model_path}')
        except Exception as e:
            print(f'[mock] Warning: failed to save model file: {e}')
        
        docs_path = ''
        if docs and docs.filename:
            try:
                docs_path = out_dir / f'mock_docs_{ts}_{docs.filename}'
                docs.save(str(docs_path))
                print(f'[mock] Saved docs to {docs_path}')
            except Exception as e:
                print(f'[mock] Warning: failed to save docs file: {e}')

        # Create a dummy report mimicking the real pipeline structure
        bundle = {
            'pipeline_run_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_file': model.filename,
            'governance': {
                'agent_name': 'Governance Policy Agent',
                'score': 78,
                'status': 'WARNING',
                'findings': ['Documentation sparse', 'No explainability report'],
                'recommendations': ['Add model card', 'Generate SHAP report'],
            },
            'performance': {
                'agent': 'Performance Monitor Agent',
                'status': 'PASS',
                'performance_score': 88.5,
                'deployment_readiness': 'READY',
                'metrics': {'accuracy': 0.9, 'precision': 0.88, 'recall': 0.87}
            },
            'risk': {
                'agent_name': 'Risk Analysis Agent',
                'overall_risk_level': 'LOW',
                'risk_score': 22,
                'recommendation': 'APPROVED',
                'bias_risk': {'level': 'LOW', 'details': ''},
                'operational_risk': {'level': 'LOW', 'details': ''},
                'regulatory_risk': {'level': 'LOW', 'details': ''}
            },
            'summary': {
                'agent_name': 'Governance Summarizer',
                'final_decision': 'CONDITIONAL APPROVAL',
                'decision_rationale': 'Composite score 80/100',
                'composite_score': 80,
                'agent_scores': {'governance': 78, 'performance': 88.5, 'risk': 22},
                'next_steps': ['Address governance warnings', 'Re-run pipeline']
            }
        }
        
        print(f'[mock] Returning bundle for {model.filename}')
        return jsonify(bundle)
    
    except Exception as e:
        print(f'[mock] ERROR: {e}')
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

if __name__ == '__main__':
    print('Mock orchestrator running on http://0.0.0.0:5002')
    app.run(host='0.0.0.0', port=5002, debug=True)
