import os
import joblib
import pickle
import PyPDF2
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def extract_documentation(file_path: str) -> str:
    """Extracts text from a PDF, MD, or TXT file."""
    if not file_path or not os.path.exists(file_path):
        return "No formal documentation was provided for this model."
    
    _, ext = os.path.splitext(file_path)
    extracted_text = ""
    try:
        if ext.lower() == '.pdf':
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"
        elif ext.lower() in ['.md', '.txt']:
            with open(file_path, 'r', encoding='utf-8') as file:
                extracted_text = file.read()
        return extracted_text.strip() if extracted_text else "Empty documentation file."
    except Exception as e:
        return f"Error reading documentation: {str(e)}"

def load_model_package(file_path: str) -> Dict[str, Any]:
    """
    Loads model or model package from a pickle/joblib file.
    Normalizes the structure into a common metadata schema.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        package = joblib.load(file_path)
    except Exception:
        try:
            with open(file_path, 'rb') as f:
                package = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Could not parse pickle/joblib file: {str(e)}")

    normalized_data = {
        "is_package": False,
        "metadata": {},
        "performance_metrics": {},
        "data_profile": {},
        "feature_metadata": {},
        "drift_baseline": {},
        "explainability_report": {},
        "raw_model": None
    }

    # Case 1: Already structured model package
    if isinstance(package, dict) and ("metadata" in package or "performance_metrics" in package):
        normalized_data["is_package"] = True
        normalized_data.update(package)
        if "model" in package:
            normalized_data["raw_model"] = package["model"]
        return normalized_data

    # Case 2: Raw ML Model (Scikit-Learn, XGBoost, etc.)
    normalized_data["raw_model"] = package
    
    # Infer basic model info
    model_type = type(package).__name__
    model_module = type(package).__module__
    
    params = {}
    if hasattr(package, "get_params"):
        params = package.get_params()
    
    # Check features
    features = []
    if hasattr(package, "feature_names_in_"):
        features = list(package.feature_names_in_)
    elif hasattr(package, "feature_names") and package.feature_names:
        features = list(package.feature_names)
        
    normalized_data["metadata"] = {
        "model_name": f"Raw {model_type}",
        "version": "1.0-raw",
        "algorithm": model_type,
        "training_date": "Unknown (Raw model uploaded)",
        "feature_names": features,
        "hyperparameters": params
    }
    
    normalized_data["feature_metadata"] = {
        f: {"dtype": "float64"} for f in features
    }
    
    return normalized_data

def generate_synthetic_validation_data(feature_names: list, n_samples: int = 1000) -> pd.DataFrame:
    """
    Generates synthetic validation data for testing drift and fairness metrics.
    Attempts to generate realistic data based on known features of the Loan Approval Dataset.
    """
    np.random.seed(42)
    data = {}
    
    # Let's map typical loan approval feature distributions
    for col in feature_names:
        clean_col = col.strip()
        if clean_col == "cibil_score":
            data[col] = np.random.randint(300, 900, size=n_samples)
        elif clean_col == "income_annum":
            data[col] = np.random.randint(200000, 10000000, size=n_samples)
        elif clean_col == "loan_amount":
            data[col] = np.random.randint(100000, 8000000, size=n_samples)
        elif clean_col == "loan_term":
            data[col] = np.random.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], size=n_samples)
        elif clean_col == "no_of_dependents":
            data[col] = np.random.randint(0, 6, size=n_samples)
        elif clean_col == "education":
            # 0: Graduate, 1: Not Graduate (Standard encoding)
            data[col] = np.random.choice([0, 1], p=[0.75, 0.25], size=n_samples)
        elif clean_col == "self_employed":
            # 0: No, 1: Yes
            data[col] = np.random.choice([0, 1], p=[0.5, 0.5], size=n_samples)
        elif clean_col == "residential_assets_value":
            data[col] = np.random.randint(0, 15000000, size=n_samples)
        elif clean_col == "commercial_assets_value":
            data[col] = np.random.randint(0, 15000000, size=n_samples)
        elif clean_col == "luxury_assets_value":
            data[col] = np.random.randint(0, 30000000, size=n_samples)
        elif clean_col == "bank_asset_value":
            data[col] = np.random.randint(0, 15000000, size=n_samples)
        elif clean_col == "total_assets":
            # Derived asset value
            res = data.get("residential_assets_value", np.random.randint(0, 15000000, size=n_samples))
            comm = data.get("commercial_assets_value", np.random.randint(0, 15000000, size=n_samples))
            lux = data.get("luxury_assets_value", np.random.randint(0, 30000000, size=n_samples))
            bank = data.get("bank_asset_value", np.random.randint(0, 15000000, size=n_samples))
            data[col] = res + comm + lux + bank
        elif clean_col == "loan_income_ratio":
            # Derived
            amt = data.get("loan_amount", np.random.randint(100000, 8000000, size=n_samples))
            inc = data.get("income_annum", np.random.randint(200000, 10000000, size=n_samples))
            data[col] = amt / (inc + 1.0)
        elif clean_col == "loan_asset_ratio":
            # Derived
            amt = data.get("loan_amount", np.random.randint(100000, 8000000, size=n_samples))
            assets = data.get("total_assets", np.random.randint(0, 90000000, size=n_samples))
            data[col] = amt / (assets + 1.0)
        elif clean_col == "assets_income_ratio":
            # Derived
            assets = data.get("total_assets", np.random.randint(0, 90000000, size=n_samples))
            inc = data.get("income_annum", np.random.randint(200000, 10000000, size=n_samples))
            data[col] = assets / (inc + 1.0)
        else:
            # General fallback continuous variable
            data[col] = np.random.normal(loc=10.0, scale=3.0, size=n_samples)
            
    return pd.DataFrame(data)
