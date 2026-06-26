import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import ks_2samp
from typing import Dict, Any, List, Tuple

class DriftDetector:
    """
    Performs data drift analysis using both Evidently (if available) and
    robust statistical tests (Kolmogorov-Smirnov) as a fallback.
    """
    
    @staticmethod
    def detect_drift_statistical(
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Runs a two-sample Kolmogorov-Smirnov test on all continuous features
        to identify distribution changes between reference and validation data.
        """
        drift_results = {}
        drifted_features = []
        scores = {}
        
        # Align columns
        common_cols = [c for c in reference_df.columns if c in current_df.columns]
        
        for col in common_cols:
            # Check if numeric
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                ref_data = reference_df[col].dropna()
                cur_data = current_df[col].dropna()
                
                if len(ref_data) > 10 and len(cur_data) > 10:
                    # Run KS test
                    ks_stat, p_value = ks_2samp(ref_data, cur_data)
                    drift_detected = p_value < threshold
                    
                    drift_results[col] = {
                        "drift_detected": bool(drift_detected),
                        "p_value": float(p_value),
                        "ks_stat": float(ks_stat),
                        "ref_mean": float(ref_data.mean()),
                        "cur_mean": float(cur_data.mean()),
                        "ref_std": float(ref_data.std()),
                        "cur_std": float(cur_data.std())
                    }
                    
                    scores[col] = float(ks_stat)
                    if drift_detected:
                        drifted_features.append(col)
                        
        drift_share = len(drifted_features) / len(common_cols) if common_cols else 0.0
        
        return {
            "status": "PASS" if drift_share < 0.3 else "WARNING",
            "detected": drift_share >= 0.3,
            "drift_share": drift_share,
            "drifted_features": drifted_features,
            "metrics": drift_results,
            "scores": scores
        }

    @staticmethod
    def detect_drift_from_package(
        baseline_stats: Dict[str, Any],
        current_df: pd.DataFrame,
        threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Detects drift when only baseline statistics are available (no raw reference dataset).
        Compares validation means to reference means normalized by baseline standard deviations.
        """
        drift_results = {}
        drifted_features = []
        scores = {}
        
        feature_stats = baseline_stats.get("feature_statistics", {})
        
        for col, stats in feature_stats.items():
            if col in current_df.columns:
                ref_mean = stats.get("mean")
                ref_std = stats.get("std", 1.0)
                
                cur_data = current_df[col].dropna()
                if len(cur_data) > 0:
                    cur_mean = float(cur_data.mean())
                    cur_std = float(cur_data.std())
                    
                    # Compute a simple standardized mean difference (Z-score of difference)
                    diff = abs(cur_mean - ref_mean)
                    normalized_diff = diff / (ref_std + 1e-9)
                    
                    # If mean shifts by more than 'threshold' standard deviations, flag drift
                    drift_detected = normalized_diff > threshold
                    
                    drift_results[col] = {
                        "drift_detected": bool(drift_detected),
                        "drift_score": float(normalized_diff),
                        "ref_mean": float(ref_mean),
                        "cur_mean": float(cur_mean),
                        "ref_std": float(ref_std),
                        "cur_std": float(cur_std)
                    }
                    
                    scores[col] = float(normalized_diff)
                    if drift_detected:
                        drifted_features.append(col)
                        
        drift_share = len(drifted_features) / len(feature_stats) if feature_stats else 0.0
        
        return {
            "status": "PASS" if drift_share < 0.3 else "WARNING",
            "detected": drift_share >= 0.3,
            "drift_share": drift_share,
            "drifted_features": drifted_features,
            "metrics": drift_results,
            "scores": scores
        }

    @staticmethod
    def generate_drift_chart(drift_results: Dict[str, Any]) -> go.Figure:
        """
        Generates a horizontal bar chart showing drift scores for each feature.
        """
        fig = go.Figure()
        
        features = list(drift_results["scores"].keys())
        scores = list(drift_results["scores"].values())
        drifted_features = drift_results["drifted_features"]
        
        # Set colors based on whether feature drifted
        colors = []
        for f in features:
            if f in drifted_features:
                colors.append('#FF4B4B') # Red for drifted
            else:
                colors.append('#00C49F') # Green for stable
                
        fig.add_trace(go.Bar(
            y=features,
            x=scores,
            orientation='h',
            marker_color=colors,
            text=[f"Drifted" if f in drifted_features else "Stable" for f in features],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Feature Data Drift Scores",
            xaxis_title="Drift Score (KS Statistic or Z-Shift)",
            yaxis_title="Features",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(autorange="reversed")
        )
        
        return fig
