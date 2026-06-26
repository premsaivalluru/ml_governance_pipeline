import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Tuple

class FairnessAuditor:
    """
    Computes fairness metrics and generates Plotly charts.
    Does not strictly require Fairlearn/AIF360, ensuring 100% compatibility.
    """
    
    @staticmethod
    def audit_fairness(
        df: pd.DataFrame,
        protected_attribute: str,
        predictions: np.ndarray,
        ground_truth: np.ndarray = None,
        privileged_value: Any = 0, # e.g. Graduate (0) is privileged compared to Not Graduate (1)
        positive_outcome: Any = 0  # e.g. Loan approved (0) is the positive outcome
    ) -> Dict[str, Any]:
        """
        Computes Demographic Parity Difference, Disparate Impact Ratio, and Equal Opportunity Difference.
        """
        df = df.copy()
        df["predictions"] = predictions
        
        # Calculate selection rates
        privileged_df = df[df[protected_attribute] == privileged_value]
        unprivileged_df = df[df[protected_attribute] != privileged_value]
        
        if len(privileged_df) == 0 or len(unprivileged_df) == 0:
            return {
                "error": f"One of the groups is empty for protected attribute '{protected_attribute}'. Make sure the privileged value {privileged_value} exists in the data."
            }
            
        priv_selection_rate = (privileged_df["predictions"] == positive_outcome).mean()
        unpriv_selection_rate = (unprivileged_df["predictions"] == positive_outcome).mean()
        
        # Demographic Parity Difference
        demographic_parity_diff = abs(priv_selection_rate - unpriv_selection_rate)
        
        # Disparate Impact Ratio (avoid division by zero)
        if priv_selection_rate > 0:
            disparate_impact_ratio = unpriv_selection_rate / priv_selection_rate
        else:
            disparate_impact_ratio = 1.0 if unpriv_selection_rate == 0 else float('inf')
            
        equal_opportunity_diff = None
        priv_tpr = None
        unpriv_tpr = None
        
        # Equal Opportunity Difference (requires ground truth)
        if ground_truth is not None:
            df["ground_truth"] = ground_truth
            
            # Recalculate with ground truth positive outcomes (Y = positive_outcome)
            priv_pos = privileged_df[ground_truth[privileged_df.index] == positive_outcome]
            unpriv_pos = unprivileged_df[ground_truth[unprivileged_df.index] == positive_outcome]
            
            if len(priv_pos) > 0:
                priv_tpr = (priv_pos["predictions"] == positive_outcome).mean()
            else:
                priv_tpr = 0.0
                
            if len(unpriv_pos) > 0:
                unpriv_tpr = (unpriv_pos["predictions"] == positive_outcome).mean()
            else:
                unpriv_tpr = 0.0
                
            equal_opportunity_diff = abs(priv_tpr - unpriv_tpr)
            
        return {
            "protected_attribute": protected_attribute,
            "privileged_group_size": len(privileged_df),
            "unprivileged_group_size": len(unprivileged_df),
            "privileged_selection_rate": float(priv_selection_rate),
            "unprivileged_selection_rate": float(unpriv_selection_rate),
            "demographic_parity_difference": float(demographic_parity_diff),
            "disparate_impact_ratio": float(disparate_impact_ratio),
            "privileged_tpr": float(priv_tpr) if priv_tpr is not None else None,
            "unprivileged_tpr": float(unpriv_tpr) if unpriv_tpr is not None else None,
            "equal_opportunity_difference": float(equal_opportunity_diff) if equal_opportunity_diff is not None else None
        }
        
    @staticmethod
    def generate_fairness_charts(results: Dict[str, Any], protected_name: str) -> go.Figure:
        """
        Creates a dual bar chart showing Selection Rate and TPR comparison between groups.
        """
        fig = go.Figure()
        
        # Groups
        groups = ["Privileged Group", "Unprivileged Group"]
        
        # Selection Rates
        selection_rates = [
            results["privileged_selection_rate"] * 100,
            results["unprivileged_selection_rate"] * 100
        ]
        
        fig.add_trace(go.Bar(
            name='Selection Rate (Approval %)',
            x=groups,
            y=selection_rates,
            marker_color='#4B70F5',
            text=[f"{val:.1f}%" for val in selection_rates],
            textposition='auto',
        ))
        
        # TPR
        if results.get("privileged_tpr") is not None:
            tprs = [
                results["privileged_tpr"] * 100,
                results["unprivileged_tpr"] * 100
            ]
            fig.add_trace(go.Bar(
                name='True Positive Rate (Opportunity %)',
                x=groups,
                y=tprs,
                marker_color='#3D30A2',
                text=[f"{val:.1f}%" for val in tprs],
                textposition='auto',
            ))
            
        fig.update_layout(
            title=f"Fairness Metric Breakdown (Attribute: {protected_name})",
            xaxis_title="Groups",
            yaxis_title="Percentage (%)",
            barmode='group',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
