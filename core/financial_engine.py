"""
core/financial_engine.py
────────────────────────────────────────────────────────────────────────────────
Pure Python business logic for the financial allocation model.
No Streamlit calls — safe to test independently.
"""

from __future__ import annotations

import os
import json
import logging
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "models", "multioutput_xgb_model.pkl")

_model = None


def _load_model():
    """Lazy-load model once."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


# ── Helpers ───────────────────────────────────────────────────────────────────

def sum_json_amount(json_list) -> float:
    """Sum amounts from a JSON list safely."""
    if not json_list or (isinstance(json_list, float) and pd.isna(json_list)):
        return 0.0
    if isinstance(json_list, str):
        try:
            json_list = json.loads(json_list)
        except Exception:
            return 0.0
    return float(sum(item.get("amount", 0) for item in json_list))


def calculate_features(data: pd.DataFrame):
    """
    Prepare features for ML model from raw user-input DataFrame.
    Returns (enriched_df, feature_cols_list).
    """
    # Categorical encoding
    data["marital_status_encoded"]     = data["marital_status"].map({"single": 0, "married": 1})
    data["saving_preference_encoded"]  = data["saving_preference"].map({"Low": 0, "Medium": 0.5, "High": 1})
    data["risk_tolerance_encoded"]     = data["risk_tolerance"].map({"Low": 0, "Medium": 0.5, "High": 1})
    data["living_cost_level_encoded"]  = data["living_cost_level"].map({"Low": 0.8, "Medium": 1.0, "High": 1.2})

    # Priority encoding
    priority_map = {"Food": 0, "Transport": 1, "Optional": 2, "Savings": 3, "Emergency": 4}
    for i in range(1, 5):
        data[f"priority_{i}_num"] = data[f"priority_{i}"].map(priority_map).fillna(2)

    # Totals
    data["total_services_cost"]       = data["rent"] + data["utilities"] + data["transportation"] + data["optional_services_cost"]
    data["total_monthly_debt"]        = data["monthly_debts_json"].apply(sum_json_amount)
    data["monthly_allocated_expenses"]= data["annual_expenses_json"].apply(lambda x: sum_json_amount(x) / 12)
    data["available_income"]          = (
        data["monthly_salary"]
        - data["total_services_cost"]
        - data["total_monthly_debt"]
        - data["monthly_allocated_expenses"]
    )
    data["services_percentage"]       = (data["total_services_cost"] / data["monthly_salary"]).fillna(0) * 100

    feature_cols = [
        "monthly_salary", "number_of_family_members",
        "marital_status_encoded", "living_cost_level_encoded",
        "income_stability", "saving_preference_encoded", "risk_tolerance_encoded",
        "priority_1_num", "priority_2_num", "priority_3_num", "priority_4_num",
        "rent", "utilities", "transportation",
        "total_services_cost", "services_percentage",
        "optional_services_cost", "total_monthly_debt",
        "monthly_allocated_expenses", "available_income",
    ]
    return data, feature_cols


def predict_ratios(df_features: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Run ML inference and return ratio predictions."""
    model = _load_model()
    X = df_features[feature_cols]
    y_pred = model.predict(X)
    return pd.DataFrame(
        y_pred,
        columns=["food_ratio", "saving_ratio", "emergency_ratio", "optional_ratio"],
    )


def build_allocation(monthly_salary: float, predicted_ratios: pd.DataFrame,
                     available_income: float) -> dict:
    """
    Convert ratio predictions to actual EGP amounts.
    Returns a dict with labelled allocations.
    """
    r = predicted_ratios.iloc[0]
    return {
        "available_income": round(available_income, 2),
        "food":             round(r["food_ratio"]      * available_income, 2),
        "saving":           round(r["saving_ratio"]    * available_income, 2),
        "emergency":        round(r["emergency_ratio"] * available_income, 2),
        "optional":         round(r["optional_ratio"]  * available_income, 2),
        "ratios": {
            "food":      round(float(r["food_ratio"]),      4),
            "saving":    round(float(r["saving_ratio"]),    4),
            "emergency": round(float(r["emergency_ratio"]), 4),
            "optional":  round(float(r["optional_ratio"]),  4),
        },
    }
