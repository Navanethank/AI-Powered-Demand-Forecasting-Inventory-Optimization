import numpy as np
import pandas as pd
from typing import Dict

def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE).
    Preferred in retail demand forecasting over MAPE because it avoids zero-division issues.
    Formula: sum(|y_true - y_pred|) / sum(y_true)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual)

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def calculate_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Pinball Loss (Quantile Loss) for evaluating probabilistic forecasts.
    L_q(y, y_hat) = max(q * (y - y_hat), (1 - q) * (y_hat - y))
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    err = y_true - y_pred
    loss = np.maximum(quantile * err, (quantile - 1) * err)
    return float(np.mean(loss))

def calculate_quantile_coverage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates empirical fraction of actual observations that fell below the predicted quantile."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true <= y_pred))

def evaluate_forecast_system(y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Evaluates point predictions (WAPE, RMSE) and quantiles (Pinball, Empirical Coverage)."""
    results = {}
    
    if "p50" in predictions:
        results["P50_Median"] = {
            "WAPE": round(calculate_wape(y_true, predictions["p50"]), 4),
            "RMSE": round(calculate_rmse(y_true, predictions["p50"]), 2),
            "Coverage_Pct": round(calculate_quantile_coverage(y_true, predictions["p50"]) * 100, 2),
            "Pinball_Loss": round(calculate_pinball_loss(y_true, predictions["p50"], 0.50), 3)
        }

    if "p10" in predictions:
        results["P10_Pessimistic"] = {
            "Coverage_Pct": round(calculate_quantile_coverage(y_true, predictions["p10"]) * 100, 2),
            "Pinball_Loss": round(calculate_pinball_loss(y_true, predictions["p10"], 0.10), 3)
        }

    if "p90" in predictions:
        results["P90_Optimistic"] = {
            "Coverage_Pct": round(calculate_quantile_coverage(y_true, predictions["p90"]) * 100, 2),
            "Pinball_Loss": round(calculate_pinball_loss(y_true, predictions["p90"], 0.90), 3)
        }

    if "xgboost_baseline" in predictions:
        results["XGBoost_Baseline"] = {
            "WAPE": round(calculate_wape(y_true, predictions["xgboost_baseline"]), 4),
            "RMSE": round(calculate_rmse(y_true, predictions["xgboost_baseline"]), 2)
        }

    return results
