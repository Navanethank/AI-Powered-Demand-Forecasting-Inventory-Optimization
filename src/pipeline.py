import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import timedelta
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import MODELS_DIR, TARGET_COL
from db.queries import fetch_sku_history, fetch_products, fetch_training_dataset
from src.features import FeatureEngineer
from src.optimization.inventory import InventoryOptimizer
from src.optimization.simulator import InventoryBacktestSimulator

class ForecastingPipeline:
    """
    Central inference engine serving predictions, inventory triggers, and simulations.
    """

    def __init__(self):
        self.bundle_path = MODELS_DIR / "forecast_bundle.pkl"
        self.bundle = None
        self.optimizer = InventoryOptimizer()
        self._load_bundle()

    def _load_bundle(self):
        if not self.bundle_path.exists():
            print("[*] Model bundle not found. Training models now...")
            from src.models.trainer import ForecastModelTrainer
            trainer = ForecastModelTrainer()
            self.bundle = trainer.run_training_pipeline()
        else:
            self.bundle = joblib.load(self.bundle_path)

    def forecast_sku(self, sku_id: str, store_id: str = "STR-01", horizon_days: int = 14) -> dict:
        """
        Generates multi-day probabilistic forecast (P10, P50, P90) for a specific SKU & Store.
        """
        history_df = fetch_sku_history(sku_id, store_id)
        if history_df.empty:
            raise ValueError(f"No history found for SKU: {sku_id} and Store: {store_id}")

        last_date = history_df["date"].max()
        product_info = fetch_products()
        prod = product_info[product_info["sku_id"] == sku_id].iloc[0]

        # Construct future horizon rows
        future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon_days)]
        future_rows = []
        for d in future_dates:
            future_rows.append({
                "date": d,
                "sku_id": sku_id,
                "store_id": store_id,
                "product_name": prod["name"],
                "category": prod["category"],
                "unit_cost": prod["unit_cost"],
                "unit_price": prod["unit_price"],
                "lead_time_days": prod["lead_time_days"],
                "min_order_qty": prod["min_order_qty"],
                "region": "Northeast",
                "store_tier": "Tier-1",
                "demand": np.nan,
                "is_promoted": 0,
                "discount_pct": 0.0,
                "stockout_flag": 0
            })
        
        future_df = pd.DataFrame(future_rows)
        # Concatenate history to compute trailing lags/rolling statistics
        full_df = pd.concat([history_df.assign(
            product_name=prod["name"],
            category=prod["category"],
            region="Northeast",
            store_tier="Tier-1",
            min_order_qty=prod["min_order_qty"],
            demand=history_df["units_sold"]
        ), future_df], ignore_index=True)

        # Transform using saved feature engineer
        fe: FeatureEngineer = self.bundle["feature_engineer"]
        feats_df = fe.fit_transform(full_df, is_training=False)
        
        # Filter back to future horizon
        horizon_feats = feats_df[feats_df["date"].isin(future_dates)]
        feature_cols = self.bundle["feature_cols"]

        X_future = horizon_feats[feature_cols]

        p10 = np.maximum(0, self.bundle["models"]["p10"].predict(X_future))
        p50 = np.maximum(0, self.bundle["models"]["p50"].predict(X_future))
        p90 = np.maximum(0, self.bundle["models"]["p90"].predict(X_future))

        # Recent 30-day historical points for charting
        recent_history = history_df.tail(30)

        return {
            "sku_id": sku_id,
            "store_id": store_id,
            "product_name": prod["name"],
            "unit_cost": float(prod["unit_cost"]),
            "unit_price": float(prod["unit_price"]),
            "default_lead_time": int(prod["lead_time_days"]),
            "min_order_qty": int(prod["min_order_qty"]),
            "historical_dates": recent_history["date"].dt.strftime("%Y-%m-%d").tolist(),
            "historical_demand": recent_history["units_sold"].tolist(),
            "forecast_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
            "p10": [round(float(v), 1) for v in p10],
            "p50": [round(float(v), 1) for v in p50],
            "p90": [round(float(v), 1) for v in p90],
            "historical_daily_std": float(np.std(recent_history["units_sold"]))
        }

    def get_inventory_recommendation(
        self,
        sku_id: str,
        store_id: str = "STR-01",
        current_stock: int = 40,
        lead_time_override: int = None,
        service_level: float = 0.95
    ) -> dict:
        """
        Combines model forecast with inventory optimization math to issue operational replenishment orders.
        """
        forecast_data = self.forecast_sku(sku_id, store_id, horizon_days=14)
        lead_time = lead_time_override if lead_time_override is not None else forecast_data["default_lead_time"]

        avg_p10 = float(np.mean(forecast_data["p10"]))
        avg_p50 = float(np.mean(forecast_data["p50"]))
        avg_p90 = float(np.mean(forecast_data["p90"]))

        rec = self.optimizer.generate_replenishment_recommendation(
            sku_id=sku_id,
            current_stock=current_stock,
            lead_time_days=lead_time,
            min_order_qty=forecast_data["min_order_qty"],
            unit_cost=forecast_data["unit_cost"],
            unit_price=forecast_data["unit_price"],
            forecast_p10=avg_p10,
            forecast_p50=avg_p50,
            forecast_p90=avg_p90,
            historical_daily_std=forecast_data["historical_daily_std"],
            service_level=service_level
        )
        rec["product_name"] = forecast_data["product_name"]
        rec["unit_price"] = forecast_data["unit_price"]
        rec["unit_cost"] = forecast_data["unit_cost"]
        rec["avg_daily_demand"] = round(avg_p50, 1)
        return rec

    def run_backtest_simulation(self, sku_id: str, store_id: str = "STR-01") -> dict:
        """
        Runs an out-of-time historical backtest for a specific SKU.
        """
        history_df = fetch_sku_history(sku_id, store_id)
        prod = fetch_products().set_index("sku_id").loc[sku_id]
        
        # Take last 60 days for simulation
        sim_df = history_df.tail(60).reset_index(drop=True)
        daily_demands = sim_df["units_sold"].values

        # Generate rolling pseudo-forecasts and safety stocks
        fe: FeatureEngineer = self.bundle["feature_engineer"]
        full_df = fetch_training_dataset()
        sku_full = full_df[(full_df["sku_id"] == sku_id) & (full_df["store_id"] == store_id)]
        sku_feats = fe.fit_transform(sku_full).tail(60)
        
        feature_cols = self.bundle["feature_cols"]
        X = sku_feats[feature_cols]

        pred_p50 = np.maximum(0, self.bundle["models"]["p50"].predict(X))
        pred_p90 = np.maximum(0, self.bundle["models"]["p90"].predict(X))
        safety_stocks = np.maximum(5, pred_p90 - pred_p50)

        simulator = InventoryBacktestSimulator(
            lead_time_days=int(prod["lead_time_days"]),
            unit_cost=float(prod["unit_cost"]),
            unit_price=float(prod["unit_price"]),
            annual_holding_rate=0.20
        )

        results = simulator.run_simulation(
            daily_demands=daily_demands,
            ai_forecast_p50=pred_p50,
            ai_safety_stocks=safety_stocks
        )
        results["sku_id"] = sku_id
        results["product_name"] = prod["name"]
        return results
