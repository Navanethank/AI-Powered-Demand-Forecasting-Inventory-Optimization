import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import MODELS_DIR, QUANTILES, TARGET_COL
from db.queries import fetch_training_dataset
from src.features import FeatureEngineer
from src.models.evaluate import evaluate_forecast_system

class ForecastModelTrainer:
    """
    Trains and persists LightGBM Quantile models and XGBoost point baselines.
    """

    def __init__(self, test_days: int = 60):
        self.test_days = test_days
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.feature_cols = []

    def run_training_pipeline(self):
        print("[*] Fetching training data from database...")
        raw_df = fetch_training_dataset()

        print("[*] Transforming features with leakage-safe rolling statistics...")
        df_feats = self.feature_engineer.fit_transform(raw_df)
        self.feature_cols = self.feature_engineer.get_feature_columns(df_feats)
        print(f"[+] Engineered {len(self.feature_cols)} features: {self.feature_cols[:6]}...")

        # Temporal Out-Of-Time Train / Test Split
        cutoff_date = df_feats["date"].max() - pd.Timedelta(days=self.test_days)
        train_df = df_feats[df_feats["date"] <= cutoff_date]
        test_df = df_feats[df_feats["date"] > cutoff_date]

        print(f"[*] Train set: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} rows)")
        print(f"[*] Test set:  {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} rows)")

        X_train = train_df[self.feature_cols]
        y_train = train_df[TARGET_COL]
        X_test = test_df[self.feature_cols]
        y_test = test_df[TARGET_COL]

        predictions = {}

        # 1. Train LightGBM Quantile Models (P10, P50, P90)
        for q in QUANTILES:
            q_name = f"p{int(q*100)}"
            print(f"[*] Training LightGBM Quantile Regressor ({q_name.upper()} - {q*100}% quantile)...")
            
            lgb_reg = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=180,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            lgb_reg.fit(X_train, y_train)
            self.models[q_name] = lgb_reg
            predictions[q_name] = np.maximum(0, lgb_reg.predict(X_test))

        # 2. Train XGBoost Baseline Point Forecast
        print("[*] Training XGBoost Baseline Regressor (MSE)...")
        # Ensure category dtypes are handled for XGBoost
        X_train_xgb = X_train.copy()
        X_test_xgb = X_test.copy()
        for col in X_train_xgb.select_dtypes(include=["category"]).columns:
            X_train_xgb[col] = X_train_xgb[col].cat.codes
            X_test_xgb[col] = X_test_xgb[col].cat.codes

        xgb_reg = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
        xgb_reg.fit(X_train_xgb, y_train)
        self.models["xgboost_baseline"] = xgb_reg
        predictions["xgboost_baseline"] = np.maximum(0, xgb_reg.predict(X_test_xgb))

        # 3. Model Evaluation
        print("\n" + "="*50)
        print("          OUT-OF-TIME EVALUATION RESULTS")
        print("="*50)
        eval_metrics = evaluate_forecast_system(y_test.values, predictions)
        for model_name, metrics in eval_metrics.items():
            print(f"\n[{model_name}]")
            for k, v in metrics.items():
                print(f"  - {k}: {v}")

        # 4. Save Artifacts
        print("\n[*] Saving model artifacts to disk...")
        artifact_bundle = {
            "models": self.models,
            "feature_engineer": self.feature_engineer,
            "feature_cols": self.feature_cols,
            "evaluation_metrics": eval_metrics,
            "last_training_date": str(df_feats["date"].max().date())
        }
        joblib.dump(artifact_bundle, MODELS_DIR / "forecast_bundle.pkl")
        print(f"[+] Saved bundle to: {MODELS_DIR / 'forecast_bundle.pkl'}")
        return artifact_bundle

if __name__ == "__main__":
    trainer = ForecastModelTrainer()
    trainer.run_training_pipeline()
