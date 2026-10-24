import numpy as np
import pandas as pd
from typing import Tuple, List
from src.config import LAG_DAYS, ROLLING_WINDOWS, TARGET_COL

class FeatureEngineer:
    """
    Transforms raw retail transaction series into leakage-free ML features.
    Enforces cutoff lag boundaries to support realistic operational lead times.
    """

    def __init__(self, lag_days: List[int] = None, rolling_windows: List[int] = None):
        self.lag_days = lag_days or LAG_DAYS
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        self.categorical_cols = ["sku_id", "store_id", "category", "region", "store_tier"]
        self.category_maps = {}

    def fit_transform(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Transforms training data and learns categorical encodings."""
        df = df.copy()
        df = df.sort_values(by=["sku_id", "store_id", "date"]).reset_index(drop=True)

        # 1. Calendar & Cyclical Seasonality
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Sine/Cosine cyclical transforms
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # 2. Economic & Margin Features
        df["profit_margin_pct"] = (df["unit_price"] - df["unit_cost"]) / (df["unit_price"] + 1e-6)
        df["effective_price"] = df["unit_price"] * (1 - df["discount_pct"])

        # 3. Lag Features (grouped by SKU and Store)
        # Shift at least 7 days to simulate a 7-day planning/lead-time horizon
        for lag in self.lag_days:
            df[f"demand_lag_{lag}"] = df.groupby(["sku_id", "store_id"])[TARGET_COL].shift(lag)

        # 4. Rolling Window Features on top of base lag (lag 7)
        base_series = df.groupby(["sku_id", "store_id"])[TARGET_COL].shift(7)
        for w in self.rolling_windows:
            df[f"demand_roll_mean_{w}"] = (
                df.groupby(["sku_id", "store_id"])[TARGET_COL]
                .shift(7)
                .transform(lambda s: s.rolling(w, min_periods=3).mean())
            )
            df[f"demand_roll_std_{w}"] = (
                df.groupby(["sku_id", "store_id"])[TARGET_COL]
                .shift(7)
                .transform(lambda s: s.rolling(w, min_periods=3).std())
            )
            df[f"demand_roll_max_{w}"] = (
                df.groupby(["sku_id", "store_id"])[TARGET_COL]
                .shift(7)
                .transform(lambda s: s.rolling(w, min_periods=3).max())
            )

        # 5. Price Elasticity / Promotion Interaction
        df["promo_lag_interaction"] = df["discount_pct"] * df["demand_lag_7"].fillna(0)

        # 6. Categorical Encoding (Label Encoding for Tree Models)
        for col in self.categorical_cols:
            if col in df.columns:
                if is_training or col not in self.category_maps:
                    categories = {val: idx for idx, val in enumerate(df[col].unique())}
                    self.category_maps[col] = categories
                df[f"{col}_cat"] = df[col].map(self.category_maps[col]).fillna(0).astype("category")

        # If training, drop warm-up rows containing NaN in features or target
        feature_cols = self.get_feature_columns(df)
        if is_training:
            clean_df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
        else:
            # In inference, forward fill any rolling feature gaps for the forward horizon
            df[feature_cols] = df.groupby(["sku_id", "store_id"])[feature_cols].ffill().bfill()
            clean_df = df.reset_index(drop=True)
        return clean_df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Returns the list of training feature names, excluding metadata and target."""
        exclude_cols = [
            "date", "sku_id", "store_id", "product_name", "category",
            "region", "store_tier", TARGET_COL, "stockout_flag"
        ]
        return [c for c in df.columns if c not in exclude_cols]
