import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.features import FeatureEngineer

@pytest.fixture
def sample_sales_data():
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(120)]
    records = []
    for d in dates:
        records.append({
            "date": d,
            "sku_id": "SKU-TEST",
            "store_id": "STR-01",
            "category": "Electronics",
            "region": "Northeast",
            "store_tier": "Tier-1",
            "unit_cost": 50.0,
            "unit_price": 100.0,
            "lead_time_days": 7,
            "min_order_qty": 10,
            "demand": 20 + int(np.sin(len(records)) * 5),
            "is_promoted": 0,
            "discount_pct": 0.0,
            "stockout_flag": 0
        })
    return pd.DataFrame(records)

def test_feature_generation_no_leakage(sample_sales_data):
    fe = FeatureEngineer()
    df_feats = fe.fit_transform(sample_sales_data)
    
    # Check that lag 7 is present and lag 1 is absent (leakage avoidance for 7d horizon)
    assert "demand_lag_7" in df_feats.columns
    assert "demand_lag_1" not in df_feats.columns
    
    # Check cyclical transforms exist
    assert "dow_sin" in df_feats.columns
    assert "dow_cos" in df_feats.columns
    
    # Verify no NaN values remained after feature transformation
    assert df_feats.isnull().sum().sum() == 0

def test_feature_columns_selector(sample_sales_data):
    fe = FeatureEngineer()
    df_feats = fe.fit_transform(sample_sales_data)
    feature_cols = fe.get_feature_columns(df_feats)
    
    assert "demand" not in feature_cols
    assert "date" not in feature_cols
    assert "profit_margin_pct" in feature_cols
