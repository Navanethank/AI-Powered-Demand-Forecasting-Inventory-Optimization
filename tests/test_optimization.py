import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.optimization.inventory import InventoryOptimizer

def test_safety_stock_scaling():
    opt = InventoryOptimizer()
    
    # Higher service level must require higher safety stock
    ss_90 = opt.calculate_safety_stock(lead_time_days=7, daily_demand_std=5.0, service_level=0.90)
    ss_95 = opt.calculate_safety_stock(lead_time_days=7, daily_demand_std=5.0, service_level=0.95)
    ss_99 = opt.calculate_safety_stock(lead_time_days=7, daily_demand_std=5.0, service_level=0.99)

    assert ss_90 < ss_95 < ss_99
    assert ss_90 > 0

def test_reorder_point_logic():
    opt = InventoryOptimizer()
    lead_time = 5
    avg_demand = 20.0
    safety_stock = 15.0

    rop = opt.calculate_reorder_point(
        forecast_daily_demand=avg_demand,
        lead_time_days=lead_time,
        safety_stock=safety_stock
    )
    # ROP = (20 * 5) + 15 = 115
    assert rop == 115.0

def test_newsvendor_critical_ratio():
    opt = InventoryOptimizer()
    nv = opt.calculate_newsvendor_policy(
        unit_price=100.0,
        unit_cost=40.0,
        demand_forecast_p10=50.0,
        demand_forecast_p50=100.0,
        demand_forecast_p90=150.0
    )
    # Underage cost Cu = 100 - 40 = 60
    assert nv["underage_cost_cu"] == 60.0
    assert 0.50 <= nv["critical_ratio"] <= 1.0
    assert nv["optimal_order_quantity"] >= 100
