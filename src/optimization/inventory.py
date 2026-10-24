import numpy as np
from scipy import stats
from typing import Dict, Any
from src.config import DEFAULT_SERVICE_LEVEL, DEFAULT_ANNUAL_HOLDING_RATE, DAYS_PER_YEAR

class InventoryOptimizer:
    """
    Translates probabilistic demand forecasts into cost-optimal inventory replenishment policies:
    Safety Stock (SS), Reorder Point (ROP), and Newsvendor Cost-Minimization.
    """

    def __init__(self, annual_holding_rate: float = DEFAULT_ANNUAL_HOLDING_RATE):
        self.annual_holding_rate = annual_holding_rate

    def calculate_safety_stock(
        self,
        lead_time_days: int,
        daily_demand_std: float,
        service_level: float = DEFAULT_SERVICE_LEVEL
    ) -> float:
        """
        Calculates stochastic safety stock under demand uncertainty:
        SS = Z * std_dev_daily * sqrt(lead_time)
        """
        # Z-score for target cycle service level (e.g., Z = 1.645 for 95%)
        z_score = stats.norm.ppf(service_level)
        ss = z_score * daily_demand_std * np.sqrt(lead_time_days)
        return float(max(0, np.ceil(ss)))

    def calculate_reorder_point(
        self,
        forecast_daily_demand: float,
        lead_time_days: int,
        safety_stock: float
    ) -> float:
        """
        Calculates the inventory trigger point (ROP):
        ROP = (Expected Daily Demand * Lead Time) + Safety Stock
        """
        rop = (forecast_daily_demand * lead_time_days) + safety_stock
        return float(np.ceil(rop))

    def calculate_newsvendor_policy(
        self,
        unit_price: float,
        unit_cost: float,
        demand_forecast_p10: float,
        demand_forecast_p50: float,
        demand_forecast_p90: float,
        holding_days: int = 14
    ) -> Dict[str, float]:
        """
        Calculates cost-minimizing order quantity based on the Newsvendor problem.
        Underage cost (Cu) = Profit Margin lost on lost sale
        Overage cost (Co) = Capital holding cost over period
        Critical Fractile = Cu / (Cu + Co)
        """
        cu = unit_price - unit_cost  # Gross profit margin lost per unit of stockout
        # Daily holding cost based on annual capital carrying rate
        daily_holding_cost = unit_cost * (self.annual_holding_rate / DAYS_PER_YEAR)
        co = max(0.05, daily_holding_cost * holding_days)

        # Critical fractile (target non-stockout probability)
        critical_ratio = cu / (cu + co)
        critical_ratio = float(np.clip(critical_ratio, 0.50, 0.99))

        # Interpolate optimal order quantity across quantile forecasts
        # If critical ratio is ~0.90, order near P90; if ~0.50, order near P50
        if critical_ratio >= 0.50:
            weight = (critical_ratio - 0.50) / 0.40  # scaled between 0 and 1
            optimal_qty = demand_forecast_p50 + weight * (demand_forecast_p90 - demand_forecast_p50)
        else:
            weight = (0.50 - critical_ratio) / 0.40
            optimal_qty = demand_forecast_p50 - weight * (demand_forecast_p50 - demand_forecast_p10)

        return {
            "underage_cost_cu": round(cu, 2),
            "overage_cost_co": round(co, 3),
            "critical_ratio": round(critical_ratio, 4),
            "optimal_order_quantity": int(np.ceil(optimal_qty))
        }

    def generate_replenishment_recommendation(
        self,
        sku_id: str,
        current_stock: int,
        lead_time_days: int,
        min_order_qty: int,
        unit_cost: float,
        unit_price: float,
        forecast_p10: float,
        forecast_p50: float,
        forecast_p90: float,
        historical_daily_std: float,
        service_level: float = DEFAULT_SERVICE_LEVEL
    ) -> Dict[str, Any]:
        """
        Full decision policy: Evaluates stock status, calculates ROP, and triggers order recommendations.
        """
        safety_stock = self.calculate_safety_stock(
            lead_time_days=lead_time_days,
            daily_demand_std=historical_daily_std,
            service_level=service_level
        )

        rop = self.calculate_reorder_point(
            forecast_daily_demand=forecast_p50,
            lead_time_days=lead_time_days,
            safety_stock=safety_stock
        )

        nv_policy = self.calculate_newsvendor_policy(
            unit_price=unit_price,
            unit_cost=unit_cost,
            demand_forecast_p10=forecast_p10 * lead_time_days,
            demand_forecast_p50=forecast_p50 * lead_time_days,
            demand_forecast_p90=forecast_p90 * lead_time_days,
            holding_days=lead_time_days
        )

        # Order Trigger Logic:
        # If current on-hand stock <= ROP, order up to Target Inventory (ROP + Safety Stock)
        needs_reorder = bool(current_stock <= rop)
        suggested_order_qty = 0

        if needs_reorder:
            deficit = (rop + safety_stock) - current_stock
            suggested_order_qty = max(min_order_qty, int(np.ceil(deficit)))

        # Risk Analysis
        days_of_supply = round(current_stock / (forecast_p50 + 1e-6), 1)
        stockout_risk = "HIGH" if current_stock < safety_stock else ("MEDIUM" if needs_reorder else "LOW")

        return {
            "sku_id": sku_id,
            "current_stock": current_stock,
            "days_of_supply": days_of_supply,
            "lead_time_days": lead_time_days,
            "safety_stock": int(safety_stock),
            "reorder_point": int(rop),
            "needs_reorder": needs_reorder,
            "suggested_order_qty": suggested_order_qty,
            "stockout_risk": stockout_risk,
            "estimated_order_cost": round(suggested_order_qty * unit_cost, 2),
            "newsvendor_metrics": nv_policy
        }
