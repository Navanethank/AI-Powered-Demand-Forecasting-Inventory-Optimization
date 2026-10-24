import numpy as np
import pandas as pd
from typing import Dict, Any

class InventoryBacktestSimulator:
    """
    Simulates inventory replenishment over a historical test horizon to compare:
    1. AI-Driven Stochastic Policy (LightGBM Quantile ROP)
    2. Naive Baseline Heuristic (Static average + fixed buffer)
    """

    def __init__(
        self,
        lead_time_days: int = 7,
        unit_cost: float = 40.0,
        unit_price: float = 100.0,
        annual_holding_rate: float = 0.20
    ):
        self.lead_time = lead_time_days
        self.unit_cost = unit_cost
        self.unit_price = unit_price
        self.daily_holding_cost = unit_cost * (annual_holding_rate / 365.0)
        self.stockout_penalty_per_unit = unit_price - unit_cost

    def run_simulation(
        self,
        daily_demands: np.ndarray,
        ai_forecast_p50: np.ndarray,
        ai_safety_stocks: np.ndarray
    ) -> Dict[str, Any]:
        """Runs comparative daily simulation between AI policy and Heuristic."""
        n_days = len(daily_demands)

        # Baseline Heuristic parameters: fixed average demand of first 30 days + 7 days buffer
        baseline_avg_demand = max(1.0, np.mean(daily_demands[:min(30, n_days)]))
        baseline_rop = baseline_avg_demand * self.lead_time + (baseline_avg_demand * 5)
        baseline_target_stock = baseline_rop * 1.5

        def simulate_policy(use_ai: bool):
            inventory = int(ai_safety_stocks[0] * 2) if use_ai else int(baseline_rop)
            pipeline_orders = []  # List of tuples: (arrival_day, qty)
            
            total_fulfilled = 0
            total_lost_sales = 0
            total_holding_cost = 0.0
            daily_inventory_log = []
            stockout_days = 0

            for t in range(n_days):
                # 1. Receive incoming shipments scheduled to arrive on day t
                arrived_qty = sum(qty for arr_day, qty in pipeline_orders if arr_day == t)
                inventory += arrived_qty
                pipeline_orders = [(arr_day, qty) for arr_day, qty in pipeline_orders if arr_day > t]

                # 2. Daily demand arrives
                demand = daily_demands[t]
                if inventory >= demand:
                    fulfilled = demand
                    inventory -= demand
                    lost = 0
                else:
                    fulfilled = inventory
                    lost = demand - inventory
                    inventory = 0
                    stockout_days += 1

                total_fulfilled += fulfilled
                total_lost_sales += lost
                total_holding_cost += inventory * self.daily_holding_cost
                daily_inventory_log.append(inventory)

                # 3. Check Reorder Trigger at end of day
                inventory_position = inventory + sum(qty for _, qty in pipeline_orders)

                if use_ai:
                    current_rop = (ai_forecast_p50[t] * self.lead_time) + ai_safety_stocks[t]
                    target_stock = current_rop + ai_safety_stocks[t]
                    if inventory_position <= current_rop:
                        order_qty = max(10, int(target_stock - inventory_position))
                        pipeline_orders.append((t + self.lead_time, order_qty))
                else:
                    if inventory_position <= baseline_rop:
                        order_qty = max(10, int(baseline_target_stock - inventory_position))
                        pipeline_orders.append((t + self.lead_time, order_qty))

            total_demand = sum(daily_demands)
            service_level = (total_fulfilled / total_demand) if total_demand > 0 else 1.0
            total_stockout_penalty = total_lost_sales * self.stockout_penalty_per_unit
            total_supply_chain_cost = total_holding_cost + total_stockout_penalty

            return {
                "total_demand": int(total_demand),
                "total_fulfilled": int(total_fulfilled),
                "total_lost_sales": int(total_lost_sales),
                "stockout_days": int(stockout_days),
                "service_level_pct": round(service_level * 100, 2),
                "avg_daily_inventory": round(float(np.mean(daily_inventory_log)), 1),
                "total_holding_cost": round(total_holding_cost, 2),
                "total_stockout_penalty": round(total_stockout_penalty, 2),
                "total_supply_chain_cost": round(total_supply_chain_cost, 2)
            }

        ai_results = simulate_policy(use_ai=True)
        baseline_results = simulate_policy(use_ai=False)

        cost_reduction_pct = 0.0
        if baseline_results["total_supply_chain_cost"] > 0:
            cost_reduction_pct = (
                (baseline_results["total_supply_chain_cost"] - ai_results["total_supply_chain_cost"])
                / baseline_results["total_supply_chain_cost"]
            ) * 100

        return {
            "simulation_days": n_days,
            "ai_policy": ai_results,
            "baseline_heuristic": baseline_results,
            "cost_reduction_pct": round(cost_reduction_pct, 2)
        }
