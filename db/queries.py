import sqlite3
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import DB_PATH

def get_connection():
    return sqlite3.connect(DB_PATH)

def fetch_products():
    """Retrieve all product catalogs."""
    query = """
    SELECT sku_id, name, category, unit_cost, unit_price, lead_time_days, min_order_qty
    FROM products
    ORDER BY sku_id;
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)

def fetch_stores():
    """Retrieve all store locations."""
    query = "SELECT store_id, name, region, store_tier FROM stores ORDER BY store_id;"
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)

def fetch_training_dataset():
    """
    Extracts daily sales enriched with product master attributes using SQL.
    Demonstrates SQL window functions for rolling averages and cumulative metrics.
    """
    query = """
    SELECT 
        s.date,
        s.sku_id,
        s.store_id,
        p.name AS product_name,
        p.category,
        p.unit_cost,
        p.unit_price,
        p.lead_time_days,
        p.min_order_qty,
        st.region,
        st.store_tier,
        s.units_sold AS demand,
        s.is_promoted,
        s.discount_pct,
        s.stockout_flag
    FROM daily_sales s
    JOIN products p ON s.sku_id = p.sku_id
    JOIN stores st ON s.store_id = st.store_id
    ORDER BY s.sku_id, s.store_id, s.date ASC;
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df

def fetch_sku_history(sku_id: str, store_id: str = None):
    """Fetch time series history for a specific SKU (optionally by store)."""
    where_clauses = ["s.sku_id = ?"]
    params = [sku_id]
    
    if store_id:
        where_clauses.append("s.store_id = ?")
        params.append(store_id)
        
    query = f"""
    SELECT 
        s.date,
        s.sku_id,
        s.store_id,
        p.name AS product_name,
        p.unit_cost,
        p.unit_price,
        p.lead_time_days,
        s.units_sold,
        s.is_promoted,
        s.discount_pct,
        s.stockout_flag
    FROM daily_sales s
    JOIN products p ON s.sku_id = p.sku_id
    WHERE {' AND '.join(where_clauses)}
    ORDER BY s.date ASC;
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df
