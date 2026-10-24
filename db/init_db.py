import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import DB_PATH, BASE_DIR

def seed_database():
    print(f"[*] Initializing database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Read and execute schema
    schema_file = BASE_DIR / "db" / "schema.sql"
    with open(schema_file, "r") as f:
        cursor.executescript(f.read())

    # Check if data already exists
    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] > 0:
        print("[!] Products already exist in database. Skipping generation.")
        conn.close()
        return

    # Seed Products
    products = [
        ("SKU-101", "Noise-Canceling Wireless Headphones", "Electronics", 45.0, 119.99, 7, 20),
        ("SKU-102", "Ergonomic Mechanical Keyboard", "Electronics", 32.0, 89.50, 5, 15),
        ("SKU-103", "4K Ultra-HD Smart Monitor 27-inch", "Electronics", 160.0, 299.99, 14, 10),
        ("SKU-201", "Organic Peruvian Coffee Beans 1kg", "Grocery", 8.50, 22.00, 3, 30),
        ("SKU-202", "Cold-Pressed Extra Virgin Olive Oil", "Grocery", 6.20, 16.50, 4, 25),
        ("SKU-301", "Performance Dri-Fit Training Hoodie", "Apparel", 14.0, 48.00, 10, 20),
        ("SKU-302", "Water-Resistant Commuter Backpack", "Apparel", 18.5, 59.99, 8, 15),
        ("SKU-401", "Aromatherapy Ceramic Essential Diffuser", "Home", 11.0, 34.50, 6, 25),
        ("SKU-402", "Memory Foam Orthopedic Pillow", "Home", 15.0, 42.00, 7, 20),
        ("SKU-501", "Stainless Steel Insulated Tumbler 32oz", "Lifestyle", 5.50, 24.99, 5, 40),
    ]
    cursor.executemany(
        "INSERT INTO products (sku_id, name, category, unit_cost, unit_price, lead_time_days, min_order_qty) VALUES (?, ?, ?, ?, ?, ?, ?)",
        products
    )

    # Seed Stores
    stores = [
        ("STR-01", "Downtown Flagship", "Northeast", "Tier-1"),
        ("STR-02", "Metro Suburbs Mall", "Midwest", "Tier-2"),
        ("STR-03", "Airport Express", "South", "Tier-3"),
    ]
    cursor.executemany(
        "INSERT INTO stores (store_id, name, region, store_tier) VALUES (?, ?, ?, ?)",
        stores
    )

    conn.commit()
    print(f"[+] Inserted {len(products)} products and {len(stores)} stores.")

    # Generate 2 Years of Daily Time Series Data
    print("[*] Generating 730 days of multi-SKU retail sales with trend, seasonality, and promotional lifts...")
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    n_days = 730  # 2 years

    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    sales_records = []

    # Baseline daily sales multipliers per store tier
    store_multipliers = {"STR-01": 1.4, "STR-02": 1.0, "STR-03": 0.7}
    
    # Base daily volume per SKU
    sku_base_volume = {
        "SKU-101": 25, "SKU-102": 18, "SKU-103": 8,
        "SKU-201": 45, "SKU-202": 35, "SKU-301": 22,
        "SKU-302": 15, "SKU-401": 19, "SKU-402": 24,
        "SKU-501": 38
    }

    for d_idx, cur_date in enumerate(dates):
        date_str = cur_date.strftime("%Y-%m-%d")
        day_of_week = cur_date.weekday()
        day_of_year = cur_date.timetuple().tm_yday
        month = cur_date.month

        # Day-of-week seasonality (Friday=4, Saturday=5, Sunday=6 peak)
        dow_factor = 1.0 + 0.35 * (day_of_week in [4, 5]) + 0.15 * (day_of_week == 6) - 0.15 * (day_of_week in [0, 1])

        # Annual seasonality (Holiday season Q4 boost)
        holiday_factor = 1.0
        if month == 11 and cur_date.day >= 20:  # Black Friday week
            holiday_factor = 2.1
        elif month == 12 and cur_date.day <= 24:  # Christmas rush
            holiday_factor = 1.85
        elif month in [6, 7]:  # Summer mini-surge
            holiday_factor = 1.15

        # Year trend (gentle 8% annual growth)
        trend_factor = 1.0 + (d_idx / 730.0) * 0.12

        for sku_id, base_vol in sku_base_volume.items():
            for store_id, s_mult in store_multipliers.items():
                # Random promotional campaign (~8% probability)
                is_promoted = 1 if np.random.rand() < 0.08 else 0
                discount_pct = float(np.random.choice([0.10, 0.15, 0.20, 0.25])) if is_promoted else 0.0
                promo_lift = 1.0 + (discount_pct * 2.8) if is_promoted else 1.0

                # Expected demand
                expected_demand = base_vol * s_mult * dow_factor * holiday_factor * trend_factor * promo_lift
                
                # Sample Poisson / Negative Binomial noise
                actual_demand = max(0, int(np.random.poisson(lam=expected_demand)))

                # Stockout simulation (~2% random supplier delay or inventory depletion)
                stockout_flag = 1 if (np.random.rand() < 0.02 and actual_demand > 5) else 0
                units_sold = int(actual_demand * 0.4) if stockout_flag else actual_demand

                sales_records.append((
                    date_str, sku_id, store_id, units_sold, is_promoted, discount_pct, stockout_flag
                ))

    # Bulk insert sales
    cursor.executemany(
        """INSERT INTO daily_sales (date, sku_id, store_id, units_sold, is_promoted, discount_pct, stockout_flag)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        sales_records
    )
    conn.commit()
    print(f"[+] Successfully seeded {len(sales_records):,} sales records into SQLite.")
    conn.close()

if __name__ == "__main__":
    seed_database()
