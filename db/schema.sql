-- Relational Schema for Retail Demand Forecasting & Inventory Optimization

CREATE TABLE IF NOT EXISTS products (
    sku_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_cost REAL NOT NULL,
    unit_price REAL NOT NULL,
    lead_time_days INTEGER NOT NULL DEFAULT 7,
    min_order_qty INTEGER NOT NULL DEFAULT 10
);

CREATE TABLE IF NOT EXISTS stores (
    store_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,
    store_tier TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    sku_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    units_sold INTEGER NOT NULL,
    is_promoted INTEGER NOT NULL DEFAULT 0,
    discount_pct REAL NOT NULL DEFAULT 0.0,
    stockout_flag INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (sku_id) REFERENCES products(sku_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    UNIQUE(date, sku_id, store_id)
);

CREATE INDEX IF NOT EXISTS idx_sales_date_sku ON daily_sales(date, sku_id);
CREATE INDEX IF NOT EXISTS idx_sales_sku_store ON daily_sales(sku_id, store_id);

CREATE TABLE IF NOT EXISTS inventory_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    sku_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    on_hand_units INTEGER NOT NULL,
    reorder_point INTEGER NOT NULL,
    safety_stock INTEGER NOT NULL,
    FOREIGN KEY (sku_id) REFERENCES products(sku_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);
