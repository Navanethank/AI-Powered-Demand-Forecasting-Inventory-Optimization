import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "retail_inventory.db"
MODELS_DIR = BASE_DIR / "models_artifact"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Forecasting settings
QUANTILES = [0.10, 0.50, 0.90]
FORECAST_HORIZON_DAYS = 14
TARGET_COL = "demand"

# Feature configuration
LAG_DAYS = [7, 14, 21, 28]  # Note: avoiding lag 1-6 prevents data leakage with a 7-day lead-time!
ROLLING_WINDOWS = [7, 14, 30]

# Inventory default assumptions
DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_ANNUAL_HOLDING_RATE = 0.20  # 20% annual holding cost rate
DAYS_PER_YEAR = 365
