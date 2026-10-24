import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="OptiSupply AI: Demand Forecasting & Inventory Optimization")
    parser.add_argument("--train", action="store_true", help="Retrain LightGBM Quantile & XGBoost models")
    parser.add_argument("--seed", action="store_true", help="Re-seed SQLite database with retail transaction series")
    parser.add_argument("--port", type=int, default=5000, help="Port to run Flask server (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")

    args = parser.parse_args()

    if args.seed:
        print("[*] Re-seeding database...")
        from db.init_db import seed_database
        seed_database()

    if args.train:
        print("[*] Starting model training pipeline...")
        from src.models.trainer import ForecastModelTrainer
        trainer = ForecastModelTrainer()
        trainer.run_training_pipeline()

    print(f"\n=======================================================")
    print(f"  OptiSupply AI Dashboard Running at: http://{args.host}:{args.port}")
    print(f"=======================================================\n")
    from app.server import app
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
