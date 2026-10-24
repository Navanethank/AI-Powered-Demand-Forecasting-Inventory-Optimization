from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.pipeline import ForecastingPipeline
from db.queries import fetch_products, fetch_stores

app = Flask(__name__)

# Initialize pipeline lazily / on startup
pipeline = ForecastingPipeline()

@app.route("/")
def index():
    products_df = fetch_products()
    stores_df = fetch_stores()
    products = products_df.to_dict(orient="records")
    stores = stores_df.to_dict(orient="records")
    eval_metrics = pipeline.bundle.get("evaluation_metrics", {})
    return render_template(
        "index.html",
        products=products,
        stores=stores,
        eval_metrics=eval_metrics
    )

@app.route("/api/products", methods=["GET"])
def get_products():
    products_df = fetch_products()
    return jsonify({"status": "success", "data": products_df.to_dict(orient="records")})

@app.route("/api/forecast", methods=["GET"])
def get_forecast():
    sku_id = request.args.get("sku_id", "SKU-101")
    store_id = request.args.get("store_id", "STR-01")
    horizon = int(request.args.get("horizon_days", 14))

    try:
        data = pipeline.forecast_sku(sku_id, store_id, horizon_days=horizon)
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/inventory-plan", methods=["POST"])
def get_inventory_plan():
    payload = request.get_json() or {}
    sku_id = payload.get("sku_id", "SKU-101")
    store_id = payload.get("store_id", "STR-01")
    current_stock = int(payload.get("current_stock", 50))
    lead_time_override = payload.get("lead_time_override")
    if lead_time_override is not None:
        lead_time_override = int(lead_time_override)
    service_level = float(payload.get("service_level", 0.95))

    try:
        plan = pipeline.get_inventory_recommendation(
            sku_id=sku_id,
            store_id=store_id,
            current_stock=current_stock,
            lead_time_override=lead_time_override,
            service_level=service_level
        )
        return jsonify({"status": "success", "data": plan})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/simulate", methods=["GET"])
def get_simulation():
    sku_id = request.args.get("sku_id", "SKU-101")
    store_id = request.args.get("store_id", "STR-01")

    try:
        results = pipeline.run_backtest_simulation(sku_id, store_id)
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    metrics = pipeline.bundle.get("evaluation_metrics", {})
    return jsonify({"status": "success", "metrics": metrics})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
