// OptiSupply AI - Frontend Controller

let forecastChart = null;
let benchmarkChart = null;

// DOM Elements
const selectSku = document.getElementById("select-sku");
const selectStore = document.getElementById("select-store");
const inputStock = document.getElementById("input-current-stock");
const labelStock = document.getElementById("label-current-stock");
const inputLeadTime = document.getElementById("input-lead-time");
const labelLeadTime = document.getElementById("label-lead-time");
const inputServiceLevel = document.getElementById("input-service-level");
const labelServiceLevel = document.getElementById("label-service-level");
const btnRecalculate = document.getElementById("btn-recalculate");

// Decision Elements
const decisionBanner = document.getElementById("decision-banner");
const riskBadge = document.getElementById("risk-badge");
const decisionTitle = document.getElementById("decision-title");
const decisionDesc = document.getElementById("decision-desc");
const metricSafetyStock = document.getElementById("metric-safety-stock");
const metricRop = document.getElementById("metric-rop");
const metricDos = document.getElementById("metric-dos");

// Newsvendor Elements
const nvCu = document.getElementById("nv-cu");
const nvCo = document.getElementById("nv-co");
const nvCr = document.getElementById("nv-cr");
const nvOrderCost = document.getElementById("nv-order-cost");
const backtestSavingLabel = document.getElementById("backtest-saving-label");
const kpiCostReduction = document.getElementById("kpi-cost-reduction");
const kpiServiceLevel = document.getElementById("kpi-service-level");

// Event Listeners for Live Sliders
inputStock.addEventListener("input", (e) => {
    labelStock.textContent = `${e.target.value} units`;
});

inputLeadTime.addEventListener("input", (e) => {
    labelLeadTime.textContent = `${e.target.value} days`;
});

inputServiceLevel.addEventListener("input", (e) => {
    labelServiceLevel.textContent = `${e.target.value}%`;
    kpiServiceLevel.textContent = `${e.target.value}.0%`;
});

// Update default lead time when SKU changes
selectSku.addEventListener("change", () => {
    const selected = selectSku.options[selectSku.selectedIndex];
    const defaultLead = selected.getAttribute("data-lead") || 7;
    inputLeadTime.value = defaultLead;
    labelLeadTime.textContent = `${defaultLead} days`;
    refreshAll();
});

selectStore.addEventListener("change", refreshAll);
btnRecalculate.addEventListener("click", refreshAll);

async function refreshAll() {
    const skuId = selectSku.value;
    const storeId = selectStore.value;
    const currentStock = parseInt(inputStock.value);
    const leadTime = parseInt(inputLeadTime.value);
    const serviceLevel = parseInt(inputServiceLevel.value) / 100.0;

    await Promise.all([
        updateForecastChart(skuId, storeId),
        updateInventoryPlan(skuId, storeId, currentStock, leadTime, serviceLevel),
        updateSimulation(skuId, storeId)
    ]);
}

async function updateForecastChart(skuId, storeId) {
    try {
        const res = await fetch(`/api/forecast?sku_id=${skuId}&store_id=${storeId}&horizon_days=14`);
        const json = await res.json();
        if (json.status !== "success") return;
        const d = json.data;

        const histDates = d.historical_dates;
        const foreDates = d.forecast_dates;
        const allLabels = [...histDates, ...foreDates];

        // Format datasets
        const histData = [...d.historical_demand, ...new Array(foreDates.length).fill(null)];
        // Connect history to forecast
        const lastHistVal = d.historical_demand[d.historical_demand.length - 1];
        
        const p10Data = [...new Array(histDates.length - 1).fill(null), lastHistVal, ...d.p10];
        const p50Data = [...new Array(histDates.length - 1).fill(null), lastHistVal, ...d.p50];
        const p90Data = [...new Array(histDates.length - 1).fill(null), lastHistVal, ...d.p90];

        const ctx = document.getElementById("forecastChart").getContext("2d");
        if (forecastChart) forecastChart.destroy();

        forecastChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: allLabels,
                datasets: [
                    {
                        label: "Historical Actuals",
                        data: histData,
                        borderColor: "#94a3b8",
                        backgroundColor: "rgba(148, 163, 184, 0.1)",
                        borderWidth: 2,
                        pointRadius: 1,
                        tension: 0.1
                    },
                    {
                        label: "P50 Expected Forecast",
                        data: p50Data,
                        borderColor: "#3b82f6",
                        borderWidth: 2.5,
                        pointRadius: 2,
                        tension: 0.2
                    },
                    {
                        label: "P90 Peak Forecast",
                        data: p90Data,
                        borderColor: "rgba(168, 85, 247, 0.4)",
                        borderDash: [4, 4],
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: "+1",
                        backgroundColor: "rgba(168, 85, 247, 0.12)"
                    },
                    {
                        label: "P10 Floor Forecast",
                        data: p10Data,
                        borderColor: "rgba(168, 85, 247, 0.4)",
                        borderDash: [4, 4],
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { mode: "index", intersect: false }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(75, 85, 99, 0.15)" },
                        ticks: { color: "#64748b", maxTicksLimit: 10, font: { size: 10 } }
                    },
                    y: {
                        grid: { color: "rgba(75, 85, 99, 0.15)" },
                        ticks: { color: "#64748b", font: { size: 10 } }
                    }
                }
            }
        });

    } catch (err) {
        console.error("Forecast error:", err);
    }
}

async function updateInventoryPlan(skuId, storeId, currentStock, leadTime, serviceLevel) {
    try {
        const res = await fetch("/api/inventory-plan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sku_id: skuId,
                store_id: storeId,
                current_stock: currentStock,
                lead_time_override: leadTime,
                service_level: serviceLevel
            })
        });
        const json = await res.json();
        if (json.status !== "success") return;
        const d = json.data;

        metricSafetyStock.textContent = d.safety_stock;
        metricRop.textContent = d.reorder_point;
        metricDos.textContent = `${d.days_of_supply} d`;

        // Update Decision Banner
        if (d.stockout_risk === "HIGH") {
            decisionBanner.className = "glass-card p-5 rounded-2xl border-l-4 border-l-rose-500 flex flex-col md:flex-row md:items-center justify-between gap-4";
            riskBadge.className = "px-2.5 py-0.5 rounded text-xs font-semibold bg-rose-500/20 text-rose-400";
            riskBadge.textContent = "CRITICAL RISK";
            decisionTitle.textContent = "Immediate Stockout Hazard!";
            decisionDesc.innerHTML = `Stock (${d.current_stock}) is below Safety Stock (${d.safety_stock}). Order <strong class="text-rose-400">${d.suggested_order_qty} units</strong> immediately.`;
        } else if (d.needs_reorder) {
            decisionBanner.className = "glass-card p-5 rounded-2xl border-l-4 border-l-amber-500 flex flex-col md:flex-row md:items-center justify-between gap-4";
            riskBadge.className = "px-2.5 py-0.5 rounded text-xs font-semibold bg-amber-500/20 text-amber-300";
            riskBadge.textContent = "REORDER RECOMMENDED";
            decisionTitle.textContent = "Replenishment Order Advised";
            decisionDesc.innerHTML = `Current stock (${d.current_stock}) has pierced Reorder Point (${d.reorder_point}). Order <strong class="text-amber-300">${d.suggested_order_qty} units</strong>.`;
        } else {
            decisionBanner.className = "glass-card p-5 rounded-2xl border-l-4 border-l-emerald-500 flex flex-col md:flex-row md:items-center justify-between gap-4";
            riskBadge.className = "px-2.5 py-0.5 rounded text-xs font-semibold bg-emerald-500/20 text-emerald-400";
            riskBadge.textContent = "OPTIMAL";
            decisionTitle.textContent = "Inventory Position Healthy";
            decisionDesc.innerHTML = `Stock (${d.current_stock}) safely exceeds Reorder Point (${d.reorder_point}). No procurement needed right now.`;
        }

        // Update Newsvendor
        nvCu.textContent = `$${d.newsvendor_metrics.underage_cost_cu.toFixed(2)} / unit`;
        nvCo.textContent = `$${d.newsvendor_metrics.overage_cost_co.toFixed(3)} / unit`;
        nvCr.textContent = d.newsvendor_metrics.critical_ratio.toFixed(3);
        nvOrderCost.textContent = `$${d.estimated_order_cost.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;

    } catch (err) {
        console.error("Inventory plan error:", err);
    }
}

async function updateSimulation(skuId, storeId) {
    try {
        const res = await fetch(`/api/simulate?sku_id=${skuId}&store_id=${storeId}`);
        const json = await res.json();
        if (json.status !== "success") return;
        const d = json.data;

        backtestSavingLabel.textContent = `+${d.cost_reduction_pct}% Net Cost Savings`;
        kpiCostReduction.textContent = `-${d.cost_reduction_pct}%`;

        const ctx = document.getElementById("benchmarkChart").getContext("2d");
        if (benchmarkChart) benchmarkChart.destroy();

        benchmarkChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Holding Cost ($)", "Stockout Penalties ($)", "Total Cost ($)"],
                datasets: [
                    {
                        label: "Baseline Static Policy",
                        data: [
                            d.baseline_heuristic.total_holding_cost,
                            d.baseline_heuristic.total_stockout_penalty,
                            d.baseline_heuristic.total_supply_chain_cost
                        ],
                        backgroundColor: "#64748b"
                    },
                    {
                        label: "AI Stochastic Policy",
                        data: [
                            d.ai_policy.total_holding_cost,
                            d.ai_policy.total_stockout_penalty,
                            d.ai_policy.total_supply_chain_cost
                        ],
                        backgroundColor: "#3b82f6"
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: "#94a3b8", font: { size: 10 } } }
                },
                scales: {
                    x: { ticks: { color: "#64748b", font: { size: 10 } } },
                    y: { ticks: { color: "#64748b", font: { size: 10 } } }
                }
            }
        });

    } catch (err) {
        console.error("Simulation error:", err);
    }
}

// Initial bootstrap on load
document.addEventListener("DOMContentLoaded", refreshAll);
