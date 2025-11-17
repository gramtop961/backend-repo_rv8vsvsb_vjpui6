import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

from database import db, create_document, get_documents
from schemas import TruffleObservation, WTIIndex, QualityStandard, SeasonalityWindow, WeeklyReport

app = FastAPI(title="World Truffle Index (WTI)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRUFFLE_TYPES = ["white", "black", "summer", "uncinatum", "bianchetto", "others"]
REGIONS = ["italy", "france", "spain", "australia", "us", "middle_east"]

# ---------- Utility functions ----------

def normalize_quality(obs: TruffleObservation, weights: Dict[str, float]) -> float:
    # Map inputs to 0-100 scale
    import math
    size_score = min(obs.size_mm / 60.0, 1.0) * 100  # 60mm ~ top
    aroma = (obs.aroma_score / 10.0) * 100
    # freshness: exponential decay per day, half-life ~ 3 days
    freshness = max(0.0, 100 * math.exp(-obs.freshness_days * 0.231))
    color = (obs.color_score / 10.0) * 100
    texture = (obs.texture_score / 10.0) * 100
    origin = 100.0 if obs.origin_certified else 60.0
    handling = (obs.handling_score / 10.0) * 100

    scores = {
        "size_mm": size_score,
        "aroma_score": aroma,
        "freshness_days": freshness,
        "color_score": color,
        "texture_score": texture,
        "origin_certified": origin,
        "handling_score": handling,
    }

    total = 0.0
    for k, w in weights.items():
        total += scores[k] * w
    return total


def weekly_key(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def aggregate_index(week: str) -> WTIIndex:
    # Load quality weights (latest standard or default)
    qs_docs = get_documents("qualitystandard", {}, limit=1)
    if qs_docs:
        weights = qs_docs[-1].get("weights", {})
    else:
        weights = QualityStandard().weights

    # Fetch observations for week
    obs = get_documents("truffleobservation", {"week": week})

    # Avoid division by zero
    if not obs:
        return WTIIndex(
            week=week,
            global_index=0.0,
            by_type={},
            by_region={},
            by_type_region={},
            supply_signal={},
            demand_signal={},
            volatility={},
            forecast_next_week={},
        )

    # Weighted price index by quality and quantity
    import statistics

    by_type: Dict[str, List[float]] = {t: [] for t in TRUFFLE_TYPES}
    by_region: Dict[str, List[float]] = {r: [] for r in REGIONS}
    by_tr: Dict[str, Dict[str, List[float]]] = {t: {r: [] for r in REGIONS} for t in TRUFFLE_TYPES}

    supply_signal: Dict[str, float] = {t: 50.0 for t in TRUFFLE_TYPES}
    demand_signal: Dict[str, float] = {t: 50.0 for t in TRUFFLE_TYPES}
    volatility: Dict[str, float] = {t: 50.0 for t in TRUFFLE_TYPES}

    weighted_prices: List[float] = []

    for d in obs:
        try:
            o = TruffleObservation(**{k: d[k] for k in d if k != "_id"})
        except Exception:
            # Skip invalid
            continue
        q = normalize_quality(o, weights)  # 0-100
        quality_weight = max(q / 100.0, 0.01)
        qty_weight = max(float(o.quantity_kg), 0.01)
        w = quality_weight * qty_weight
        weighted_prices.append(o.price_per_kg * w)

        by_type[o.truffle_type].append(o.price_per_kg)
        by_region[o.region].append(o.price_per_kg)
        by_tr[o.truffle_type][o.region].append(o.price_per_kg)

        # Market signals: map qualitative supply/demand to 0-100
        map_lv = {"very_low": 20, "low": 35, "balanced": 50, "high": 65, "very_high": 80}
        supply_signal[o.truffle_type] = (supply_signal[o.truffle_type] + map_lv[o.supply_level]) / 2.0
        demand_signal[o.truffle_type] = (demand_signal[o.truffle_type] + map_lv[o.demand_level]) / 2.0
        volatility[o.truffle_type] = (volatility[o.truffle_type] + (o.volatility_score * 10)) / 2.0

    # Compute medians to reduce outliers
    def median_or0(arr: List[float]) -> float:
        return float(statistics.median(arr)) if arr else 0.0

    idx_by_type = {t: median_or0(by_type[t]) for t in TRUFFLE_TYPES}
    idx_by_region = {r: median_or0(by_region[r]) for r in REGIONS}
    idx_by_tr = {t: {r: median_or0(by_tr[t][r]) for r in REGIONS} for t in TRUFFLE_TYPES}

    # Global index: median of all weighted prices scaled back to price level
    global_index = median_or0(weighted_prices) if weighted_prices else 0.0

    # Forecast: naive momentum from previous week medians
    from datetime import timedelta
    y, wk = week.split("-W")
    y = int(y)
    wk = int(wk)
    prev_week = f"{y}-W{wk-1:02d}" if wk > 1 else f"{y-1}-W52"

    prev_obs = get_documents("truffleobservation", {"week": prev_week})
    prev_prices_by_type: Dict[str, List[float]] = {t: [] for t in TRUFFLE_TYPES}
    for d in prev_obs:
        try:
            o = TruffleObservation(**{k: d[k] for k in d if k != "_id"})
        except Exception:
            continue
        prev_prices_by_type[o.truffle_type].append(o.price_per_kg)

    forecast_next_week: Dict[str, float] = {}
    for t in TRUFFLE_TYPES:
        prev_med = median_or0(prev_prices_by_type[t])
        cur_med = idx_by_type[t]
        if prev_med > 0:
            pct = ((cur_med - prev_med) / prev_med) * 100.0
        else:
            pct = 0.0
        # Momentum damped by volatility (higher vol -> lower confidence)
        vol = max(1.0, volatility[t])
        forecast_next_week[t] = round(pct * (100.0 / (100.0 + vol)), 2)

    return WTIIndex(
        week=week,
        global_index=round(global_index, 2),
        by_type={k: round(v, 2) for k, v in idx_by_type.items()},
        by_region={k: round(v, 2) for k, v in idx_by_region.items()},
        by_type_region={t: {r: round(v, 2) for r, v in idx_by_tr[t].items()} for t in TRUFFLE_TYPES},
        supply_signal={k: round(v, 1) for k, v in supply_signal.items()},
        demand_signal={k: round(v, 1) for k, v in demand_signal.items()},
        volatility={k: round(v, 1) for k, v in volatility.items()},
        forecast_next_week=forecast_next_week,
    )

# ---------- API Models ----------

class ObservationIn(TruffleObservation):
    pass

class ObservationOut(BaseModel):
    id: str

# ---------- Routes ----------

@app.get("/")
def root():
    return {"message": "World Truffle Index Backend ready"}

@app.get("/test")
def test_database():
    status = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set" if not os.getenv("DATABASE_URL") else "✅ Set",
        "database_name": "❌ Not Set" if not os.getenv("DATABASE_NAME") else os.getenv("DATABASE_NAME"),
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            status["database"] = "✅ Connected"
            status["connection_status"] = "Connected"
            status["collections"] = db.list_collection_names()
    except Exception as e:
        status["database"] = f"⚠️ {str(e)[:80]}"
    return status

@app.post("/observations", response_model=ObservationOut)
def submit_observation(obs: ObservationIn):
    # Store raw observation
    obs_dict = obs.model_dump()
    oid = create_document("truffleobservation", obs_dict)
    return {"id": oid}

@app.get("/index/{week}", response_model=WTIIndex)
def get_week_index(week: str):
    try:
        index = aggregate_index(week)
        # Store or upsert snapshot
        existing = get_documents("wtiindex", {"week": week}, limit=1)
        if not existing:
            create_document("wtiindex", index.model_dump())
        return index
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/seasonality", response_model=List[SeasonalityWindow])
def get_seasonality():
    # Static authoritative windows; could be stored in DB for edits
    return [
        SeasonalityWindow(truffle_type="white", start_month=10, end_month=12, peak_months=[11], regions=["italy", "france"]),
        SeasonalityWindow(truffle_type="black", start_month=12, end_month=3, peak_months=[1,2], regions=["france", "spain", "italy", "australia"]),
        SeasonalityWindow(truffle_type="summer", start_month=5, end_month=8, peak_months=[6,7], regions=["italy", "france", "spain", "us"]),
        SeasonalityWindow(truffle_type="uncinatum", start_month=9, end_month=12, peak_months=[10,11], regions=["france", "italy", "spain", "us"]),
        SeasonalityWindow(truffle_type="bianchetto", start_month=1, end_month=4, peak_months=[2,3], regions=["italy"]) ,
    ]

@app.get("/quality-standard", response_model=QualityStandard)
def get_quality_standard():
    docs = get_documents("qualitystandard", {}, limit=1)
    if docs:
        d = docs[-1]
        return QualityStandard(version=d.get("version", "1.0"), weights=d.get("weights", QualityStandard().weights), notes=d.get("notes"))
    return QualityStandard()

@app.post("/quality-standard", response_model=QualityStandard)
def set_quality_standard(std: QualityStandard):
    create_document("qualitystandard", std.model_dump())
    return std

@app.get("/weekly-report/{week}", response_model=WeeklyReport)
def get_weekly_report(week: str):
    # Auto-generate from index as a sample output
    idx = aggregate_index(week)
    highlights = [
        f"Global price index: {idx.global_index}",
        f"Top priced type: {max(idx.by_type, key=idx.by_type.get)}",
        f"Most active region: {max(idx.by_region, key=idx.by_region.get)}",
    ] if idx.by_type and idx.by_region else ["Insufficient data for the week"]

    risks = [
        f"Volatility elevated in {max(idx.volatility, key=idx.volatility.get)}",
    ] if idx.volatility else []

    commentary = "WTI powered by insights from House of Tartufo – The Italian Truffle Hub."
    return WeeklyReport(week=week, highlights=highlights, risks=risks, commentary=commentary)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
