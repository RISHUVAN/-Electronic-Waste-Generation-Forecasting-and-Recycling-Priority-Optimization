"""Unit tests (plain asserts; run with `pytest -q tests` or `python tests/test_core.py`)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.analysis import gap as G, priority as PR
from src.data import loader
from src.models import forecast as F


def _cy(n_years=6, city="A", base=100.0, slope=10.0, start=2018):
    yrs = np.arange(start, start + n_years)
    return pd.DataFrame({"city": city, "state": "S", "region": "R", "city_category": "Tier-1", "year": yrs,
                         F.Y: base + slope * (yrs - start), "collection_efficiency_pct": 50.0, "recycling_rate_pct": 40.0,
                         "population": 1e6, "smart_bins": 10, "collection_vehicles": 5, "workers": 100,
                         "landfill_remaining_pct": 50.0, "municipal_score": 5.0})


def test_gap_formulas_and_missing_not_zero():
    d = _cy(2)
    d.loc[1, "recycling_rate_pct"] = np.nan
    g = G.compute_gap(d)
    assert np.isclose(g.collected_tonnes[0], 50.0) and np.isclose(g.collection_gap_pct[0], 50.0)
    assert np.isclose(g.recycled_tonnes_est[0], 20.0) and np.isclose(g.recycling_gap_pct[0], 80.0)
    assert np.isnan(g.recycling_gap_pct[1]) and not np.isnan(g.collection_gap_pct[1])
    alt = G.compute_gap(d, G.BASIS_GENERATED)
    assert np.isclose(alt.recycled_tonnes_est[0], 40.0)
    bad = _cy(1); bad["collection_efficiency_pct"] = 140
    assert np.isnan(G.compute_gap(bad).collection_gap_pct[0])


def test_forecast_requires_min_years_and_no_leakage():
    res, why = F.forecast_city(_cy(C.MIN_YEARS - 1), 3, 2026)
    assert res is None and "required" in why
    res, _ = F.forecast_city(_cy(8), 3, 2026)
    assert res["metrics"]["Linear"]["MAE"] < 1e-6 and res["metrics"]["Naive"]["MAE"] > 5   # perfect line; naive lags
    x = np.arange(8.0); y = 100 + 10 * x
    y_leak = y.copy(); y_leak[-1] = 1e9                 # corrupt only the held-out value
    ev = F.rolling_eval(x[:-1], y_leak[:-1])            # evaluation must not see it
    assert ev["Linear"]["MAE"] < 1e-6


def test_forecast_years_horizon_and_clipping():
    res, _ = F.forecast_city(_cy(6, slope=-60.0, base=300.0), 5, 2026)
    c = res["curve"]
    assert c.year.max() == 2031 and (c[["Linear", "Polynomial", "Naive"]] >= 0).all().all()
    assert "negative" in res["reasons"]["Linear"] or res["curve"]["Linear"].min() >= 0


def test_confidence_levels():
    res, _ = F.forecast_city(_cy(9), 1, 2026)
    assert res["confidence"]["Linear"] in ("Moderate", "Unreliable")
    short, _ = F.forecast_city(_cy(5), 1, 2026)
    assert short["confidence"]["Polynomial"] == "Unreliable" and "limited history" in short["reasons"]["Linear"]
    stale, _ = F.forecast_city(_cy(6, start=2012), 5, 2026)
    assert "extrapolates" in stale["reasons"]["Linear"]


def test_forecast_all_and_holdout():
    cy = pd.concat([_cy(6, "A"), _cy(3, "B"), _cy(7, "C", slope=5)])
    out = F.forecast_all(cy, 2, 2026)
    assert set(out["details"]) == {"A", "C"} and out["skipped"].city.tolist() == ["B"]
    assert out["future"].year.min() == 2027 and out["future"].predicted.min() >= 0
    h, s = F.cross_city_holdout(cy)
    assert len(h) == 2 and set(s.model) == {"Linear", "Polynomial", "Naive"}


def test_priority_ranking_missing_data_and_weights():
    inp = pd.DataFrame({"city": list("ABCD"), "state": "S", "projected_generation": [1., 2., 3., np.nan],
                        "unaccounted_quantity": [1., 2., 3., 4.], "gap_percentage": [10., 20., 30., 40.],
                        "infrastructure_deficiency": [10., 20., 30., 40.], "basis": "forecast", "confidence": "Low"})
    ranked, un = PR.rank_cities(inp, {"projected_generation": 1, "unaccounted_quantity": 1, "gap_percentage": 1, "infrastructure_deficiency": 1})
    assert ranked.city.tolist() == ["C", "B", "A"] and un.city.tolist() == ["D"] and "missing" in un.reason.iloc[0]
    assert np.isclose(ranked.priority_score.iloc[0], 100.0) and np.isclose(ranked.priority_score.iloc[-1], 0.0)
    only = PR.rank_cities(inp, {"projected_generation": 0, "unaccounted_quantity": 0, "gap_percentage": 5, "infrastructure_deficiency": 0})[0]
    assert only.city.iloc[0] == "C" and "gap percentage" in only.explanation.iloc[0]
    try:
        PR.rank_cities(inp, {c: 0 for c in PR.COLS}); assert False
    except ValueError:
        pass
    assert len(PR.sensitivity(inp, C.DEFAULT_WEIGHTS)) == 3


def test_infrastructure_index_direction():
    a, b = _cy(1, "A"), _cy(1, "B")
    b["smart_bins"], b["collection_vehicles"], b["workers"], b["municipal_score"], b["landfill_remaining_pct"] = 500, 90, 5000, 9, 90
    idx = PR.infrastructure_index(pd.concat([a, b])).set_index("city")["infrastructure_deficiency"]
    assert idx["A"] > idx["B"]


def test_future_rows_excluded_and_annualisation():
    raw = pd.DataFrame({"City": ["X"] * 3, "State": "S", "Region": "R", "City_Category": "Tier-1", "Year": [2025, 2026, 2026],
                        "Month": [1, 10, 12], "Population": 1000, "Waste_Type": "E-Waste", "Daily_Waste_Generation_Tons": [2.0, 4.0, 9.0],
                        "Collection_Efficiency_Percentage": 50.0, "Recycling_Rate": 40.0, "Latitude": 20.0, "Longitude": 78.0,
                        **{c: 1.0 for c in C.INFRA_COLUMNS}})
    e, fut, bad = loader.extract_ewaste(raw, pd.Timestamp("2026-10-18"))
    assert len(e) == 2 and len(fut) == 1 and bad == 0
    cy = loader.annualise(e, None, pd.Timestamp("2026-10-18"))
    assert np.isclose(cy[cy.year == 2026].ewaste_generated_tonnes.iloc[0], 4.0 * 365)
    assert np.isclose(cy.collection_efficiency_pct.iloc[0], 50.0) and np.isclose(cy.recycling_rate_pct.iloc[0], 40.0)
    assert bool(cy[cy.year == 2026].year_partial.iloc[0]) and not bool(cy[cy.year == 2025].year_partial.iloc[0])


def test_real_dataset_integration():
    if not C.MAIN_CSV.exists():
        return
    ds = loader.build_dataset(as_of="2026-10-18")
    r = ds["report"]
    assert r["raw_rows"] == 25203 and r["raw_cols"] == 42 and r["raw_cities"] == 130
    assert r["ewaste_rows_used"] + r["future_rows_excluded"] == r["ewaste_rows_total"] == 594
    assert not ds["cityyear"].duplicated(["city", "year"]).any()
    assert ds["cityyear"][["collection_efficiency_pct", "ewaste_generated_tonnes"]].notna().all().all()
    fc = F.forecast_all(ds["cityyear"], 3, 2026)
    assert len(fc["details"]) == r["cities_forecastable"] and (fc["future"].predicted >= 0).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
