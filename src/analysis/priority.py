"""Transparent weighted city priority score (decision-support only, not scientifically validated)."""
import numpy as np
import pandas as pd
from src.analysis import gap as G

Y = G.Y
COLS = ["projected_generation", "unaccounted_quantity", "gap_percentage", "infrastructure_deficiency"]
LABELS = {"projected_generation": "Projected generation", "unaccounted_quantity": "Unaccounted-for quantity",
          "gap_percentage": "Gap percentage", "infrastructure_deficiency": "Infrastructure deficiency"}


def minmax(s: pd.Series) -> pd.Series:
    r = s.max() - s.min()
    return (s - s.min()) / r if r and r > 0 else s * 0.0


def infrastructure_index(cy: pd.DataFrame) -> pd.DataFrame:
    """0-100 deficiency index from each city's latest observation. Adequacy of five city-level fields is the
    percentile rank across cities (robust to heavy tails); deficiency = 100 x (1 - mean adequacy).
    Needs >= 3 of 5 fields. These fields describe the whole city, not e-waste specifically."""
    last = G.latest_per_city(cy)
    pop = last["population"].replace(0, np.nan)
    last["bins_per_100k"] = last["smart_bins"] / pop * 1e5
    last["vehicles_per_100k"] = last["collection_vehicles"] / pop * 1e5
    last["workers_per_100k"] = last["workers"] / pop * 1e5
    ad = pd.DataFrame({c: last[c].rank(pct=True) for c in
                       ["bins_per_100k", "vehicles_per_100k", "workers_per_100k", "landfill_remaining_pct", "municipal_score"]})
    last["infrastructure_deficiency"] = np.where(ad.notna().sum(axis=1) >= 3, (1 - ad.mean(axis=1)) * 100, np.nan)
    return last[["city", "infrastructure_deficiency", "bins_per_100k", "vehicles_per_100k", "workers_per_100k",
                 "landfill_remaining_pct", "municipal_score"]]


def build_inputs(future: pd.DataFrame, gap_df: pd.DataFrame, cy: pd.DataFrame, target_year: int,
                 gap_kind: str = "collection", fallback: bool = False) -> pd.DataFrame:
    """Assumption: each city's latest observed gap % persists, so unaccounted = projected generation x gap %.
    Cities without a forecast are kept only if `fallback` (then latest observed generation is used and labelled)."""
    gp = G.gap_columns(gap_kind)["pct"]
    last = G.latest_per_city(gap_df)[["city", "state", "region", "city_category", "year", gp, Y]].rename(
        columns={"year": "last_obs_year", gp: "gap_percentage", Y: "latest_observed_generation"})
    fc = future[future["year"] == target_year][["city", "predicted", "model_used", "confidence"]].rename(
        columns={"predicted": "projected_generation"})
    fc["basis"] = "forecast"
    d = last.merge(fc, on="city", how="left")
    if fallback:
        miss = d["projected_generation"].isna()
        d.loc[miss, "projected_generation"] = d.loc[miss, "latest_observed_generation"]
        d.loc[miss, ["basis", "confidence", "model_used"]] = ["latest observed", "n/a (no forecast)", "none"]
    d["unaccounted_quantity"] = d["projected_generation"] * d["gap_percentage"] / 100
    return d.merge(infrastructure_index(cy), on="city", how="left")


def _w(weights: dict) -> pd.Series:
    w = pd.Series({c: float(weights.get(c, 0)) for c in COLS})
    if w.sum() <= 0:
        raise ValueError("At least one weight must be positive.")
    return w / w.sum()


def _fmt(col, v):
    return {"projected_generation": f"{v:,.0f} t/yr", "unaccounted_quantity": f"{v:,.0f} t/yr",
            "gap_percentage": f"{v:.1f}%", "infrastructure_deficiency": f"{v:.0f}/100"}[col]


def rank_cities(inp: pd.DataFrame, weights: dict):
    """Returns (ranked, unranked). Cities missing any indicator are not ranked and no value is imputed."""
    w = _w(weights)
    ok = inp.dropna(subset=COLS).copy()
    for c in COLS:
        ok[f"{c}_norm"] = minmax(ok[c])
        ok[f"{c}_points"] = ok[f"{c}_norm"] * w[c] * 100
    ok["priority_score"] = ok[[f"{c}_points" for c in COLS]].sum(axis=1)
    ok["rank"] = ok["priority_score"].rank(ascending=False, method="min").astype(int)
    n = len(ok)

    def explain(r):
        top = sorted(COLS, key=lambda c: r[f"{c}_points"], reverse=True)[:2]
        s = (f"Rank {r['rank']} of {n}, score {r['priority_score']:.1f}/100. Main drivers: "
             + " and ".join(f"{LABELS[c].lower()} ({r[f'{c}_points']:.1f} pts, {_fmt(c, r[c])})" for c in top) + ".")
        if r["basis"] == "latest observed":
            s += " Uses latest observed generation (no forecast available)."
        elif r["confidence"] != "Moderate":
            s += f" Forecast confidence: {r['confidence']}."
        return s

    if not ok.empty:
        ok["explanation"] = ok.apply(explain, axis=1)
    un = inp[~inp["city"].isin(ok["city"])].copy()
    if not un.empty:
        un["reason"] = un.apply(lambda r: "missing: " + ", ".join(LABELS[c] for c in COLS if pd.isna(r[c])), axis=1)
    return ok.sort_values("rank").reset_index(drop=True), un


def sensitivity(inp: pd.DataFrame, weights: dict, n: int = 500, seed: int = 0, top: int = 10) -> pd.DataFrame:
    """Re-rank under n random weight vectors drawn around the chosen weights (Dirichlet)."""
    ok = inp.dropna(subset=COLS).reset_index(drop=True)
    if len(ok) < 2:
        return pd.DataFrame()
    w = _w(weights).to_numpy()
    X = np.column_stack([minmax(ok[c]).to_numpy() for c in COLS])
    W = np.random.default_rng(seed).dirichlet(np.maximum(w * 30, 0.5), n)
    ranks = (-(X @ W.T)).argsort(0).argsort(0) + 1
    base = (-(X @ w)).argsort().argsort() + 1
    k = min(top, len(ok))
    return pd.DataFrame({"city": ok["city"], "base_rank": base, "mean_rank": ranks.mean(1), "best_rank": ranks.min(1),
                         "worst_rank": ranks.max(1), f"top{k}_share_pct": (ranks <= k).mean(1) * 100}
                        ).sort_values("base_rank").reset_index(drop=True)
