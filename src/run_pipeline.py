"""Headless run (no browser): writes sample outputs to outputs/.  Usage: python -m src.run_pipeline"""
import pandas as pd
from src import config as C
from src.analysis import gap as G, priority as PR
from src.data import loader
from src.models import forecast as F


def main(horizon: int = 3):
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    ds = loader.build_dataset()
    cy, year = ds["cityyear"], ds["as_of"].year
    fc = F.forecast_all(cy, horizon, year)
    gap = G.compute_gap(cy)
    inp = PR.build_inputs(fc["future"], gap, cy, year + 1, "collection", False)
    ranked, unranked = PR.rank_cities(inp, C.DEFAULT_WEIGHTS)
    holdout, summary = F.cross_city_holdout(cy)
    fc["future"].to_csv(C.OUTPUT_DIR / "forecasts.csv", index=False)
    fc["metrics"].to_csv(C.OUTPUT_DIR / "model_metrics.csv", index=False)
    summary.to_csv(C.OUTPUT_DIR / "model_holdout_summary.csv", index=False)
    gap.to_csv(C.OUTPUT_DIR / "gap_results.csv", index=False)
    ranked.drop(columns=[c for c in ranked.columns if c.endswith("_norm")]).to_csv(C.OUTPUT_DIR / "city_priority_ranking.csv", index=False)
    print(f"as_of={ds['as_of'].date()} forecast_cities={len(fc['details'])} ranked={len(ranked)} unranked={len(unranked)}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
