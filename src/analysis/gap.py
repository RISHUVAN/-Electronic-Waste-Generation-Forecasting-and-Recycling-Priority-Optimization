"""Collection / recycling gap estimates derived from the dataset's RATE columns.

The dataset provides percentages (collection efficiency, recycling rate), not documented tonnes, so every
tonnage here is a rate-derived estimate. A gap is a statistical shortfall, not proof of illegal dumping."""
import numpy as np
import pandas as pd

Y = "ewaste_generated_tonnes"
BASIS_COLLECTED = "collected"   # Recycling_Rate is a share of COLLECTED waste (default assumption)
BASIS_GENERATED = "generated"   # Recycling_Rate is a share of GENERATED waste (alternative)


def compute_gap(cy: pd.DataFrame, recycling_basis: str = BASIS_COLLECTED) -> pd.DataFrame:
    d = cy.copy()
    gen = d[Y].where(d[Y] > 0)
    coll = d["collection_efficiency_pct"].where(d["collection_efficiency_pct"].between(0, 100))
    rate = d["recycling_rate_pct"].where(d["recycling_rate_pct"].between(0, 100))
    d["collected_tonnes"] = gen * coll / 100
    d["collection_coverage_pct"] = d["collected_tonnes"] / gen * 100
    d["collection_gap_tonnes"] = (gen - d["collected_tonnes"]).clip(lower=0)
    d["collection_gap_pct"] = d["collection_gap_tonnes"] / gen * 100
    base = d["collected_tonnes"] if recycling_basis == BASIS_COLLECTED else gen
    d["recycled_tonnes_est"] = base * rate / 100
    d["recycling_coverage_pct"] = d["recycled_tonnes_est"] / gen * 100
    d["recycling_gap_tonnes"] = (gen - d["recycled_tonnes_est"]).clip(lower=0)
    d["recycling_gap_pct"] = d["recycling_gap_tonnes"] / gen * 100
    d["recycling_basis"] = recycling_basis
    return d


def gap_columns(kind: str) -> dict:
    """kind: 'collection' or 'recycling'."""
    if kind == "collection":
        return dict(cov="collection_coverage_pct", tonnes="collection_gap_tonnes", pct="collection_gap_pct",
                    label="Collection gap")
    return dict(cov="recycling_coverage_pct", tonnes="recycling_gap_tonnes", pct="recycling_gap_pct",
                label="Estimated recycling gap")


def latest_per_city(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("year").groupby("city").tail(1).reset_index(drop=True)
