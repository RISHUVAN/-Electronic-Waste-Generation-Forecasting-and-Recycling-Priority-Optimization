"""Load the supplied dataset, isolate E-Waste records and build a city-year table.

Nothing is invented: missing values stay NaN, future-dated rows are excluded, and every
transformation is counted in the returned quality report."""
from pathlib import Path
import numpy as np
import pandas as pd
from src import config as C


class DataError(Exception):
    """Raised with a user-readable message when the dataset cannot be used."""


NUMERIC = ["Year", "Month", "Population", "Daily_Waste_Generation_Tons", "Collection_Efficiency_Percentage",
           "Recycling_Rate", "Latitude", "Longitude"] + C.INFRA_COLUMNS


def load_raw(path=None) -> pd.DataFrame:
    path = Path(path or C.MAIN_CSV)
    if not path.exists():
        raise DataError(f"Dataset not found: {path}. Place Waste_Management_India_20K.csv in the data/ folder.")
    df = pd.read_csv(path)
    missing = [c for c in C.REQUIRED_COLUMNS + C.INFRA_COLUMNS if c not in df.columns]
    if missing:
        raise DataError("Dataset is missing required column(s): " + ", ".join(missing))
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_master(path=None):
    path = Path(path or C.MASTER_CSV)
    return pd.read_csv(path) if path.exists() else None


def extract_ewaste(raw: pd.DataFrame, as_of: pd.Timestamp):
    """Return (usable_rows, future_rows, n_invalid). Future = dated after the as_of month."""
    e = raw[raw["Waste_Type"] == C.TARGET_WASTE_TYPE].copy()
    if e.empty:
        raise DataError(f"No '{C.TARGET_WASTE_TYPE}' records found in the dataset.")
    future = (e["Year"] * 12 + e["Month"]) > (as_of.year * 12 + as_of.month)
    fut, e = e[future], e[~future]
    valid = (e["Daily_Waste_Generation_Tons"] > 0) & e["Year"].notna() & e["Month"].notna()
    return e[valid].copy(), fut, int((~valid).sum())


def annualise(e: pd.DataFrame, master, as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per city-year. Daily averages (mean of the monthly rows present) x 365.
    Collection/recycling rates are re-derived from tonnage-weighted daily quantities."""
    e = e.copy()
    e["_coll"] = e["Daily_Waste_Generation_Tons"] * e["Collection_Efficiency_Percentage"] / 100
    e["_rec"] = e["_coll"] * e["Recycling_Rate"] / 100
    a = e.groupby(["City", "Year"], as_index=False).agg(
        state=("State", "first"), region=("Region", "first"), city_category=("City_Category", "first"),
        gen_daily=("Daily_Waste_Generation_Tons", "mean"), coll_daily=("_coll", "mean"), rec_daily=("_rec", "mean"),
        n_months=("Month", "nunique"), population=("Population", "mean"),
        smart_bins=("Smart_Bins_Count", "mean"), collection_vehicles=("Waste_Collection_Vehicles", "mean"),
        workers=("Worker_Count", "mean"), landfill_remaining_pct=("Landfill_Remaining_Capacity", "mean"),
        municipal_score=("Municipal_Efficiency_Score", "mean"), budget_inr=("Waste_Management_Budget_INR", "mean"),
        latitude=("Latitude", "first"), longitude=("Longitude", "first"))
    a = a.rename(columns={"City": "city", "Year": "year"})
    a["year"] = a["year"].astype(int)
    a["ewaste_generated_tonnes"] = a["gen_daily"] * C.DAYS_PER_YEAR
    a["collection_efficiency_pct"] = a["coll_daily"] / a["gen_daily"] * 100
    a["recycling_rate_pct"] = np.where(a["coll_daily"] > 0, a["rec_daily"] / a["coll_daily"].replace(0, np.nan) * 100, np.nan)
    a["ewaste_kg_per_capita"] = a["ewaste_generated_tonnes"] * 1000 / a["population"].replace(0, np.nan)
    a["year_partial"] = (a["year"] >= as_of.year) & (as_of.month < 12)
    if master is not None and {"City", "Is_Tourist", "Is_Hill_Station"} <= set(master.columns):
        m = master[["City", "Is_Tourist", "Is_Hill_Station"]].rename(
            columns={"City": "city", "Is_Tourist": "is_tourist", "Is_Hill_Station": "is_hill_station"})
        a = a.merge(m, on="city", how="left")
    cols = ["city", "state", "region", "city_category", "year", "ewaste_generated_tonnes", "collection_efficiency_pct",
            "recycling_rate_pct", "ewaste_kg_per_capita", "n_months", "year_partial", "population", "smart_bins",
            "collection_vehicles", "workers", "landfill_remaining_pct", "municipal_score", "budget_inr",
            "latitude", "longitude"] + [c for c in ("is_tourist", "is_hill_station") if c in a.columns]
    return a[cols].sort_values(["city", "year"]).reset_index(drop=True)


def build_dataset(as_of=None, path=None, master_path=None) -> dict:
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    raw = load_raw(path)
    master = load_master(master_path)
    e, fut, n_invalid = extract_ewaste(raw, as_of)
    cy = annualise(e, master, as_of)
    years_per_city = cy.groupby("city")["year"].nunique()
    report = dict(
        as_of=str(as_of.date()), raw_rows=len(raw), raw_cols=raw.shape[1], raw_cities=int(raw["City"].nunique()),
        raw_states=int(raw["State"].nunique()), raw_years=(int(raw["Year"].min()), int(raw["Year"].max())),
        waste_type_counts=raw["Waste_Type"].value_counts().to_dict(),
        ewaste_rows_total=int((raw["Waste_Type"] == C.TARGET_WASTE_TYPE).sum()), ewaste_rows_used=len(e),
        future_rows_excluded=len(fut), invalid_rows_dropped=n_invalid, city_years=len(cy),
        cities_with_ewaste=int(cy["city"].nunique()),
        cities_forecastable=int((years_per_city >= C.MIN_YEARS).sum()),
        cities_without_ewaste=sorted(set(raw["City"]) - set(cy["city"])))
    return dict(raw=raw, master=master, ewaste_rows=e, future_rows=fut, cityyear=cy, report=report, as_of=as_of)
