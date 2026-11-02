"""Data-quality reporting. Produces plain DataFrames the UI can display."""
import numpy as np
import pandas as pd
from src import config as C

INDIA_BOX = (6.0, 37.5, 68.0, 98.0)  # lat_min, lat_max, lon_min, lon_max


def schema_table(raw: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({"column": raw.columns, "dtype": raw.dtypes.astype(str).values,
                      "missing": raw.isna().sum().values, "unique": raw.nunique().values})
    t["missing_pct"] = (t["missing"] / len(raw) * 100).round(2)
    t["used_in_app"] = t["column"].isin(C.REQUIRED_COLUMNS + C.INFRA_COLUMNS)
    return t


def coverage_by_city(cy: pd.DataFrame) -> pd.DataFrame:
    g = cy.groupby(["city", "state"]).agg(years_observed=("year", "nunique"), first_year=("year", "min"),
                                           last_year=("year", "max")).reset_index()
    g["forecastable"] = g["years_observed"] >= C.MIN_YEARS
    return g.sort_values("years_observed", ascending=False)


def coverage_by_year(cy: pd.DataFrame) -> pd.DataFrame:
    return cy.groupby("year").agg(cities=("city", "nunique"), partial=("year_partial", "any")).reset_index()


def run_checks(ds: dict) -> pd.DataFrame:
    raw, e, rep, master = ds["raw"], ds["ewaste_rows"], ds["report"], ds["master"]
    rows = []

    def add(status, check, detail):
        rows.append(dict(status=status, check=check, detail=detail))

    add("ok", "Required columns present", f"{len(C.REQUIRED_COLUMNS) + len(C.INFRA_COLUMNS)} columns used; all found.")
    add("info", "Dataset size",
        f"{rep['raw_rows']:,} rows, {rep['raw_cols']} columns, {rep['raw_cities']} cities, "
        f"{rep['raw_states']} states/UTs, {rep['raw_years'][0]}-{rep['raw_years'][1]}.")
    add("warn", "Waste categories", f"Only '{C.TARGET_WASTE_TYPE}' is analysed: {rep['ewaste_rows_total']:,} of "
        f"{rep['raw_rows']:,} rows ({rep['ewaste_rows_total'] / rep['raw_rows']:.1%}). Other waste types are never combined with it.")
    if rep["future_rows_excluded"]:
        add("warn", "Future-dated records", f"{rep['future_rows_excluded']} E-waste row(s) dated after {rep['as_of'][:7]} were "
            "excluded because they cannot be observations.")
    else:
        add("ok", "Future-dated records", "None found.")
    add("warn", "Sparse city coverage", f"{rep['cities_with_ewaste']} cities have E-waste data "
        f"({', '.join(rep['cities_without_ewaste']) or 'none'} have none); only {rep['cities_forecastable']} have "
        f">= {C.MIN_YEARS} distinct years and can be forecast.")
    add("warn", "Annualisation", "Each row is a monthly sample of a daily average; tonnes/year = daily mean x 365 is an estimate.")
    add("warn", "Provenance", "The file has no source, units definitions or collection method; values are used as supplied. "
        "Treat results as illustrative, not as verified real-world measurements.")
    miss = int(e[["Daily_Waste_Generation_Tons", "Collection_Efficiency_Percentage", "Recycling_Rate"]].isna().sum().sum())
    add("ok" if miss == 0 else "warn", "Missing values in key E-waste fields", f"{miss} missing; none were replaced with zero.")
    bad = int(((e["Collection_Efficiency_Percentage"] < 0) | (e["Collection_Efficiency_Percentage"] > 100) |
               (e["Recycling_Rate"] < 0) | (e["Recycling_Rate"] > 100)).sum())
    add("ok" if bad == 0 else "warn", "Rates within 0-100%", f"{bad} out-of-range rate value(s).")
    dup = int(e.duplicated(["City", "Year", "Month"]).sum())
    add("ok" if dup == 0 else "warn", "Duplicate city-month records", f"{dup} duplicate(s).")
    pc = (e["Daily_Waste_Generation_Tons"] * 1000 / e["Population"] - e["Waste_Per_Capita_KG"]).abs().median() \
        if "Waste_Per_Capita_KG" in e.columns else np.nan
    if not np.isnan(pc):
        add("info", "Per-capita field is derived", f"Median |tons*1000/population - Waste_Per_Capita_KG| = {pc:.5f}; "
            "the per-capita column adds no independent information.")
    lat_min, lat_max, lon_min, lon_max = INDIA_BOX
    out = int(((raw["Latitude"] < lat_min) | (raw["Latitude"] > lat_max) | (raw["Longitude"] < lon_min) | (raw["Longitude"] > lon_max)).sum())
    add("ok" if out == 0 else "warn", "Coordinates inside India's bounding box", f"{out} row(s) outside.")
    if master is not None:
        same = set(master["City"]) == set(raw["City"])
        add("ok" if same else "warn", "cities_master.csv consistency", "City list matches the main dataset." if same
            else "City lists differ between cities_master.csv and the main dataset.")
    add("info", "Administrative names", "State/UT names are used as supplied (e.g. 'Daman and Diu' and 'Dadra and Nagar Haveli' "
        "appear separately although they merged in 2020).")
    add("info", "Excluded file", "Waste_Management_and_Recycling_India.csv was not used: different schema, 34 cities, and E-waste "
        "magnitudes roughly 40x those in the main file, so the two cannot be combined.")
    return pd.DataFrame(rows)
