"""City-level forecasting: Linear (baseline), Polynomial(2) and a Naive last-value benchmark.

Validation is rolling-origin (each test year is predicted using only earlier years), so there is no
data leakage. With <= 8 annual points per city, results are low-confidence by construction and flagged."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from src import config as C

Y = "ewaste_generated_tonnes"
MODELS = ["Linear", "Polynomial", "Naive"]
MIN_TRAIN = {"Linear": 3, "Polynomial": 4, "Naive": 1}


def _fit_predict(kind, x, y, xn):
    if kind == "Naive":
        return np.repeat(y[-1], len(xn))
    m = LinearRegression() if kind == "Linear" else make_pipeline(PolynomialFeatures(2), LinearRegression())
    m.fit(np.asarray(x).reshape(-1, 1), y)
    return m.predict(np.asarray(xn).reshape(-1, 1))


def rolling_eval(x, y, n_test=2):
    """Predict each of the last <= n_test points from earlier points only."""
    n = len(y)
    idx = list(range(max(n - n_test, max(MIN_TRAIN.values())), n))
    out = {}
    for k in MODELS:
        pred = [_fit_predict(k, x[:i], y[:i], np.array([x[i]]))[0] for i in idx]
        act = [y[i] for i in idx]
        if not idx:
            out[k] = dict(MAE=np.nan, RMSE=np.nan, R2=np.nan, n_test=0)
            continue
        out[k] = dict(MAE=mean_absolute_error(act, pred), RMSE=float(np.sqrt(mean_squared_error(act, pred))),
                      R2=r2_score(act, pred) if len(idx) >= 3 else np.nan, n_test=len(idx))
    return out


def forecast_city(g: pd.DataFrame, horizon: int, as_of_year: int):
    """Returns (result_dict, None) or (None, reason). Forecast years are as_of_year+1 .. as_of_year+horizon."""
    g = g.dropna(subset=[Y]).sort_values("year")
    n = len(g)
    if n < C.MIN_YEARS:
        return None, f"{n} annual observation(s); at least {C.MIN_YEARS} are required"
    years, y = g["year"].to_numpy(float), g[Y].to_numpy(float)
    x = years - years[0]
    ev = rolling_eval(x, y)
    grid = np.arange(int(years[0]), as_of_year + horizon + 1)
    curve = pd.DataFrame({"year": grid})
    negative = {}
    for k in MODELS:
        raw = _fit_predict(k, x, y, grid - years[0])
        negative[k] = bool((raw < 0).any())
        curve[k] = np.clip(raw, 0, None)
    rmse = {k: (ev[k]["RMSE"] if not np.isnan(ev[k]["RMSE"]) else np.inf) for k in ("Linear", "Polynomial")}
    best = "Linear" if np.isinf(min(rmse.values())) else min(rmse, key=rmse.get)
    last_year = int(years[-1])
    extrap = as_of_year + horizon - last_year
    reasons, confidence = {}, {}
    for k in MODELS:
        hard, soft = [], []
        if n < 8:
            soft.append(f"limited history ({n} annual points)")
        if k != "Naive":
            if np.isinf(rmse[k]):
                hard.append("could not be validated")
            elif not rmse[k] < ev["Naive"]["RMSE"]:
                hard.append("did not beat the naive baseline in validation")
            if negative[k]:
                hard.append("negative values clipped to 0")
        if k == "Polynomial" and n < 8:
            hard.append("quadratic fit on <8 points (overfit risk)")
        if extrap > C.MAX_EXTRAPOLATION:
            hard.append(f"extrapolates {extrap} years beyond last observation ({last_year})")
        reasons[k] = "; ".join(hard + soft) or "none"
        confidence[k] = "Unreliable" if hard else ("Low" if soft else "Moderate")
    return dict(n_years=n, first_year=int(years[0]), last_year=last_year, curve=curve, metrics=ev, best_model=best,
                reasons=reasons, confidence=confidence, observed=g[["year", Y]].reset_index(drop=True)), None


def forecast_all(cy: pd.DataFrame, horizon: int, as_of_year: int, model: str = "Auto") -> dict:
    """Forecast every eligible city. `model` is 'Auto' (best validation RMSE of Linear/Polynomial), 'Linear' or 'Polynomial'."""
    details, rows, mets, skipped = {}, [], [], []
    meta = cy.drop_duplicates("city").set_index("city")[["state", "region", "city_category"]]
    for city, g in cy.groupby("city"):
        res, why = forecast_city(g, horizon, as_of_year)
        if res is None:
            skipped.append(dict(city=city, state=meta.loc[city, "state"], years_observed=g["year"].nunique(), reason=why))
            continue
        details[city] = res
        used = res["best_model"] if model == "Auto" else model
        c = res["curve"]
        fut = c[c["year"] > as_of_year].copy()
        fut["city"], fut["state"] = city, meta.loc[city, "state"]
        fut["model_used"], fut["predicted"] = used, fut[used]
        fut["confidence"], fut["caveats"] = res["confidence"][used], res["reasons"][used]
        rows.append(fut.rename(columns={"Linear": "linear", "Polynomial": "polynomial", "Naive": "naive"}))
        m = dict(city=city, state=meta.loc[city, "state"], n_years=res["n_years"], last_observed_year=res["last_year"],
                 best_model=res["best_model"], n_validation_points=res["metrics"]["Linear"]["n_test"])
        for k in MODELS:
            for s in ("MAE", "RMSE", "R2"):
                m[f"{k}_{s}"] = res["metrics"][k][s]
            m[f"{k}_confidence"] = res["confidence"][k]
        mets.append(m)
    cols = ["city", "state", "year", "model_used", "predicted", "linear", "polynomial", "naive", "confidence", "caveats"]
    return dict(future=pd.concat(rows)[cols].reset_index(drop=True) if rows else pd.DataFrame(columns=cols),
                metrics=pd.DataFrame(mets), skipped=pd.DataFrame(skipped, columns=["city", "state", "years_observed", "reason"]),
                details=details)


def cross_city_holdout(cy: pd.DataFrame):
    """Honest pooled evidence: for every eligible city, predict its LAST observed year from earlier years only.
    Returns (per_city_df, summary_df)."""
    rows = []
    for city, g in cy.groupby("city"):
        g = g.dropna(subset=[Y]).sort_values("year")
        if len(g) < C.MIN_YEARS:
            continue
        yrs, y = g["year"].to_numpy(float), g[Y].to_numpy(float)
        x = yrs - yrs[0]
        r = dict(city=city, held_out_year=int(yrs[-1]), actual=y[-1], n_train=len(y) - 1)
        for k in MODELS:
            r[k] = float(_fit_predict(k, x[:-1], y[:-1], np.array([x[-1]]))[0])
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()
    s = []
    naive_mae = mean_absolute_error(df["actual"], df["Naive"])
    for k in MODELS:
        err = (df[k] - df["actual"]).abs()
        mae = float(err.mean())
        s.append(dict(model=k, cities=len(df), MAE_t=mae, RMSE_t=float(np.sqrt(mean_squared_error(df["actual"], df[k]))),
                      R2_pooled=float(r2_score(df["actual"], df[k])),
                      median_abs_pct_error=float((err / df["actual"] * 100).median()),
                      skill_vs_naive=float(1 - mae / naive_mae) if naive_mae else np.nan,
                      beats_naive_pct=float((err < (df["Naive"] - df["actual"]).abs()).mean() * 100) if k != "Naive" else np.nan))
    return df, pd.DataFrame(s)
