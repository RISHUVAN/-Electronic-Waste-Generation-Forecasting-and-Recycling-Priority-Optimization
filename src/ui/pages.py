"""Page renderers. Each takes a `ctx` dict prepared in app.py (filtered data, forecasts callable, settings)."""
import numpy as np
import pandas as pd
import streamlit as st
from src import config as C
from src.analysis import gap as G, priority as PR
from src.data import quality as Q
from src.models import forecast as F
from src.ui import components as ui
from src.viz import charts as ch

latest = G.latest_per_city
GEN = "ewaste_generated_tonnes"


# ------------------------------------------------------------------ 1. Executive Overview
def overview(ctx):
    ds, cy, gap, rep = ctx["ds"], ctx["cy"], ctx["gap"], ctx["ds"]["report"]
    ui.header("Executive Overview", "Observed urban e-waste indicators across Indian cities, taken from the supplied dataset.")
    ui.callout("warn", f"<b>Scope and caution.</b> Only the E-Waste records are analysed ({rep['ewaste_rows_used']:,} of {rep['raw_rows']:,} rows). "
                       "The dataset documents no source, so figures are used as supplied. Medians below are across cities, not national totals.")
    last = latest(gap)
    years_per = cy.groupby("city")["year"].nunique()
    ui.kpis([("Cities with e-waste data", f"{cy['city'].nunique()}", f"of {rep['raw_cities']} cities in the dataset"),
             ("City-year observations", f"{len(cy):,}", f"{cy['year'].min()}-{cy['year'].max()} in current filter"),
             ("Median collection efficiency", f"{last['collection_efficiency_pct'].median():.1f}%", "latest observation per city"),
             ("Median est. recycling coverage", f"{last['recycling_coverage_pct'].median():.1f}%", "of generated, rate-derived estimate"),
             ("Forecastable cities", f"{int((years_per >= C.MIN_YEARS).sum())}", f">= {C.MIN_YEARS} observed years")])
    tab_map, tab_trend, tab_state = st.tabs(["Map and coverage", "Historical trends", "State comparison"])

    with tab_map:
        opts = {"Collection gap (%)": "collection_gap_pct", "Estimated recycling gap (%)": "recycling_gap_pct",
                "E-waste per capita (kg/yr)": "ewaste_kg_per_capita", "Annualised generation (t/yr)": GEN}
        sel = st.selectbox("Colour cities by", list(opts), key="ov_map_metric")
        ui.show(ch.india_map(last, opts[sel], sel, title=f"Latest observation per city: {sel}"), key="ov_map")
        st.caption("Marker size = annualised generation. Each marker is the city's most recent observation, so cities may be shown for "
                   "different years. Base map: Plotly's Natural Earth layer; only city markers are dataset values.")
        c1, c2 = st.columns(2)
        with c1:
            ui.show(ch.coverage_by_year_chart(Q.coverage_by_year(cy)), key="ov_cov_year")
        with c2:
            ui.show(ch.years_per_city_hist(Q.coverage_by_city(cy), C.MIN_YEARS), key="ov_cov_city")

    with tab_trend:
        top = last.nlargest(5, GEN)["city"].tolist()
        cities = st.multiselect("Cities", sorted(cy["city"].unique()), default=top)
        metrics = {"Annualised generation (t/yr)": GEN, "E-waste per capita (kg/yr)": "ewaste_kg_per_capita",
                   "Collection efficiency (%)": "collection_efficiency_pct", "Recycling rate (% of collected)": "recycling_rate_pct"}
        m = st.selectbox("Indicator", list(metrics), key="ov_trend_metric")
        d = cy[cy["city"].isin(cities)]
        if d.empty:
            ui.empty("Select at least one city", "Observed values appear here.")
        else:
            ui.show(ch.trend_lines(d, metrics[m], m, title=f"Observed {m.lower()} by city"), key="ov_trend")
            st.caption("Markers are actual observations; lines only join observed years (years without data are not interpolated). "
                       "The final year may be partial.")

    with tab_state:
        s = last.groupby("state").agg(cities=("city", "nunique"), median_collection_gap_pct=("collection_gap_pct", "median"),
                                      median_generation_t=(GEN, "median")).reset_index()
        ui.show(ch.state_bar(s, "median_collection_gap_pct", "Median collection gap (%)",
                             "Median collection gap by state/UT (latest observation per city; n = cities)"), key="ov_state")
        st.caption("States with few cities rest on very small samples; read the n labels before comparing.")
        ui.table(s.sort_values("median_collection_gap_pct", ascending=False).round(1))


# ------------------------------------------------------------------ 2. Forecast Studio
def forecast_studio(ctx):
    ui.header("Forecast Studio", "City-level e-waste generation forecasts with transparent validation and reliability flags.")
    ui.callout("info", f"Cities need <b>at least {C.MIN_YEARS} observed years</b> to be forecast. Forecasts use each city's full history "
                       "(the year filter does not apply) and cover years after the current year. Validation is rolling-origin "
                       "(each test year predicted only from earlier years).")
    horizon = st.slider("Forecast horizon (years ahead)", 1, C.MAX_HORIZON, 3, key="fs_horizon")
    FC = ctx["forecasts"](horizon, ctx["model"])
    eligible = sorted(c for c in FC["details"] if c in ctx["cities"])
    if not eligible:
        ui.empty("No forecastable city in the current selection",
                 f" Widen the state/city filter. Cities need {C.MIN_YEARS}+ observed years.")
    else:
        city = st.selectbox("City", eligible)
        res = FC["details"][city]
        used = res["best_model"] if ctx["model"] == "Auto" else ctx["model"]
        target = ctx["as_of_year"] + horizon
        fut = FC["future"][(FC["future"]["city"] == city) & (FC["future"]["year"] == target)].iloc[0]
        last_obs = res["observed"].iloc[-1]
        change = (fut["predicted"] / last_obs[GEN] - 1) * 100
        conf = res["confidence"][used]
        ui.kpis([("Last observed", ui.fmt_t(last_obs[GEN]), f"year {int(last_obs['year'])}"),
                 (f"Forecast {target}", ui.fmt_t(fut["predicted"]), f"{used} model, {change:+.1f}% vs last observed"),
                 ("Reliability", conf, res["reasons"][used]),
                 ("History used", f"{res['n_years']} years", f"{res['first_year']}-{res['last_year']}")])
        (ui.callout("danger", f"<b>Unreliable forecast.</b> {res['reasons'][used]}.") if conf == "Unreliable" else
         ui.callout("warn", f"<b>Low confidence.</b> {res['reasons'][used]}.") if conf == "Low" else None)
        ui.show(ch.forecast_chart(res, city, used, ctx["as_of_year"]), key="fs_chart")
        st.caption("White markers are observed (annualised from monthly samples); the shaded band is the forecast period. "
                   "Alternative models are shown dotted for comparison. Predictions never go below zero.")
        st.subheader("Model comparison for this city")
        mt = pd.DataFrame([{"Model": k, "Validation MAE (t)": res["metrics"][k]["MAE"], "Validation RMSE (t)": res["metrics"][k]["RMSE"],
                            "R2": res["metrics"][k]["R2"], "Validation points": res["metrics"][k]["n_test"],
                            "Confidence": res["confidence"][k], "Caveats": res["reasons"][k]} for k in F.MODELS]).round(2)
        ui.table(mt)
        st.caption("R2 is shown only when there are 3+ validation points (otherwise n/a, because it would be meaningless). "
                   "Cross-city evidence on which model works better is on the Methodology & Model Performance page.")
        ui.table(FC["future"][FC["future"]["city"] == city][["year", "predicted", "linear", "polynomial", "naive", "model_used"]].round(1))
        out = FC["future"][FC["future"]["city"].isin(ctx["cities"])]
        ui.download(out, "Download forecasts (filtered cities, CSV)", "forecasts.csv", "dl_fc")
    if len(FC["skipped"]):
        sk = FC["skipped"][FC["skipped"]["city"].isin(ctx["cities"])]
        with st.expander(f"Cities not forecast in current selection ({len(sk)})"):
            ui.table(sk)


# ------------------------------------------------------------------ 3. Recycling Gap Analysis
def gap_analysis(ctx):
    ui.header("Recycling Gap Analysis", "Generated versus rate-derived collected and recycled quantities, by city.")
    c1, c2, c3 = st.columns(3)
    kind_lbl = c1.radio("Gap type", ["Collection gap", "Estimated recycling gap"], key="gap_kind")
    kind = "collection" if kind_lbl == "Collection gap" else "recycling"
    basis_lbl = c2.radio("Recycling-rate assumption", ["Share of collected waste", "Share of generated waste"],
                         key="gap_basis", disabled=(kind == "collection"))
    basis = G.BASIS_GENERATED if basis_lbl == "Share of generated waste" and kind == "recycling" else G.BASIS_COLLECTED
    view = c3.radio("Observation", ["Latest per city", "Selected year"], key="gap_view")
    gap = G.compute_gap(ctx["cy"], basis)
    cols = G.gap_columns(kind)
    d = latest(gap)
    if view == "Selected year":
        y = st.selectbox("Year", sorted(gap["year"].unique(), reverse=True))
        d = gap[gap["year"] == y]
    d = d.dropna(subset=[cols["pct"]])
    if d.empty:
        ui.empty("No observations for this selection", " Adjust the state, city or year filters.")
        return
    if kind == "collection":
        ui.callout("info", "<b>Collection gap</b> = generated minus collected, where collected = generated x collection efficiency. "
                           "It measures waste <b>not collected</b> by the formal system, not waste that was recycled or dumped.")
    else:
        ui.callout("warn", "<b>Estimated recycling gap</b> = generated minus estimated recycled. The dataset gives <b>rates, not documented tonnes</b>, "
                           "and does not define the Recycling_Rate denominator. " +
                   ("Assumed here: the rate applies to <b>collected</b> waste." if basis == G.BASIS_COLLECTED else
                    "Assumed here: the rate applies to <b>all generated</b> waste.") + " A statistical gap does not prove illegal dumping.")
    top = d.loc[d[cols["pct"]].idxmax()]
    ui.kpis([("Cities shown", f"{len(d)}", "latest observation" if view == "Latest per city" else f"year {y}"),
             (f"Median {cols['label'].lower()}", f"{d[cols['pct']].median():.1f}%", "across cities shown"),
             ("Median documented coverage", f"{d[cols['cov']].median():.1f}%", "of generated"),
             ("Largest gap %", f"{top['city']}", f"{top[cols['pct']]:.1f}% of generated")])
    t1, t2, t3 = st.tabs(["Generated vs collected/recycled", "Gap trends", "City comparison"])
    with t1:
        n = st.slider("Cities to show (largest generators)", 5, min(30, max(5, len(d))), min(12, max(5, len(d))))
        ui.show(ch.gap_grouped_bars(d.nlargest(n, GEN), title="Generated, collected and estimated recycled tonnes per year"), key="gap_bars")
        st.caption("Collected and recycled bars are derived from percentage rates in the dataset, not from documented tonnage.")
    with t2:
        default = d.nlargest(5, cols["pct"])["city"].tolist()
        sel = st.multiselect("Cities", sorted(gap["city"].unique()), default=default)
        gd = gap[gap["city"].isin(sel)].dropna(subset=[cols["pct"]])
        if gd.empty:
            ui.empty("Select cities with observations", "")
        else:
            ui.show(ch.trend_lines(gd, cols["pct"], f"{cols['label']} (%)", title=f"{cols['label']} over observed years"), key="gap_trend")
    with t3:
        a, b = st.columns(2)
        with a:
            ui.show(ch.gap_scatter(d, cols["pct"], cols["label"], cols["tonnes"], "Scale vs severity (top-right = large and poorly covered)"), key="gap_scatter")
        with b:
            ui.show(ch.gap_rank_bar(d.nlargest(15, cols["pct"]), cols["pct"], cols["label"], "15 largest gap percentages"), key="gap_rank")
    show_cols = ["city", "state", "year", GEN, "collected_tonnes", "recycled_tonnes_est", cols["cov"], cols["tonnes"], cols["pct"]]
    with st.expander("Data table"):
        ui.table(d[show_cols].sort_values(cols["pct"], ascending=False).round(1))
    ui.download(gap[gap["city"].isin(ctx["cities"])], "Download gap results (filtered, CSV)", "gap_results.csv", "dl_gap")


# ------------------------------------------------------------------ 4. City Priority Ranking
def priority(ctx):
    ui.header("City Priority Ranking", "A transparent, configurable decision-support score for where recycling attention may matter most.")
    ui.callout("warn", "<b>Decision-support ranking only.</b> The score is a weighted combination of normalised indicators; it is not a scientifically "
                       "validated measure. Forecast inputs are low-confidence for most cities (see each explanation).")
    asof = ctx["as_of_year"]
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Target year", list(range(asof + 1, asof + C.MAX_HORIZON + 1)), index=0, key="pr_target")
    kind_lbl = c2.radio("Gap basis", ["Collection gap", "Estimated recycling gap"], key="pr_kind")
    fallback = c3.toggle("Include cities without a forecast (use latest observed generation)", value=False, key="pr_fallback")
    kind = "collection" if kind_lbl == "Collection gap" else "recycling"
    with st.expander("Indicator weights (automatically normalised to 100%)"):
        wc = st.columns(4)
        W = {c: wc[i].slider(PR.LABELS[c], 0, 100, C.DEFAULT_WEIGHTS[c], key=f"w_{c}") for i, c in enumerate(PR.COLS)}
        tot = sum(W.values())
        if tot == 0:
            st.error("At least one weight must be above zero.")
            return
        st.caption("Effective weights: " + ", ".join(f"{PR.LABELS[c]} {W[c] / tot:.0%}" for c in PR.COLS))
    cy_all = ctx["cy_all"]
    FC = ctx["forecasts"](target - asof, ctx["model"])
    inp = PR.build_inputs(FC["future"], G.compute_gap(cy_all), cy_all, target, kind, fallback)
    ranked, unranked = PR.rank_cities(inp, W)
    if ranked.empty:
        ui.empty("No city has all four indicators", " Try enabling cities without a forecast.")
        return
    view = ranked[ranked["city"].isin(ctx["cities"])]
    if view.empty:
        ui.empty("No ranked city matches the current filters", " Scores and ranks are computed across all eligible cities; filters only choose what is displayed.")
        return
    ui.kpis([("Cities ranked", f"{len(ranked)}", f"{len(view)} shown after filters"),
             ("Highest priority", view.iloc[0]["city"], f"score {view.iloc[0]['priority_score']:.1f} (rank {view.iloc[0]['rank']})"),
             ("Based on forecasts", f"{int((ranked['basis'] == 'forecast').sum())}", "others use latest observed"),
             ("Target year", f"{target}", f"gap basis: {kind_lbl.lower()}")])
    t1, t2, t3 = st.tabs(["Ranking", "Why this score", "Sensitivity"])
    with t1:
        n = st.slider("Cities in chart", 5, min(40, max(5, len(view))), min(15, max(5, len(view))))
        ui.show(ch.priority_bar(view.head(n), f"Top {min(n, len(view))} cities by priority score"), key="pr_bar")
        tbl = view[["rank", "city", "state", "city_category", "priority_score", "projected_generation", "unaccounted_quantity",
                    "gap_percentage", "infrastructure_deficiency", "basis", "confidence"]].copy()
        ui.table(tbl.round(1), column_config={"priority_score": st.column_config.ProgressColumn("priority_score", min_value=0, max_value=100, format="%.1f")})
    with t2:
        city = st.selectbox("City", view["city"].tolist())
        row = ranked[ranked["city"] == city].iloc[0]
        st.markdown(f"**{city}**, {row['state']}. {row['explanation']}")
        ui.show(ch.contribution_bar(row), key="pr_contrib")
        with st.expander("All explanations"):
            ui.table(view[["rank", "city", "explanation"]])
    with t3:
        st.caption("500 random weight vectors drawn around your weights; the bar shows each city's best-to-worst rank.")
        sens = PR.sensitivity(inp, W)
        sens = sens[sens["city"].isin(view["city"])].head(25)
        if sens.empty:
            ui.empty("Not enough ranked cities for sensitivity analysis", "")
        else:
            ui.show(ch.sensitivity_chart(sens, "Rank stability (top 25 shown)"), key="pr_sens")
            ui.table(sens.round(1))
    if len(unranked):
        with st.expander(f"Not ranked ({len(unranked)} cities, missing indicators; nothing imputed)"):
            ui.table(unranked[["city", "state", "reason"]])
    ui.download(view.drop(columns=[c for c in view.columns if c.endswith("_norm")]), "Download ranking (filtered, CSV)", "city_priority_ranking.csv", "dl_rank")


# ------------------------------------------------------------------ 5. City Comparison
def comparison(ctx):
    ui.header("City Comparison", "Side-by-side view of generation, performance, gaps and infrastructure for selected cities.")
    cy_all, gap_all = ctx["cy_all"], G.compute_gap(ctx["cy_all"])
    infra = PR.infrastructure_index(cy_all)[["city", "infrastructure_deficiency", "bins_per_100k", "vehicles_per_100k", "workers_per_100k"]]
    last_all = latest(gap_all).merge(infra, on="city", how="left")
    avail = sorted(set(ctx["cities"]) & set(last_all["city"]))
    if len(avail) < 2:
        ui.empty("Need at least two cities in the current selection", " Widen the state/city filters.")
        return
    default = last_all[last_all["city"].isin(avail)].nlargest(4, GEN)["city"].tolist()
    sel = st.multiselect("Cities to compare (2-6)", avail, default=default, max_selections=6)
    if len(sel) < 2:
        ui.empty("Select at least two cities", "")
        return
    pct = pd.DataFrame({"city": last_all["city"]})
    pct["Generation"] = last_all[GEN].rank(pct=True) * 100
    pct["Per-capita e-waste"] = last_all["ewaste_kg_per_capita"].rank(pct=True) * 100
    pct["Collection efficiency"] = last_all["collection_efficiency_pct"].rank(pct=True) * 100
    pct["Recycling rate"] = last_all["recycling_rate_pct"].rank(pct=True) * 100
    pct["Infrastructure adequacy"] = (100 - last_all["infrastructure_deficiency"]).rank(pct=True) * 100
    pct["Municipal score"] = last_all["municipal_score"].rank(pct=True) * 100
    axes = [c for c in pct.columns if c != "city"]
    d = last_all[last_all["city"].isin(sel)]
    a, b = st.columns([1.1, 1])
    with a:
        ui.show(ch.radar(pct[pct["city"].isin(sel)], axes, "Percentile rank among all cities with data (higher = more)"), key="cmp_radar")
    with b:
        ui.show(ch.grouped_city_bars(d, GEN, "Annualised generation (t/yr)", "Generation (latest observation)"), key="cmp_gen")
        ui.show(ch.grouped_city_bars(d, "collection_gap_pct", "Collection gap (%)", "Collection gap"), key="cmp_gap")
    st.caption("Percentiles are computed across every city with e-waste data (not just the selection), using each city's latest observation. "
               "Infrastructure fields describe the whole city's waste system, not e-waste alone.")
    ui.show(ch.trend_lines(cy_all[cy_all["city"].isin(sel)], GEN, "Annualised generation (t/yr)", "Observed generation over time"), key="cmp_trend")
    cols = ["city", "state", "city_category", "year", GEN, "ewaste_kg_per_capita", "collection_efficiency_pct", "recycling_rate_pct",
            "collection_gap_pct", "recycling_gap_pct", "bins_per_100k", "vehicles_per_100k", "workers_per_100k",
            "landfill_remaining_pct", "municipal_score", "infrastructure_deficiency"]
    ui.table(d[cols].round(1))


# ------------------------------------------------------------------ 6. Data Explorer
def data_explorer(ctx):
    ui.header("Data Explorer", "Browse, filter and export the underlying E-waste observations.")
    cy, ds = ctx["cy"], ctx["ds"]
    t1, t2, t3 = st.tabs(["City-year table", "Raw E-waste rows", "Relationship explorer"])
    with t1:
        q = st.text_input("Search city", "", key="de_search")
        d = cy[cy["city"].str.contains(q, case=False, na=False)] if q else cy
        cols = st.multiselect("Columns", list(d.columns), default=["city", "state", "year", GEN, "collection_efficiency_pct",
                              "recycling_rate_pct", "ewaste_kg_per_capita", "n_months", "year_partial"], key="de_cols")
        if d.empty or not cols:
            ui.empty("Nothing to show", " Adjust the search, columns or filters.")
        else:
            ui.table(d[cols].round(2))
            ui.download(d, "Download city-year table (CSV)", "city_year_ewaste.csv", "dl_cy")
    with t2:
        e = ds["ewaste_rows"]
        e = e[e["City"].isin(ctx["cities"]) & e["Year"].between(*ctx["year_range"])]
        if e.empty:
            ui.empty("No raw rows in the current selection", "")
        else:
            st.caption(f"{len(e):,} original E-Waste rows (one row = one city-month sample), after excluding future-dated records.")
            ui.table(e)
    with t3:
        num = [c for c in cy.select_dtypes("number").columns if c not in ("year",)]
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis", num, index=num.index("population") if "population" in num else 0, key="de_x")
        y = c2.selectbox("Y axis", num, index=num.index(GEN) if GEN in num else 0, key="de_y")
        col = c3.selectbox("Colour by", ["region", "city_category", "state"], key="de_c")
        d = latest(cy).dropna(subset=[x, y])
        if d.empty:
            ui.empty("No data to plot", "")
        else:
            import plotly.express as px
            fig = px.scatter(d, x=x, y=y, color=col, hover_name="city", custom_data=["state", "year"])
            fig.update_traces(marker=dict(size=9, opacity=0.85), hovertemplate="<b>%{hovertext}</b>, %{customdata[0]}<br>Year %{customdata[1]}<br>"
                              + x + ": %{x:,.1f}<br>" + y + ": %{y:,.1f}<extra></extra>")
            ui.show(ch._style(fig, f"{y} vs {x} (latest observation per city)", 460), key="de_scatter")
            st.caption(f"Pearson correlation: {d[x].corr(d[y]):.2f}. Correlation in this dataset does not imply causation.")


# ------------------------------------------------------------------ 7. Methodology & Model Performance
def methodology(ctx):
    ui.header("Methodology & Model Performance", "How the numbers are produced, how well the models do, and where the data is weak.")
    ds, rep = ctx["ds"], ctx["ds"]["report"]
    t1, t2, t3 = st.tabs(["Model performance", "Data quality", "Methodology"])
    with t1:
        h, s = F.cross_city_holdout(ds["cityyear"])
        if s.empty:
            ui.empty("No city has enough history for validation", "")
        else:
            best = s[s["model"] != "Naive"].sort_values("MAE_t").iloc[0]
            ui.callout("info", f"<b>Hold-out test across {int(s.iloc[0]['cities'])} cities:</b> each city's last observed year is predicted using only earlier years. "
                               f"Best model by MAE: <b>{best['model']}</b> (beats the naive last-value baseline in "
                               f"{best['beats_naive_pct']:.0f}% of cities).")
            a, b = st.columns(2)
            with a:
                ui.show(ch.model_skill_chart(s), key="mp_skill")
            with b:
                ui.show(ch.holdout_scatter(h), key="mp_scatter")
            ui.table(s.round(3).rename(columns={"MAE_t": "MAE (t)", "RMSE_t": "RMSE (t)", "R2_pooled": "R2 (pooled)",
                     "median_abs_pct_error": "Median abs % error", "skill_vs_naive": "Skill vs naive", "beats_naive_pct": "% cities beating naive"}))
            st.caption("Pooled R2 is dominated by differences between large and small cities, so it flatters every model including Naive. "
                       "Use median % error, skill versus naive and the share of cities beating the baseline instead. Skill > 0 means better than naive.")
        FC = ctx["forecasts"](3, ctx["model"])
        with st.expander("Per-city rolling-origin validation (horizon 3)"):
            ui.table(FC["metrics"].round(2))
            ui.download(FC["metrics"], "Download model metrics (CSV)", "model_metrics.csv", "dl_mm")
    with t2:
        ui.kpis([("Rows in file", f"{rep['raw_rows']:,}", f"{rep['raw_cols']} columns"), ("E-waste rows used", f"{rep['ewaste_rows_used']:,}",
                 f"{rep['future_rows_excluded']} future-dated excluded"), ("Cities with data", f"{rep['cities_with_ewaste']}", f"of {rep['raw_cities']}"),
                 ("Forecastable", f"{rep['cities_forecastable']}", f">= {C.MIN_YEARS} years")])
        chk = Q.run_checks(ds)
        chk["status"] = chk["status"].map({"ok": "OK", "warn": "Warning", "info": "Note"})
        ui.table(chk)
        with st.expander("Schema and missing values (full file)"):
            ui.table(Q.schema_table(ds["raw"]))
        with st.expander("Coverage by city"):
            ui.table(Q.coverage_by_city(ds["cityyear"]))
    with t3:
        st.markdown(f"""
**Scope.** India only. Source: `Waste_Management_India_20K.csv` (supplied). Only rows with `Waste_Type = E-Waste` are used; other waste
categories are never combined with it. `cities_master.csv` supplies tourist/hill-station flags and a consistency check. No source documentation was
provided with the data, so it is treated as supplied and results are illustrative.

**Annualisation.** Each row is a monthly sample of a *daily* average. Per city-year: tonnes/year = mean daily tonnes x {C.DAYS_PER_YEAR}.
Collection efficiency and recycling rate are re-derived from tonnage-weighted daily quantities. Records dated after the current month are excluded.

**Forecasting.** Per city, year to annual generation. Models: Linear regression (baseline), Polynomial (degree 2) and a Naive last-value benchmark.
Cities need >= {C.MIN_YEARS} distinct observed years. Validation is rolling-origin on the last two points (training only on earlier years). MAE/RMSE are always
reported, R2 only with 3+ validation points. Reliability: *Unreliable* (did not beat naive, negative values, quadratic on <8 points, or extrapolation
> {C.MAX_EXTRAPOLATION} years), *Low* (fewer than 8 annual points), otherwise *Moderate*. Forecasts are clipped at zero. No prediction intervals are produced.

**Gap formulas.**
- Collected = generated x collection efficiency / 100. Collection gap (t) = max(0, generated - collected); gap % = gap / generated x 100.
- Estimated recycled = collected x recycling rate / 100 (default) or generated x recycling rate / 100 (alternative). Recycling gap defined analogously.
- These are rate-derived estimates, not documented tonnes. A gap is a statistical shortfall, not proof of illegal dumping.

**Priority score.** Four indicators are min-max normalised across eligible cities and combined with configurable weights (default
{C.DEFAULT_WEIGHTS['projected_generation']}/{C.DEFAULT_WEIGHTS['unaccounted_quantity']}/{C.DEFAULT_WEIGHTS['gap_percentage']}/{C.DEFAULT_WEIGHTS['infrastructure_deficiency']}):
projected generation, unaccounted quantity (projected generation x latest gap %), gap %, and an infrastructure deficiency index
(100 x (1 - mean percentile rank of bins, vehicles and workers per 100k people, landfill capacity remaining and municipal score)).
Cities missing any indicator are listed as unranked, never imputed. Sensitivity analysis re-ranks under 500 Dirichlet-perturbed weight sets.
This is a decision-support heuristic, not a scientifically validated measure.

**Limitations.** Sparse city histories (most cities have 2-5 annual points), monthly-sample annualisation, undefined rate denominators,
undocumented provenance, no uncertainty intervals, infrastructure fields that describe the whole city rather than e-waste, and historical admin names.
""")
