# INDIA E-WASTE INTELLIGENCE
*Urban E-Waste Forecasting, Recycling Gap Assessment & City Priority Optimization*

Dark-mode-only Streamlit dashboard for Indian cities, built on the supplied `Waste_Management_India_20K.csv`.

## Run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Data files live in `data/` (`Waste_Management_India_20K.csv`, `cities_master.csv`). The theme is set in `.streamlit/config.toml` plus custom CSS in `src/viz/theme.py`.

## Pages
Executive Overview (India map, coverage, historical trends, states) · Forecast Studio · Recycling Gap Analysis · City Priority Ranking · City Comparison · Data Explorer · Methodology & Model Performance.
Sidebar filters (state/UT, city, year) apply to displayed data; forecasts always train on each city's full history. CSV exports: forecasts, gap results, priority ranking, city-year data, model metrics.

## What the data supports (verified by inspection)
- 25,203 rows, 42 columns, 130 cities, 2015-2026; 9 waste types. Only `Waste_Type = E-Waste` is used (594 rows; 12 dated after the current month are excluded as not yet observed).
- E-waste is sparse: 128 cities have data, 42 have >= 5 distinct years (needed to forecast). Chennai and Surat have none.
- Collection efficiency and recycling rate are **percentages, not documented tonnes**; the Recycling_Rate denominator is undefined. Gaps are therefore rate-derived estimates, with a selectable assumption.
- The dataset has no documented source. Treat results as illustrative.
- `Waste_Management_and_Recycling_India.csv` is not used (different schema; E-waste magnitudes ~40x different).

## Method (short)
- **Annualise:** tonnes/year = mean daily tonnes x 365 per city-year (monthly samples).
- **Forecast:** Linear (baseline), Polynomial (degree 2), Naive last-value; rolling-origin validation on the last 2 points; MAE/RMSE always, R2 only with 3+ points; reliability = Unreliable / Low / Moderate; clipped at 0; no intervals.
- **Gap:** collected = generated x collection %; recycled = collected x recycling % (or generated x recycling %); gap = max(0, generated - documented-estimate); gap % = gap / generated x 100.
- **Priority:** min-max normalised projected generation, unaccounted quantity, gap %, infrastructure deficiency; configurable weights (default 25/30/25/20); missing indicators -> unranked (never imputed); 500-run Dirichlet sensitivity. Decision-support only.

## Test
```bash
python tests/test_core.py        # or: pytest -q tests      (9 unit/integration tests, real dataset included)
python tests/smoke_ui.py         # executes every page with a Streamlit stand-in (no browser)
python -m src.run_pipeline       # writes sample outputs to outputs/
```
`smoke_ui.py` checks that page code runs and builds figures; it cannot judge visual appearance, so open the app and review each page.

## Layout
`app.py` · `src/data/` (loader, quality) · `src/models/forecast.py` · `src/analysis/` (gap, priority) · `src/viz/` (theme, charts) · `src/ui/` (components, pages) · `tests/` · `outputs/` (sample results).

## Limitations
Short city histories, annualisation from monthly samples, undefined rate denominators, undocumented provenance, no prediction intervals, whole-city (not e-waste-specific) infrastructure fields, and historical administrative names kept as supplied.
