import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src import config as C
from src.analysis import gap as G
from src.data import loader
from src.models import forecast as F
from src.ui import components as ui, pages
from src.viz import theme

st.set_page_config(page_title="India E-Waste Intelligence", layout="wide", initial_sidebar_state="expanded")
st.markdown(theme.CSS, unsafe_allow_html=True)

PAGES = {"Executive Overview": pages.overview, "Forecast Studio": pages.forecast_studio,
         "Recycling Gap Analysis": pages.gap_analysis, "City Priority Ranking": pages.priority,
         "City Comparison": pages.comparison, "Data Explorer": pages.data_explorer,
         "Methodology & Model Performance": pages.methodology}


@st.cache_data(show_spinner="Loading dataset...")
def get_dataset(day: str):
    return loader.build_dataset(as_of=pd.Timestamp(day))


@st.cache_data(show_spinner="Training models...")
def run_forecasts(cy, horizon, as_of_year, model):
    return F.forecast_all(cy, horizon, as_of_year, model)


def main():
    try:
        ds = get_dataset(str(pd.Timestamp.today().date()))
    except loader.DataError as e:
        st.error(str(e))
        st.stop()
    cy_all, as_of_year = ds["cityyear"], ds["as_of"].year

    st.sidebar.markdown('<div class="brand"><div class="brand-mark">E</div><div><div class="brand-name">INDIA E-WASTE<br>INTELLIGENCE</div>'
                        '<div class="brand-sub">Forecasting, gap and priority</div></div></div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed", key="nav")

    with st.sidebar:
        ui.label("Filters")
        states = st.multiselect("State / UT", sorted(cy_all["state"].unique()), key="f_states", placeholder="All states")
        pool = cy_all[cy_all["state"].isin(states)] if states else cy_all
        cities = st.multiselect("City", sorted(pool["city"].unique()), placeholder="All cities")
        ymin, ymax = int(cy_all["year"].min()), int(cy_all["year"].max())
        year_range = st.slider("Year", ymin, ymax, (ymin, ymax), key="f_years")
        ui.label("Forecast model")
        model = st.selectbox("Model", ["Auto", "Linear", "Polynomial"], key="f_model", label_visibility="collapsed",
                             help="Auto picks the lower validation RMSE of Linear and Polynomial for each city.")
        st.caption("Filters choose which cities and years are displayed. Forecasts always use each city's full history.")

    sel = pool[pool["city"].isin(cities)] if cities else pool
    city_list = sorted(sel["city"].unique())
    cy = cy_all[cy_all["city"].isin(city_list) & cy_all["year"].between(*year_range)]
    if cy.empty and page != "Methodology & Model Performance":
        ui.header(page, "")
        ui.empty("No observations match the current filters", " Widen the state, city or year selection in the sidebar.")
        st.stop()
    ctx = dict(ds=ds, cy_all=cy_all, cy=cy, cities=city_list, year_range=year_range, model=model, as_of_year=as_of_year,
               gap=G.compute_gap(cy) if not cy.empty else cy,
               forecasts=lambda h, m: run_forecasts(cy_all, h, as_of_year, m))
    PAGES[page](ctx)


main()
