
# Import system and file path handling libraries
import sys
from pathlib import Path

# Add the current project directory to Python's module search path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import Pandas for data processing and Streamlit for web application development
import pandas as pd
import streamlit as st

# Import project configuration settings
from src import config as C

# Import recycling gap analysis module
from src.analysis import gap as G

# Import dataset loading and preprocessing module
from src.data import loader

# Import forecasting models and prediction functions
from src.models import forecast as F

# Import reusable UI components and dashboard pages
from src.ui import components as ui, pages

# Import custom visualization theme and CSS styling
from src.viz import theme

# Configure the Streamlit page title, layout and initial sidebar state
st.set_page_config(page_title="India E-Waste Intelligence", layout="wide", initial_sidebar_state="expanded")

# Apply custom CSS styling to the entire application
st.markdown(theme.CSS, unsafe_allow_html=True)

# Define the seven dashboard pages and connect them to their respective functions
PAGES = {"Executive Overview": pages.overview, "Forecast Studio": pages.forecast_studio,
         "Recycling Gap Analysis": pages.gap_analysis, "City Priority Ranking": pages.priority,
         "City Comparison": pages.comparison, "Data Explorer": pages.data_explorer,
         "Methodology & Model Performance": pages.methodology}


# Cache the dataset to avoid repeated loading and processing
@st.cache_data(show_spinner="Loading dataset...")
def get_dataset(day: str):
    # Load and prepare the dataset using the specified date
    return loader.build_dataset(as_of=pd.Timestamp(day))


# Cache forecasting results to improve application performance
@st.cache_data(show_spinner="Training models...")
def run_forecasts(cy, horizon, as_of_year, model):
    # Generate future forecasts using the selected model and historical data
    return F.forecast_all(cy, horizon, as_of_year, model)


# Define the main function that controls the complete application
def main():
    # Load the prepared dataset using the current date
    try:
        ds = get_dataset(str(pd.Timestamp.today().date()))

    # Handle errors that occur during dataset loading
    except loader.DataError as e:
        # Display the dataset loading error
        st.error(str(e))
        # Stop application execution if dataset loading fails
        st.stop()

    # Extract complete city-year data and the latest reference year
    cy_all, as_of_year = ds["cityyear"], ds["as_of"].year

    # Display application logo, name and description in the sidebar
    st.sidebar.markdown('<div class="brand"><div class="brand-mark">E</div><div><div class="brand-name">INDIA E-WASTE<br>INTELLIGENCE</div>'
                        '<div class="brand-sub">Forecasting, gap and priority</div></div></div>', unsafe_allow_html=True)

    # Create sidebar navigation to switch between dashboard pages
    page = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed", key="nav")

    # Create sidebar filters for the dashboard
    with st.sidebar:
        # Display the filter section heading
        ui.label("Filters")

        # Allow users to select multiple states or Union Territories
        states = st.multiselect("State / UT", sorted(cy_all["state"].unique()), key="f_states", placeholder="All states")

        # Filter available city records according to selected states
        # If no states are selected, use all available records
        pool = cy_all[cy_all["state"].isin(states)] if states else cy_all

        # Allow users to select multiple cities from the filtered state list
        cities = st.multiselect("City", sorted(pool["city"].unique()), placeholder="All cities")

        # Identify the minimum and maximum years available in the dataset
        ymin, ymax = int(cy_all["year"].min()), int(cy_all["year"].max())

        # Create a slider to select the required year range
        year_range = st.slider("Year", ymin, ymax, (ymin, ymax), key="f_years")

        # Display the forecasting model selection heading
        ui.label("Forecast model")

        # Allow users to choose Auto, Linear Regression or Polynomial Regression
        model = st.selectbox("Model", ["Auto", "Linear", "Polynomial"], key="f_model", label_visibility="collapsed",
                             help="Auto picks the lower validation RMSE of Linear and Polynomial for each city.")

        # Explain that dashboard filters do not restrict historical forecasting data
        st.caption("Filters choose which cities and years are displayed. Forecasts always use each city's full history.")

    # Apply the selected city filter; otherwise, include all cities in the selected states
    sel = pool[pool["city"].isin(cities)] if cities else pool

    # Create a sorted list of selected cities
    city_list = sorted(sel["city"].unique())

    # Filter city-year records using selected cities and year range
    cy = cy_all[cy_all["city"].isin(city_list) & cy_all["year"].between(*year_range)]

    # Check whether the selected filters return any observations
    # Allow the methodology page to open even if the filtered dataset is empty
    if cy.empty and page != "Methodology & Model Performance":
        # Display the selected dashboard page heading
        ui.header(page, "")
        # Show a message asking users to widen their filters
        ui.empty("No observations match the current filters", " Widen the state, city or year selection in the sidebar.")
        # Stop execution when no matching records are available
        st.stop()

    # Create a context dictionary containing the data and settings required by dashboard pages
    ctx = dict(ds=ds, cy_all=cy_all, cy=cy, cities=city_list, year_range=year_range, model=model, as_of_year=as_of_year,
               # Calculate the recycling gap using the filtered dataset
               gap=G.compute_gap(cy) if not cy.empty else cy,
               # Provide a function to generate forecasts for the requested horizon and model
               forecasts=lambda h, m: run_forecasts(cy_all, h, as_of_year, m))

    # Execute the selected dashboard page and pass the context dictionary
    PAGES[page](ctx)


# Call the main function to start the Streamlit application
main()
