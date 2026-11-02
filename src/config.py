from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MAIN_CSV = DATA_DIR / "Waste_Management_India_20K.csv"
MASTER_CSV = DATA_DIR / "cities_master.csv"

TARGET_WASTE_TYPE = "E-Waste"
DAYS_PER_YEAR = 365
MIN_YEARS = 5          # distinct annual observations needed to forecast a city
MAX_HORIZON = 5        # maximum forecast horizon (years after the current year)
MAX_EXTRAPOLATION = 5  # flag forecasts extending further than this beyond a city's last observation

REQUIRED_COLUMNS = ["City", "State", "Region", "City_Category", "Year", "Month", "Population", "Waste_Type",
                    "Daily_Waste_Generation_Tons", "Collection_Efficiency_Percentage", "Recycling_Rate",
                    "Latitude", "Longitude"]
INFRA_COLUMNS = ["Smart_Bins_Count", "Waste_Collection_Vehicles", "Worker_Count",
                 "Landfill_Remaining_Capacity", "Municipal_Efficiency_Score", "Waste_Management_Budget_INR"]

DEFAULT_WEIGHTS = {"projected_generation": 25, "unaccounted_quantity": 30,
                   "gap_percentage": 25, "infrastructure_deficiency": 20}
