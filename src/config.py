"""Central configuration: paths, column groups and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SUBMISSIONS_DIR = ROOT / "submissions"

RANDOM_STATE = 42

# --- Schema -----------------------------------------------------------------
ID_COL = "TransactionID"
TARGET = "TargetValue"
DATE_COL = "TransactionDate"

# Nearly empty columns (>90% missing) that carry no usable signal.
EMPTY_COLS = ["col18", "col19", "col5", "col4"]

# ManufactureYear contains a sentinel value (~10.6% of rows) that is not a real year.
MANUFACTURE_YEAR_SENTINEL = 1001

# --- Feature groups ---------------------------------------------------------
NUMERIC_FEATURES = [
    "Age",
    "OperationalHoursMeter_log",
    "sale_year",
    "sale_month",
    "sale_elapsed",
]
LOW_CARD_CAT = [
    "InventoryGroupCategory",
    "UtilizationTier",
    "DataOriginCode",
    "RegionCode",
    "AssetScaleFactor",
    "CabinType",
    "DrivetrainType",
]
HIGH_CARD_CAT = [
    "ProductConfigID",
    "Spec_BaseClass",
    "Spec_SubClass",
    "Spec_ReleaseSeries",
    "Spec_VariantModifier",
    "FunctionalClassification",
]
CAT_FEATURES = LOW_CARD_CAT + HIGH_CARD_CAT

# --- Validation / outlier settings -----------------------------------------
# Chronological split: the most recent 15% of transactions are held out.
EVAL_FRACTION_CUTOFF = 0.85
# Hours above this quantile of the *training* data are treated as sensor/entry errors.
HOURS_CAP_QUANTILE = 0.995
MAX_ASSET_AGE = 100
