
"""Project configuration for the Airbnb price prediction experiment."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Airbnb_Data.csv.gz"
RESULTS_DIR = ROOT / "results"

SEED = 42
TEST_SIZE = 0.20

# Random Forest settings
RF_N_ESTIMATORS = 250
RF_MIN_SAMPLES_LEAF = 2

# XGBoost settings
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 8
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE = 0.85

# Numerical features
NUMERIC_COLUMNS = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "number_of_reviews",
    "review_scores_rating",
    "latitude",
    "longitude",
]

# Categorical features
CATEGORICAL_COLUMNS = [
    "property_type",
    "room_type",
    "bed_type",
    "cancellation_policy",
    "city",
    "neighbourhood",
    "zipcode",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
    "cleaning_fee",
]

# Text and list features
TEXT_COLUMNS = ["name", "description"]
LIST_COLUMNS = ["amenities"]
