"""Project configuration for the Airbnb price prediction experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Airbnb_Data.csv.gz"
RESULTS_DIR = ROOT / "results"
SEED = 42
TEST_SIZE = 0.20

# Model settings selected for this dataset and kept fixed for reproducibility.
RIDGE_ALPHA = 1.0
RF_N_ESTIMATORS = 250
RF_MIN_SAMPLES_LEAF = 2
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 8
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE = 0.85

NUMERIC_COLUMNS = [
    "accommodates", "bathrooms", "bedrooms", "beds", "number_of_reviews",
    "review_scores_rating", "latitude", "longitude",
]
CATEGORICAL_COLUMNS = [
    "property_type", "room_type", "bed_type", "cancellation_policy", "city",
    "neighbourhood", "zipcode", "host_has_profile_pic",
    "host_identity_verified", "instant_bookable", "cleaning_fee",
]
TEXT_COLUMNS = ["name", "description"]
LIST_COLUMNS = ["amenities"]
