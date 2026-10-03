"""Central configuration. Values follow Luo, Zhou & Zhou (2019) unless noted."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DATASETS = {
    "nyc": DATA_DIR / "nyc.csv",
    "paris": DATA_DIR / "paris.csv",
    "berlin": DATA_DIR / "berlin.csv",
}

SEED = 42
PRICE_CAP = 500.0                 # "data thresholding": drop price > $500 (~1% of listings)
SPLIT = (0.7, 0.2, 0.1)           # train : validation : test
CORR_THRESHOLD = 0.8              # drop one of each highly-correlated continuous pair (assumed value)
SVD_DIM = 50                      # per text column (12 columns -> 600 dims)

# Neural network (Section 6.2 of the paper)
NN_HIDDEN = (64, 32)
NN_LR = 0.005
NN_MOMENTUM = 0.9                 # SGD + Nesterov
NN_L2 = 0.005
NN_BATCH = 64
NN_MAX_EPOCHS = 200
NN_ES_VAL_FRAC = 0.10             # 10% of training data for early stopping
NN_ES_TOL = 1e-4                  # stop if change in val R^2 < tol ...
NN_ES_PATIENCE = 10               # ... for 10 consecutive iterations

# Columns in the Inside-Airbnb style listings table. Only those present in a
# given CSV are used, so the pipeline works on all three cities.
PRICE_FILL_ZERO = ["security_deposit", "cleaning_fee"]
MONEY_COLUMNS = ["price", "security_deposit", "cleaning_fee", "extra_people"]

CONTINUOUS = [
    "accommodates", "bathrooms", "bedrooms", "beds", "guests_included",
    "security_deposit", "cleaning_fee", "extra_people",
    "minimum_nights", "maximum_nights",
    "minimum_minimum_nights", "maximum_minimum_nights", "minimum_nights_avg_ntm",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
    "latitude", "longitude",
    "host_listings_count", "host_total_listings_count",
    "calculated_host_listings_count",
]

CATEGORICAL = [
    "property_type", "room_type", "bed_type", "cancellation_policy",
    "neighbourhood_cleansed", "neighbourhood_group_cleansed", "city", "zipcode",
    "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
    "is_location_exact", "instant_bookable", "require_guest_profile_picture",
    "require_guest_phone_verification", "is_business_travel_ready",
    "host_response_time", "requires_license",
]
LIST_COLUMNS = ["amenities", "host_verifications"]
LIST_MIN_FREQ = 0.01              # keep list tokens present in >=1% of training listings

TEXT_COLUMNS = [
    "name", "summary", "space", "description", "neighborhood_overview",
    "notes", "transit", "access", "interaction", "house_rules",
    "host_about", "host_neighbourhood",
]
DATE_COLUMNS = ["host_since", "first_review", "last_review"]
