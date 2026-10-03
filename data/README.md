# Data

The CSVs are **not** committed (they are large). Download the three Kaggle listing
files (the ones with 96 raw columns, as used in the paper) and save them as:

| City   | Expected path     | Listings in paper |
|--------|-------------------|-------------------|
| NYC    | `data/nyc.csv`    | 44,317            |
| Paris  | `data/paris.csv`  | 59,881            |
| Berlin | `data/berlin.csv` | 22,552            |

Kaggle sources cited by the paper: "Airbnb Open Data in NYC", "Airbnb Paris",
"Berlin Airbnb Data" (all 2019). Use the detailed `listings` table
(not the summary one). Row counts may differ slightly from the paper if you
download a different snapshot.
