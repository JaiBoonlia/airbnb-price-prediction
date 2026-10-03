# Predicting Airbnb Listing Price Across Different Cities

UE24CS352A – Machine Learning · Mini-Project

**Team:** Jai Boonlia – PES1UG24CS197, Kanishka Sarraf – PES1UG24CS215 · **Section:** D · **Problem #:** 71

A reimplementation of *"Predicting Airbnb Listing Price Across Different Cities"*
(Luo, Zhou & Zhou, 2019). We predict nightly Airbnb prices for New York City,
Paris and Berlin with Ridge/Linear regression, KNN, Random Forest, XGBoost and a
small neural network, and test whether a network trained on NYC + Paris transfers
to an unseen city (Berlin).

## Repository layout
```
config.py              all paths and hyper-parameters (paper values)
run_experiments.py     reproduces Tables 1-7 of the paper
src/data.py            loading, $-parsing, $500 threshold, 7:2:1 split, label transforms
src/features.py        continuous / interaction / categorical / text (tf-idf+SVD) / date features
src/models.py          Ridge, KNN, RF, XGBoost, PyTorch MLP (64-32, SGD+Nesterov, early stopping)
data/                  put the Kaggle CSVs here (see data/README.md)
results/               generated CSVs and plots
docs/PAPER_RESULTS.md  numbers reported in the paper, for comparison
docs/writeup.pdf       project write-up
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Download the three Kaggle datasets and save them as `data/nyc.csv`,
`data/paris.csv`, `data/berlin.csv` (details in `data/README.md`).

## Run
```bash
python run_experiments.py --table all     # everything (slow: text features + RF)
python run_experiments.py --table 6       # one table, e.g. NN individual vs combined
python run_experiments.py --table 7       # transfer to Berlin (+ scatter plots in results/)
```
| Table | Experiment |
|---|---|
| 1 | 5 models on continuous features, NYC, price ≤ $500 |
| 2 | XGBoost with none / sqrt / log label (and + interaction terms) |
| 3 | Text features: unigram count vs tf-idf vs uni+bigram tf-idf |
| 4 | Date, categorical, combined C+O+T features |
| 5 | Same XGBoost setting on Paris |
| 6 | Neural net on NYC, Paris, NYC+Paris |
| 7 | Neural net trained on NYC / Paris / NYC+Paris, tested on Berlin |

Each run writes `results/tableN.csv` (train/test MSE and R²). MSE/R² for log and
sqrt labels are computed in the transformed space, as in the paper.

## Method summary
- **Label:** drop price > $500 (≈1 % of listings) *or* transform with log / sqrt.
- **Continuous:** drop highly-correlated columns, 0-fill security deposit & cleaning fee,
  mean-fill the rest, standardise; optional degree-2 interaction terms.
- **Categorical:** one-hot encoding; `amenities` / `host_verifications` via dictionary multi-hot.
- **Text:** tf-idf (uni+bigrams) → truncated SVD to 50 dims per text column.
- **Date:** null → mean date, converted to days since earliest date.
- **Neural net (final):** 2 hidden layers (64, 32), ReLU, MSE loss, SGD + Nesterov
  (lr 0.005, momentum 0.9), L2 0.005, early stopping on 10 % of train
  (Δ val-R² < 1e-4 for 10 epochs).
- **Split:** 7 : 2 : 1 train / validation / test per city; all preprocessing is fitted on train only.

## Notes and assumptions
- The paper does not list exact column names, the correlation threshold, the
  random-forest size or the thresholding choice for the log-label runs. We use
  `CORR_THRESHOLD = 0.8`, sklearn defaults for RF/XGBoost, and **no** $500 cap when a
  log label is used (the paper presents the cap and the transform as alternatives).
  All of these are in `config.py` / `run_experiments.py`.
- Our numbers will differ somewhat from the paper's (different snapshot, split seed,
  feature list). The paper's reported values are in `docs/PAPER_RESULTS.md`.

## Reference
Y. Luo, X. Zhou, Y. Zhou. *Predicting Airbnb Listing Price Across Different Cities.* 2019.
