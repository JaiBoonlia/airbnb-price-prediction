# Airbnb Price Prediction

A machine learning project that explores Airbnb listing data and predicts listing prices using regression models. The project covers data preprocessing, exploratory data analysis, feature engineering, model training, and performance evaluation.

## Project Overview

Airbnb listing prices depend on several factors, including location, room type, property characteristics, and availability. This project uses listing data from multiple cities to investigate these relationships and build a regression model for price prediction.

The main objectives are to:
- Explore the dataset and identify patterns in listing prices.
- Clean and preprocess the data for machine learning.
- Develop features suitable for regression.
- Train and evaluate regression models.
- Compare model performance using appropriate evaluation metrics.

## Dataset

The project uses `Airbnb_Data.csv.gz`, a compressed CSV dataset containing 74,111 Airbnb listings across six cities:

- New York City
- Los Angeles
- San Francisco
- Washington, D.C.
- Chicago
- Boston

The dataset includes listing information such as price, location, room type, and other property-related attributes.

The dataset is stored in the `data/` directory. See [`data/README.md`](data/README.md) for additional details.

## Project Structure

```text
airbnb-price-prediction/
├── data/
│   ├── Airbnb_Data.csv.gz
│   └── README.md
├── docs/
│   ├── figures/
│   ├── RESULTS.md
│   ├── make_writeup.py
│   └── writeup.pdf
├── results/
│   ├── model_comparison.csv
│   └── results.csv
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   └── models.py
├── config.py
├── requirements.txt
├── run_experiments.py
└── README.md
```

## Technologies Used

- Python
- pandas and NumPy for data processing
- scikit-learn for preprocessing, regression, and evaluation
- Matplotlib and Seaborn for data visualization

Refer to `requirements.txt` for the project dependencies.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/JaiBoonlia/airbnb-price-prediction.git
cd airbnb-price-prediction
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Run the experiment pipeline from the repository root:

```bash
python run_experiments.py
```

The script loads the dataset, applies the configured preprocessing and feature engineering steps, trains the configured models, and evaluates their predictions.

Generated results are stored in the `results/` directory. The exact models and metrics available depend on the experiment configuration and completed runs.

## Exploratory Data Analysis

The project includes visualizations that examine important properties of the dataset, including:

- Listing counts across cities.
- The distribution of log-transformed prices.
- Median listing prices by city.
- Relationships between selected numerical variables.

These figures are available in `docs/figures/`.

## Model Evaluation

Regression performance is assessed using metrics such as:

- **R²:** Measures the proportion of target variance explained by the model.
- **Root Mean Squared Error (RMSE):** Measures the typical prediction error while penalizing larger errors more heavily.
- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.

The experiment outputs are recorded in `results/results.csv` and `results/model_comparison.csv`. Additional discussion is available in [`docs/RESULTS.md`](docs/RESULTS.md).

Metrics must be interpreted according to the scale of the target variable. When the target is log-transformed, errors reported on that scale should not be interpreted directly as errors in currency units.

## Results and Reproducibility

The results files contain the metrics recorded by the project's completed experiments. The Ridge regression baseline is documented in the current results.

For reproducibility, install the listed dependencies, use the provided dataset, and run the experiment pipeline. Results can vary depending on the model configuration, preprocessing choices, random seed, and library versions.

Only models that have actually been trained and evaluated should be included in performance comparisons.

## Limitations

- Listing prices can vary with seasonality, demand, and market conditions that may not be fully represented in the dataset.
- The dataset reflects the period and locations from which it was collected; results may not generalize to other markets or time periods.
- Observational relationships do not establish that a particular listing feature causes a price change.
- Prediction quality depends on data quality, feature selection, and model configuration.

## Future Improvements

Potential extensions include evaluating additional regression algorithms, improving feature selection, tuning hyperparameters, analyzing prediction errors by city, and testing the model on separate or more recent data.

## Authors

Developed as a team mini-project for the Machine Learning course at PES University.

## License

This repository was created for academic project purposes.