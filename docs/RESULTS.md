# Experimental Results

The results below are generated from `data/Airbnb_Data.csv.gz` by the repository code.

| Model | MAE (log price) | RMSE (log price) | R² |
|---|---:|---:|---:|
| Ridge | 0.2857 | 0.3967 | 0.6992 |

The dataset contains 74,111 listings. The split contains 59,288 training rows and 14,823 test rows, using city-stratified sampling with random seed 42.

To regenerate the result:

```bash
python run_experiments.py --models ridge
```
