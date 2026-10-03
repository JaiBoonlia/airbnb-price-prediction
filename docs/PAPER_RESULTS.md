# Results reported in the reference paper

Source: Y. Luo, X. Zhou, Y. Zhou, *Predicting Airbnb Listing Price Across Different
Cities*, Dec 14, 2019 (course project report). **These are the paper's numbers, not
numbers produced by this repository.** Run `python run_experiments.py --table all`
and compare your own `results/table*.csv` against them.

Legend: C = continuous, I = 2-degree interactions, O = categorical, T = text
(uni+bigram tf-idf), D = date. TH = price ≤ $500. Log/Sqrt = label transform.

## Table 1 – continuous features, NYC (TH)
| Model | Train MSE | Train R² | Test MSE | Test R² |
|---|---|---|---|---|
| Linear regression (L2) | 3514.64 | 0.489 | 3541.99 | 0.497 |
| K-nearest neighbor | 4466.30 | 0.351 | 4760.61 | 0.323 |
| Random forest | 1706.86 | 0.752 | 2427.05 | 0.655 |
| Neural network | 2059.43 | 0.702 | 2367.44 | 0.665 |
| XGBoost | 2273.98 | 0.669 | 2357.83 | 0.665 |

## Table 2 – XGBoost label transformations (NYC, continuous)
| Feature | Label | Train MSE | Train R² | Test MSE | Test R² |
|---|---|---|---|---|---|
| C | N/A | 16748.6 | 0.558 | 58003.1 | 0.195 |
| C | Sqrt | 7.17806 | 0.652 | 10.4435 | 0.537 |
| C | Log | 0.13905 | 0.695 | 0.14765 | 0.669 |
| C+I | Log | 0.13061 | 0.708 | 0.14704 | 0.676 |

## Table 3 – XGBoost text features (TH)
| Text feature | Train MSE | Train R² | Test MSE | Test R² |
|---|---|---|---|---|
| unigram count | 4127.11 | 0.400 | 4632.63 | 0.342 |
| unigram tf-idf | 3868.55 | 0.438 | 4337.49 | 0.384 |
| unigram & bigram tf-idf | 3771.55 | 0.452 | 4231.74 | 0.399 |

## Table 4 – XGBoost date / categorical / text (NYC)
| Feature | Label | Train MSE | Train R² | Test MSE | Test R² |
|---|---|---|---|---|---|
| D | TH | 6654.41 | 0.033 | 6927.05 | 0.016 |
| O | TH | 3134.42 | 0.546 | 3219.57 | 0.532 |
| C+O+T | TH | 1885.25 | 0.724 | 2078.78 | 0.706 |
| C+O+T | Log | 0.11383 | 0.749 | 0.11731 | 0.740 |

## Table 5 – XGBoost, Paris, C+O+T
| Label | Train MSE | Train R² | Test MSE | Test R² |
|---|---|---|---|---|
| TH | 1308.47 | 0.697 | 1413.89 | 0.691 |
| Log | 0.11114 | 0.722 | 0.11667 | 0.704 |

## Table 6 – neural network, individual vs combined (C+O, Log)
| Data | Train MSE | Train R² | Test MSE | Test R² |
|---|---|---|---|---|
| NYC | 0.10525 | 0.769 | 0.11723 | 0.741 |
| Paris | 0.09448 | 0.762 | 0.11277 | 0.716 |
| NYC and Paris | 0.08331 | 0.816 | 0.10357 | **0.773** |

## Table 7 – transfer to unseen Berlin (neural network, C+O, Log)
| Trained on | Train MSE | Train R² | Test MSE | Test R² |
|---|---|---|---|---|
| NYC only | 0.10653 | 0.765 | 1.78007 | -3.512 |
| Paris only | 0.09494 | 0.761 | 7.26637 | -17.42 |
| NYC and Paris | 0.10461 | 0.763 | 0.19163 | **0.514** |

## Inconsistencies inside the paper (be ready for these in Q&A)
- Sec 6.1 says the first NN used "4 fully connected layers (hidden size 128, 512, 64)" while Sec 5.4 describes 3 layers; Sec 6.2 (the final model) uses hidden sizes 64 and 32.
- The KNN prediction formula writes `1/n Σ_{i=1..n}` where it should be `1/k` over the k nearest neighbours.
- "NYC and Paris" train R² is 0.816 in Table 6 but 0.763 in Table 7 (different runs/feature sets are not explained).
- The paper says 28 continuous / 20 categorical features were obtained; the exact column lists are not given, so `config.py` uses a best-effort candidate list.
- Typos: "Aribnb", "Berline", "deposite".
