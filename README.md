# Deep Learning – Lab 1: MLPs with PyTorch
**Université Abdelmalek Essaadi – FSTT – Master MBD – Pr. Elaachak Lotfi**
**Student:** Souka

## 1. Objective
Get familiar with PyTorch by building DNN/MLP architectures for (i) a regression task
(NYSE stock prices) and (ii) a multi-class classification task (machine predictive maintenance).

## 2. Environment
Google Colab (GPU T4), Python 3.x, PyTorch, scikit-learn, skorch, imbalanced-learn. Seed = 42.

## 3. Part 1 – Regression (NYSE)
### 3.1 Data and EDA
`prices-split-adjusted.csv`, 851k rows, 501 stocks, 2010–2016, no missing values.
Observations: open/high/low/close correlation ≈ 1; volume is right-skewed;
prices span several orders of magnitude, so standardization is required.
### 3.2 Problem and preprocessing
Predict next-day close from open, high, low, close, volume, 1-day return, MA5, MA10 on 20 random stocks.
Chronological split (train < 2016, test = 2016), StandardScaler fitted on train only.
Persistence baseline: RMSE = XX, R² = XX.
### 3.3 Architecture and GridSearch
MLP (input 8 → [hidden] × [layers] ReLU → 1), loss MSE.
GridSearchCV (TimeSeriesSplit, 3 folds) over learning rate, optimizer, epochs, hidden size, depth.
Best: `XX`.
### 3.4 Results (baseline)
| Set | RMSE ($) | MAE ($) | R² |
|---|---|---|---|
| Train | XX | XX | XX |
| Test | XX | XX | XX |

![curves](figures/Regression_baseline.png)
Interpretation: (convergence speed, gap train/test, signs of over/underfitting).
### 3.5 Regularization
Dropout 0.2 + BatchNorm + AdamW (weight decay 1e-3) + early stopping (patience 10).
| Model | Test RMSE | Test R² | Train–test gap | Epochs |
|---|---|---|---|---|
| Baseline | XX | XX | XX | XX |
| Regularized | XX | XX | XX | XX |

Discussion: (did regularization help? why / why not? compare to the persistence baseline).

## 4. Part 2 – Multi-class classification (Predictive Maintenance)
### 4.1 Data, cleaning, EDA
10,000 machines, 5 numeric features + type. Removed 27 inconsistent rows, dropped IDs and `Target` (label leakage),
encoded Type (L/M/H → 0/1/2), engineered Temp_diff, Power, Strain.
Class distribution is extremely imbalanced (~97% "No Failure"). (Insert EDA figures and observations.)
### 4.2 Preprocessing and augmentation
Stratified 80/20 split → standardization (train statistics) → SMOTE applied **only to the training set**.
Class counts before: `XX` → after: `XX`.
### 4.3 Architecture and GridSearch
MLP (input → hidden × layers ReLU → logits), CrossEntropyLoss.
GridSearchCV with SMOTE inside each fold (imblearn pipeline), scoring = macro-F1. Best: `XX`.
### 4.4 Results (baseline)
| Metric | Train | Test |
|---|---|---|
| Accuracy | XX | XX |
| Sensitivity (macro recall) | XX | XX |
| Precision (macro) | XX | XX |
| F1 (macro) | XX | XX |

(Insert Loss/Accuracy curves, confusion matrix, per-class report and an interpretation.)
### 4.5 Regularization
Dropout 0.3 + BatchNorm + AdamW + early stopping.
| Model | Test Acc | Test Sensitivity | Test F1 (macro) | Gap |
|---|---|---|---|---|
| Baseline | XX | XX | XX | XX |
| Regularized | XX | XX | XX | XX |

Discussion: (effect on gap, on minority-class recall, on stability).

## 5. Limitations
- Regression: price prediction is close to a random walk; the persistence baseline is strong.
- Classification: very few real samples for some failure types; SMOTE creates synthetic points (fractional `Type`),
  and test metrics on tiny classes are noisy.
- Validation for early stopping in Part 2 comes from the augmented set.

## 6. Personal synthesis (what I learned)
In this lab I learned how to structure a complete PyTorch workflow: tensors and DataLoaders, a configurable MLP,
the training/evaluation loop, and how loss and metric curves reveal under/overfitting.
I learned that preprocessing choices matter as much as the model: scaling on the training set only,
time-based splits for time series, and augmenting only the training data to avoid leakage.
GridSearch through skorch showed how to tune hyper-parameters (learning rate, optimizer, epochs, depth, width),
and that the metric must match the problem: R² and RMSE for regression, macro-F1 and sensitivity for imbalanced classification
instead of plain accuracy. Finally, regularization (dropout, weight decay, batch norm, early stopping) reduced/did not reduce
the generalization gap depending on the task (replace with your own observation), which taught me that regularization
fixes overfitting but cannot create information that the data does not contain.

## 7. How to run
Open `Lab1_DL.ipynb` in Colab, enable GPU, Runtime → Run all. Datasets are downloaded with `kagglehub`.
