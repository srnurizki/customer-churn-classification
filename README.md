# Customer Churn Classification

**Bank customer churn prediction with rigorous missing-data analysis and SHAP-based interpretability.**

## Overview

This project predicts whether a bank customer will churn, with particular emphasis on principled handling of missing data and understanding *why* the model makes the predictions it does, not just how accurate it is.

## Missing Data Analysis

30.07% of rows contained "Unknown" values across three columns (`marital_status`, `education_level`, `income_category`). Rather than assuming these were missing at random, Chi-Square and Mann-Whitney U-tests were used to test the missingness mechanism, indicating the data was likely Missing Not At Random (MNAR). Based on documented evidence that MAR-based imputation on MNAR data can introduce substantial bias, missing values were retained as an explicit "Undisclosed" category instead of being imputed or dropped.

## Feature Engineering & Selection

- Correlation analysis via Phi-K (suited to mixed categorical/numeric data, unlike standard Pearson correlation)
- Feature selection via mutual information scoring, with multicollinear pairs resolved by keeping the higher-MI feature (e.g. `avg_open_to_buy` over `credit_limit`)
- Final feature set: 9 predictors

## Modeling

11 classification algorithms were benchmarked, narrowing to 3 candidates (SVM, Random Forest, XGBoost). Two imbalance-handling strategies were compared independently, class weight balancing and SMOTE resampling, each tuned via Optuna (150 trials).

## Results

| Approach | Best Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|---|
| Class weight balancing | Random Forest | 0.906 | — | 0.96 |
| SMOTE | XGBoost | 0.93 | 0.88 | 0.97 |

**XGBoost + SMOTE** was selected as the final model based on precision-recall performance and more stable learning curves (less overfitting than the weight-balanced Random Forest).

## Feature Importance

SHAP (TreeExplainer) was applied to the final model for both global and instance-level interpretation. `total_trans_amt` emerged as the most influential feature, consistent with the earlier mutual information ranking. Findings were translated into retention-relevant business insights tied to Customer Acquisition Cost (CAC), Retention Cost, and Customer Lifetime Value (CLTV).

## Tech Stack

| Layer | Technology |
|---|---|
| Modeling | scikit-learn, XGBoost |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Hyperparameter tuning | Optuna |
| Interpretability | SHAP |

## Running Locally

```bash
pip install -r requirements.txt
jupyter notebook
```

## Author

Satryo Akbar Nurizki - [GitHub](https://github.com/srnurizki)
