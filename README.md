# ASML Classification Project

## Overview

This repository contains a data analysis project aimed at predicting whether a bank customer would accept a personal loan offer. The goal is to optimize customer targeting for loan upselling using various machine learning models.
The dataset includes demographic and financial information for 5000 customers. After exploring several models, **Extreme Gradient Boosting (XGBoost)** was found to be the most effective, achieving a **98.1% accuracy** in predicting loan acceptance.


## Data Description

The dataset contains 5000 customer records with the following key features:
- **Demographics**: Age, Education, Family, ZIP code, etc.
- **Financials**: Income, CreditCard usage, Securities Account, Mortgage, etc.
- **Banking Behavior**: Online banking use, CD Account, and Personal Loan acceptance (target variable).

## Models Implemented

1. **Baseline Model**: Featureless classifier as a baseline for comparison.
2. **Logistic Regression**: Linear classification model for binary outcomes.
3. **Random Forest**: Ensemble model of decision trees with bootstrapped samples.
4. **Extreme Gradient Boosting (XGBoost)**: Boosting model to minimize residual errors iteratively, which achieved the best performance.
5. **Neural Networks**: Used with dropout layers and sigmoid activation for non-linear pattern recognition.

## Key Metrics
- **Accuracy**: Measures the overall correctness of predictions.
- **AUC (Area Under the Curve)**: Indicates the model's ability to distinguish between classes.
- **False Positive Rate (FPR)**: Percentage of incorrect positive predictions.
- **False Negative Rate (FNR)**: Percentage of missed positive predictions.

| Model               | Accuracy   | AUC    | FPR     | FNR     |
|---------------------|------------|--------|---------|---------|
| Baseline            | 90.1%      | 0.50   | 1.00    | 0.00    |
| Logistic Regression  | 94.7%      | 0.93   | 0.41    | 0.01    |
| Random Forest        | 97.8%      | 0.99   | 0.21    | 0.00    |
| **XGBoost**          | **98.1%**  | **0.99** | **0.24** | **0.00** |
| Neural Network       | 97.7%      | 0.96   | 0.04    | 0.00    |

## Hyperparameter Tuning

Hyperparameter tuning was performed for both Random Forest and XGBoost models to find the optimal number of trees, depth, and boosting rounds. Random Forest achieved optimal performance with a **tree depth of 10** and **20 trees**, while XGBoost continued to improve with more rounds but showed diminishing returns after a certain point.

- **R Libraries Used**:
  - `data.table`
  - `mlr3verse`
  - `skimr`
  - `dplyr`
  - `ggplot2`
  - `tensorflow`
  - `keras`
  

  
