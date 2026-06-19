# Customer Churn Prediction

A machine learning pipeline that predicts customer churn on the IBM Telco Customer Churn dataset, combining a stacked ensemble of gradient boosting models with SHAP-based explainability and a leakage-free decision threshold tuning process.

## Project Overview

Customer churn is a major problem for subscription-based businesses such as telecom providers. This project builds an end-to-end pipeline to identify customers who are likely to churn, so that retention strategies can be applied proactively.

**Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features (demographics, account information, services subscribed).

## Methodology

- **Exploratory Data Analysis (EDA):** distribution analysis, correlation checks, churn-driver identification
- **Preprocessing:** Yeo-Johnson power transformation for skewed numerical features, `ColumnTransformer` pipeline for mixed numerical/categorical data, K-Means cluster labels injected as an additional feature
- **Modeling:** Stacking ensemble with XGBoost, LightGBM, and CatBoost as base learners and Logistic Regression as the meta-learner
- **Hyperparameter Optimization:** Optuna-based Bayesian optimization across all base learners and the meta-learner
- **Threshold Tuning:** Out-of-fold (OOF) predictions used to select a leakage-free decision threshold targeting a churn recall of at least 70%
- **Explainability:** SHAP values to interpret feature contributions to individual and global predictions

## Results

Final leakage-free pipeline performance on the held-out test set (decision threshold = 0.59):

| Metric | Score |
|---|---|
| Accuracy | 78.14% |
| Macro F1-Score | 74.02% |
| Churn Precision | 56.96% |
| Churn Recall | 72.19% |

The model is tuned to prioritize **recall on the churn class**, since in a real business setting the cost of missing a customer who churns is typically higher than the cost of a false alarm.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · SHAP · Streamlit

## Repository Contents

| File | Description |
|---|---|
| `churn_prediction_model.ipynb` | Full notebook: EDA, preprocessing, modeling, tuning, evaluation, SHAP analysis |
| `demo_app.py` | Interactive Streamlit demo for live churn prediction |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset used for training and evaluation |

## Running the Demo

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm catboost
streamlit run demo_app.py
```

## Live Demo

![App Screenshot](images/app_screenshot.png)

## Live Demo

Try the app here:
https://customer-churn-prediction-telco-8sbzcddyv5txzj4u5uyftx.streamlit.app/
## Authors

- Rabia Geyik
- Büşra Sarı

Developed as part of a Big Data Mining graduate course project at Gebze Technical University (GTÜ).
