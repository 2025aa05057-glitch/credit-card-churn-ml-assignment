# Credit Card Churn Prediction - ML Classification Models

## Problem Statement
This project implements and compares 6 different machine learning classification models to predict credit card customer churn using an interactive Streamlit web application.

## Dataset Description
- **Source**: Credit Card Churn Dataset
- **Size**: 10,000+ instances with 20+ features
- **Target Variable**: Attrition_Flag (Existing Customer vs Attrited Customer)
- **Features**: Customer demographics, account information, and transaction behavior

## Models Implemented
All models are saved as pickle files and can process uploaded CSV data:

| Model | File | Description |
|-------|------|-------------|
| Logistic Regression | logistic_regression_model.pkl | Linear classification with regularization |
| Decision Tree | decision_tree_model.pkl | Tree-based interpretable model |
| KNN | knn_model.pkl | Instance-based learning algorithm |
| Naive Bayes | naive_bayes_model.pkl | Probabilistic classifier |
| Random Forest | random_forest_model.pkl | Ensemble of decision trees |
| XGBoost | xgboost_model.pkl | Gradient boosting ensemble |

## Models Performance
| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.846 | 0.5491 | 0.7157 | 0.846 | 0.7754 | 0 |
| Decision Tree | 0.72 | 0.5158 | 0.747 | 0.72 | 0.7327 | 0.0289 |
| KNN | 0.826 | 0.5081 | 0.7396 | 0.826 | 0.7726 | 0.0006 |
| Naive Bayes | 0.846 | 0.5466 | 0.7157 | 0.846 | 0.7754 | 0 |
| Random Forest | 0.846 | 0.5324 | 0.7157 | 0.846 | 0.7754 | 0 |
| XGBoost | 0.84 | 0.5134 | 0.7415 | 0.84 | 0.7751 | 0.0033 |

## Model Performance Observations
| Model | Observations |
|-------|--------------|
| Logistic Regression | Good baseline performance with interpretable coefficients |
| Decision Tree | Interpretable model but may overfit, good for feature importance |
| KNN | Instance-based learning, performance depends on neighborhood size |
| Naive Bayes | Fast and simple, assumes feature independence |
| Random Forest | Ensemble method, usually robust and handles overfitting well |
| XGBoost | Advanced ensemble method, often achieves highest performance |

## Features
- **CSV Upload**: Upload customer data for churn prediction
- **Model Selection**: Choose from 6 different ML models
- **Real-time Predictions**: Get instant churn predictions
- **Evaluation Metrics**: View accuracy, AUC, precision, recall, F1, MCC
- **Visualization**: Confusion matrix and performance charts
- **Download Results**: Export predictions as CSV

## How to Use
1. Select a model from the dropdown
2. Upload a CSV file with customer data
3. View predictions and metrics
4. Download results

## Repository Structure
- app.py
- requirements.txt
- README.md
- decision_tree_model.pkl
- knn_model.pkl
- logistic_regression_model.pkl
- model_performance.pkl
- naive_bayes_model.pkl
- preprocessing_components.pkl
- random_forest_model.pkl
- xgboost_model.pkl
- model/
- model/Credit_Card_Churn.csv
- model/2025AA05057_ML_Assignment_2.ipynb
