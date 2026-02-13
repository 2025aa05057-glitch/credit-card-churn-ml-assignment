import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Credit Card Churn Prediction - ML Models Comparison")
st.write("This app demonstrates 6 different classification models on credit card churn data")

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest', 'XGBoost']
)

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write(df.head())
    
    # Display model performance (you'll need to load your trained models)
    st.write(f"### {model_choice} Performance")
    
    # Load pre-computed results (you'll create this)
    results = {
        'Logistic Regression': {'Accuracy': 0.85, 'AUC': 0.82, 'Precision': 0.84, 'Recall': 0.85, 'F1': 0.84, 'MCC': 0.65},
        'Decision Tree': {'Accuracy': 0.82, 'AUC': 0.79, 'Precision': 0.81, 'Recall': 0.82, 'F1': 0.81, 'MCC': 0.60},
        # Add other models...
    }
    
    if model_choice in results:
        metrics_df = pd.DataFrame([results[model_choice]])
        st.table(metrics_df)

# Display comparison table
st.write("### Model Comparison Table")
comparison_data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest', 'XGBoost'],
    'Accuracy': [0.85, 0.82, 0.83, 0.80, 0.87, 0.89],
    'AUC': [0.82, 0.79, 0.80, 0.77, 0.85, 0.87],
    'Precision': [0.84, 0.81, 0.82, 0.79, 0.86, 0.88],
    'Recall': [0.85, 0.82, 0.83, 0.80, 0.87, 0.89],
    'F1': [0.84, 0.81, 0.82, 0.79, 0.86, 0.88],
    'MCC': [0.65, 0.60, 0.61, 0.57, 0.69, 0.72]
}

comparison_df = pd.DataFrame(comparison_data)
st.table(comparison_df)
