import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Credit Card Churn Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Churn Prediction - ML Models Comparison")
st.write("Upload your credit card customer data to predict churn using 6 different ML models")

# Load preprocessing components
@st.cache_resource
def load_preprocessing_components():
    try:
        with open('preprocessing_components.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Preprocessing components not found. Please ensure all model files are uploaded.")
        return None

# Load model performance
@st.cache_resource
def load_model_performance():
    try:
        with open('model_performance.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Load individual model
@st.cache_resource
def load_model(model_name):
    model_file = f'{model_name.replace(" ", "_").lower()}_model.pkl'
    try:
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file {model_file} not found.")
        return None

def preprocess_uploaded_data(df, preprocessing_components):
    """Preprocess uploaded data using saved components"""
    data = df.copy()
    
    # Remove ID column if exists
    if 'CLIENTNUM' in data.columns:
        data = data.drop('CLIENTNUM', axis=1)
    
    # Handle target variable if exists
    if 'Attrition_Flag' in data.columns:
        target_encoder = preprocessing_components['target_encoder']
        data['Attrition_Flag'] = target_encoder.transform(data['Attrition_Flag'])
        has_target = True
    else:
        has_target = False
    
    # Handle categorical variables
    feature_encoders = preprocessing_components['feature_encoders']
    for col, encoder in feature_encoders.items():
        if col in data.columns:
            try:
                data[col] = encoder.transform(data[col].astype(str))
            except ValueError as e:
                st.warning(f"Unknown category in column {col}: {e}")
                # Handle unknown categories by using most frequent class
                data[col] = encoder.transform([encoder.classes_[0]] * len(data))
    
    return data, has_target

def make_predictions(model, data, preprocessing_components, model_name):
    """Make predictions using the loaded model"""
    # Prepare features (excluding target if present)
    if 'Attrition_Flag' in data.columns:
        X = data.drop('Attrition_Flag', axis=1)
        y_true = data['Attrition_Flag']
    else:
        X = data
        y_true = None
    
    # Apply scaling if required
    if model_name in preprocessing_components['models_requiring_scaling']:
        scaler = preprocessing_components['scaler']
        X_processed = scaler.transform(X)
    else:
        X_processed = X
    
    # Make predictions
    y_pred = model.predict(X_processed)
    y_pred_proba = model.predict_proba(X_processed)
    
    return y_pred, y_pred_proba, y_true

# Sidebar
st.sidebar.header("Model Selection")
model_options = [
    'Logistic Regression',
    'Decision Tree', 
    'KNN',
    'Naive Bayes',
    'Random Forest',
    'XGBoost'
]

selected_model = st.sidebar.selectbox("Choose a model:", model_options)

# File upload
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

# Load components
preprocessing_components = load_preprocessing_components()
model_performance = load_model_performance()

if preprocessing_components is None:
    st.error("Please ensure all required files are present in the repository.")
    st.stop()

# Main content
if uploaded_file is not None:
    try:
        # Load uploaded data
        df = pd.read_csv(uploaded_file)
        
        st.write("### Dataset Overview")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        
        with st.expander("View Dataset Sample"):
            st.dataframe(df.head(10))
        
        # Preprocess data
        processed_data, has_target = preprocess_uploaded_data(df, preprocessing_components)
        
        # Load selected model
        model = load_model(selected_model)
        
        if model is not None:
            st.write(f"### {selected_model} - Analysis Results")
            
            # Make predictions
            y_pred, y_pred_proba, y_true = make_predictions(
                model, processed_data, preprocessing_components, selected_model
            )
            
            # Display predictions
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("#### Prediction Distribution")
                pred_counts = pd.Series(y_pred).value_counts()
                pred_labels = ['Existing Customer', 'Churn Customer']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.bar(pred_labels, [pred_counts.get(0, 0), pred_counts.get(1, 0)])
                ax.set_title(f'{selected_model} - Churn Predictions')
                ax.set_ylabel('Number of Customers')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
                
                st.pyplot(fig)
            
            with col2:
                st.write("#### Prediction Confidence")
                confidence_scores = np.max(y_pred_proba, axis=1)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(confidence_scores, bins=20, alpha=0.7, color='skyblue')
                ax.set_title('Prediction Confidence Distribution')
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            
            # If ground truth is available, show evaluation metrics
            if has_target and y_true is not None:
                st.write("#### Evaluation Metrics")
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                precision = precision_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')
                f1 = f1_score(y_true, y_pred, average='weighted')
                mcc = matthews_corrcoef(y_true, y_pred)
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    st.metric("AUC Score", f"{auc:.4f}")
                
                with col2:
                    st.metric("Precision", f"{precision:.4f}")
                    st.metric("Recall", f"{recall:.4f}")
                
                with col3:
                    st.metric("F1 Score", f"{f1:.4f}")
                    st.metric("MCC", f"{mcc:.4f}")
                
                # Confusion Matrix
                st.write("#### Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Existing', 'Churn'],
                           yticklabels=['Existing', 'Churn'], ax=ax)
                ax.set_title(f'{selected_model} - Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                
                # Classification Report
                st.write("#### Classification Report")
                class_report = classification_report(y_true, y_pred, 
                                                   target_names=['Existing Customer', 'Churn Customer'],
                                                   output_dict=True)
                st.dataframe(pd.DataFrame(class_report).transpose())
            
            # Show detailed predictions
            st.write("#### Detailed Predictions")
            result_df = df.copy()
            result_df['Predicted_Churn'] = ['Churn Customer' if pred == 1 else 'Existing Customer' for pred in y_pred]
            result_df['Churn_Probability'] = y_pred_proba[:, 1]
            
            # Sort by churn probability
            result_df = result_df.sort_values('Churn_Probability', ascending=False)
            
            st.dataframe(result_df[['Predicted_Churn', 'Churn_Probability'] + df.columns.tolist()[:5]])
            
            # Download predictions
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f'{selected_model}_predictions.csv',
                mime='text/csv'
            )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please ensure your CSV file has the same structure as the training data.")

# Model Comparison Section
st.write("## 📊 Model Performance Comparison")

if model_performance:
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(model_performance).T
    comparison_df = comparison_df.round(4)
    
    st.write("### Performance Metrics Table")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualization
    st.write("### Performance Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(model_performance.keys())
        accuracies = [model_performance[model]['Accuracy'] for model in models]
        
        bars = ax.bar(models, accuracies, color='lightblue')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2., acc + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        # AUC comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        auc_scores = [model_performance[model]['AUC'] for model in models]
        
        bars = ax.bar(models, auc_scores, color='lightgreen')
        ax.set_title('Model AUC Score Comparison')
        ax.set_ylabel('AUC Score')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, auc in zip(bars, auc_scores):
            ax.text(bar.get_x() + bar.get_width()/2., auc + 0.01,
                   f'{auc:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)

else:
    st.write("Model performance data not available. Please ensure model_performance.pkl is in the repository.")

# Instructions
with st.write("### Instructions")
    st.write("""
    **How to use this app:**
    
    1. **Select a Model:** Choose from 6 different ML models in the sidebar
    2. **Upload Data:** Upload a CSV file with the same structure as the training data
    3. **View Results:** 
       - See prediction distribution and confidence scores
       - If ground truth is available, view evaluation metrics
       - Download predictions as CSV
    4. **Compare Models:** Check the performance comparison section below
    
    **Required CSV Format:**
    - Same columns as the original Credit Card Churn dataset
    - Include 'Attrition_Flag' column for evaluation (optional)
    - Categorical columns will be automatically encoded
    """)
