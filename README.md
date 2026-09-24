# DiabeticLens

## Explainable Machine Learning for Diabetes Prediction

DiabeticLens is a machine learning project that predicts the likelihood of diabetes using patient-related health and lifestyle information.

The project compares multiple machine learning algorithms, performs preprocessing and model evaluation, tunes the best-performing model, and provides a Streamlit application for making predictions.

> Note: DiabeticLens is an educational machine learning project and is not intended to provide medical diagnosis or replace professional medical advice.

---

## Features

- Exploratory Data Analysis (EDA)
- Data cleaning and duplicate removal
- Numerical feature scaling
- Categorical feature encoding
- Multiple machine learning models
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Decision threshold tuning
- Precision, Recall, F1-score and ROC-AUC evaluation
- Streamlit prediction interface
- Saved trained model and preprocessing pipeline
- Optional Explainable AI using SHAP planned for future development

---

## Machine Learning Workflow

The project follows the following workflow:

```text
Dataset
   ↓
Data Understanding
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Train-Test Split
   ↓
Preprocessing
   ↓
Model Training
   ↓
Model Comparison
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Threshold Tuning
   ↓
Final Model
   ↓
Streamlit Application


 python -m streamlit run app.py   