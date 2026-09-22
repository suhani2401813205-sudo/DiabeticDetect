import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="DiabeticLens",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Main content width */
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #17324d;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #667085;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-title {
        font-size: 24px;
        font-weight: 650;
        color: #17324d;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Information card */
    .info-card {
        background-color: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e4e7ec;
        margin-bottom: 20px;
    }

    /* Result cards */
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 14px;
        border: 1px solid #e4e7ec;
        text-align: center;
        margin-top: 15px;
    }

    .probability {
        font-size: 38px;
        font-weight: 700;
        color: #17324d;
    }

    .result-label {
        font-size: 15px;
        color: #667085;
        margin-bottom: 5px;
    }

    .prediction-safe {
        background-color: #ecfdf3;
        border: 1px solid #abefc6;
        color: #067647;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 19px;
        font-weight: 600;
        margin-top: 20px;
    }

    .prediction-risk {
        background-color: #fef3f2;
        border: 1px solid #fecdca;
        color: #b42318;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 19px;
        font-weight: 600;
        margin-top: 20px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 10px;
        font-size: 17px;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 13px;
        margin-top: 35px;
        padding-top: 20px;
        border-top: 1px solid #e4e7ec;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_models():

    preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    model = joblib.load(
        "models/final_diabetes_model.pkl"
    )

    threshold = joblib.load(
        "models/decision_threshold.pkl"
    )

    return preprocessor, model, threshold


preprocessor, model, threshold = load_models()


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">DiabeticLens</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Diabetes Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the patient's health information to generate "
    "a diabetes prediction using the trained machine learning model."
)


# ==========================================================
# PATIENT INFORMATION
# ==========================================================

st.markdown(
    '<div class="section-title">Patient Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-card">',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


# ---------------- COLUMN 1 ----------------

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=5.0,
        max_value=100.0,
        value=27.0,
        step=0.1
    )

    hba1c = st.number_input(
        "HbA1c Level",
        min_value=3.0,
        max_value=10.0,
        value=5.5,
        step=0.1
    )


# ---------------- COLUMN 2 ----------------

with col2:

    blood_glucose = st.number_input(
        "Blood Glucose Level",
        min_value=50.0,
        max_value=400.0,
        value=120.0,
        step=1.0
    )

    smoking_history = st.selectbox(
        "Smoking History",
        [
            "never",
            "former",
            "current",
            "not current",
            "ever",
            "No Info"
        ]
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x:
        "Yes" if x == 1 else "No"
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1],
        format_func=lambda x:
        "Yes" if x == 1 else "No"
    )


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# PREDICT BUTTON
# ==========================================================

st.write("")

predict_button = st.button(
    "Predict Diabetes"
)


# ==========================================================
# PREDICTION
# ==========================================================

if predict_button:

    # Create input DataFrame

    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [blood_glucose]
    })


    # Preprocess

    input_processed = preprocessor.transform(
        input_data
    )


    # Probability

    probability = model.predict_proba(
        input_processed
    )[0][1]


    # Apply tuned threshold

    prediction = int(
        probability >= threshold
    )


    # ======================================================
    # RESULT
    # ======================================================

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.markdown(
            '<div class="result-card">'
            '<div class="result-label">'
            'Diabetes Probability'
            '</div>'
            f'<div class="probability">'
            f'{probability * 100:.2f}%'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    with result_col2:

        st.markdown(
            '<div class="result-card">'
            '<div class="result-label">'
            'Decision Threshold'
            '</div>'
            f'<div class="probability">'
            f'{threshold:.2f}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    with result_col3:

        prediction_text = (
            "Diabetes"
            if prediction == 1
            else "No Diabetes"
        )

        st.markdown(
            '<div class="result-card">'
            '<div class="result-label">'
            'Model Prediction'
            '</div>'
            f'<div class="probability">'
            f'{prediction_text}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # ======================================================
    # PREDICTION MESSAGE
    # ======================================================

    if prediction == 1:

        st.markdown(
            '<div class="prediction-risk">'
            'The model predicts diabetes based on the '
            'provided information.'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="prediction-safe">'
            'The model does not predict diabetes based on '
            'the provided information.'
            '</div>',
            unsafe_allow_html=True
        )


    # ======================================================
    # MODEL INFORMATION
    # ======================================================

    st.write("")

    with st.expander("Model Information"):

        st.write(
            "**Model:** Tuned Gradient Boosting"
        )

        st.write(
            f"**Decision Threshold:** {threshold:.2f}"
        )

        st.write(
            "**Purpose:** Diabetes prediction based on "
            "the input health features."
        )


# ==========================================================
# DISCLAIMER
# ==========================================================

st.markdown(
    """
    <div class="footer">

    <b>Disclaimer:</b> DiabeticLens is an educational
    machine-learning project and is not a medical
    diagnostic tool. Predictions should not replace
    professional medical advice or clinical testing.

    <br><br>

    DiabeticLens • Machine Learning Project

    </div>
    """,
    unsafe_allow_html=True
)