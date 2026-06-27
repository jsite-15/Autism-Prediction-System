import streamlit as st
import pandas as pd
import pickle


st.set_page_config(
    page_title="Autism Prediction System",
    page_icon="🧠",
    layout="wide"
)
st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

h1 {
    text-align: center;
    color: #0F4C81;
}

.stButton>button {
    width: 100%;
    background-color: #0F4C81;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    padding: 12px;
}

.stButton>button:hover {
    background-color: #1565C0;
    color: white;
}

div[data-testid="stMetric"]{
    background-color:#1E1E1E;
    border-radius:12px;
    padding:15px;
    border:1px solid #3a3a3a;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model & Encoders
# -----------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# -----------------------------
# Title
# -----------------------------
st.title("🧠 Autism Prediction System")
st.write("""
This application predicts whether an individual is likely to show signs of **Autism Spectrum Disorder (ASD)** based on questionnaire responses and demographic information.

**Note:** This tool is for educational purposes only and should not be used as a medical diagnosis.
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("About")
st.sidebar.info(
    """
    **Machine Learning Model**

    ✔ Random Forest / XGBoost (Best Model)

    ✔ SMOTE Applied

    ✔ Hyperparameter Tuning

    ✔ Cross Validation

    ✔ Feature Importance Analysis

    Developed using **Python, Scikit-Learn, Streamlit**
    """
)

# -----------------------------
# Questionnaire Section
# -----------------------------
st.header("Autism Screening Questions")

col1, col2 = st.columns(2)

with col1:
    A1 = st.selectbox("A1 Score", [0,1])
    A2 = st.selectbox("A2 Score", [0,1])
    A3 = st.selectbox("A3 Score", [0,1])
    A4 = st.selectbox("A4 Score", [0,1])
    A5 = st.selectbox("A5 Score", [0,1])

with col2:
    A6 = st.selectbox("A6 Score", [0,1])
    A7 = st.selectbox("A7 Score", [0,1])
    A8 = st.selectbox("A8 Score", [0,1])
    A9 = st.selectbox("A9 Score", [0,1])
    A10 = st.selectbox("A10 Score", [0,1])

# -----------------------------
# Personal Information
# -----------------------------
st.header("Personal Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=10
    )

    gender = st.selectbox(
        "Gender",
        ["f","m"]
    )

    ethnicity = st.selectbox(
    "Ethnicity",
    list(encoders["ethnicity"].classes_)
    )

    jaundice = st.selectbox(
        "Born with Jaundice?",
        ["no","yes"]
    )

with col2:

    austim = st.selectbox(
        "Family Member with Autism?",
        ["no","yes"]
    )

    countries = list(encoders["contry_of_res"].classes_)
    country = st.selectbox(
    "Country of Residence",
    countries,
    index=countries.index("India")
    )

    used_app_before = st.selectbox(
        "Used Autism Screening App Before?",
        ["no","yes"]
    )

    relation = st.selectbox(
    "Relation",
    list(encoders["relation"].classes_)
    )

# -----------------------------
# Screening Score
# -----------------------------
st.header("Screening Result")

result = st.slider(
    "Screening Score",
    0,
    10,
    5
)

st.divider()

predict_button = st.button(
    "🔍 Predict Autism",
    use_container_width=True
)# =====================================================
# Prediction Logic
# =====================================================

if predict_button:

    # Create input dictionary
    input_data = {
        "A1_Score": A1,
        "A2_Score": A2,
        "A3_Score": A3,
        "A4_Score": A4,
        "A5_Score": A5,
        "A6_Score": A6,
        "A7_Score": A7,
        "A8_Score": A8,
        "A9_Score": A9,
        "A10_Score": A10,
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "jaundice": jaundice,
        "austim": austim,
        "contry_of_res": country,
        "used_app_before": used_app_before,
        "result": result,
        "relation": relation
    }

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    categorical_columns = [
        "gender",
        "ethnicity",
        "jaundice",
        "austim",
        "contry_of_res",
        "used_app_before",
        "relation"
    ]

    try:
        for column in categorical_columns:
            input_df[column] = encoders[column].transform(input_df[column])

        # Make Prediction
        prediction = model.predict(input_df)[0]

        # Prediction Probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0]
            confidence = max(probability) * 100
        else:
            confidence = None

        st.divider()
        st.subheader("🧠 Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Probability of Autism Spectrum Disorder (ASD)")
            st.warning(
                "The model predicts that the individual may exhibit signs associated with Autism Spectrum Disorder. "
                "Please consult a qualified healthcare professional for further evaluation."
            )
        else:
            st.success("✅ Low Probability of Autism Spectrum Disorder (ASD)")
            st.info(
                "The model predicts a low probability of Autism Spectrum Disorder based on the provided information."
            )

        # Show confidence score
        if confidence is not None:
            st.metric("Prediction Confidence", f"{confidence:.2f}%")
            st.progress(confidence / 100)

        # Display entered information
        with st.expander("📋 View Submitted Information"):
            st.dataframe(
                pd.DataFrame([input_data]),
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        

# ---------- About Model ----------

st.divider()

st.subheader("About the Machine Learning Model")

st.write("""

This application predicts Autism Spectrum Disorder (ASD)
using a trained Machine Learning model.

### Workflow

✔ Data Cleaning

✔ Exploratory Data Analysis (EDA)

✔ Label Encoding

✔ SMOTE for Class Balancing

✔ Decision Tree

✔ Random Forest

✔ XGBoost

✔ Hyperparameter Tuning

✔ Cross Validation

✔ Feature Importance Analysis

✔ Predictive System

""")


# ---------- Medical Disclaimer ----------

with st.expander("⚠️ Medical Disclaimer"):

    st.write("""

This application is developed **only for educational and research purposes**.

The prediction generated by this model **must not** be considered as a medical diagnosis.

Always consult a qualified healthcare professional for proper evaluation,
screening, and diagnosis.

The developers are not responsible for any medical decisions taken
based on this application's prediction.

""")


# ---------- Footer ----------

st.markdown("---")

st.markdown(
"""
<div style="text-align:center; color:gray;">

<h4>🧠 Autism Prediction System</h4>

Developed by <b>Jasmine Singh</b>

Computer Science Engineering Student

Machine Learning • Data Science • Artificial Intelligence

</div>
""",
unsafe_allow_html=True
)
