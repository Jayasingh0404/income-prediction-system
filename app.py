import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Income Level Classifier (Census Income >$50K)",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .result-badge-high {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.8rem 1.4rem;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .result-badge-low {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 0.8rem 1.4rem;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #FDE68A;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle():
    bundle_path = os.path.join(os.path.dirname(__file__), "model", "model_bundle.pkl")
    if not os.path.exists(bundle_path):
        st.error("Model bundle not found! Please run train.py first.")
        return None
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_bundle()

st.markdown('<div class="main-header">💼 Income Prediction ML System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict whether an individual earns more than $50,000/year based on demographic and employment attributes (Adult Census Dataset).</div>', unsafe_allow_html=True)

if bundle is None:
    st.stop()

scaler = bundle["scaler"]
lr_model = bundle["model"]
ridge_model = bundle["ridge_model"]
nb_model = bundle["nb_model"]
accuracies = bundle["accuracies"]
feature_names = bundle["feature_names"]
categories_dict = bundle["categories_dict"]

# Sidebar - Configuration & Metrics
with st.sidebar:
    st.header("⚙️ Model Selector")
    model_choice = st.selectbox(
        "Choose Classifier",
        ["Logistic Regression (Best)", "Ridge Classifier", "Gaussian Naive Bayes"]
    )
    
    st.markdown("---")
    st.subheader("📊 Model Benchmarks")
    st.write(f"**Logistic Regression**: `{accuracies['Logistic Regression'] * 100:.2f}%`")
    st.write(f"**Ridge Classifier**: `{accuracies['Ridge Classifier'] * 100:.2f}%`")
    st.write(f"**Gaussian Naive Bayes**: `{accuracies['Gaussian Naive Bayes'] * 100:.2f}%`")
    
    st.markdown("---")
    st.info("💡 **Dataset:** UCI Adult Census Dataset\n\n🎯 **Target:** Income `> $50K` vs `<= $50K`")

# Main Interface: Input Form
st.subheader("📋 Enter Individual Demographic & Employment Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", min_value=17, max_value=90, value=38)
    workclass = st.selectbox("Workclass", categories_dict.get("workclass", ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Self-emp-inc", "Federal-gov"]))
    educational_num = st.slider("Educational Level (Years of Education)", min_value=1, max_value=16, value=10, 
                                help="1: Preschool, 9: HS-grad, 10: Some-college, 13: Bachelors, 14: Masters, 16: Doctorate")
    hours_per_week = st.slider("Hours per Week", min_value=1, max_value=100, value=40)

with col2:
    marital_status = st.selectbox("Marital Status", categories_dict.get("marital-status", ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"]))
    occupation = st.selectbox("Occupation", categories_dict.get("occupation", ["Prof-specialty", "Exec-managerial", "Craft-repair", "Sales", "Adm-clerical", "Other-service"]))
    relationship = st.selectbox("Relationship", categories_dict.get("relationship", ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"]))
    race = st.selectbox("Race", categories_dict.get("race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]))

with col3:
    gender = st.selectbox("Gender", categories_dict.get("gender", ["Male", "Female"]))
    capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=100000, value=0, step=500)
    capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=10000, value=0, step=100)
    fnlwgt = st.number_input("Final Weight (Census Demographic Sampling Weight)", min_value=10000, max_value=1500000, value=189673, step=5000)
    native_country = st.selectbox("Native Country", categories_dict.get("native-country", ["United-States", "India", "Canada", "Germany", "Mexico", "Philippines", "England"]))

# Compute Region
north_america = ["United-States", "Mexico", "Canada", "Puerto-Rico", "Cuba"]
asia = ["Japan", "South", "Philippines", "China", "Vietnam", "Thailand", "India", "Cambodia", "Yugoslavia", "Iran", "Taiwan", "Hong"]
europe = ["Portugal", "Italy", "England", "Germany", "Poland", "Greece", "Ireland", "Scotland", "France", "Hungary"]
latin_america = ["Ecuador", "Dominican-Republic", "Jamaica", "El-Salvador", "Guatemala", "Columbia", "Haiti", "Peru", "Nicaragua", "Trinadad&Tobago"]

if native_country in north_america:
    region = "North America"
elif native_country in asia:
    region = "Asia"
elif native_country in europe:
    region = "Europe"
elif native_country in latin_america:
    region = "Latin America"
else:
    region = "Other"

# Prediction Button
st.markdown("---")
if st.button("🚀 Predict Income Level", type="primary", use_container_width=True):
    # Construct input row
    input_dict = {f: 0 for f in feature_names}
    
    # Assign numerical values
    input_dict["age"] = age
    input_dict["fnlwgt"] = fnlwgt
    input_dict["educational-num"] = educational_num
    input_dict["capital-gain"] = capital_gain
    input_dict["capital-loss"] = capital_loss
    input_dict["hours-per-week"] = hours_per_week
    
    # One-hot encoded assignments
    for prefix, val in [
        ("workclass", workclass),
        ("marital-status", marital_status),
        ("occupation", occupation),
        ("relationship", relationship),
        ("race", race),
        ("gender", gender),
        ("native-country", native_country),
        ("Region", region)
    ]:
        col_name = f"{prefix}_{val}"
        if col_name in input_dict:
            input_dict[col_name] = 1
            
    input_df = pd.DataFrame([input_dict])[feature_names]
    input_scaled = scaler.transform(input_df)
    
    # Selected Model
    if "Logistic" in model_choice:
        pred = lr_model.predict(input_scaled)[0]
        prob = lr_model.predict_proba(input_scaled)[0][1]
    elif "Ridge" in model_choice:
        pred = ridge_model.predict(input_scaled)[0]
        decision = ridge_model.decision_function(input_scaled)[0]
        prob = 1 / (1 + np.exp(-decision))
    else:
        pred = nb_model.predict(input_scaled)[0]
        prob = nb_model.predict_proba(input_scaled)[0][1]
        
    st.markdown("### 🎯 Prediction Results")
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if pred == 1:
            st.markdown('<div class="result-badge-high">✅ High Income (> $50,000/year)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-badge-low">⚠️ Moderate/Low Income (≤ $50,000/year)</div>', unsafe_allow_html=True)
            
    with res_col2:
        st.metric("Estimated Probability of Income > $50K", f"{prob * 100:.2f}%")
        st.progress(float(prob))
        
    st.caption(f"Evaluated using **{model_choice}**.")
