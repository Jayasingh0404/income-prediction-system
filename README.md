# 💼 Adult Census Income Prediction System

A complete Machine Learning application and interactive Streamlit web interface for predicting individual income levels (`> $50K` vs `<= $50K`) based on demographic and employment attributes from the **Adult Census (UCI Machine Learning Repository)** dataset.

---

## 📌 Project Overview
- **Objective:** Predict whether an individual's annual income exceeds $50,000 using machine learning classification algorithms.
- **Algorithms Evaluated:**
  - 🥇 **Logistic Regression**: **85.15% Accuracy** (Primary Production Model)
  - 🥈 **Ridge Classifier**: **77.56% Accuracy**
  - 🥉 **Gaussian Naive Bayes**: **51.48% Accuracy**
- **Deployment Interface:** Interactive web app with customizable inputs, probability estimation, and real-time inference.

---

## 🗂️ Project Structure
```
├── app.py                            # Streamlit web application
├── train.py                          # Data fetching, preprocessing & ML training script
├── income_prediction_notebook.ipynb  # Comprehensive EDA and model experiments notebook
├── requirements.txt                  # Python dependencies
├── model/
│   └── model_bundle.pkl              # Serialized scaler, trained models & schema metadata
├── data/
│   └── sample_adult_data.csv         # Sample dataset for demonstration
└── README.md                         # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Jayasingh0404/income-prediction-system.git
cd income-prediction-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the interactive web app
```bash
streamlit run app.py
```

### 4. (Optional) Re-train the models
```bash
python3 train.py
```

---

## 🌐 Cloud Deployment Options
This application is ready to deploy directly on:
- **Streamlit Community Cloud**: Connect this GitHub repository and point to `app.py`.
- **Hugging Face Spaces**: Select Streamlit SDK and connect this repo.
- **Render / Railway**: Run `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
