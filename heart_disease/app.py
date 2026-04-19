import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details:")

# Inputs
age = st.slider("Age", 20, 80)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
resting_bps = st.slider("Resting BP", 80, 200)
chol = st.slider("Cholesterol", 100, 400)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalach = st.slider("Max Heart Rate", 70, 210)
exang = st.selectbox("Exercise Angina", [0,1])
oldpeak = st.slider("Oldpeak", 0.0, 6.0)
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("CA", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

# Convert to array
features = np.array([[age, sex, cp, resting_bps, chol, fasting_bs,
                      restecg, thalach, exang, oldpeak,
                      slope, ca, thal]])

# Scale
features = scaler.transform(features)

# Predict
if st.button("Predict"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({prob:.2f})")
    else:
        st.success(f"✅ Low Risk ({prob:.2f})")