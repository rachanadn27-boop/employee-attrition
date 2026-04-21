import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from model_pipeline import predict

# ---------------- LOAD DATA ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"))

st.set_page_config(layout="wide")
st.title(" Employee Attrition Predictor")

# ---------------- INPUT SECTION ----------------
st.subheader("🧾 Enter Employee Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 60, 30)
    distance = st.slider("Distance From Home", 1, 30, 5)
    years = st.slider("Years at Company", 0, 40, 5)
    total_working_years = st.slider("Total Working Years", 0, 40, 10)

with col2:
    income = st.slider("Monthly Income", 1000, 20000, 5000)
    hike = st.slider("Percent Salary Hike", 10, 25, 15)
    companies = st.slider("Companies Worked", 0, 10, 2)
    training = st.slider("Training Times Last Year", 0, 10, 2)

with col3:
    job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    work_life = st.slider("Work Life Balance (1-4)", 1, 4, 3)
    overtime = st.selectbox("OverTime", ["No", "Yes"])

overtime = 1 if overtime == "Yes" else 0

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    data = {
        "Age": age,
        "DistanceFromHome": distance,
        "YearsAtCompany": years,
        "TotalWorkingYears": total_working_years,
        "MonthlyIncome": income,
        "PercentSalaryHike": hike,
        "NumCompaniesWorked": companies,
        "TrainingTimesLastYear": training,
        "JobSatisfaction": job_sat,
        "EnvironmentSatisfaction": env_sat,
        "WorkLifeBalance": work_life,
        "OverTime": overtime
    }

    prob = predict(data)

    st.subheader(f"📊 Leave Probability: {prob*100:.2f}%")

    st.progress(float(prob))

    if prob < 0.3:
        st.success("🟢 Low Risk Employee")
    elif prob < 0.6:
        st.warning("🟡 Medium Risk Employee")
    else:
        st.error("🔴 High Risk Employee")

# ---------------- VISUALIZATION ----------------
st.subheader("📊 Insights Dashboard")

sns.set_style("whitegrid")

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots()
    sns.countplot(x='Attrition', data=df, ax=ax)
    ax.set_title("Attrition Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, ax=ax)
    ax.set_title("Income vs Attrition")
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots()
    sns.countplot(x='OverTime', hue='Attrition', data=df, ax=ax)
    ax.set_title("OverTime Impact")
    st.pyplot(fig)

col4, col5 = st.columns(2)

with col4:
    fig, ax = plt.subplots()
    sns.countplot(x='JobSatisfaction', hue='Attrition', data=df, ax=ax)
    ax.set_title("Job Satisfaction vs Attrition")
    st.pyplot(fig)

with col5:
    fig, ax = plt.subplots()
    sns.boxplot(x='Attrition', y='YearsAtCompany', data=df, ax=ax)
    ax.set_title("Experience vs Attrition")
    st.pyplot(fig)
