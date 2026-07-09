import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Household Power Prediction",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = joblib.load("random_forest_model.pkl")
except Exception as e:
    st.error(f"Unable to load model:\n{e}")
    st.stop()

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main{
background:#eef4ff;
}

[data-testid="stSidebar"]{
background:linear-gradient(180deg,#141E30,#243B55);
}

[data-testid="stSidebar"] *{
color:white;
}

.big-title{
font-size:48px;
font-weight:bold;
color:white;
}

.sub-title{
font-size:22px;
color:white;
}

.header-box{
background:linear-gradient(90deg,#4A00E0,#8E2DE2,#ff0080);
padding:35px;
border-radius:20px;
color:white;
box-shadow:0px 5px 20px rgba(0,0,0,0.3);
}

.card{
background:linear-gradient(135deg,#ffffff,#f8fbff);
padding:25px;
border-radius:18px;
box-shadow:0px 8px 20px rgba(0,0,0,0.18);
text-align:center;
border-left:8px solid #6C63FF;
transition:0.3s;
}

.card h2{
color:#1E3A8A;
font-size:28px;
font-weight:bold;
margin-bottom:10px;
}

.card h3{
color:#111827;
font-size:30px;
font-weight:bold;
}

.card p{
color:#4B5563;
font-size:18px;
font-weight:600;
}

.result{
background:linear-gradient(90deg,#00c853,#64dd17);
padding:30px;
border-radius:20px;
text-align:center;
color:white;
font-size:38px;
font-weight:bold;
}

.footer{
background:#111827;
padding:20px;
border-radius:15px;
color:white;
text-align:center;
}

</style>
""",unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""

<div class="header-box">

<div class="big-title">
⚡ Household Electric Power Consumption Prediction
</div>

<div class="sub-title">
Random Forest Regression Dashboard
</div>

</div>

""",unsafe_allow_html=True)

st.write("")

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("⚙ Input Panel")

reactive = st.sidebar.number_input(
"Global Reactive Power",
value=0.10)

voltage = st.sidebar.number_input(
"Voltage",
value=240.00)

intensity = st.sidebar.number_input(
"Global Intensity",
value=10.00)

sub1 = st.sidebar.number_input(
"Sub Metering 1",
value=0.00)

sub2 = st.sidebar.number_input(
"Sub Metering 2",
value=1.00)

sub3 = st.sidebar.number_input(
"Sub Metering 3",
value=17.00)

predict = st.sidebar.button("⚡ Predict")

st.write("")

# -----------------------------
# TOP CARDS
# -----------------------------

c1,c2,c3,c4 = st.columns(4)

with c1:

    st.markdown("""

<div class="card">

## 📊 Dataset

### 2.07 Million

Records

</div>

""",unsafe_allow_html=True)

with c2:

    st.markdown("""

<div class="card">

## 🤖 Model

### Random Forest

Regression

</div>

""",unsafe_allow_html=True)

with c3:

    st.markdown("""

<div class="card">

## 🎯 Features

### 6

Input Features

</div>

""",unsafe_allow_html=True)

with c4:

    st.markdown("""

<div class="card">

## ⚡ Target

Global Active Power

</div>

""",unsafe_allow_html=True)

st.write("")
# -----------------------------
# PREDICTION SECTION
# -----------------------------

left, right = st.columns([2,1])

with left:

    st.subheader("📋 Input Summary")

    input_df = pd.DataFrame({
        "Feature":[
            "Global Reactive Power",
            "Voltage",
            "Global Intensity",
            "Sub Metering 1",
            "Sub Metering 2",
            "Sub Metering 3"
        ],
        "Value":[
            reactive,
            voltage,
            intensity,
            sub1,
            sub2,
            sub3
        ]
    })

    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )

with right:

    st.subheader("⚙ Model Status")

    st.success("✅ Model Loaded Successfully")

    st.progress(100)

    st.info("Algorithm : Random Forest Regression")

st.write("")

# -----------------------------
# PREDICTION
# -----------------------------

if predict:

    features = np.array([[
        reactive,
        voltage,
        intensity,
        sub1,
        sub2,
        sub3
    ]])

    prediction = model.predict(features)


    st.markdown(f"""

    <div class="result">

    ⚡ Predicted Global Active Power

    <br><br>

    {prediction[0]:.4f} kW

    </div>

    """, unsafe_allow_html=True)

    st.write("")

    # -----------------------------
    # GAUGE
    # -----------------------------

    value = float(prediction[0])

    if value < 1:
        percent = 20
        status = "🟢 Low Consumption"

    elif value < 3:
        percent = 55
        status = "🟡 Medium Consumption"

    else:
        percent = 90
        status = "🔴 High Consumption"

    st.subheader("⚡ Power Usage Level")

    st.progress(percent)

    st.success(status)

    st.write("")

    # -----------------------------
    # METRICS
    # -----------------------------

    m1,m2,m3 = st.columns(3)

    with m1:
        st.metric(
            "Voltage",
            f"{voltage:.2f} V"
        )

    with m2:
        st.metric(
            "Current",
            f"{intensity:.2f} A"
        )

    with m3:
        st.metric(
            "Prediction",
            f"{prediction[0]:.3f} kW"
        )

    st.write("")

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------

st.subheader("📊 Feature Contribution")

fig2, ax2 = plt.subplots(figsize=(8,4))
names = [
    "Reactive Power",
    "Voltage",
    "Intensity",
    "Sub Metering 1",
    "Sub Metering 2",
    "Sub Metering 3"
]
importance = model.feature_importances_
colors = [
    "#ff4b4b",
    "#ff9800",
    "#4caf50",
    "#03a9f4",
    "#9c27b0",
    "#009688"
]

ax2.barh(names, importance, color=colors)

ax2.set_xlabel("Importance Score")
ax2.set_title("Feature Contribution")

for i, v in enumerate(importance):
    ax2.text(v + 0.002, i, f"{v:.3f}")

st.pyplot(fig2)
    # -----------------------------
    # PIE CHART
    # -----------------------------

st.subheader("🥧 Feature Contribution")

feature_names = [
    "Reactive Power",
    "Voltage",
    "Intensity",
    "Sub Metering 1",
    "Sub Metering 2",
    "Sub Metering 3"
]

importance = model.feature_importances_

fig, ax = plt.subplots(figsize=(10,5))

bars = ax.barh(
    feature_names,
    importance,
    color=[
        "#6C63FF",
        "#00C853",
        "#FF9800",
        "#03A9F4",
        "#E91E63",
        "#9C27B0"
    ]
)

ax.set_title(
    "Random Forest Feature Importance",
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel("Importance Score")

ax.set_xlim(0, max(importance) + 0.05)

for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.003,
        bar.get_y() + bar.get_height()/2,
        f"{width:.3f}",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

st.pyplot(fig)
    # -----------------------------
# MODEL PERFORMANCE
# -----------------------------

st.write("")
st.markdown("---")

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card">
    <h2 style="color:#ff4b4b;">MAE</h2>
    <h3>0.0188</h3>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h2 style="color:#2196F3;">MSE</h2>
    <h3>0.0011</h3>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <h2 style="color:#4CAF50;">RMSE</h2>
    <h3>0.0331</h3>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
    <h2 style="color:#9C27B0;">R² Score</h2>
    <h3>0.9990</h3>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------
# ABOUT PROJECT
# -----------------------------

st.markdown("---")

st.subheader("📖 About This Project")

st.info("""
This project predicts **Global Active Power Consumption**
using a **Random Forest Regression** model trained on the
**Individual Household Electric Power Consumption Dataset**.

### Features Used
- Global Reactive Power
- Voltage
- Global Intensity
- Sub Metering 1
- Sub Metering 2
- Sub Metering 3

### Target
- Global Active Power

### Tools & Technologies
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
""")

st.write("")

# -----------------------------
# PROJECT HIGHLIGHTS
# -----------------------------

st.subheader("🚀 Project Highlights")

h1, h2, h3 = st.columns(3)

with h1:
    st.success("""
    ✔ Machine Learning

    ✔ Random Forest

    ✔ Regression Model
    """)

with h2:
    st.success("""
    ✔ Data Cleaning

    ✔ Feature Engineering

    ✔ Model Evaluation
    """)

with h3:
    st.success("""
    ✔ Streamlit Dashboard

    ✔ Data Visualization

    ✔ Interactive Prediction
    """)

st.write("")

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("""

<div class="footer">

<h2>⚡ Household Electric Power Consumption Prediction</h2>

<h4>Random Forest Regression Dashboard</h4>

<hr>

<p>
Built with ❤️ using
<b>Python • Streamlit • Scikit-learn • Pandas • NumPy</b>
</p>

<p>
Developed by <b>Arun Kaushik</b>
</p>

</div>

""", unsafe_allow_html=True)