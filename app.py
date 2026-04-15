import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- SESSION STATE --------------------
if "section" not in st.session_state:
    st.session_state.section = None

if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="ML Model Comparator", layout="wide")

# -------------------- CSS --------------------
st.markdown("""
<style>

.block-container {
    max-width: 1100px;
    margin: auto;
}

.center-box {
    text-align: center;
}

/* Title */
h1 {
    text-align: center;
    font-size: 42px;
    white-space: nowrap;
    color: #FFA500;
}

/* FIX: make ONLY headings yellow */
h2, h3, h4 {
    color: #FFA500 !important;
}

/* Streamlit subheaders (important fix) */
[data-testid="stMarkdownContainer"] h2 {
    color: #FFA500 !important;
}

/* ONLY upload label white */
[data-testid="stFileUploader"] label {
    color: white !important;
}

/* Upload box */
[data-testid="stFileUploader"] {
    width: 350px;
    margin: auto;
}

section[data-testid="stFileUploader"] > div {
    border: 2px dashed orange;
    border-radius: 12px;
    padding: 25px;
    background-color: #1a1a1a;
}

/* Buttons */
.stButton {
    display: flex;
    justify-content: center;
    margin-top: 10px;
}

div.stButton > button {
    background-color: #FFA500;
    color: black;
    border-radius: 8px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="center-box">', unsafe_allow_html=True)

st.markdown("<h1>🔥 ML Model Comparator Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("### Upload Dataset")
uploaded_file = st.file_uploader("", type=["csv"])

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

# -------------------- MAIN --------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    target_col = st.selectbox("Select Target Column", df.columns)

    # -------------------- RUN BUTTON --------------------
    if st.button("Run Models 🚀"):
        st.session_state.results_ready = True
        st.session_state.section = None

    # -------------------- MAIN LOGIC --------------------
    if st.session_state.results_ready:

        # -------- PREPROCESSING --------
        for col in df.columns:
            if df[col].dtype != 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

            if df[col].dtype == 'object':
                if df[col].mode().empty:
                    df[col] = df[col].fillna("Unknown")
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
            else:
                if df[col].isna().all():
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].mean())

        df = df.fillna(0)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col])

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # -------- VALIDATION --------
        if len(X) == 0:
            st.error("Dataset became empty.")
            st.stop()

        if len(df) < 10:
            st.error("Dataset too small.")
            st.stop()

        if len(np.unique(y)) < 2:
            st.error("Target must have at least 2 classes.")
            st.stop()

        # -------- SPLIT --------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------- MODELS --------
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        results = []
        eval_metrics = {}
        confusion_data = {}

        for name, model in models.items():
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

            error_rate = 1 - test_acc
            overfit_gap = train_acc - test_acc

            results.append({
                "Model": name,
                "Train Accuracy": train_acc,
                "Test Accuracy": test_acc,
                "Error Rate": error_rate,
                "Overfit Gap": overfit_gap,
                "Training Time": train_time
            })

            eval_metrics[name] = {
                "Accuracy": test_acc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Sensitivity": recall,
                "Specificity": 1 - error_rate,
                "Error Rate": error_rate
            }

            confusion_data[name] = confusion_matrix(y_test, y_test_pred)

        results_df = pd.DataFrame(results)

        # -------- TABLE --------
        st.subheader("📋 Model Comparison Table")
        st.dataframe(results_df, use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # -------- BUTTONS --------
        col1, col2, col3 = st.columns(3)

        if col1.button("📊 Evaluation Parameters"):
            st.session_state.section = "eval"

        if col2.button("📉 Confusion Matrix"):
            st.session_state.section = "confusion"

        if col3.button("📈 Overfitting Graph"):
            st.session_state.section = "overfit"

        st.markdown("---")

        # -------- DISPLAY --------
        if st.session_state.section == "eval":
            st.subheader("📊 Evaluation Metrics")
            st.dataframe(pd.DataFrame(eval_metrics).T, use_container_width=True)

        elif st.session_state.section == "confusion":
            st.subheader("📉 Confusion Matrices")

            model_names = list(confusion_data.keys())

            # Create 2x2 grid
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            cols = [col1, col2, col3, col4]

            for i, (name, cm) in enumerate(confusion_data.items()):
                with cols[i]:
                    st.markdown(f"### {name}")

                    fig, ax = plt.subplots(figsize=(5, 4))

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt='d',
                        cmap='Oranges',
                        cbar=False,
                        linewidths=1,
                        ax=ax
                    )

                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")

                    st.pyplot(fig, use_container_width=True)

        elif st.session_state.section == "overfit":
            st.subheader("📈 Overfitting Graph")
            fig, ax = plt.subplots()
            ax.plot(results_df["Model"], results_df["Train Accuracy"], marker='o')
            ax.plot(results_df["Model"], results_df["Test Accuracy"], marker='o')
            ax.legend(["Train", "Test"])
            st.pyplot(fig)

        # -------- FINAL RESULT --------
        st.markdown("---")
        st.subheader("🏆 Final Result")

        best_model = results_df.sort_values(by="Test Accuracy", ascending=False).iloc[0]

        st.write(f"### Best Model: {best_model['Model']}")
        st.write("- Highest Test Accuracy")
        st.write("- Lower Overfitting Gap preferred")

        st.write("### Model Ranking")
        st.dataframe(results_df.sort_values(by="Test Accuracy", ascending=False), use_container_width=True)