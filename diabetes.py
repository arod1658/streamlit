"""Diabetes classification demo built with Streamlit.

Run: python -m streamlit run diabetes.py
Place diabetes.csv alongside this script.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_FILE = Path(__file__).with_name("diabetes.csv")
CLASS_NAMES = ["Non-Diabetes", "Diabetes"]


@st.cache_data
def load_data(modified_time):
    """Invalidate the data cache when diabetes.csv changes."""
    return pd.read_csv(DATA_FILE)


@st.cache_resource(show_spinner="Training model (only for new settings)...")
def get_trained_model(classifier, train_percent, parameters, modified_time):
    """Train once for each data version, classifier, split and parameter combination."""
    df = load_data(modified_time)
    features = df.drop(columns=["Outcome"])
    target = df["Outcome"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        train_size=train_percent / 100,
        random_state=42,
        stratify=target,
    )

    if classifier == "Decision Tree":
        criterion, splitter = parameters
        model = DecisionTreeClassifier(
            criterion=criterion, splitter=splitter, random_state=42
        )
    elif classifier == "Logistic Regression":
        c_value, max_iter = parameters
        model = LogisticRegression(C=c_value, max_iter=max_iter)
    else:
        n_estimators, max_depth, bootstrap = parameters
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(x_train, y_train)
    return model, x_test, y_test, len(x_train)


def plot_metrics(selected_metrics, model, x_test, y_test):
    if "Confusion Matrix" in selected_metrics:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(
            model, x_test, y_test, display_labels=CLASS_NAMES,
            cmap="Blues", ax=ax, colorbar=False,
        )
        st.pyplot(fig)
        plt.close(fig)

    if "ROC Curve" in selected_metrics:
        st.subheader("ROC Curve")
        probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        area = roc_auc_score(y_test, probabilities)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC (AUC = {area:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random guess")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)


def main():
    st.title("Predicting Diabetes Web App")
    st.markdown("Predict diabetes classification using a selected machine-learning model.")
    st.caption("For classroom demonstrations only; not for medical diagnosis or treatment decisions.")

    if not DATA_FILE.is_file():
        st.error(f"Cannot find diabetes.csv. Put it alongside {Path(__file__).name}.")
        st.stop()

    modified_time = DATA_FILE.stat().st_mtime_ns
    df = load_data(modified_time)
    if "Outcome" not in df.columns or df["Outcome"].nunique() != 2:
        st.error("Dataset must contain a binary Outcome column (0 and 1).")
        st.stop()

    st.sidebar.title("Model Selection Panel")
    classifier = st.sidebar.selectbox(
        "Classifier", ("Decision Tree", "Logistic Regression", "Random Forest")
    )
    train_percent = st.sidebar.slider(
        "Training data proportion (%)", min_value=50, max_value=90,
        value=70, step=5,
        help="The remaining observations are used for testing.",
    )
    st.sidebar.caption(f"Training: {train_percent}% | Testing: {100 - train_percent}%")

    st.sidebar.subheader("Model parameters")
    if classifier == "Decision Tree":
        criterion = st.sidebar.radio("Criterion", ("gini", "entropy"))
        splitter = st.sidebar.radio("Splitter", ("best", "random"))
        parameters = (criterion, splitter)
    elif classifier == "Logistic Regression":
        c_value = st.sidebar.number_input(
            "C (Regularization parameter)", min_value=0.01,
            max_value=10.0, value=1.0, step=0.01,
        )
        max_iter = st.sidebar.slider("Maximum iterations", 100, 1000, 500, 50)
        parameters = (c_value, max_iter)
    else:
        n_estimators = st.sidebar.number_input(
            "Number of trees", min_value=10, max_value=500,
            value=100, step=10,
        )
        max_depth = st.sidebar.number_input(
            "Maximum tree depth", min_value=1, max_value=20,
            value=5, step=1,
        )
        bootstrap = st.sidebar.checkbox("Bootstrap", value=True)
        parameters = (n_estimators, max_depth, bootstrap)

    selected_metrics = st.sidebar.multiselect(
        "Charts", ("Confusion Matrix", "ROC Curve"),
        default=("Confusion Matrix", "ROC Curve"),
    )
    if st.sidebar.checkbox("Show raw data", value=False):
        st.subheader("Diabetes Raw Dataset")
        st.dataframe(df)

    # Streamlit reruns on every widget interaction. Cached models do not refit
    # unless the dataset, model type, split, or hyperparameters change.
    model, x_test, y_test, training_size = get_trained_model(
        classifier, train_percent, parameters, modified_time
    )
    predictions = model.predict(x_test)

    st.subheader(f"{classifier} Results")
    st.write(f"Training samples: **{training_size}** | Test samples: **{len(x_test)}**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, predictions, zero_division=0):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, predictions, zero_division=0):.3f}")
    col4.metric("F1 Score", f"{f1_score(y_test, predictions, zero_division=0):.3f}")

    plot_metrics(selected_metrics, model, x_test, y_test)

    st.divider()
    st.subheader("Single Patient Prediction")
    st.caption("Enter the feature values below, then click Predict. The existing model is reused.")
    features = df.drop(columns=["Outcome"])
    inputs = {}
    with st.form("single_prediction_form"):
        cols = st.columns(2)
        for idx, feature in enumerate(features.columns):
            default_value = float(features[feature].median())
            with cols[idx % 2]:
                if pd.api.types.is_integer_dtype(features[feature]):
                    inputs[feature] = st.number_input(
                        feature, min_value=0, value=max(0, int(round(default_value))),
                        step=1, key=f"patient_{feature}",
                    )
                else:
                    inputs[feature] = st.number_input(
                        feature, min_value=0.0, value=max(0.0, default_value),
                        step=0.1, format="%.3f", key=f"patient_{feature}",
                    )
        submitted = st.form_submit_button("Predict")

    model_settings = (classifier, train_percent, parameters, modified_time)
    if submitted:
        # Preserve the original feature names and order used in model.fit.
        patient_df = pd.DataFrame([inputs], columns=features.columns)
        predicted_class = int(model.predict(patient_df)[0])
        probabilities = model.predict_proba(patient_df)[0]
        by_class = dict(zip(model.classes_, probabilities))
        st.session_state["single_prediction_result"] = (
            model_settings,
            predicted_class,
            float(by_class.get(0, 0)),
            float(by_class.get(1, 0)),
        )

    result = st.session_state.get("single_prediction_result")
    if result is not None and result[0] == model_settings:
        _, predicted_class, non_diabetes_prob, diabetes_prob = result
        st.success(f"Prediction: {CLASS_NAMES[predicted_class]}")
        st.write(f"Non-Diabetes probability: **{non_diabetes_prob:.2%}**")
        st.write(f"Diabetes probability: **{diabetes_prob:.2%}**")
        fig, ax = plt.subplots()
        ax.pie(
            [non_diabetes_prob, diabetes_prob],
            labels=CLASS_NAMES,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#5DADE2", "#F1948A"],
        )
        ax.set_title("Prediction Probability Distribution")
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
