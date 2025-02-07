import pandas as pd
import streamlit as st
import joblib


def create_demo():
    TARGET = "churn"

    presentation_df_address = "dataset/processed/presentation.csv"
    model_address = "models/v1/lightgbm_model_final.pkl"
    best_features = [
        "age",
        "service_type",
        "overdue_payments",
        "auto_payment",
        "avg_top_up_count",
    ]

    complete_columns = best_features.copy()
    complete_columns.append(TARGET)

    presentation_df = pd.read_csv(presentation_df_address)
    presentation_df = presentation_df.loc[:, complete_columns]

    model = joblib.load(model_address)

    X = presentation_df.loc[:, presentation_df.columns != TARGET]

    st.title("Churn Prediction Demo")

    if st.button("Predict Churn"):
        st.dataframe(presentation_df)
        prediction = model.predict(X)
        prediction_prob = model.predict_proba(X)[:, -1]
        presentation_df["prediction"] = prediction
        presentation_df["churn_predicted_probability"] = prediction_prob
        result_cols = ["churn", "prediction", "churn_predicted_probability"]
        result_df = presentation_df.loc[:, result_cols]
        st.dataframe(result_df.sort_values(by=["churn"], ascending=False))
    else:
        st.dataframe(presentation_df)
