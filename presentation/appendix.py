import streamlit as st


def data_cleaning():
    st.header("Data Cleaning")
    code = """
        df = (
    df
    .pipe(insert_apps)
    .drop(columns=["id", "apps"])
    .drop_duplicates()
    .pipe(preprocess_service_type)
    .pipe(preprocess_churn)
    .pipe(impute_with_xgboost, column_to_impute="avg_call_duration")
    .pipe(impute_with_xgboost, column_to_impute="roaming_usage")
    .pipe(impute_with_xgboost, column_to_impute="auto_payment", job="classification")
    .pipe(impute_with_xgboost, column_to_impute="call_drops")
    .assign(**{
        "data_usage": lambda df: df["data_usage"].fillna(df["data_usage"].mean()),
        "tenure": lambda df: df["tenure"].fillna(df["tenure"].mean()),
        "monthly_charge": lambda df: df["monthly_charge"].fillna(df["monthly_charge"].median())
    })
    .pipe(winsorize_outliers)
    .astype({col: "int32" for col in df.select_dtypes("int64").columns})
    .astype({col: "float32" for col in df.select_dtypes("float64").columns})
 )

        """
    st.code(code)


def feature_extraction():
    st.header("Feature Extraction")
    code = """
    df = ( 
    df
    .pipe(insert_binary_app_combinations, columns=app_cols)
    .assign(**{
        "yearly_tenure" : lambda df: df["tenure"] * 12,
        "total_avg_call" : lambda df: df["avg_call_duration"] + df["roaming_usage"],
    })
    .astype({col: "int32" for col in df.select_dtypes("int64").columns})
    .astype({col: "float32" for col in df.select_dtypes("float64").columns})
)
    """
    st.code(code)


def feature_selection():
    st.header("Feature Selection")
    st.write("Selected Features:")
    selected_features = [
        "age",
        "service_type",
        "overdue_payments",
        "auto_payment",
        "avg_top_up_count",
    ]

    for feature in selected_features:
        st.markdown(f"- {feature}")


def modelling():
    st.header("Modelling")
    st.markdown("### Random Classifier")
    st.image("figures/4-random_model_classification_report.png")
    st.image("figures/5-random_classifier_auroc.png")
    st.markdown("#### f1 score: 0.0131")

    st.markdown("### LightGBM Classifier")
    st.image("figures/1-classification_report.png")
    st.image("figures/3-auroc.png")
    st.image("figures/2-feature_importance.png")
