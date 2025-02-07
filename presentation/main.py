import streamlit as st
from demo import create_demo
from appendix import data_cleaning, feature_extraction, feature_selection, modelling


def main():
    create_demo()
    st.header("Appendix")
    data_cleaning()
    feature_extraction()
    feature_selection()
    modelling()


if __name__ == "__main__":
    main()
