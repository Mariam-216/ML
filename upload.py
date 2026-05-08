import streamlit as st
import pandas as pd

if "data" not in st.session_state:
    st.session_state["data"] = None
if "data_processed" not in st.session_state:
    st.session_state["data_processed"] = None
if "filename" not in st.session_state:
    st.session_state["filename"] = ""



st.title(" File Upload")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state["data"] = df
        st.session_state["filename"] = uploaded_file.name

        st.success(f" File '{uploaded_file.name}' uploaded successfully!")

        st.subheader("Data Preview")
        st.dataframe(df.head(10))

        st.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

        st.subheader("Quick Info")
        st.write(f"Missing Values: {df.isnull().sum().sum()}")
        st.dataframe(df.dtypes.rename("Type").reset_index())
        st.dataframe(df.describe())

    except Exception as e:
        st.error(f" Error reading file: {e}")

else:
    st.warning(" Please upload a CSV or Excel file to continue.")
