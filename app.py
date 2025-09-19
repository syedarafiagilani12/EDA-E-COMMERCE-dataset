import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("📊 Exploratory Data Analysis (EDA) App")

# Load dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset preview
    st.subheader("🔍 Dataset Preview")
    st.write(df.head())

    # Shape of dataset
    st.write("**Shape of dataset:**", df.shape)

    # Missing values
    st.subheader("🧹 Missing Values")
    st.write(df.isnull().sum())

    # Summary statistics
    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include="all"))

    # Correlation heatmap (for numeric columns)
    st.subheader("🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns available for correlation heatmap.")

    # Column-wise analysis
    st.subheader("📊 Column-wise Analysis")
    column = st.selectbox("Select a column to analyze", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(df[column].describe())
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[column], kde=True, bins=20, ax=ax)
        ax.set_title(f"Distribution of {column}")
        st.pyplot(fig)
    else:
        st.write(df[column].value_counts())
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=df[column], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title(f"Count plot of {column}")
        st.pyplot(fig)

    # Pairplot (optional for smaller datasets)
    if st.checkbox("Show Pairplot (may be slow for large data)"):
        fig = sns.pairplot(df.select_dtypes(include=['int64','float64']))
        st.pyplot(fig)
