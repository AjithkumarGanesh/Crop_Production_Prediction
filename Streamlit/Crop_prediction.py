import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Production Prediction App", layout="wide")

# Sidebar with information and branding
with st.sidebar:
    st.image("Logo.jpeg", use_container_width=True)
    st.markdown("## About This App")
    st.write("This app predicts crop production based on input features using a machine learning model.")
    st.markdown("---")

# Function to load dataset and encode categories
@st.cache_data
def load_data():
    df = pd.read_csv("D:/Crop_Production_Prediction/Streamlit/Updated_data.csv")

    label_encoders = {}
    category_mappings = {}
    categorical_cols = ["Domain", "Area", "Item"]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        category_mappings[col] = dict(enumerate(le.classes_))  # {0: 'Value1', 1: 'Value2', ...}

    return df, label_encoders, category_mappings

# Function to train the model
@st.cache_resource
def train_model(df):
    best_params = {'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': None}
    model = RandomForestRegressor(**best_params, random_state=42)
    X = df.drop(columns=["Production"])
    y = df["Production"]
    model.fit(X, y)
    return model

# Load dataset and model once
df, label_encoders, category_mappings = load_data()
model = train_model(df)

# Main Content Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.title("Production Prediction App")
    st.write("Enter the details below to predict the production value.")

    domain = st.selectbox("Select Domain", category_mappings["Domain"].values())
    area = st.selectbox("Select Area", category_mappings["Area"].values())
    item = st.selectbox("Select Item", category_mappings["Item"].values())
    year = st.number_input("Enter Year", min_value=2000, max_value=2030, step=1)
    area_harvested = st.number_input("Enter Area Harvested", min_value=0.0, step=0.1)
    yield_value = st.number_input("Enter Yield", min_value=0.0, step=0.1)

with col2:
    st.markdown("### Input Summary")
    st.write(f"**Domain:** {domain}")
    st.write(f"**Area:** {area}")
    st.write(f"**Item:** {item}")
    st.write(f"**Year:** {year}")
    st.write(f"**Area Harvested:** {area_harvested} hectares")
    st.write(f"**Yield:** {yield_value} kg/ha")

# Function to encode user input safely
def safe_encode(value, mapping):
    return list(mapping.keys())[list(mapping.values()).index(value)] if value in mapping.values() else -1

# Predict production on button click
if st.button("Predict Production"):
    domain_enc = safe_encode(domain, category_mappings["Domain"])
    area_enc = safe_encode(area, category_mappings["Area"])
    item_enc = safe_encode(item, category_mappings["Item"])

    input_data = np.array([[domain_enc, area_enc, item_enc, year, area_harvested, yield_value]])
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Production: {prediction:,.2f} Metric Tons")
