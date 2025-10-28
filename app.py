import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import joblib
import os

# Define paths to saved artifacts
ARTIFACTS_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'logistic_regression_model.pkl')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'standard_scaler.pkl')

# --- Step 1: Loading Saved Model and Scaler ---

@st.cache_resource
def load_artifacts():
    """Loads the pre-trained model and fitted scaler."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model or Scaler not found. Please run 'training_script.py' first to generate '{ARTIFACTS_DIR}' directory.")
        st.stop() # Stop the app if artifacts are missing

model, scaler = load_artifacts()

# Get feature names from the dataset (needed for slider defaults)
cancer_data = load_breast_cancer()
feature_names = cancer_data.feature_names
target_names = cancer_data.target_names


# --- Step 2: Streamlit GUI Setup and Input ---

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

st.title("🔬 Breast Cancer Diagnostic Predictor")
st.markdown("""
Use the sliders on the left to input tumor measurements. This application uses a pre-trained **Logistic Regression** model to predict the diagnosis.
""")

st.sidebar.header("Tumor Feature Input")

def user_input_features():
    """Creates sidebar input widgets for the user to enter feature values."""
    
    # Use key predictive features for user input
    
    # Find min, max, and mean values across the entire dataset for slider defaults
    df_full = pd.DataFrame(cancer_data.data, columns=feature_names)
    
    radius_mean = st.sidebar.slider(
        'Mean Radius (size of the core)', 
        df_full['mean radius'].min(), 
        df_full['mean radius'].max(), 
        df_full['mean radius'].mean()
    )
    
    perimeter_mean = st.sidebar.slider(
        'Mean Perimeter (outline length)', 
        df_full['mean perimeter'].min(), 
        df_full['mean perimeter'].max(), 
        df_full['mean perimeter'].mean()
    )
    
    area_mean = st.sidebar.slider(
        'Mean Area (total surface area)', 
        df_full['mean area'].min(), 
        df_full['mean area'].max(), 
        df_full['mean area'].mean()
    )

    # Create the full 30-feature input array, filling un-controlled features with the dataset's mean
    data = dict(zip(feature_names, np.mean(cancer_data.data, axis=0)))
    data['mean radius'] = radius_mean
    data['mean perimeter'] = perimeter_mean
    data['mean area'] = area_mean
    
    # Convert to DataFrame for consistent processing
    features = pd.DataFrame([data])
    return features

# Get the user input
input_df = user_input_features()

st.subheader('User Input Features')
st.dataframe(input_df[['mean radius', 'mean perimeter', 'mean area']])


# --- Step 3: Prediction and Display ---

# 1. Scale the input data using the *fitted* scaler
# The scaler expects all 30 features
scaled_data = scaler.transform(input_df)

# 2. Make the prediction
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

st.subheader('Prediction')

# Determine the result message and color
if prediction[0] == 0:
    result_text = f"**Malignant ({target_names[0].capitalize()})**"
    color = "red"
    prob = prediction_proba[0][0] # Probability of Class 0
else:
    result_text = f"**Benign ({target_names[1].capitalize()})**"
    color = "green"
    prob = prediction_proba[0][1] # Probability of Class 1

# Display the main prediction with custom styling
st.markdown(f"""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {color};'>
    <h3 style='color: {color}; margin-top: 0;'>Diagnosis: {result_text}</h3>
</div>
""", unsafe_allow_html=True)


# Display the probability/confidence
st.markdown(f"""
<br>
**Prediction Confidence for {target_names[prediction[0]].capitalize()}:** **{prob * 100:.2f}%**
""")
st.progress(prob)