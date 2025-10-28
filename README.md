Breast Cancer Diagnostic Predictor 🔬
This project demonstrates a minimal viable product (MVP) machine learning workflow, from data exploration and model training to an interactive web-based GUI built with Streamlit.

Project Goal
Primary Goal: Successfull implement a full MLOps cycle, from data loading to deployment (via Streamlit GUI).
Business Objective: Provide a fast, easy-to-use proof-of-concept tool to demonstrate the diagnostic power of breast cancer dataset.

Targeted Audience
Primary: Data Scientists/Engneers (for reviewing the code and pipeline).
Secondary: Medical Students or Researchers (for exploring the relationship between features and diagnosis).

Scope 
Feature ID,Feature Name,Description
FEA-001,Data Pipeline,"Script for data loading, EDA, and model persistence (training_script.py)."
FEA-002,Model Artifacts,Trained Logistic Regression model and fitted StandardScaler saved using joblib.
FEA-003,Interactive GUI,Web application built with Streamlit that loads the artifacts.
FEA-004,User Input,"Interactive sliders for mean radius, mean perimeter, and mean area."
FEA-005,Prediction Output,Clear display of the final predicted class (Malignant/Benign) and the model's confidence probability.

Techincal Requirements 
Technology Stack: Python (3.8+), pandas, NumPy, Scikit-learn, matplotlib, Seaborn, Joblib, Streamlit.

model: Logistic Regression (Simple, Interpreatable Baseline).
Preprocessing: Standard Scaling applied to all 30 features. 
Persistence: Model and Scaler must be serialized to .pkl files.

Success Criteria 

Metric,Target Value,Notes
Model Performance,Accuracy ≥0.95,Achievable with Logistic Regression on this dataset.
Artifact Loading,Artifacts loaded in ≤5 seconds,Ensures fast GUI startup via @st.cache_resource.
Code Quality,PEP 8 Compliant,"Code must be clean, commented, and follow Python standards."



🚀 Getting Started
Prerequisites
You must have Python 3.8+ installed.

Bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate
Installation
Install all required libraries:

Bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
Step 1: Train the Model
The training_script.py handles data preprocessing, model fitting, evaluation, and saves the final artifacts needed for the GUI.

Bash
python training_script.py
This step will create a directory named model_artifacts/ containing two crucial files:

model_artifacts/logistic_regression_model.pkl

model_artifacts/standard_scaler.pkl

Step 2: Run the Web Application (GUI)
Once the artifacts are saved, run the Streamlit application:

Bash
streamlit run app.py
A new window will open in your default browser at http://localhost:8501. You can use the sliders in the sidebar to test predictions in real-time.

📁 File Structure
breast_cancer_predictor/
├── training_script.py      # Core script for data science workflow (EDA, Training, Saving)
├── app.py                  # Streamlit web application (Loading, Prediction, GUI)
├── model_artifacts/        # Directory for persistent model components
│   ├── logistic_regression_model.pkl
│   └── standard_scaler.pkl
├── README.md               # This file
├── ... (other documentation files)
🛠 Model Details
Model Type: Logistic Regression

Preprocessing: Standard Scaling (StandardScaler)

Evaluation (on Test Set): Typically achieves an Accuracy of 95% - 98%.

💡 Key Features of the GUI
Interactive Input: Control the three most influential features (mean radius, mean perimeter, mean area).

Safe Prediction: The GUI correctly scales the user input before passing it to the model, preventing prediction errors.

Confidence Score: Displays the probability of the predicted class, giving insight into the model's certainty.