import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# --- Setup and Data Loading ---
print("--- Starting Project Setup and Training ---")

# Load the dataset
cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target

# Create directory for saving model artifacts
ARTIFACTS_DIR = 'model_artifacts'
if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)
    print(f"Created directory: {ARTIFACTS_DIR}")


# --- Exploratory Data Analysis (EDA) ---
print("\n--- Running EDA ---")

# 1. Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='viridis')
plt.title('Target Distribution (0: Malignant, 1: Benign)')
plt.savefig(os.path.join(ARTIFACTS_DIR, 'target_distribution.png'))
# plt.show() # Uncomment to see the plot immediately

# 2. Key Feature Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x='target', y='mean radius', data=df, palette='coolwarm')
plt.title('Mean Radius vs. Target')
plt.savefig(os.path.join(ARTIFACTS_DIR, 'mean_radius_boxplot.png'))
# plt.show() # Uncomment to see the plot immediately

print("EDA plots saved to 'model_artifacts' directory.")


# --- Preprocessing and Modeling ---

# 1. Define X and y
X = df.drop('target', axis=1)
y = df['target']

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training (Logistic Regression)
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

print("\nModel trained successfully!")


# --- Model Evaluation ---
y_pred = model.predict(X_test_scaled)

print("\n--- Model Evaluation Results ---")
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer_data.target_names))


# --- Saving Model and Scaler (Serialization) ---
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'logistic_regression_model.pkl')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'standard_scaler.pkl')

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")
print("--- Training Script Complete ---")