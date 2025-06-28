# preprocess.py
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pickle
from sklearn.feature_selection import SelectKBest, f_classif

# Load data
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv("https://raw.githubusercontent.com/madhav481010/Diabetes-Prediction/main/diabetes.csv", 
                 names=columns, header=0)

# Replace biologically implausible zeroes with NaN
na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[na_columns] = df[na_columns].replace(0, np.nan)

# Feature Engineering: Create new relevant features
df['Glucose_BMI_Interaction'] = df['Glucose'] * df['BMI']
df['Age_Glucose_Ratio'] = df['Age'] / df['Glucose']
df['BloodPressure_Age_Interaction'] = df['BloodPressure'] * df['Age']
df['Metabolic_Score'] = (df['Glucose'] * df['Insulin']) / (df['BMI'] + 1e-6)
df['Pregnancies_Age_Interaction'] = df['Pregnancies'] * df['Age']

# Select most relevant features based on domain knowledge and statistical significance
selected_features = [
    'Glucose',  # Strong diabetes indicator
    'BMI',      # Obesity factor
    'Age',      # Diabetes risk increases with age
    'Glucose_BMI_Interaction',  # Combined effect important
    'Pregnancies',  # Gestational diabetes history
    'BloodPressure',  # Hypertension connection
    'Metabolic_Score'  # Custom metabolic indicator
]

# Separate features and labels
X = df[selected_features]
y = df['Outcome']

# Apply KNN imputation (after standardizing for distance-based algorithm)
scaler_before_impute = StandardScaler()
X_scaled_for_knn = scaler_before_impute.fit_transform(X)

# KNN Imputer (n_neighbors=5 is common)
imputer = KNNImputer(n_neighbors=5)
X_imputed_scaled = imputer.fit_transform(X_scaled_for_knn)

# Convert back to DataFrame with original column names
X_imputed = pd.DataFrame(scaler_before_impute.inverse_transform(X_imputed_scaled), 
                         columns=X.columns)

# Clip values after imputation to ensure physiological plausibility
X_imputed['BloodPressure'] = X_imputed['BloodPressure'].clip(lower=40, upper=140)
X_imputed['BMI'] = X_imputed['BMI'].clip(lower=15, upper=50)
X_imputed['Glucose'] = X_imputed['Glucose'].clip(lower=50, upper=200)

# Feature selection using ANOVA F-value
selector = SelectKBest(f_classif, k=5)  # Select top 5 features
X_selected = selector.fit_transform(X_imputed, y)

# Get selected feature names
selected_mask = selector.get_support()
selected_feature_names = X_imputed.columns[selected_mask]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, 
                                                   test_size=0.2, 
                                                   random_state=42,
                                                   stratify=y)

# Final scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data and feature names
os.makedirs('data', exist_ok=True)
with open('data/X_train.pkl', 'wb') as f: pickle.dump(X_train_scaled, f)
with open('data/X_test.pkl', 'wb') as f: pickle.dump(X_test_scaled, f)
with open('data/y_train.pkl', 'wb') as f: pickle.dump(y_train, f)
with open('data/y_test.pkl', 'wb') as f: pickle.dump(y_test, f)
with open('data/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('data/imputer.pkl', 'wb') as f: pickle.dump(imputer, f)
with open('data/feature_names.pkl', 'wb') as f: pickle.dump(selected_feature_names, f)
with open('data/feature_selector.pkl', 'wb') as f: pickle.dump(selector, f)

print("Enhanced preprocessing with feature engineering completed.")
print(f"Selected features: {list(selected_feature_names)}")
