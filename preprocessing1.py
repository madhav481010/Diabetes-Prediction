# preprocess.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv("https://raw.githubusercontent.com/madhav481010/Diabetes-Prediction/main/diabetes.csv", names=columns, header=0)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs('data', exist_ok=True)
with open('data/X_train.pkl', 'wb') as f: pickle.dump(X_train_scaled, f)
with open('data/X_test.pkl', 'wb') as f: pickle.dump(X_test_scaled, f)
with open('data/y_train.pkl', 'wb') as f: pickle.dump(y_train, f)
with open('data/y_test.pkl', 'wb') as f: pickle.dump(y_test, f)
with open('data/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)

print("Data preprocessing completed.")
