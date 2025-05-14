import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Create directory for models
print("Current working directory:", os.getcwd())

import traceback
try:
    os.makedirs('models', exist_ok=True)
except Exception as e:
    print("Failed to create directory:", e)
    traceback.print_exc()

# Define model path before checking
model_save_path = 'models/best_model.pkl'

# Save a dummy model file to simulate save
with open(model_save_path, 'wb') as f:
    f.write(b'Test model content')


print("\nChecking saved files...")
print(f"Model file exists: {os.path.exists(model_save_path)}")
