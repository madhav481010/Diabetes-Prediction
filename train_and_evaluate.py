import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
with open('data/X_train.pkl', 'rb') as f: X_train = pickle.load(f)
with open('data/X_test.pkl', 'rb') as f: X_test = pickle.load(f)
with open('data/y_train.pkl', 'rb') as f: y_train = pickle.load(f)
with open('data/y_test.pkl', 'rb') as f: y_test = pickle.load(f)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Debugging paths
model_path = os.path.abspath('models/best_model.pkl')
scaler_path = os.path.abspath('data/scaler.pkl')
print("Saving model to:", model_path)
print("Saving scaler to:", scaler_path)

# Initialize models
models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'DecisionTree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier()
}

best_model = None
best_accuracy = 0
best_name = ""

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

# Save best model
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

# Save scaler
scaler = StandardScaler().fit(X_train)
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Output best model info
print(f"Best model saved: {best_name} with accuracy {best_accuracy}")
print(os.getcwd())

# Check if files exist
print("Checking if files exist...")
print(f"Model exists: {os.path.exists(model_path)}")
print(f"Scaler exists: {os.path.exists(scaler_path)}")
