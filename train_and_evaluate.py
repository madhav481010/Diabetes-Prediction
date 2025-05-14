import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load processed data
with open('data/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('data/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('data/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Create directory for models
os.makedirs('models', exist_ok=True)

# Initialize classifiers
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

best_model = None
best_accuracy = 0
best_model_name = ""

# Train and evaluate models
print("\nTraining and evaluating models...\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Save the best model
model_save_path = 'models/best_model.pkl'
with open(model_save_path, 'wb') as f:
    pickle.dump(best_model, f)

print("\n✔️ Model training complete.")
print(f"🏆 Best Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
print(f"📁 Saved model at: {os.path.abspath(model_save_path)}")

# Final file existence check
print("\nChecking saved files...")
print(f"Model file exists: {os.path.exists(model_save_path)}")
