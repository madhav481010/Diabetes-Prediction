import os
import pickle
import traceback
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc 
from xgboost import XGBClassifier

# Create directories
artifacts_dir = Path('artifacts')
artifacts_dir.mkdir(exist_ok=True)
os.makedirs('models', exist_ok=True)

# Show working directory
print("Current working directory:", os.getcwd())
os.makedirs('models', exist_ok=True)

# Load processed data
try:
    with open('data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('data/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
except Exception as e:
    print("Error loading data:", e)
    traceback.print_exc()
    exit(1)

# Initialize classifiers
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Define Stacking Classifier
stacking_model = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ],
    final_estimator=LogisticRegression()
)
models['Stacking Ensemble'] = stacking_model

# Metric Weights for Healthcare Priority
weights = {
    'Accuracy': 0.05,
    'Precision': 0.05,
    'Recall': 0.4,
    'F1-Score': 0.3,
    'ROC-AUC': 0.2
}

# Initialize tracking
best_model = None
best_score = 0.0
best_model_name = ""
results = {}

print("\nTraining and evaluating models...\n")

# Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Needed for ROC-AUC

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Weighted Score
    score = (weights['Accuracy'] * acc +
             weights['Precision'] * prec +
             weights['Recall'] * rec +
             weights['F1-Score'] * f1 +
             weights['ROC-AUC'] * roc_auc)

    # Store results
    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'weighted_score': score
    }

    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(artifacts_dir / f'roc_{name.lower().replace(" ", "_")}.png')
    plt.close()

    # Display results
    print(f"\n{name}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  Weighted Score: {score:.4f}")

    # Track best model
    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

# Save metrics

# Save the best model
model_save_path = 'models/best_model.pkl'
metrics_save_path = 'models/model_metrics.json'
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(metrics_save_path, 'wb') as f:
        pickle.dump(results, f)   
except Exception as e:
    print("Error saving model:", e)
    traceback.print_exc()
    exit(1)

print("\nModel training complete.")
print("\nBest Model Based on Weighted Healthcare Metrics:")
print(f"{best_model_name} with Weighted Score: {best_score:.4f}")
print(f"Saved model at: {os.path.abspath(model_save_path)}")
print(f"Saved model at: {os.path.abspath(metrics_save_path)}")

# Final file existence check
print("\nChecking saved files...")
print(f"Model file exists: {os.path.exists(model_save_path)}")
print(f"Model file exists: {os.path.exists(metrics_save_path)}")
print(f"Artifacts directory exists: {artifacts_dir.exists()}")
print("Training complete. Artifacts saved in artifacts/ directory")
