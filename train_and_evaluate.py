import os
import pickle
import traceback
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc 
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

# Show working directory
print("Current working directory:", os.getcwd())

# Create directory for models
try:
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)  # For saving plots
except Exception as e:
    print("Failed to create directories:", e)
    traceback.print_exc()

# Load processed data and feature names
try:
    with open('data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('data/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    with open('data/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except Exception as e:
    print("Error loading data:", e)
    traceback.print_exc()
    exit(1)

print("\nSelected Features:", list(feature_names))

# Initialize classifiers with optimized parameters for the selected features
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced_subsample',
        random_state=42
    ),
    'Support Vector Machine': SVC(
        probability=True,
        class_weight='balanced',
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=7,
        weights='distance'
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
}

# Define Stacking Classifier with the top performing base models
stacking_model = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)
models['Stacking Ensemble'] = stacking_model

# Metric Weights for Healthcare Priority (emphasizing recall for diabetes detection)
weights = {
    'Accuracy': 0.05,
    'Precision': 0.15,
    'Recall': 0.45,  # Higher weight to minimize false negatives
    'F1-Score': 0.25,
    'ROC-AUC': 0.10
}

# Initialize tracking
best_model = None
best_score = 0.0
best_model_name = ""
results = {}

print("\nTraining and evaluating models...\n")

# Evaluate models
for name, model in models.items():
    try:
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

        # Plot feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.title(f"{name} - Feature Importance")
            plt.bar(range(X_train.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig(f'plots/{name.replace(" ", "_")}_feature_importance.png')
            plt.close()

    except Exception as e:
        print(f"Error with {name}: {e}")
        traceback.print_exc()

# Save the best model and all results
model_save_path = 'models/best_model.pkl'
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save results as JSON
    with open('models/model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # Save feature names for later use
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
except Exception as e:
    print("Error saving model or results:", e)
    traceback.print_exc()
    exit(1)

print("\nModel training complete.")
print("\nBest Model Based on Weighted Healthcare Metrics:")
print(f"{best_model_name} with Weighted Score: {best_score:.4f}")
print(f"Saved model at: {os.path.abspath(model_save_path)}")

# Generate and save ROC curve for best model
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {results[best_model_name]["roc_auc"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Best Model')
plt.legend(loc='lower right')
plt.savefig('plots/best_model_roc_curve.png')
plt.close()

# Final file existence check
print("\nChecking saved files...")
print(f"Model file exists: {os.path.exists(model_save_path)}")
print(f"Results file exists: {os.path.exists('models/model_results.json')}")
print(f"Feature names file exists: {os.path.exists('models/feature_names.pkl')}")
