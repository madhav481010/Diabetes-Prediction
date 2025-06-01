import os
import pickle
import time
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import numpy as np

# Create directory for models
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
    print(f"Error loading data: {e}")
    traceback.print_exc()
    exit(1)

# Hyperparameter Tuning Function
def tune_model(model, param_grid, X_train, y_train, time_budget=60):
    """
    Generic hyperparameter tuning function with time budget
    
    Args:
        model: sklearn model instance
        param_grid: Dictionary of parameters to tune
        X_train, y_train: Training data
        time_budget: Maximum tuning time in seconds
        
    Returns:
        best_model: Tuned model instance
        best_params: Dictionary of best parameters
    """
    # Adjust n_iter based on time budget (heuristic: 5 iterations per minute)
    n_iter = max(5, min(50, int(time_budget / 60 * 5)))
    
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=make_scorer(recall_score),  # Focus on recall for healthcare
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    
    start_time = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    
    print(f"\nTuning completed in {elapsed:.2f}s")
    print(f"Best recall: {search.best_score_:.4f}")
    print("Best parameters:", search.best_params_)
    
    return search.best_estimator_, search.best_params_

# Parameter grids for different models
PARAM_GRIDS = {
    'LogisticRegression': {
        'C': np.logspace(-3, 2, 6),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVC': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    },
    'DecisionTreeClassifier': {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    },
    'XGBClassifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

# Initialize models with basic configuration
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Metric weights for healthcare priority
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
all_results = {}

print("\nStarting model training with hyperparameter tuning...\n")

# Train and evaluate each model
for name, model in models.items():
    print(f"\n=== Processing {name} ===")
    
    # Get model class name for parameter grid lookup
    model_class = type(model).__name__
    
    # Perform tuning
    tuned_model, best_params = tune_model(
        model=model,
        param_grid=PARAM_GRIDS[model_class],
        X_train=X_train,
        y_train=y_train,
        time_budget=120  # 2 minutes per model
    )
    
    # Evaluate on test set
    y_pred = tuned_model.predict(X_test)
    y_proba = tuned_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Parameters': best_params
    }
    
    # Calculate weighted score
    metrics['Weighted_Score'] = sum(weights[metric] * value 
                                for metric, value in metrics.items() 
                                if metric in weights)
    
    # Store results
    all_results[name] = metrics
    
    # Print results
    print(f"\n{name} Results:")
    for metric, value in metrics.items():
        if metric not in ['Parameters', 'Weighted_Score']:
            print(f"  {metric:<12}: {value:.4f}")
    print(f"  Weighted Score: {metrics['Weighted_Score']:.4f}")
    
    # Track best model
    if metrics['Weighted_Score'] > best_score:
        best_score = metrics['Weighted_Score']
        best_model = tuned_model
        best_model_name = name

# Add Stacking Ensemble (using best base models)
print("\n=== Creating Stacking Ensemble ===")
stacking_model = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(**all_results['Logistic Regression']['Parameters'],
                                 max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(**all_results['Random Forest']['Parameters'],
                                    random_state=42)),
        ('xgb', XGBClassifier(**all_results['XGBoost']['Parameters'],
                             use_label_encoder=False, eval_metric='logloss',
                             random_state=42))
    ],
    final_estimator=LogisticRegression(),
    n_jobs=-1
)

stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
y_proba = stacking_model.predict_proba(X_test)[:, 1]

# Calculate metrics first
stack_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_proba),
    'Parameters': 'Stacking of best models'
}

# Calculate weighted score using the fully populated dictionary
stack_metrics['Weighted_Score'] = sum(weights[metric] * stack_metrics[metric]
                                      for metric in weights) # Iterate over weights keys

all_results['Stacking Ensemble'] = stack_metrics

# Final evaluation
print("\n=== Final Results ===")
for name, metrics in sorted(all_results.items(),
                          key=lambda x: x[1]['Weighted_Score'],
                          reverse=True):
    print(f"\n{name}:")
    for metric, value in metrics.items():
        if metric != 'Parameters':
            print(f"  {metric:<15}: {value:.4f}" if isinstance(value, float)
                  else f"  {metric:<15}: {value}")

# Save the best model
model_save_path = 'models/best_model.pkl'
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save all results for analysis
    results_save_path = 'models/training_results.pkl'
    with open(results_save_path, 'wb') as f:
        pickle.dump({
            'best_model_name': best_model_name,
            'best_score': best_score,
            'best_params': all_results[best_model_name]['Parameters'],
            'all_metrics': all_results,
            'weights_used': weights
        }, f)
        
except Exception as e:
    print(" Error saving model:", e)
    traceback.print_exc()
    exit(1)

print("\nModel training and tuning complete.")
print("\n=== Best Model Summary ===")
print(f"Model Name: {best_model_name}")
print(f"Weighted Score: {best_score:.4f}")
print(f"Recall: {all_results[best_model_name]['Recall']:.4f}")
print(f"ROC-AUC: {all_results[best_model_name]['ROC-AUC']:.4f}")
print("\nBest Parameters:")
for param, value in all_results[best_model_name]['Parameters'].items():
    print(f"  {param}: {value}")
print(f"\nSaved model at: {os.path.abspath(model_save_path)}")
print(f"Saved results at: {os.path.abspath(results_save_path)}")

# Final file existence check
print("\nChecking saved files...")
print(f"Model file exists: {os.path.exists(model_save_path)}")
print(f"Results file exists: {os.path.exists(results_save_path)}")

print("\n=== Top 3 Models ===")
top_models = sorted(all_results.items(), key=lambda x: x[1]['Weighted_Score'], reverse=True)[:3]
for i, (name, metrics) in enumerate(top_models, 1):
    print(f"\n{i}. {name}")
    print(f"   Weighted Score: {metrics['Weighted_Score']:.4f}")
    print(f"   Recall: {metrics['Recall']:.4f}")
    print(f"   ROC-AUC: {metrics['ROC-AUC']:.4f}")

print("\nTraining pipeline completed successfully!")
