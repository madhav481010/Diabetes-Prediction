# train_and_evaluate.py
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

with open('data/X_train.pkl', 'rb') as f: X_train = pickle.load(f)
with open('data/X_test.pkl', 'rb') as f: X_test = pickle.load(f)
with open('data/y_train.pkl', 'rb') as f: y_train = pickle.load(f)
with open('data/y_test.pkl', 'rb') as f: y_test = pickle.load(f)

models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'DecisionTree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

os.makedirs('models', exist_ok=True)
with open('models/best_model.pkl', 'wb') as f: pickle.dump(best_model, f)

print(f"✅ Best model saved: {best_name} with accuracy {best_accuracy}")
