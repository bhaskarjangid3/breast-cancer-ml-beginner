from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

print("=== BREAST CANCER CLASSIFICATION v1 ===")

# 1. Load data
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 3. Train Logistic Regression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
print("Model trained!")

# 4. Predict & evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ FINAL ACCURACY: {accuracy:.3f} ({accuracy*100:.1f}%)")

print("\nDetailed report:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
