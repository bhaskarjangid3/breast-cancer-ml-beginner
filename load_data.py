from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

# Load dataset
data = load_breast_cancer(as_frame=True)
X = data.data      # features (30 columns)
y = data.target    # labels (0=malignant, 1=benign)

# Print dataset overview
print("=== DATASET OVERVIEW ===")
print(f"Total samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Target distribution:\n{y.value_counts()}")
print(f"\nFeature names (first 5): {data.feature_names[:5]}")

print("\n=== BASIC STATISTICS ===")
print(X.describe())

print("\n=== MISSING VALUES ===")
print("Any missing values?", X.isnull().sum().sum())

print("\n=== FIRST FEW SAMPLES WITH TARGET ===")
df_sample = X.copy()
df_sample['target'] = y
print(df_sample.head())
