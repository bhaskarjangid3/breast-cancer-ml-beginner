from sklearn.datasets import load_breast_cancer
import pandas as pd

# 1. Load dataset
data = load_breast_cancer(as_frame=True)
X = data.data          # features
y = data.target        # target (0 = malignant, 1 = benign)

# 2. Combine into one DataFrame
df = X.copy()
df["target"] = y

# 3. Basic EDA
print("=== SHAPE ===")
print(df.shape)

print("\n=== TARGET COUNTS ===")
print(df["target"].value_counts())

print("\n=== BASIC STATS (first 5 features) ===")
print(df.iloc[:, :5].describe())

print("\n=== FIRST 5 ROWS ===")
print(df.head())
