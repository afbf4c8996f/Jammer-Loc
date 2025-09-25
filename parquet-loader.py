import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# 1) Load the processed dataset (NOT scaled!)

df = pd.read_parquet("data/processed_dataset.parquet")
print(f"Loaded {len(df)} samples with {df.shape[1]} columns.")
print(df.head(5))
print(df.columns)


df = pd.read_parquet("data/external_processed_dataset.parquet")
print(f"Loaded {len(df)} samples with {df.shape[1]} columns.")
print(df.head(5))
print(df.columns)

num_classes = df["label"].nunique()
print(f"Number of unique classes in 'label': {num_classes}")

classes = sorted(df["label"].unique())
print(classes)
