# test_model.py
import pandas as pd
import numpy as np

# Load your training data
train = pd.read_csv("train.csv")
print(f"Data loaded: {len(train)} rows")
print(f"\nColumns: {train.columns.tolist()}")
print(f"\nFirst few rows:")
print(train.head())

# Check the target column
if "TARGET(PRICE_IN_LACS)" in train.columns:
    print(f"\nPrice statistics:")
    print(f"Min: {train['TARGET(PRICE_IN_LACS)'].min()}")
    print(f"Max: {train['TARGET(PRICE_IN_LACS)'].max()}")
    print(f"Mean: {train['TARGET(PRICE_IN_LACS)'].mean()}")
    print(f"Std: {train['TARGET(PRICE_IN_LACS)'].std()}")
    
    # Check for constant values
    unique_prices = train['TARGET(PRICE_IN_LACS)'].unique()
    print(f"\nUnique price values: {len(unique_prices)}")
    if len(unique_prices) < 10:
        print(f"Prices: {unique_prices}")
else:
    print("\nERROR: 'TARGET(PRICE_IN_LACS)' column not found!")
    print(f"Available columns: {train.columns.tolist()}")

# Check feature columns
features_needed = ["UNDER_CONSTRUCTION", "RERA", "BHK_NO.", "SQUARE_FT", "READY_TO_MOVE", "RESALE"]
print(f"\nChecking feature columns:")
for feature in features_needed:
    if feature in train.columns:
        unique_vals = train[feature].unique()[:5]
        print(f"{feature}: ✓ ({len(train[feature].unique())} unique values, sample: {unique_vals})")
    else:
        print(f"{feature}: ✗ MISSING!")

# Check for all 1's or 0's
print(f"\nChecking for constant columns:")
for col in train.columns:
    if train[col].nunique() == 1:
        print(f"Warning: {col} has only 1 unique value: {train[col].iloc[0]}")