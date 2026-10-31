
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris(as_frame=True)

# Convert to DataFrame
df = iris.data.copy()
df["target"] = iris.target

# Create data directory
Path("data").mkdir(exist_ok=True)

# Save CSV file
df.to_csv("data/raw.csv", index=False)

print("Iris dataset created successfully!")
print("Total records:", len(df))
print("Columns:", list(df.columns))
print(df.head())
