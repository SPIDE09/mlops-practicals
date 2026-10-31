from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Iris dataset:
    1. Validate required columns.
    2. Convert features to numeric.
    3. Handle missing values.
    4. Create petal_area_cm2 feature.
    """

    if "target" not in df.columns:
        raise ValueError("Required target column is missing")

    result = df.copy()

    for col in FEATURES:
        if col not in result.columns:
            raise ValueError(f"Missing feature column: {col}")

        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        )

        result[col] = result[col].fillna(
            result[col].median()
        )

    result["target"] = pd.to_numeric(
        result["target"],
        errors="coerce"
    )

    if result[FEATURES + ["target"]].isna().any().any():
        raise ValueError("Null values remain after cleaning")

    result["target"] = result["target"].astype(int)

    # Feature engineering
    result["petal_area_cm2"] = (
        result["petal length (cm)"]
        * result["petal width (cm)"]
    )

    return result


def run_pipeline(input_path="data/raw.csv", output_dir="data"):
    """
    Execute the complete preprocessing pipeline.
    """

    # Load raw dataset
    raw = pd.read_csv(input_path)

    print("Raw shape:", raw.shape)

    print(
        "Missing values before cleaning:",
        int(raw.isna().sum().sum())
    )

    # Clean dataset
    processed = clean_data(raw)

    print(
        "Missing values after cleaning:",
        int(processed.isna().sum().sum())
    )

    # Split dataset
    train_df, test_df = train_test_split(
        processed,
        test_size=0.2,
        random_state=42,
        stratify=processed["target"]
    )

    # Create output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save datasets
    processed_path = out / "processed.csv"
    train_path = out / "train.csv"
    test_path = out / "test.csv"

    processed.to_csv(processed_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Processed:", processed.shape)
    print("Train:", train_df.shape)
    print("Test:", test_df.shape)

    # Return output file paths
    return {
        "processed": processed_path,
        "train": train_path,
        "test": test_path
    }


if __name__ == "__main__":
    run_pipeline()