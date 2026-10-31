
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.pipeline import clean_data, run_pipeline


def sample_data():
    return pd.DataFrame({
        "sepal length (cm)": [
            5.1, 4.9, 5.0, 6.0, 6.1, 6.2,
            6.3, 6.4, 6.5, 4.8, 4.7, 4.6,
            5.4, 5.5, 5.6
        ],
        "sepal width (cm)": [
            3.5, 3.0, None, 2.8, 2.9, 3.0,
            3.3, 3.2, 3.1, 3.1, 3.2, 3.3,
            3.7, 3.6, 3.5
        ],
        "petal length (cm)": [
            1.4, 1.4, 1.5, 4.5, 4.6, 4.7,
            5.0, 5.1, 5.2, 1.3, 1.2, 1.1,
            1.5, 1.6, 1.7
        ],
        "petal width (cm)": [
            0.2, 0.2, 0.2, 1.3, 1.4, 1.5,
            1.8, 1.8, 1.9, 0.2, 0.2, 0.1,
            0.4, 0.4, 0.5
        ],
        "target": [
            0, 0, 0, 1, 1, 1, 2, 2,
            2, 0, 0, 0, 1, 1, 1
        ],
    })


def test_clean_data_fills_missing_and_engineers_feature():
    cleaned = clean_data(sample_data())

    assert cleaned.isna().sum().sum() == 0
    assert "petal_area_cm2" in cleaned.columns
    assert (cleaned["petal_area_cm2"] >= 0).all()


def test_pipeline_writes_processed_train_and_test(tmp_path):
    raw_path = tmp_path / "sample.csv"
    out_dir = tmp_path / "output"

    sample_data().to_csv(raw_path, index=False)

    outputs = run_pipeline(
        input_path=raw_path,
        output_dir=out_dir
    )

    assert outputs["processed"].exists()
    assert outputs["train"].exists()
    assert outputs["test"].exists()

    train = pd.read_csv(outputs["train"])
    test = pd.read_csv(outputs["test"])

    assert len(train) + len(test) == 15
    assert "target" in train.columns
    assert "target" in test.columns


def test_model_predictions_have_valid_shape_and_labels():
    cleaned = clean_data(sample_data())

    features = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "petal_area_cm2"
    ]

    model = LogisticRegression(max_iter=1000).fit(
        cleaned[features],
        cleaned["target"]
    )

    predictions = model.predict(cleaned[features])

    assert len(predictions) == len(cleaned)
    assert set(predictions).issubset({0, 1, 2})
