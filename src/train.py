
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

FEATURES = [
    "sepal length (cm)", "sepal width (cm)",
    "petal length (cm)", "petal width (cm)", "petal_area_cm2",
]

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

X_train, y_train = train_df[FEATURES], train_df["target"]
X_test, y_test = test_df[FEATURES], test_df["target"]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris MLOps Practical")
mlflow.sklearn.autolog(registered_model_name="IrisClassifier")

model = LogisticRegression(max_iter=1000, random_state=42)

with mlflow.start_run(run_name="baseline-logistic-regression"):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_macro", f1)

    print("Test accuracy:", round(accuracy, 4))
    print("Test macro F1:", round(f1, 4))

Path("models").mkdir(exist_ok=True)

joblib.dump(
    {"model": model, "feature_names": FEATURES},
    "models/iris_model.joblib"
)

print("Saved models/iris_model.joblib")
