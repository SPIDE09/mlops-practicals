
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score


# 1. Define features
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "petal_area_cm2",
]


# 2. Load datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")


# 3. Separate features and target
X_train = train_df[FEATURES]
y_train = train_df["target"]

X_test = test_df[FEATURES]
y_test = test_df["target"]


# 4. Define hyperparameter search space
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 4],
}


# 5. Create baseline Random Forest model
base = RandomForestClassifier(random_state=42)


# 6. Perform Grid Search
print("\nRunning Grid Search...")

grid = GridSearchCV(
    estimator=base,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

grid.fit(X_train, y_train)

print("Grid Search completed.")
print("Grid Search best parameters:", grid.best_params_)


# 7. Perform Randomized Search
print("\nRunning Randomized Search...")

random_search = RandomizedSearchCV(
    estimator=base,
    param_distributions=param_grid,
    n_iter=6,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_train, y_train)

print("Randomized Search completed.")
print("Randomized Search best parameters:", random_search.best_params_)


# 8. Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris Hyperparameter Search")


# 9. Evaluate and track both search methods
results = []

search_methods = [
    ("GridSearchCV", grid),
    ("RandomizedSearchCV", random_search),
]

for name, search in search_methods:

    best_model = search.best_estimator_

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    results.append({
        "method": name,
        "best_cv_accuracy": search.best_score_,
        "test_accuracy": accuracy,
        "test_f1_macro": f1,
        "best_params": str(search.best_params_),
    })

    print(f"\nTracking {name} in MLflow...")

    with mlflow.start_run(run_name=name):

        mlflow.log_param("search_method", name)

        mlflow.log_params(search.best_params_)

        mlflow.log_metric(
            "best_cv_accuracy",
            search.best_score_,
        )

        mlflow.log_metric(
            "test_accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "test_f1_macro",
            f1,
        )

        mlflow.sklearn.log_model(
            best_model,
            name="best_model",
            serialization_format="cloudpickle",
        )

    print(f"{name} logged successfully.")


# 10. Generate comparison report
comparison = pd.DataFrame(results)

Path("reports").mkdir(
    parents=True,
    exist_ok=True,
)

comparison.to_csv(
    "reports/search_comparison.csv",
    index=False,
)

print("\nHYPERPARAMETER SEARCH COMPARISON")
print(comparison.to_string(index=False))


# 11. Select the best model using cross-validation score
winner_name, winner = max(
    search_methods,
    key=lambda item: item[1].best_score_,
)


# 12. Save the selected model
Path("models").mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    {
        "model": winner.best_estimator_,
        "feature_names": FEATURES,
    },
    "models/iris_model.joblib",
)


# 13. Display final results
print("\nFINAL RESULTS")
print("------------------------------")

print("Selected method:", winner_name)

print("Best parameters:", winner.best_params_)

print("Best CV accuracy:", round(winner.best_score_, 4))

print("Saved model: models/iris_model.joblib")

print("Saved report: reports/search_comparison.csv")

print("\nExperiment 7 completed successfully!")
