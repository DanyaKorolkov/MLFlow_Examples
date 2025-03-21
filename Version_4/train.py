import mlflow
from config import parse_args
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.utils import collect_data, save_model, collect_metrics

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("ML_Experiment")
            
def main():
    args = parse_args()

    # === Collect Data ===
    X_train, X_test, y_train, y_test = collect_data(train_size=args.train_size)
    np.save("artefacts/X_test.npy", X_test)
    np.save("artefacts/y_test.npy", y_test)

    # === Define Model ===
    params = {
        "solver": "lbfgs",
        "penalty": args.penalty,
        "C": args.C,
        "fit_intercept": args.fit_intercept
    }
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # === Predict ===
    y_pred = model.predict(X_train)

    # === Metrics ===
    metrics: dict = collect_metrics(y_true=y_train, y_pred=y_pred)

    save_model(model)

    with mlflow.start_run(run_name="Training") as run:
        
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="Logistic Regression v4",
        )

if __name__ == "__main__":
    main()