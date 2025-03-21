import os
import sys
import logging
import warnings
import mlflow
import numpy as np
from logger.Logger import LoggingStream
from config import parse_args
from sklearn.linear_model import LogisticRegression
from src.utils import collect_data, save_model, collect_metrics

warnings.filterwarnings('ignore')
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
)
logger = logging.getLogger("mlflow_logger")
mlflow_logging_stream = LoggingStream(logger, logging.INFO)
sys.stdout = mlflow_logging_stream
sys.stderr = mlflow_logging_stream

try:
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    logging.log(
        level=logging.INFO,
        msg=f"MLFlow Tracking URI: {mlflow.get_tracking_uri()}"
    )
except Exception as e:
    logging.log(
        level=logging.ERROR,
        msg=f"Failed to set mlflow tracking uri: {e}"
    )
            
def main():
    args = parse_args()

    # === Collect Data ===
    X_train, X_test, y_train, y_test = collect_data(train_size=args.train_size)
    logging.log(level=logging.INFO, msg="Data collected")

    # === Save Data ===
    if not os.path.exists(f"./artefacts_v4/"):
        os.makedirs(f"./artefacts_v4/")
    np.save(f"./artefacts_v4/X_test.npy", X_test)
    np.save(f"./artefacts_v4/y_test.npy", y_test)
    logging.log(level=logging.INFO, msg="Data Saved")

    # === Define Model ===
    params = {
        "solver": "lbfgs",
        "penalty": args.penalty,
        "C": args.C,
        "fit_intercept": args.fit_intercept
    }
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    logging.log(level=logging.INFO, msg="Model trained")

    # === Predict ===
    y_pred = model.predict(X_train)

    # === Metrics ===
    metrics: dict = collect_metrics(y_true=y_train, y_pred=y_pred)

    save_model(model, path="./artefacts_v4/model.pkl")
    logging.log(level=logging.INFO, msg="Model saved")

    with mlflow.start_run() as run:
        
        mlflow.log_params(params)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="Logistic Regression v4",
        )

if __name__ == "__main__":
    main()