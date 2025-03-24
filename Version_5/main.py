from logger import Logger
import mlflow
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    mlflow.run(
        uri=".",
        entry_point="load_train_data",
        parameters={"config_path": "stages/config.yaml"},
        experiment_name="ml-experiment",
        run_name="load_data",
    )