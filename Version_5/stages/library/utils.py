from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
import pandas as pd
import numpy as np
import yaml


def read_config(config_path: str) -> dict:
    """
    Read config from YAML file.
    """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def save_config(config_path: str, config: dict):
    """
    Save config into YAML file.
    """
    with open(config_path, "w") as file:
        yaml.dump(config, file)


def load_train_data(config_path: str):
    """
    Load, split and save train data.
    """
    config = read_config(config_path)

    if config["data"]["train"]["X_path"] is None:
        from sklearn.datasets import load_iris
        X,y = load_iris(return_X_y=True)

        X_train, _, y_train, _ = train_test_split(
            X,y,
            train_size=config["data"]["train"]["train_size"]
        )
        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
    else:
        X_train = pd.read_parquet(config["data"]["train"]["X_path"])
        y_train = pd.read_parquet(config["data"]["train"]["y_path"])

    X_train.to_parquet("X_train.parquet")
    y_train.to_parquet("y_train.parquet")

    config["data"]["train"]["X_path"] = "X_train.parquet"
    config["data"]["train"]["y_path"] = "y_train.parquet"
    save_config(config_path=config_path, config=config)

    return X_train, y_train


def collect_metrics(y_true, y_pred):
    """
    Collect dict of metrics.
    """
    accuracy = (y_true == y_pred).mean()
    f1 = fbeta_score(y_true=y_true, y_pred=y_pred, beta=1, average="macro")
    precision = precision_score(y_true=y_true, y_pred=y_pred, average="macro")
    recall = recall_score(y_true=y_true, y_pred=y_pred, average="macro")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def training_phase(config_path: str):
    """
    Main training phase.
    """
    config = read_config(config_path)
    X_path = config["data"]["train"]["X_path"]
    y_path = config["data"]["train"]["y_path"]

    X_train = pd.read_parquet(X_path)
    y_train = pd.read_parquet(y_path)

    params = {"solver": "lbfgs"}
    params.update(config["model_params"])

    model = LogisticRegression(**params)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)

    y_pred = np.array(y_pred)
    y_train = np.array(y_train)

    metrics: dict = collect_metrics(y_true=y_train, y_pred=y_pred)

    return model, params, metrics