import mlflow
import pickle
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score

def collect_test_data():
    """
    Load test data.
    """
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    return X_test, y_test

def load_model():
    """
    Load ML_Model.
    """
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


if __name__ == "__main__":
    X_test, y_test = collect_test_data()
    model = load_model()

    y_pred = model.predict(X_test)

    accuracy = (y_test == y_pred).mean()
    f1 = fbeta_score(y_true=y_test, y_pred=y_pred, beta=1, average="macro")
    precision = precision_score(y_true=y_test, y_pred=y_pred, average="macro")
    recall = recall_score(y_true=y_test, y_pred=y_pred, average="macro")

    with mlflow.start_run() as run:
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        })
