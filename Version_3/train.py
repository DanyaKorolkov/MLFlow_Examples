import mlflow
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("ML_Experiment")

train_size = 0.8
penalty = "l2"
fit_intercept = False

def collect_data():
    """
    Load data.
    """
    X,y = load_iris(return_X_y=True)
    return train_test_split(X,y, train_size=train_size)

def training(X_train, y_train):
    """
    Define and train ML_Model.
    """
    model = LogisticRegression(
        penalty=penalty,
        fit_intercept=fit_intercept,
    )
    model.fit(X_train, y_train)
    return model

def save_model(model):
    """
    Saving ML_Model.
    """
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = collect_data()
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    model = training(X_train, y_train)
    y_pred = model.predict(X_train)

    accuracy = (y_train == y_pred).mean()
    f1 = fbeta_score(y_true=y_train, y_pred=y_pred, beta=1, average="macro")
    precision = precision_score(y_true=y_train, y_pred=y_pred, average="macro")
    recall = recall_score(y_true=y_train, y_pred=y_pred, average="macro")

    save_model(model)

    with mlflow.start_run(run_name="Training") as run:
        
        mlflow.log_params({
            "sample_param": {"a": 1, "b": 2}
        })
        
        mlflow.log_params({
            "train_size": train_size,
            "penalty": penalty,
            "fit_intercept": fit_intercept,
        })


        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        })

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="Logistic Regression v3",
        )