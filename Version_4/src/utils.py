import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score

def collect_data(train_size: float=0.8):
    """
    Load data.
    """
    X,y = load_iris(return_X_y=True)
    return train_test_split(X,y, train_size=train_size)


def save_model(model):
    """
    Saving ML_Model.
    """
    with open("artefacts/model.pkl", "wb") as file:
        pickle.dump(model, file)


def collect_metrics(y_true, y_pred):
    """
    Collecting metrics.
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
