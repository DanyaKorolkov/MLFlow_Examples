from library.utils import load_train_data
import mlflow
import click


@click.command()
@click.option("--config_path", default="stages/config.yaml", type=str)
def main(config_path):
    X_train, y_train = load_train_data(config_path)

    mlflow.log_input(
        mlflow.data.from_pandas(X_train, name="X_train"),
        context="train"
    )
    mlflow.log_input(
        mlflow.data.from_pandas(y_train, name="y_train"),
        context="train"
    )

if __name__ == "__main__":
    main()    