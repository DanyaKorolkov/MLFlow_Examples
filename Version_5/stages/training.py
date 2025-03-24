import click
from library.utils import training_phase
import mlflow


@click.command()
@click.option("--config_path", default="stages/config.yaml", type=str)
def main(config_path):
    model, params, metrics = training_phase(config_path)
 
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="Logistic Regression",
    )

if __name__ == "__main__":
    main()
