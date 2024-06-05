import pickle
import mlflow
from mlflow.sklearn import log_model
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT_NAME = "homework_week_2"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

def to_pickle(data: pickle, filename: str):
    with open(filename, "wb") as f_in:
        pickle.dump(data, f_in)

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    dv, model = data

    #pickle the DictVectorizer
    to_pickle(data=dv, filename='dv.pickle')

    with mlflow.start_run():
        # log model
        log_model(model, 'sklearn_models')
        # log DictVectorizer
        mlflow.log_artifact('dv.pickle')

