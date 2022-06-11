import pandas as pd
import datetime
import pickle
from dateutil.relativedelta import relativedelta
from typing import Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner


@task
def get_paths(date: str = None) -> Tuple[str, str]:
    """
    Get the path where the train and validation sets are located.

    Args:
        date (str, optional): Date in the format ("%Y-%m-%d"). Defaults to None.

    Returns:
        Tuple[str, str]: Train and validation paths.
    """
    def month_format(month):
        if month < 10:
            return f"0{month}"
        else:
            return f"{month}"

    if date is None:
        run_date = datetime.date.today()
    else:
        run_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()

    train_date = run_date - relativedelta(months=2)
    train_year = train_date.year
    train_month = month_format(train_date.month)

    val_date = run_date - relativedelta(months=1)
    val_year = val_date.year
    val_month = month_format(val_date.month)

    train_path = f"./data/fhv_tripdata_{train_year}-{train_month}.parquet"
    val_path = f"./data/fhv_tripdata_{val_year}-{val_month}.parquet"

    return train_path, val_path


@task
def read_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(
    df: pd.DataFrame,
    categorical: list[str],
    train: bool = True
) -> pd.DataFrame:
    """
    Create the `duration` feature.

    Args:
        df (pd.DataFrame): Dataframe to apply the transformation.
        categorical (list[str]): List of categorical feature names.
        train (bool, optional): Flag that indicates whether the feature
                                is created on the train set. Defaults to True.

    Returns:
        df (pd.DataFrame): Dataframe with appended `duration` feature.
    """
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(
    df: pd.DataFrame,
    categorical: list[str],
    date: str
) -> Tuple[LinearRegression, DictVectorizer]:
    """
    Traina linear regression model.

    Args:
        df (pd.DataFrame): training dataframe.
        categorical (list[str]): list of categorical feature names.
        date (str): Date in the format ("%Y-%m-%d").

    Returns:
        Tuple[LinearRegression, DictVectorizer]: Return the fitted model
                                                 and dictionary vectorizer.
    """
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    dv.fit(train_dicts)
    # save fitted DictVectorizer
    with open(f"dv-{date}.b", "wb") as f_out:
        pickle.dump(dv, f_out)
    X_train = dv.transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    # save fitted model
    with open(f"model-{date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(
    df: pd.DataFrame,
    categorical: list[str],
    dv: DictVectorizer,
    lr: LinearRegression
) -> None:
    """
    Score the validation set using the linear regression model.

    Args:
        df (pd.DataFrame): validation dataframe.
        categorical (list[str]): list of categorical feature names.
        dv (DictVectorizer): fitted Dictionary Vectorizer.
        lr (LinearRegression): fitted linear regression model.
    """
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@flow(task_runner=SequentialTaskRunner())
def main(
    date: str = None
):
    """
    Main function as Prefect Flow.

    Args:
        date (str, optional): Date in the format ("%Y-%m-%d"). Defaults to None.
    """

    categorical = ['PUlocationID', 'DOlocationID']

    train_path, val_path = get_paths(date=date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical, date=date).result()
    run_model(df_val_processed, categorical, dv, lr)


main("2021-08-15")
