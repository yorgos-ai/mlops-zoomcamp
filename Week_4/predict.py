#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import os
import sys
from sklearn.feature_extraction import DictVectorizer


def read_data(input_url: str) -> pd.DataFrame:
    """
    Read data from a parquet file.

    Args:
        input_url (str): URL of the parquet file.

    Returns:
        pd.DataFrame: The data read from the parquet file.
    """
    df = pd.read_parquet(input_url)
    return df

def clean_data(df: pd.DataFrame, cat_cols: list[str] = None) -> pd.DataFrame:
    """
    Data cleaning step. Removes rows with invalid duration and 
    converts categorical columns to strings.

    Args:
        df (pd.DataFrame): The input dataframe.
        cat_cols (list[str], optional): A list of categorical columns. Defaults to None.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[cat_cols] = df[cat_cols].fillna(-1).astype('int').astype('str')
    return df
    

def prepare_data(df: pd.DataFrame, dv: DictVectorizer, cat_cols: list[str] = None) -> pd.DataFrame:
    """
    Prepare the data for the model. Converts the categorical columns to a sparse matrix.

    Args:
        df (pd.DataFrame): The input dataframe.
        dv (DictVectorizer): The fitted DictVectorizer object.
        cat_cols (list[str], optional): A list of categorical columns to be converted to sparse matrix. Defaults to None.

    Returns:
        pd.DataFrame: The sparse matrix of the categorical columns.
    """
    dicts = df[cat_cols].to_dict(orient='records')
    X = dv.transform(dicts)
    return X

def apply_model(Χ: pd.DataFrame, model) -> pd.Series:
    """
    Apply the model to the input data.

    Args:
        model (_type_): The fitted model.

    Returns:
        pd.Series: The model predictions.
    """
    y_pred = model.predict(Χ)
    print(f'The mean predicted duration is {y_pred.mean():.2f} minutes')
    return y_pred

def save_results(df: pd.DataFrame, y_pred: pd.Series, output_file: str, year: int, month: int) -> None:
    """
    Save the model predictions to a parquet file.

    Args:
        df (pd.DataFrame): initial dataframe
        y_pred (pd.Series): model predictions
        output_file (str): the name of the output file
        year (int): the year of the taxi trip records data
        month (int): the month of the taxi trip records data
    """
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.concat([df['ride_id'], pd.Series(y_pred, name='prediction')], axis=1)
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    print(f'Predictions saved to {output_file}')

def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    print(f'Running predictions for {year}-{month:02d}')
    
    input_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    output_dir = 'output'
    output_file = f'{output_dir}/predictions_{year}-{month:02d}.parquet'
    
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_url=input_url)
    df_clean = clean_data(df=df, cat_cols=categorical)
    X = prepare_data(df_clean, dv, cat_cols=categorical)
    y_pred = apply_model(X, model)
    os.makedirs(output_dir, exist_ok=True)
    save_results(df=df_clean, y_pred=y_pred, output_file=output_file, year=year, month=month)

if __name__ == '__main__':
    run()
