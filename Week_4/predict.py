#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import os
import sys
from sklearn.feature_extraction import DictVectorizer


def read_data(input_url: str, cat_cols: list[str] = None):
    df = pd.read_parquet(input_url)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[cat_cols] = df[cat_cols].fillna(-1).astype('int').astype('str')
    
    return df

def prepare_data(df: pd.DataFrame, dv: DictVectorizer, cat_cols: list[str] = None):
    dicts = df[cat_cols].to_dict(orient='records')
    X = dv.transform(dicts)
    return X

def apply_model(Χ, model):
    y_pred = model.predict(Χ)
    print(f'The mean predicted duration is {y_pred.mean():.2f} minutes')
    return y_pred

def save_results(df: pd.DataFrame, y_pred: pd.Series, output_file: str, year: int, month: int):
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

    df = read_data(input_url=input_url, cat_cols=categorical)
    X = prepare_data(df, dv, cat_cols=categorical)
    y_pred = apply_model(X, model)
    os.makedirs(output_dir, exist_ok=True)
    save_results(df=df, y_pred=y_pred, output_file=output_file, year=year, month=month)

if __name__ == '__main__':
    run()
