#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd
import math
import os

# Create an argument parser
parser = argparse.ArgumentParser(description='Script description.')

# Add the year and month arguments
parser.add_argument('--year', type=int, help='Specify the year.')
parser.add_argument('--month', type=int, help='Specify the month.')

# Parse the command-line arguments
args = parser.parse_args()

# Extract the values of year and month from the arguments
year = args.year
month = args.month

categorical = ['PULocationID', 'DOLocationID']



def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
df = read_data(url)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


def calc_std_dev(y_pred):
    n = len(y_pred)
    mean = sum(y_pred) / n
    squared_diff_sum = sum((x - mean) ** 2 for x in y_pred)
    variance = squared_diff_sum / n
    std_dev = math.sqrt(variance)
    return std_dev


std_dev = calc_std_dev(y_pred)

# print("Standard Deviation:", std_dev)


# Create an artificial ride_id column
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# Prepare the dataframe with results
df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predicted_duration': y_pred})

# Save the dataframe as a Parquet file
output_file = 'results_cli.parquet'
df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)


file_size_bytes = os.path.getsize(output_file)
file_size_mb = file_size_bytes / (1024 * 1024)

# print("Size of the output file:", file_size_mb, "MB")

mean_pred_duration = sum(y_pred) / len(y_pred)
print("Mean predicted duration:", mean_pred_duration)

