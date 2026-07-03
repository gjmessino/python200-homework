import pandas as pd
import numpy as np
from prefect import task, flow

## Pipeline Question 2
@task
def create_series(arr):
    return pd.Series(arr)
@task
def clean_data(series):
    return series.dropna()
@task
def summarize_data(series):
    my_dict = {'mean': series.mean(),
               'median': series.median(),
               'mode': series.mode()[0],
               'std': series.std()}
    return my_dict
@flow
def pipeline_flow(arr):
    series = create_series(arr)
    clean = clean_data(series)
    my_dict = summarize_data(clean)
    return(my_dict)

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

if __name__ == "__main__":
    my_dict = pipeline_flow(arr)
    print(f"Keys: {my_dict.keys()}")
    print(f"Values: {my_dict.values()}")

## ADD COMMENT