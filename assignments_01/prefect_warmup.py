import pandas as pd
import numpy as np
from prefect import task, flow

## Pipeline Question 2
@task
def create_series():
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
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
def pipeline_flow():
    series = create_series()
    clean = clean_data(series)
    my_dict = summarize_data(clean)

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

if __name__ == "__main__":
    my_dict = pipeline_flow()

## Question 1
# Prefect might be too much overhead because of the simplicity 
# of this pipeline. Prefect is intended for larger data sets 
# and simplifying workflows. This workflow is already simply. 
# It would be easier to have not added the decorators in the 
# first place and just called pipeline_flow late in the code.

## Question 1
# Prefect would be useful when working with a larger data sets, 
# especially ones that may need more cleaning. It would be helpful 
# in situation where more tasks are required even if the logic 
# stays simple.