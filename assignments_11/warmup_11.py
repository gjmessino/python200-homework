# ----- Prefect Orchestration ----- #

## Prefect Question 1 ##
# The difference between @task and @flow is that @task indicates code with a single use and @flow orchestrates and 
# orders those individual code blocks into a pipeline. In this example I would not use prefect because there is only 
# one task that needs to be performed by the code. There is no need to coordinate functions together into a pipeline. 
# I would only use @flow if the temperature conversion was part of a larger set of tasks to complete in order.

## Prefect Question 2 ##
from prefect import task, flow, get_run_logger
@task(retries= 3,retry_delay_seconds=30)

## Prefect Question 3 ##
# From the Prefect UI you can click on the pipeline you desire on the side panel labeled "Flow", which will show if 
# tasks are running or failing. From there you can click on individual tasks to learn more. You can also also click on 
# the side panel that says "Runs", which will show a recording of all your runs so you can check those that are red. 
# After selecting the what you want to look at, theres a page that diagrams your flow with an option to look at logs at 
# the bottom. In this instance we would look at transform under the correct pipeline.

# ----- Production Patterns ----- #

## Production Question 1 ##
# With raise_for_status() the code will raise an exception for any error, where as if response.status_code != 200: 
# print("error") will only raise an error if it's in the 200s or API errors. This means any other kind of error won't 
# get flagged and the code will continue on. This can be a problem is 400 and 500 level errors aren't caught, and can 
# lead to inaccurate data. In the case of 400 or 500 errors there can be empty or malformed responses.

## Production Question 2 ##
# In this scenario upsert would stop the code from processing data twice. If insert had been used an error would 
# immediately be raised because the code would read the half of the data that's been processed as new and try to process 
# it again. However, if two rows have the same key an error will be raised, which is precisely what you happen if you 
# try to re-add rows that have already been processed. With upsert, the code will recognize the already processed data 
# and update anything with matching keys without raising an error or trying to create a new row.

## Production Question 3 ##
@task
def get_info(enrichment_records: list) -> None:
    logger = get_run_logger()
    message = f"Number of Enrichment Records Upserted: {len(enrichment_records)}"
    logger.INFO(message)

## Production Question 4 ##
# An incremental check won't reprocess records that have already been processed. In my answer about upsert for Production
# Question 2 I mentioned how upsert will still update existing rows if they get updated twice, but incremental 
# processing checks first to make sure rows already exist so we don't have to upsert them more than once. So if there is 
# a data set with 100 rows, half of which are processed, instead of going through all of them again we check for what is 
# done so as not to waste time, energy or money.
