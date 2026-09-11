# ----- Prefect Orchestration ----- #

## Prefect Question 1 ##
# The difference between @task and @flow is that @task indicates code with a single use and @flow orchestrates and 
# orders those individual code blocks into a pipeline. In this example I would not use prefect because there is only 
# one task that needs to be performed by the code. There is no need to coordinate functions together into a pipeline. 
# I would only use @flow if the temperature conversion was part of a larger set of tasks to complete in order.

## Prefect Question 2 ##
# from prefect import task, flow, get_run_logger
# @task(retries= 3,retry_delay_seconds=30)

## Prefect Question 3 ##
# From the Prefect UI you can click on the pipeline you desire on the side panel labeled "Flow", which will show if 
# tasks are running or failing. From there you can click on individual tasks to learn more. You can also also click on 
# the side panel that says "Runs", which will show a recording of all your runs so you can check those that are red. 
# After selecting the what you want to look at, theres a page that diagrams your flow with an option to look at logs at 
# the bottom. In this instance we would look at transform under the correct pipeline.

# ----- Production Patterns ----- #