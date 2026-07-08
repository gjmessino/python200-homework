import pandas as pd
from prefect import task, flow, get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

## Task 1: Load Multiple Years of Data
@task(retries=3, retry_delay_seconds=2)
def load_data():
    logger = get_run_logger()
    base_url = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_20"
    num = 15
    results=[]
    for i in range(10):
        url = f"{base_url}{int(num)}.csv"
        df = pd.read_csv(url, sep=';', decimal=',')
        df['year'] = 2000 + num
        if num == 24:
            df.rename(columns = {"Ladder score": "Happiness score"})
        results.append(df)
        num+=1
    df = pd.concat(results, ignore_index=True)
    df.to_csv("assignments_01/outputs/merged_happiness.csv")
    logger.info("____ Loading Data Complete ____")
    return df

## Task 2: Descriptive Statistics
@task
def happiness_scores(df):
    logger = get_run_logger()
    logger.info(f"Mean: {df.mean()}")
    logger.info(f"Median: {df.Median()}")
    logger.info(f"Standard Deviation: {df.std()}")
    yearly = df.group_by('year').agg({'Happiness score':'mean'})
    regionally = df.group_by('region').agg({'Happiness score':'mean'})
    logger.info(f"Yearly Mean: {yearly}")
    logger.info(f"Regional Mean: {regionally}")
    logger.info("____ descriptive stats complete ____")

## Task 3: Visual Exploration
@task
def make_visuals(df):
    logger = get_run_logger()

    plt.figure(1)
    plt.hist(df['Happiness score'])
    plt.savefig("assignments_01/outputs/happiness_histogram.png")
    plt.show()
    logger.info("Histogram")

    plt.figure(2)
    sns.boxplot(x = df['year'], y = df['Happiness score'])
    plt.savefig("assignments_01/outputs/happiness_by_year.png")
    plt.show()
    logger.info("Boxplot")

    plt.figure(3)
    plt.scatter(df['GDP per capita'], df['Happiness score'])
    plt.savefig("assignments_01/outputs/gdp_vs_happiness.png")
    plt.show()
    logger.info("Scatter")

    numeric_df = df.corr(method = 'peason', numeric_only=True)

    plt.figure(4)
    sns.heatmap(numeric_df, annot=True)
    plt.savefig("assignments_01/outputs/correlation_heatmap.png")
    plt.show()
    logger.info("Heatmap")

    logger.info('____ VIsializations complete ____')

## Task 4: Hypothesis Testing
@task
def hypo_testing(df):
    logger = get_run_logger()
    before = df.loc[df['year'] == 2019, 'Happiness score']
    after = df.loc[df['year'] == 2020, 'Happiness score']
    tstat,pval = stats.ttest_rel(before, after)
    logger.info(f"Stats: {tstat}")
    logger.info(f"P Values: {pval}")
    logger.info(f"2019 Mean: {before.mean()}")
    logger.info(f"2020 Mean: {after.mean()}")

    ## ADD COMMENT
    
    alpha = 0.05
    if pval < alpha:
        logger.info("Significant")
    else:
        logger.info("Insignificant")
    ## ADD COMMENT
    ## ADD SECOND TEST
    logger.info('____ Hypo testing complete____ ')

## Task 5: Correlation and Multiple Comparisons
@task
def corr_comparison(df):
    corr, pval = stats.pearsonr(df.select_dtypes(include='number'))
    get_run_logger('____ Correlation complete )___')

## Task 6: Summary Report
@flow
def happiness_pipeline():
    df = load_data()
    happiness_scores(df)
    make_visuals(df)
    hypo_testing(df)
    corr_comparison(df)

if __name__ == "__main__":
    happiness_pipeline()