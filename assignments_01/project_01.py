import pandas as pd
from prefect import task, get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

## Task 1: Load Multiple Years of Data
@task(retries=3, retry_delay_seconds=2)
def load_data():
    base_url = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_20"
    num = 15
    results=[]
    for i in range(10):
        url = f"{base_url}{int(num)}.csv"
        df = pd.read_csv(url, sep=';', decimal=',')
        df['year'] = 2000 + num
        results.append(df)
        num+=1
    df = pd.concat(results, ignore_index=True)
    df.to_csv("assignments_01/outputs/merged_happiness.csv")

## Task 2: Descriptive Statistics
@task
def happiness_scores(df):
    get_run_logger(f"Mean: {df.mean()}")
    get_run_logger(f"Median: {df.Median()}")
    get_run_logger(f"Standard Deviation: {df.std()}")
    yearly = df.group_by('year').agg({'Happiness score':'mean'})
    regionally = df.group_by('region').agg({'Happiness score':'mean'})
    get_run_logger(f"Yearly Mean: {yearly}")
    get_run_logger(f"Regional Mean: {regionally}")

## Task 3: Visual Exploration
@task
def make_visuals(df):
    plt.figure(1)
    plt.hist(df['Happiness score'])
    plt.savefig("assignments_01/outputs/happiness_histogram.png")
    plt.show()
    get_run_logger("Histogram")

    plt.figure(2)
    sns.boxplot(x = df['year'], y = df['Happiness score'])
    plt.savefig("assignments_01/outputs/happiness_by_year.png")
    plt.show()
    get_run_logger("Boxplot")

    plt.figure(3)
    plt.scatter(df['GDP per capita'], df['Happiness score'])
    plt.savefig("assignments_01/outputs/gdp_vs_happiness.png")
    plt.show()
    get_run_logger("Scatter")

    numeric_df = df.corr(method = 'peason', numeric_only=True)

    plt.figure(4)
    sns.heatmap(numeric_df, annot=True)
    plt.savefig("assignments_01/outputs/correlation_heatmap.png")
    plt.show()
    get_run_logger("Heatmap")

## Task 4: Hypothesis Testing
@task
def hypo_testing(df):
    before = df.loc[df['year'] == 2019, 'Happiness score']
    after = df.loc[df['year'] == 2020, 'Happiness score']
    tstat,pval = stats.ttest_rel(before, after)
    get_run_logger(f"Stats: {tstat}")
    get_run_logger(f"P Values: {pval}")
    get_run_logger(f"2019 Mean: {before.mean()}")
    get_run_logger(f"2020 Mean: {after.mean()}")
    ## ADD COMMENT
    
    alpha = 0.05
    if pval < alpha:
        get_run_logger("Significant")
    else:
        get_run_logger("Insignificant")
    ## ADD COMMENT
    ## ADD SECOND TEST

## Task 5: Correlation and Multiple Comparisons
