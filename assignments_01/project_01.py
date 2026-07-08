import pandas as pd
from prefect import task, flow, get_run_logger
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
        if num == 24:
            df.rename(columns = {"Ladder score": "Happiness score"})
        results.append(df)
        num+=1
    df = pd.concat(results, ignore_index=True)
    df.to_csv("assignments_01/outputs/merged_happiness.csv")
    return df

## Task 2: Descriptive Statistics
@task
def happiness_scores(df):
    logger = get_run_logger()
    logger.info(f"Mean: {df['Happiness score'].mean()}")
    logger.info(f"Median: {df['Happiness score'].median()}")
    logger.info(f"Standard Deviation: {df['Happiness score'].std()}")
    yearly = df.groupby('year').agg({'Happiness score':'mean'})
    regionally = df.groupby('Regional indicator').agg({'Happiness score':'mean'})
    logger.info(f"Yearly Mean: {yearly}")
    logger.info(f"Regional Mean: {regionally}")

## Task 3: Visual Exploration
@task
def make_visuals(df):
    logger = get_run_logger()

    plt.figure(1)
    plt.hist(df['Happiness score'])
    plt.title('Happiness Score Histogram')
    plt.savefig("assignments_01/outputs/happiness_histogram.png")
    plt.close()
    logger.info("Histogram")

    plt.figure(2)
    sns.boxplot(x = df['year'], y = df['Happiness score'])
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.title("Happiness by Year")
    plt.savefig("assignments_01/outputs/happiness_by_year.png")
    plt.close()
    logger.info("Boxplot")

    plt.figure(3)
    plt.scatter(df['GDP per capita'], df['Happiness score'])
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.title("Happiness based on GDP")
    plt.savefig("assignments_01/outputs/gdp_vs_happiness.png")
    plt.close()
    logger.info("Scatter")

    numeric_df = df.corr(method = 'pearson', numeric_only=True)

    plt.figure(4)
    sns.heatmap(numeric_df, annot=True)
    plt.title("Happiness Heatmap")
    plt.savefig("assignments_01/outputs/correlation_heatmap.png")
    plt.close()
    logger.info("Heatmap")

## Task 4: Hypothesis Testing
@task
def hypo_testing(df):
    logger = get_run_logger()
    df_2019 = df.loc[df['year'] = 2019, ['Country', 'Happiness score']]
    df_2020 = df.loc['year' = 2020].dropna()
    before = pd.Series(df_2019, index='Country')
    after = pd.Series(df_2020, index='Country')
    tstat,pval = stats.ttest_rel(before, after)
    logger.info(df_2019['Country'])
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

## Task 5: Correlation and Multiple Comparisons
@task
def corr_comparison(df):
    corr, pval = stats.pearsonr(df.select_dtypes(include='number'))

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