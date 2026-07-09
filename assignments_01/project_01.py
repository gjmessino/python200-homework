import pandas as pd
from prefect import task, flow, get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

## Task 1: Load Multiple Years of Data
@task(retries=3, retry_delay_seconds=2)
def load_data():
    num = 2015
    results = []
    for i in range(10):
        happiness_file = f'happiness_project/world_happiness_{num}.csv'
        df = pd.read_csv(happiness_file, sep=';', decimal=',')
        df['year'] = num
        if num == 24:
            df = df.rename(columns = {"Ladder score": "Happiness score"})
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
    for year in df['year']:
        placement = [1,df['year'].nunique,1]
        plt.subplot(''.join(placement))
        sns.boxplot(x = 'year', y = df.loc(df['year'] == year, ['Happiness score']))
        plt.xlabel("Year")
        plt.ylabel("Happiness Score")
        plt.title(f"Happiness by Year: {year}")
        placement[2] += 1
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
    
    df_2019 = df[df['year'] == 2019]['Happiness score'].dropna()
    df_2020 = df[df['year'] == 2020]['Happiness score'].dropna()

    tstat, pval = stats.ttest_ind(df_2019, df_2020, nan_policy='omit') 
    mean_2019 = df_2019.mean()
    mean_2020 = df_2020.mean()
    
    logger.info(f"T Stat: {tstat}")
    logger.info(f"P Value: {pval}")
    logger.info(f"2019 Mean: {mean_2019}")
    logger.info(f"2020 Mean: {mean_2020}")
    
    alpha = 0.05
    if pval < alpha:
        direction = "decreased" if mean_2020 < mean_2019 else "increased"
        logger.info(f"Significant: Global happiness scores {direction} between 2019 and 2020.")
    else:
        logger.info("Insignificant: No statistically significant change in happiness scores was detected.")
    # Because the samples are perfectly paired/aligned by country, 
    # a paired t-test(ttest_rel) is used to check if the mean difference 
    # deviates significantly from zero.
    
    high_region = df[df['Regional indicator'] == 'Western Europe']['Happiness score']
    low_region = df[df['Regional indicator'] == 'Sub-Saharan Africa']['Happiness score']

    tstat, pval2 = stats.ttest_ind(high_region, low_region, nan_policy='omit')

    logger.info(f"Regional Test (Western Europe vs Sub-Saharan Africa):")
    logger.info(f"T Stat: {tstat}, P Value: {pval2}")
    # The second test compares the historical differences between 
    # Western Europe and Sub-Saharan Africa across all years using 
    # an independent samples t-test to confirm if the regional gap 
    # observed in descriptive stats is statistically reliable.
    
    return pval, mean_2019, mean_2020

## Task 5: Correlation and Multiple Comparisons
@task
def corr_comparison(df):
    logger = get_run_logger()
    
    explanatory_vars = [
        'GDP per capita', 'Social support', 'Healthy life expectancy', 
        'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
    ]
    
    df_clean = df.dropna(subset=['Happiness score'] + explanatory_vars)
    
    num_tests = len(explanatory_vars)
    original_alpha = 0.05
    adjusted_alpha = original_alpha / num_tests
    
    best_var = None
    best_coeff = 0.0
    
    sig_original_vars = []
    sig_bonferroni_vars = []
    
    for var in explanatory_vars:
        if var not in df.columns:
            continue
        coeff, pval = stats.pearsonr(df_clean[var], df_clean['Happiness score'])       
        
        if abs(coeff) > abs(best_coeff):
            best_coeff = coeff
            best_var = var
            
        if pval < original_alpha:
            sig_original_vars.append(var)
        if pval < adjusted_alpha:
            sig_bonferroni_vars.append(var)
        
        logger.info(f"Variable: {var}")
        logger.info(f"Pearson r: {coeff}")
        logger.info(f"P Value: {pval}")
        
    return best_var, best_coeff, sig_original_vars, sig_bonferroni_vars


## Task 6: Summary Report
@task
def summary_report(df, pval, mean_2019, mean_2020, best_var, best_coeff, sig_original, sig_bonferroni):
    logger = get_run_logger()
    
    logger.info(f"Number of Countries: {df['Country'].nunique()}")
    logger.info(f"Years: {df['year'].nunique()}")
        
    regional_means = df.groupby('Regional indicator')['Happiness score'].mean().sort_values(ascending=False)
    logger.info(f"Top 3 happiest regions: {', '.join(regional_means.head(3).index.tolist())}")
    logger.info(f"Bottom 3 happiest regions: {', '.join(regional_means.tail(3).index.tolist())}")
    
    alpha = 0.05
    if pval < alpha:
        direction = "decreased" if mean_2020 < mean_2019 else "increased"
        logger.info(f"Pandemic Impact: Global happiness scores significantly {direction} (p = {pval:.4f}).")
    else:
        logger.info("Pandemic Impact: No statistically significant change found between 2019 and 2020.")
        
    logger.info(f"Strongest Predictor: '{best_var}' (r = {best_coeff:.4f}).")
    logger.info(f"Variables significant at baseline alpha (0.05): {', '.join(sig_original)}")
    logger.info(f"Variables remaining significant after Bonferroni correction: {', '.join(sig_bonferroni)}")
  
@flow
def happiness_pipeline():
    df = load_data()
    happiness_scores(df)
    make_visuals(df)
    pval, mean_2019, mean_2020 = hypo_testing(df)
    best_var, best_coeff, sig_original, sig_bonferroni = corr_comparison(df)
    summary_report(df, pval, mean_2019, mean_2020, best_var, best_coeff, sig_original, sig_bonferroni)

if __name__ == "__main__":
    happiness_pipeline()