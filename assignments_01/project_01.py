import pandas as pd
from prefect import task, flow, get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

## Task 1: Load Multiple Years of Data
@task(retries=3, retry_delay_seconds=2)
def load_data():
    year = 2015
    results = []
    for i in range(10):
        happiness_file = f'assignments_01/happiness_project/world_happiness_{year}.csv'
        df = pd.read_csv(happiness_file, sep=';', decimal = ',')
        df['year'] = year
        results.append(df)
        year += 1
    df = pd.concat(results, ignore_index=True)
    df.to_csv('assignments_01/outputs/merged_happiness.csv',  index=False)
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
    plt.xlabel('Score')
    plt.ylabel('Commonality')
    plt.savefig("assignments_01/outputs/happiness_histogram.png")
    logger.info("Histogram")

    plt.figure(2)
    years_sorted = sorted(df['year'].unique())
    data_by_year = [df.loc[df['year'] == y, 'Happiness score'].dropna() for y in years_sorted]
    plt.boxplot(data_by_year, labels=years_sorted)
    plt.title('Happiness Boxplot')
    plt.ylabel('Happiness Score')
    plt.xlabel('Year')
    plt.savefig('assignments_01/outputs/happiness_by_year.png')
    logger.info("Boxplot")

    plt.figure(3)
    plt.scatter(df['GDP per capita'], df['Happiness score'])
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.title("Happiness based on GDP")
    plt.savefig("assignments_01/outputs/gdp_vs_happiness.png")
    logger.info("Scatter")

    numeric_df = df.corr(method = 'pearson', numeric_only=True)

    plt.figure(4)
    sns.heatmap(numeric_df, annot=True)
    plt.title("Happiness Heatmap")
    plt.savefig("assignments_01/outputs/correlation_heatmap.png")
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
        logger.info(
            f"Pandemic Impact: Global happiness scores significantly {direction} "
            f"between 2019 (mean={mean_2019:.3f}) and 2020 (mean={mean_2020:.3f}), p = {pval:.4f}. "
            f"A difference this large would be unlikely to occur by chance alone, "
            f"suggesting a real shift in global happiness around the start of the pandemic."
        )
    else:
        logger.info(
            f"Pandemic Impact: No statistically significant change was detected between 2019 "
            f"(mean={mean_2019:.3f}) and 2020 (mean={mean_2020:.3f}), p = {pval:.4f}. "
            f"Since p is well above 0.05, a gap this size could plausibly occur by chance even if "
            f"the pandemic had no real effect on global happiness — so this data does not support "
            f"the claim that happiness meaningfully changed between those two years."
    )
    # Because the samples are perfectly aligned by country, 
    # an independent t-test(ttest_ind) is used to check if the mean difference s
    # deviates significantly region by region. The P Val is .592 meaning 
    # there wasn't a significant difference in happiness levels during the
    # pandemic.
    
    logger.info('Regional Happiness Differences (ttest 2)')
    high_region = df[df['Regional indicator'] == 'Western Europe']['Happiness score']
    low_region = df[df['Regional indicator'] == 'Sub-Saharan Africa']['Happiness score']

    tstat, pval2 = stats.ttest_ind(high_region, low_region, nan_policy='omit')

    logger.info(f"Regional Test (Western Europe vs Sub-Saharan Africa):")
    if pval2 < alpha:
        logger.info(f"Significant: Western Europe's happiness scores differ significantly from Sub-Saharan Africa's (p = {pval2:.4g}).")
    else:
        logger.info("Insignificant: No statistically significant difference detected between the two regions.")
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

    logger.info(f"Adjusted alpha (Bonferroni, n={num_tests} tests): {adjusted_alpha:.5f}")

    coeff_dict = {}

    sig_original_vars = []
    sig_bonferroni_vars = []
    for var in explanatory_vars:
        if var not in df.columns:
            continue
        coeff, pval = stats.pearsonr(df_clean[var], df_clean['Happiness score'])
        coeff_dict[var] = coeff

        sig_at_05 = pval < original_alpha
        sig_after_bonferroni = pval < adjusted_alpha
        if sig_at_05:
            sig_original_vars.append(var)
        if sig_after_bonferroni:
            sig_bonferroni_vars.append(var)

        logger.info(f"Variable: {var}")
        logger.info(f"Pearson r: {coeff:.4f}, p = {pval:.5f}")
        logger.info(f"Significant at alpha=0.05: {sig_at_05} | Significant after Bonferroni: {sig_after_bonferroni}")

    logger.info(f"Significant at alpha=0.05 ({len(sig_original_vars)} of {num_tests}): {', '.join(sig_original_vars) or 'none'}")
    logger.info(f"Remain significant after Bonferroni correction ({len(sig_bonferroni_vars)} of {num_tests}): {', '.join(sig_bonferroni_vars) or 'none'}")

    best_var = None
    best_coeff = 0.0
    if sig_bonferroni_vars:
        best_var = max(sig_bonferroni_vars, key=lambda v: abs(coeff_dict[v]))
        best_coeff = coeff_dict[best_var]
    else:
        logger.info("No variables remained significant after Bonferroni correction.")

    return best_var, best_coeff, sig_original_vars, sig_bonferroni_vars

## Task 6: Summary Report
@task
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
        logger.info(
            f"Pandemic Impact: At alpha = 0.05, global happiness scores significantly {direction} "
            f"between 2019 and 2020 (p = {pval:.4f}) — a difference this large is unlikely to be due to chance."
        )
    else:
        logger.info(
            f"Pandemic Impact: At alpha = 0.05, no statistically significant change was found between "
            f"2019 and 2020 (p = {pval:.4f}) — a gap this size could plausibly happen by chance, so this "
            f"data does not support the claim that happiness meaningfully shifted at the start of the pandemic."
        )

    if best_var:
        logger.info(f"Strongest Predictor: '{best_var}' (r = {best_coeff:.4f}).")
    else:
        logger.info("Strongest Predictor: none of the variables remained significant after Bonferroni correction.")

    logger.info(f"Variables significant at baseline alpha (0.05): {', '.join(sig_original) or 'none'}")
    logger.info(f"Variables remaining significant after Bonferroni correction: {', '.join(sig_bonferroni) or 'none'}")  

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