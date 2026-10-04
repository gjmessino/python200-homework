from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
from scipy import stats
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")
api_key = os.getenv("OPENAI_API_KEY")

# ---------- Pre-task: Load the Data ---------- #
DATA_PATH = Path("../assignments_01/outputs/merged_happiness.csv")
RESOURCES_DIR = Path("resources")

# ---------- Task 1: Define Your Tools ---------- #
df = None

#Tool 1: load_happiness_data
@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory. Look specifically at merged_happiness.csv defined in DATA_PATH

    Returns: A dict with two keys: "shape" (a tuple of row and column counts) and
        "columns" (a list of column name strings). This is a summary dict, not
        the DataFrame itself -- access values with happiness_data["shape"], not
        happiness_data.shape.
    """
    global df
    if not DATA_PATH.exists():
        num = 2015
        results = []
        for i in range(10):
            happiness_file = f'.../assignments_01/happiness_project/world_happiness_{num}.csv'
            df = pd.read_csv(happiness_file, sep=';', decimal=',')
            df['year'] = num
            if num == 24:
                df = df.rename(columns = {"Ladder score": "Happiness score"})
            results.append(df)
            num+=1
        df = pd.concat(results, ignore_index=True)
        df.to_csv("assignments_01/outputs/merged_happiness.csv")
        return {"shape": df.shape,
                "columns": list(df.columns)}
    else:
        try: 
            df = pd.read_csv(DATA_PATH)
            return {"shape": df.shape,
                    "columns": list(df.columns)}
        except:
            print('An error has occured')

#Tool 2: summarize_column
@tool
def summarize_column(column: str) -> dict:
    """Return summary stats for selected column. 
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        column: Column names to summarize. If None return error message

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    return df[column].describe().to_dict()

#Tool 3: compute_correlation
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation between two columns in the loaded DataFrame.

        Args:
            col1: The first column name.
            col2: The second column name.

        Returns:
            A dictionary containing col1, col2, pearson_r, and p_value, each rounded
            to 4 decimal places.
                
    """
    if df is None:
        return {'error': 'Column not found'}
    missing = [c for c in (col1, col2) if c not in df.columns]
    if missing:
        return {"error": f"These columns are not in the data: {missing}"}
    clean = df[[col1, col2]].dropna()
    if clean.empty:
        return {"error": f"No overlapping non-missing data between '{col1}' and '{col2}'."}
    r, pval = stats.pearsonr(clean[col1],clean[col2])
    my_dict = {"col1": col1,
               "col2": col2,
               "pearson_r": round(r, 4),
               "p_value": round(pval, 4),
                }
    return my_dict

#Tool 4: get_top_n_countries
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The column to rank countries by.
        year: The year to filter the data on.
        n: The number of top rows to return. Defaults to 5.

    Returns:
        A list of dicts, each with "country" and the requested column's value,
        or an error dict on bad input.
    """
    if df is None:
        return {'error': 'Dataframe does not exist'}
    if column not in df.columns:
        return {'error': 'Column not found'}
    sorted_df = df[df['year'] == year].sort_values(by=column, ascending=False).head(n)
    country_col = "Country" if "Country" in df.columns else "country"
    return sorted_df[[country_col, column]].rename(columns={country_col: "country"}).to_dict("records")

# ---------- Task 2: Build the Agent ---------- #
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# ---------- Task 3: Run Guided Queries ---------- #
def main():
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    # ---------- Task 4: Your Own Questions ---------- #
    # My query 1
    my_query_1 = "What's the corelation on Happiness score and Perceptions of corruption"   
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?

    # My query 2
    my_query_2 = "Is happiness trending upward over time"  
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?

# ---------- Task 5: Reflection ---------- #
# 1. It listed the correlation coefficient as 0.622, p value as 0.0, and decided that was a 
# significant amount. The agent didn't explain why it was significant, but the p value is under 
# 5% (anything under 5% is significant) and indicates the program may have rounded off a tiny number 
# to 0. The r value (which was also probably rounded) shows a strong positive connection between 
# the two datasets. The numbers are in line with determining this to be a significant correlation.

# 2. The agent struggled deeply with graphs, which isn't surprising given that wasn't a tool that 
# was coded in beforehand. Happiness_trend_over_time.png is a blank graph with labels. 
# happiness_by_region.png has one line, but the legend is cut off so I'm not sure what it's 
# referencing, and there are two other random dots. The agent did get the graph for happiness over 
# time correct, and by 'correct' I mean it created a readable graph, but I'm not sure if anything 
# on it is correct. I expected this to be a little bit bad, but it was worse than expected.

# 3. Considering my comment above, a graphing tool would be very helpful. Some other tools that
#  would be helpful additions would include: expanding the loading tool to include data cleaning, 
# becuase the system really struggled with NaN values; a tool for relating multiple columns 
# (ie more than two), which could be implemented for looking at how multiple characteristics shape 
# happiness; or creating a tool than can make predictions through a chosen method 
# (logistic regression/ knn/ decision tree ect) that can be used to predict future trends in happiness.

if __name__ == "__main__":
   main()