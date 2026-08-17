import os
import pandas as pd
from scipy import stats
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

# if load_dotenv():
#     print("Successfully loaded environment variables from .env")
# else:
#     print("Warning: could not load environment variables from .env")
api_key = os.getenv("OPENAI_API_KEY")

# RESOURCES_DIR = Path("resources")

# ---------- Pre-task: Load the Data ---------- #
DATA_PATH = "assignments_01/outputs/merged_happiness.csv"

# ---------- Task 1: Define Your Tools ---------- #
df = None

#Tool 1: load_happiness_data
@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory. Look specifically at merged_happiness.csv defined in DATA_PATH

    Returns: dictionary containing the shape and columns from the datafram
    """
    my_df = pd.load_csv('assignments_01/outputs/merged_happiness.csv')
    my_dict = {'shape': df.shape(),
               'columns': df.columns()}
    df = df.concat(my_df, ignore_index = True)
    return my_dict

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
    if column is None:
        return {'error': 'Column not found'}
    df[column].describe().to_dict()

#Tool 3: compute_correlation
@tool
def compute_correlation(col1: str, col2: str) -> dict:
        """
            Compute the Pearson correlation between two columns in the loaded DataFrame.
            Returns the correlation coefficient and p-value.

            Args: 
                  col1: a column given to the code origination from the df
                  col2: a separate column given to the code origination from the df

            Returns: a dictionary containing r value for correlation, p value and both columns round decimal to 4 digits
        """
        if col1 == None or col2 == None:
            return {'error': 'Column not found'}
        r, pval = stats.pearsonr(col1,col2)
        my_dict = {"col1": col1,
                    "col2": col2,
                    "pearson_r": r,
                    "p_value": pval}
        return my_dict

#Tool 4: get_top_n_countries
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.
    
    Args: a specified column, a specified year, and an integer of how many rows should be seelcted (if not given n=5)

    Returns: a dictionary of the top rows based on the column and year given
    """
    if year == None or column == None:
        return {'error': 'Column not found'}
    sorted_df = df[df['year'] == 2026].sort_values(by='column', ascending=False)
    return sorted_df.head(n)

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


# if __name__ == "__main__":
#     # Task 3 queries
#     # Task 4 queries