from datetime import datetime
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path
from scipy import stats
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent


if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')

RESOURCES_DIR = Path("resources")

# ---------- Question 1 ---------- #
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

tools = [
    {
    'type': 'function',
    'function': {
        'name': 'celsius_to_fahrenheit',
        'description': 'Converts a Celsius temperature to Fahrenheit through given equation',
        'parameters': {
            'type': 'object',
            'properties': {
                'celsius': {'type': 'number', 'description': 'Temperature in Celsius'}
            },
            'required': ['celsius'],
        },
    },
}
]

print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# ---------- Question 2 ---------- #
def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

tools.extend([
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
])

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt}
    ]

    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto'
    )

    print("\nFirst Response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        print('\nAgentic mode engaged...')
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                 args = json.loads(tool_call.function.arguments)
                 tool_result = celsius_to_fahrenheit(**args)
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            print("Tool called:", function_name)
            print("Tool result:", tool_result)

            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

            second_response = client.chat.completions.create(
                model='gpt-4.1-mini',
                messages=messages,
            )
            print("Second response received from model...")
            print(second_response)

            final_message = second_response.choices[0].message
            return final_message.content or ''
        else:
            print("No tools needed....")

        return first_message.content or ''

## Prediction ##
# The agent should be able to complete the request in one tool_call given that it is r
# eadily availbable. The model should be able to automatically pick the correct function/tool.

message = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("\n", message)

## Actual Response ##
# The agent correctly called celsius to farenheit on its first try. 
# It did a second tool call given that its already in the function, 
# but the second call produced the first answer as the first and so 
# was unnecessary.

# ---------- Question 3 ---------- #
response_a = run_agent("\nWhat is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
## Comment ##
# This should trigger a tool call to celsius_to_fahrenheit, since the model
# now has a tool specifically built for this exact conversion and doesn't need to
# approximate the arithmetic itself.

# The agent correctly used the celsius_to_fahrenheit on the first attempt

response_b = run_agent("\nWhat is the boiling point of water in plain English?")
print("Response B:", response_b)
## Comment ##
# This should NOT trigger a tool call -- neither tool answers a plain-English
# factual question, so the model should just answer directly from its own knowledge.

# The agent's response was "None" because it doesn't have access to that information.

# ---- Setup ----#

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.
    
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
    
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
    
        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None
    
        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
    
        title_csv = self.csv_name or "current CSV"
    
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
    
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
    
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        
        return f"Plotted {y} vs {x} as a {plot_type}."
    # ---------- Question 4 ---------- #
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        # your code here
        if col1 and col2:
            coeff, pval = stats.pearsonr(self.df[col1], self.df[col2])
            return_dict = {"col1": col1,
                        "col2": col2,
                        "pearson_r": coeff,
                        "p_value": pval}
            return return_dict
        else:
            return {"error": f"These columns are not in the data"}

print("Class defined")
csv_manager = CsvManager(resources_dir=RESOURCES_DIR)

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt, 
    and then user text will be appended.  

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the 
        final answer for this react cycle, this implies that reasoning alone without 
        tool calls was enough.  
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content 

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                    
            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))
            
            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."

resources_dir = Path("resources")  # adjust to wherever your CSVs actually live
csv_backend = CsvManager(resources_dir)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}

tools.extend([
    {
        'type': 'function',
        'function': {
            'name': 'list_csv_files',
            'description': 'List available CSV files in the resources folder.',
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'load_csv',
            'description': 'Load a CSV file from resources/ and make it the active dataset.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'filename': {'type': 'string', 'description': 'CSV filename, with or without .csv extension'}
                },
                'required': ['filename'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_columns',
            'description': 'Return the column names for the currently loaded CSV.',
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'summarize_columns',
            'description': 'Return summary statistics for one or more columns (or all columns if none specified).',
            'parameters': {
                'type': 'object',
                'properties': {
                    'columns': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Column names to summarize; omit to summarize all columns',
                    }
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'describe_column',
            'description': 'Return summary statistics for a single column.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'column': {'type': 'string', 'description': 'The column name to describe'}
                },
                'required': ['column'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'plot_data',
            'description': 'Plot a column from the active CSV, optionally against another column.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'y': {'type': 'string', 'description': 'Column to plot on the y-axis'},
                    'x': {'type': 'string', 'description': 'Optional column to plot on the x-axis; defaults to row index'},
                    'plot_type': {'type': 'string', 'enum': ['line', 'scatter'], 'description': 'Type of plot to generate'},
                },
                'required': ['y'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'compute_correlation',
            'description': 'Computes the Pearson correlation between two columns in the loaded CSV.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'col1': {'type': 'string', 'description': 'First column name'},
                    'col2': {'type': 'string', 'description': 'Second column name'},
                },
                'required': ['col1', 'col2'],
            },
        },
    },
])

# ---------- Question 5 ---------- #
SYSTEM_PROMPT = """
You are a basic assistant that helps handle basic tasks based on the tools provided to you
"""
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print(result)

# ---------- Question 6 ---------- #
## Comment##
# system: This describes the agentic as it initially exists. This is the basic system of the code or os that has yet to be trained
# user: The user is me, or for more complex codes, whoever is using the model. The user is the human feeding prompts to the agent
# assistant: Assistant is the role the system takes on after it's been given the system prompt. Once the user has told the system it is an assistant, it is redefined as such
# tool: This is whatever tool was called to complete the request from the prompt. In this instance it is what is in node_tools. This role explains what action is being performed by the tools
print(json.dumps(messages, indent=2, default=str))

# ---------- Question 7 ---------- #
@tool
def compute_correlation(col1,col2) -> dict:
        """
            Compute the Pearson correlation between two columns in the loaded DataFrame.
            Returns the correlation coefficient and p-value.

            Args: 
                col1: a column of information selected from a dataframe
                col2: a second column of information selected from a dataframe

            Returns: a dictionary containing r value for correlation, p value and both columns
        """
        return csv_manager.compute_correlation(col1,col2)

print(compute_correlation.description)

## Comment ##
# smolagents need a baseline of what the code should be doing in order to expand on it. 
# This includes instructions and a direct connection to the original tool code.
# In this block we set up a link to the original compute_correlation code so the agentic 
# can use it as a tool first before it tries to make it's own code. This also means we don't 
# have to write JSON schema's for all the tools, because smolagents already defines tools 
# in its wrapper.

# ---------- Question 8 ---------- #
@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns). 
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis. 
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot. 
        Retirms a short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)

TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

model = OpenAIServerModel(
    api_key=os.environ["OPENAI_API_KEY"],
    model_id="gpt-4o",
)

SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

tool_agent = ToolCallingAgent(tools=TOOLS,
                         model=model,
                         instructions=SYSTEM_PROMPT,)

CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
  If code execution fails, do not fall back to plot_data when the user requested styling (like color). 
  Explain what failed and what you would need to proceed.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})

print(f"/nResponse Tool: {response_tool}")
print(f"Response Code: {response_code}")