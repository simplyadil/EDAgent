"""Enhanced Data Wrangling Agent with Database Support"""

from typing import Any, Optional, Dict, Union
import pandas as pd
import sqlalchemy as sql
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.types import Checkpointer

from basic_templates.agent_template import (
    BaseAgent,
    create_coding_agent_graph,
    node_func_execute_agent_code_on_data,
    node_func_execute_agent_from_sql_connection,
    node_func_fix_agent_code,
    node_func_explain_agent_code,
    node_func_human_review,
    node_func_report_agent_outputs
)
from utils.parsing import PythonOutputParser, SQLOutputParser
from utils.logging import log_ai_function
from utils.regex import add_comments_to_top, relocate_imports_inside_function

class DataWranglingAgent(BaseAgent):
    """
    Data Wrangling Agent that can work with both file data and database connections.
    """
    
    def __init__(
        self,
        model: Any,
        connection: Optional[Any] = None,
        n_samples: int = 5,
        log: bool = False,
        bypass_recommended_steps: bool = False,
        bypass_explain_code: bool = True,
        human_in_the_loop: bool = False,
        checkpointer: Optional[Checkpointer] = None,
        max_retries: int = 3,
    ):
        self._params = {
            "model": model,
            "connection": connection,
            "n_samples": n_samples,
            "log": log,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code,
            "human_in_the_loop": human_in_the_loop,
            "checkpointer": checkpointer,
            "max_retries": max_retries,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        self.response = None
        return make_data_wrangling_agent(**self._params)

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    def invoke_agent(
        self, 
        user_instructions: str,
        data_raw: Optional[Union[pd.DataFrame, dict, list]] = None,
        connection: Optional[Any] = None,
        data_source_type: str = "file",
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs
    ):
        """
        Enhanced invoke method that handles both file and database data sources.
        """
        # Only pass keys that are expected by the graph (no extra keys)
        if connection is not None or data_source_type == "database":
            # Database mode
            input_data = {
                "user_instructions": user_instructions,
                "connection": connection or self._params.get("connection"),
                "max_retries": max_retries,
                "retry_count": retry_count,
            }
        else:
            # File mode
            input_data = {
                "user_instructions": user_instructions,
                "data_raw": self._convert_data_input(data_raw) if data_raw is not None else None,
                "max_retries": max_retries,
                "retry_count": retry_count,
            }
        response = self._compiled_graph.invoke(input_data, **kwargs)
        self.response = response
        return response

    @staticmethod
    def _convert_data_input(data_raw):
        """Convert various data inputs to the expected format."""
        if isinstance(data_raw, pd.DataFrame):
            return data_raw.to_dict()
        elif isinstance(data_raw, dict):
            return data_raw
        elif isinstance(data_raw, list):
            return [item.to_dict() if isinstance(item, pd.DataFrame) else item for item in data_raw]
        else:
            raise ValueError("Invalid data input. Must be a pandas DataFrame, dict, or list.")

    def get_data_wrangled(self):
        """Get the processed data result."""
        if self.response and self.response.get("data_wrangled"):
            data = self.response["data_wrangled"]
            if isinstance(data, dict):
                return pd.DataFrame(data)
            return data
        return None

    def get_data_wrangling_function(self, markdown: bool = False):
        """Get the generated data wrangling code."""
        if self.response and self.response.get("data_wrangler_function"):
            code = self.response["data_wrangler_function"]
            if markdown:
                from IPython.display import Markdown
                return Markdown(f"```python\n{code}\n```")
            return code
        return None


def make_data_wrangling_agent(
    model: Any,
    connection: Optional[Any] = None,
    n_samples: int = 5,
    log: bool = False,
    bypass_recommended_steps: bool = False,
    bypass_explain_code: bool = True,
    human_in_the_loop: bool = False,
    checkpointer: Optional[Checkpointer] = None,
    max_retries: int = 3,
):
    """
    Create an enhanced data wrangling agent that works with both files and databases.
    """
    
    from typing import TypedDict, Annotated, Sequence
    from langchain_core.messages import BaseMessage
    import operator

    # Enhanced State Schema
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_raw: Optional[dict]
        connection: Optional[Any]
        data_source_type: str
        recommended_steps: str
        data_wrangler_function: str
        data_wrangled: dict
        data_wrangling_error: str
        max_retries: int
        retry_count: int

    # Prompts for different data source types
    FILE_DATA_PROMPT = PromptTemplate(
        template="""
        You are a Data Wrangling Agent. Your job is to write Python code using pandas to process data according to the user's instructions.

        **User Instructions:** {user_instructions}

        **Data Sample (first {n_samples} rows):**
        {data_sample}

        **Requirements:**
        1. Write a Python function called `data_wrangler(df)` that takes a pandas DataFrame as input
        2. Process the data according to the user's instructions
        3. Return the processed DataFrame
        4. Use only standard pandas operations
        5. Handle potential errors gracefully
        6. Include all necessary imports inside the function

        **Example:**
        ```python
        def data_wrangler(df):
            import pandas as pd
            import numpy as np
            
            # Your data processing logic here
            result = df.groupby('category').sum()
            return result
        ```

        Generate only the Python code for the data_wrangler function:
        """,
        input_variables=["user_instructions", "data_sample", "n_samples"]
    )

    DATABASE_PROMPT = PromptTemplate(
        template="""
        You are a SQL Data Wrangling Agent. Your job is to write Python code that executes SQL queries to process database data according to the user's instructions.

        **User Instructions:** {user_instructions}

        **Available Tables:** {table_info}

        **Requirements:**
        1. Write a Python function called `data_wrangler(connection)` that takes a SQLAlchemy connection as input
        2. Write SQL queries to process the data according to the user's instructions
        3. Use pandas.read_sql() to execute queries and return DataFrames
        4. Return the processed DataFrame
        5. Handle potential errors gracefully
        6. Include all necessary imports inside the function

        **Example:**
        ```python
        def data_wrangler(connection):
            import pandas as pd
            from sqlalchemy import text
            
            query = '''
            SELECT category, SUM(amount) as total_amount
            FROM sales 
            GROUP BY category
            ORDER BY total_amount DESC
            '''
            
            result = pd.read_sql(text(query), connection)
            return result
        ```

        Generate only the Python code for the data_wrangler function:
        """,
        input_variables=["user_instructions", "table_info"]
    )

    FIX_CODE_PROMPT = PromptTemplate(
        template="""
        You are a debugging expert. The following {function_name} function has an error. Please fix it.

        **Original Code:**
        ```python
        {code_snippet}
        ```

        **Error Message:**
        {error}

        **Instructions:**
        1. Identify and fix the error in the code
        2. Ensure the function works correctly
        3. Maintain the same function name and basic structure
        4. Include all necessary imports inside the function
        5. Return only the corrected Python code

        **Fixed Code:**
        """,
        input_variables=["code_snippet", "error", "function_name"]
    )

    # Node Functions
    def create_data_wrangler_code(state: GraphState):
        print("    * CREATE DATA WRANGLER CODE")
        
        user_instructions = state["user_instructions"]
        data_source_type = state.get("data_source_type", "file")
        
        if data_source_type == "database":
            # Database mode - generate SQL-based code
            connection = state.get("connection")
            
            # Get table information
            try:
                if connection:
                    inspector = sql.inspect(connection)
                    tables = inspector.get_table_names()
                    table_info = f"Available tables: {', '.join(tables)}"
                    
                    # Get sample data from first table for context
                    if tables:
                        sample_query = f"SELECT * FROM {tables[0]} LIMIT 3"
                        sample_df = pd.read_sql(sample_query, connection)
                        table_info += f"\n\nSample from {tables[0]}:\n{sample_df.to_string()}"
                else:
                    table_info = "No connection available"
            except Exception as e:
                table_info = f"Could not retrieve table information: {str(e)}"
            
            prompt = DATABASE_PROMPT.format(
                user_instructions=user_instructions,
                table_info=table_info
            )
        else:
            # File mode - generate pandas-based code
            data_raw = state.get("data_raw", {})
            
            if data_raw:
                df_sample = pd.DataFrame(data_raw).head(n_samples)
                data_sample = df_sample.to_string()
            else:
                data_sample = "No data sample available"
            
            prompt = FILE_DATA_PROMPT.format(
                user_instructions=user_instructions,
                data_sample=data_sample,
                n_samples=n_samples
            )
        
        # Generate code
        response = (model | PythonOutputParser()).invoke(prompt)
        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name="data_wrangler")
        
        # Log if requested
        if log:
            log_ai_function(response, "data_wrangler.py", log=True)
        
        return {"data_wrangler_function": response}

    def execute_data_wrangler_code(state: GraphState):
        print("    * EXECUTE DATA WRANGLER CODE")
        
        data_source_type = state.get("data_source_type", "file")
        
        if data_source_type == "database":
            # Use database execution function
            return node_func_execute_agent_from_sql_connection(
                state=state,
                connection=state.get("connection"),
                code_snippet_key="data_wrangler_function",
                result_key="data_wrangled",
                error_key="data_wrangling_error",
                agent_function_name="data_wrangler"
            )
        else:
            # Use file execution function
            return node_func_execute_agent_code_on_data(
                state=state,
                data_key="data_raw",
                code_snippet_key="data_wrangler_function",
                result_key="data_wrangled",
                error_key="data_wrangling_error",
                agent_function_name="data_wrangler"
            )

    def fix_data_wrangler_code(state: GraphState):
        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="data_wrangler_function",
            error_key="data_wrangling_error",
            llm=model,
            prompt_template=FIX_CODE_PROMPT.template,
            agent_name="data_wrangler",
            retry_count_key="retry_count",
            log=log,
            function_name="data_wrangler"
        )

    def explain_data_wrangler_code(state: GraphState):
        explanation_prompt = """
        Explain what this data wrangling code does in simple terms:

        {code}
        
        Describe:
        1. What data transformations are performed
        2. What the output will contain
        3. Any important business logic applied
        """
        
        return node_func_explain_agent_code(
            state=state,
            code_snippet_key="data_wrangler_function",
            result_key="messages",
            error_key="data_wrangling_error",
            llm=model,
            role="EnhancedDataWranglingAgent",
            explanation_prompt_template=explanation_prompt
        )

    def human_review_data_wrangler(state: GraphState):
        prompt_text = """
        Please review the following data wrangling approach:

        {steps}

        Does this look correct? Type 'yes' to proceed or provide modifications:
        """
        
        return node_func_human_review(
            state=state,
            prompt_text=prompt_text,
            yes_goto="execute_data_wrangler_code",
            no_goto="create_data_wrangler_code",
            user_instructions_key="user_instructions",
            recommended_steps_key="recommended_steps",
            code_snippet_key="data_wrangler_function"
        )

    def report_final_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "data_wrangler_function",
                "data_wrangled",
                "data_source_type"
            ],
            result_key="messages",
            role="EnhancedDataWranglingAgent",
            custom_title="Enhanced Data Wrangling Results"
        )

    # Create node functions dictionary
    node_functions = {
        "create_data_wrangler_code": create_data_wrangler_code,
        "execute_data_wrangler_code": execute_data_wrangler_code, 
        "fix_data_wrangler_code": fix_data_wrangler_code,
        "explain_data_wrangler_code": explain_data_wrangler_code,
        "human_review": human_review_data_wrangler,
        "report_outputs": report_final_outputs
    }

    # Create the agent graph
    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_steps", # Not used when bypassed
        create_code_node_name="create_data_wrangler_code",
        execute_code_node_name="execute_data_wrangler_code",
        fix_code_node_name="fix_data_wrangler_code", 
        explain_code_node_name="explain_data_wrangler_code",
        error_key="data_wrangling_error",
        max_retries_key="max_retries",
        retry_count_key="retry_count",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name="human_review",
        checkpointer=checkpointer,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
        agent_name="data_wrangling_agent"
    )

    return app