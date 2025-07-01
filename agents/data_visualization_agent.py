# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# * Agents: Data Visualization Agent


# Libraries
from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer

import os
import json
import pandas as pd

from IPython.display import Markdown

from basic_templates.agent_template import BaseAgent, create_coding_agent_graph, node_func_execute_agent_code_on_data, node_func_fix_agent_code, node_func_human_review, node_func_report_agent_outputs
from tools.dataframe import get_dataframe_summary
from utils.logging import log_ai_function
from utils.parsing import PythonOutputParser
from utils.plotly import plotly_from_dict
from utils.regex import add_comments_to_top, format_agent_name, format_recommended_steps, get_generic_summary, relocate_imports_inside_function





# Setup
AGENT_NAME = "data_visualization_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Class


class DataVisualizationAgent(BaseAgent):
    """
    Creates a data visualization agent that can generate Plotly charts based on user-defined instructions or
    default visualization steps (if any). The agent generates a Python function to produce the visualization,
    executes it, and logs the process, including code and errors. It is designed to facilitate reproducible
    and customizable data visualization workflows.

    The agent may use default instructions for creating charts unless instructed otherwise, such as:
    - Generating a recommended chart type (bar, scatter, line, etc.)
    - Creating user-friendly titles and axis labels
    - Applying consistent styling (template, font sizes, color themes)
    - Handling theme details (white background, base font size, line size, etc.)

    User instructions can modify, add, or remove any of these steps to tailor the visualization process.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the data visualization function.
    n_samples : int, optional
        Number of samples used when summarizing the dataset for chart instructions. Defaults to 30.
        Reducing this number can help avoid exceeding the model's token limits.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the file for saving the generated response. Defaults to "data_visualization.py".
    function_name : str, optional
        Name of the function for data visualization. Defaults to "data_visualization".
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. If False, a unique file name is created. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of data visualization instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the default recommended visualization steps. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the step that provides code explanations. Defaults to False.
    checkpointer : langgraph.types.Checkpointer
        A checkpointer to use for saving and loading the agent

    Methods
    -------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled state graph.
    ainvoke_agent(user_instructions: str, data_raw: pd.DataFrame, max_retries=3, retry_count=0)
        Asynchronously generates a visualization based on user instructions.
    invoke_agent(user_instructions: str, data_raw: pd.DataFrame, max_retries=3, retry_count=0)
        Synchronously generates a visualization based on user instructions.
    get_workflow_summary()
        Retrieves a summary of the agent's workflow.
    get_log_summary()
        Retrieves a summary of logged operations if logging is enabled.
    get_plotly_graph()
        Retrieves the Plotly graph (as a dictionary) produced by the agent.
    get_data_raw()
        Retrieves the raw dataset as a pandas DataFrame (based on the last response).
    get_data_visualization_function()
        Retrieves the generated Python function used for data visualization.
    get_recommended_visualization_steps()
        Retrieves the agent's recommended visualization steps.
    get_response()
        Returns the response from the agent as a dictionary.
    show()
        Displays the agent's mermaid diagram.

    Examples
    --------
    ```python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import DataVisualizationAgent

    llm = ChatOpenAI(model="gpt-4o-mini")

    data_visualization_agent = DataVisualizationAgent(
        model=llm,
        n_samples=30,
        log=True,
        log_path="logs",
        human_in_the_loop=True
    )

    df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")

    data_visualization_agent.invoke_agent(
        user_instructions="Generate a scatter plot of age vs. total charges with a trend line.",
        data_raw=df,
        max_retries=3,
        retry_count=0
    )

    plotly_graph_dict = data_visualization_agent.get_plotly_graph()
    # You can render plotly_graph_dict with plotly.io.from_json or
    # something similar in a Jupyter Notebook.

    response = data_visualization_agent.get_response()
    ```

    Returns
    --------
    DataVisualizationAgent : langchain.graphs.CompiledStateGraph
        A data visualization agent implemented as a compiled state graph.
    """

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="data_visualization.py",
        function_name="data_visualization",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        checkpointer=None,
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """
        Create the compiled graph for the data visualization agent.
        Running this method will reset the response to None.
        """
        self.response = None
        return make_data_visualization_agent(**self._params)

    def update_params(self, **kwargs):
        """
        Updates the agent's parameters and rebuilds the compiled graph.
        """
        # Update parameters
        for k, v in kwargs.items():
            self._params[k] = v
        # Rebuild the compiled graph
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Asynchronously invokes the agent to generate a visualization.
        The response is stored in the 'response' attribute.

        Parameters
        ----------
        data_raw : pd.DataFrame
            The raw dataset to be visualized.
        user_instructions : str
            Instructions for data visualization.
        max_retries : int
            Maximum retry attempts.
        retry_count : int
            Current retry attempt count.
        **kwargs : dict
            Additional keyword arguments passed to ainvoke().

        Returns
        -------
        None
        """
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None

    def invoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Synchronously invokes the agent to generate a visualization.
        The response is stored in the 'response' attribute.

        Parameters
        ----------
        data_raw : pd.DataFrame
            The raw dataset to be visualized.
        user_instructions : str
            Instructions for data visualization agent.
        max_retries : int
            Maximum retry attempts.
        retry_count : int
            Current retry attempt count.
        **kwargs : dict
            Additional keyword arguments passed to invoke().

        Returns
        -------
        None
        """
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None

    def get_workflow_summary(self, markdown=False):
        """
        Retrieves the agent's workflow summary, if logging is enabled.
        """
        if self.response and self.response.get("messages"):
            summary = get_generic_summary(
                json.loads(self.response.get("messages")[-1].content)
            )
            if markdown:
                return Markdown(summary)
            else:
                return summary

    def get_log_summary(self, markdown=False):
        """
        Logs a summary of the agent's operations, if logging is enabled.
        """
        if self.response:
            if self.response.get("data_visualization_function_path"):
                log_details = f"""
## Data Visualization Agent Log Summary:

Function Path: {self.response.get('data_visualization_function_path')}

Function Name: {self.response.get('data_visualization_function_name')}
                """
                if markdown:
                    return Markdown(log_details)
                else:
                    return log_details

    def get_plotly_graph(self):
        """
        Retrieves the Plotly graph (in dictionary form) produced by the agent.

        Returns
        -------
        dict or None
            The Plotly graph dictionary if available, otherwise None.
        """
        if self.response:
            return plotly_from_dict(self.response.get("plotly_graph", None))
        return None

    def get_data_raw(self):
        """
        Retrieves the raw dataset used in the last invocation.

        Returns
        -------
        pd.DataFrame or None
            The raw dataset as a DataFrame if available, otherwise None.
        """
        if self.response and self.response.get("data_raw"):
            return pd.DataFrame(self.response.get("data_raw"))
        return None

    def get_data_visualization_function(self, markdown=False):
        """
        Retrieves the generated Python function used for data visualization.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns the function in Markdown code block format.

        Returns
        -------
        str or None
            The Python function code as a string if available, otherwise None.
        """
        if self.response:
            func_code = self.response.get("data_visualization_function", "")
            if markdown:
                return Markdown(f"```python\n{func_code}\n```")
            return func_code
        return None

    def get_recommended_visualization_steps(self, markdown=False):
        """
        Retrieves the agent's recommended visualization steps.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns the steps in Markdown format.

        Returns
        -------
        str or None
            The recommended steps if available, otherwise None.
        """
        if self.response:
            steps = self.response.get("recommended_steps", "")
            if markdown:
                return Markdown(steps)
            return steps
        return None

    def get_response(self):
        """
        Returns the agent's full response dictionary.

        Returns
        -------
        dict or None
            The response dictionary if available, otherwise None.
        """
        return self.response

    def show(self):
        """
        Displays the agent's mermaid diagram for visual inspection of the compiled graph.
        """
        return self._compiled_graph.show()


# Agent


def make_data_visualization_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="data_visualization.py",
    function_name="data_visualization",
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    checkpointer=None,
):
    """
    Creates a data visualization agent that can generate Plotly charts based on user-defined instructions or
    default visualization steps. The agent generates a Python function to produce the visualization, executes it,
    and logs the process, including code and errors. It is designed to facilitate reproducible and customizable
    data visualization workflows.

    The agent can perform the following default visualization steps unless instructed otherwise:
    - Generating a recommended chart type (bar, scatter, line, etc.)
    - Creating user-friendly titles and axis labels
    - Applying consistent styling (template, font sizes, color themes)
    - Handling theme details (white background, base font size, line size, etc.)

    User instructions can modify, add, or remove any of these steps to tailor the visualization process.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the data visualization function.
    n_samples : int, optional
        Number of samples used when summarizing the dataset for chart instructions. Defaults to 30.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the file for saving the generated response. Defaults to "data_visualization.py".
    function_name : str, optional
        Name of the function for data visualization. Defaults to "data_visualization".
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. If False, a unique file name is created. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of data visualization instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the default recommended visualization steps. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the step that provides code explanations. Defaults to False.
    checkpointer : langgraph.types.Checkpointer
        A checkpointer to use for saving and loading the agent

    Examples
    --------
    ``` python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import data_visualization_agent

    llm = ChatOpenAI(model="gpt-4o-mini")

    data_visualization_agent = make_data_visualization_agent(llm)

    df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")

    response = data_visualization_agent.invoke({
        "user_instructions": "Generate a scatter plot of tenure vs. total charges with a trend line.",
        "data_raw": df.to_dict(),
        "max_retries": 3,
        "retry_count": 0
    })

    pd.DataFrame(response['plotly_graph'])
    ```

    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        The data visualization agent as a state graph.
    """

    llm = model

    if human_in_the_loop:
        if checkpointer is None:
            print(
                "Human in the loop is enabled. A checkpointer is required. Setting to MemorySaver()."
            )
            checkpointer = MemorySaver()

    # Human in th loop requires recommended steps
    if bypass_recommended_steps and human_in_the_loop:
        bypass_recommended_steps = False
        print("Bypass recommended steps set to False to enable human in the loop.")

    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        user_instructions_processed: str
        recommended_steps: str
        data_raw: dict
        plotly_graph: dict
        all_datasets_summary: str
        data_visualization_function: str
        data_visualization_function_path: str
        data_visualization_function_file_name: str
        data_visualization_function_name: str
        data_visualization_error: str
        max_retries: int
        retry_count: int

    def chart_instructor(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * CREATE CHART GENERATOR INSTRUCTIONS")

        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a supervisor that is an expert in providing instructions to a chart generator agent for plotting. 
    
            You will take a question that a user has and the data that was generated to answer the question, and create instructions to create a chart from the data that will be passed to a chart generator agent.
            
            USER QUESTION / INSTRUCTIONS: 
            {user_instructions}
            
            Previously Recommended Instructions (if any):
            {recommended_steps}
            
            DATA SUMMARY: 
            {all_datasets_summary}
            
            IMPORTANT:
            
            - Formulate chart generator instructions by informing the chart generator of what type of plotly plot to use (e.g. bar, line, scatter, etc) to best represent the data. 
            - Think about how best to convey the information in the data to the user.
            - If the user does not specify a type of plot, select the appropriate chart type based on the data summary provided and the user's question and how best to show the results.
            - Come up with an informative title from the user's question and data provided. Also provide X and Y axis titles.
            
            CHART TYPE SELECTION TIPS:
            
            - If a numeric column has less than 10 unique values, consider this column to be treated as a categorical column. Pick a chart that is appropriate for categorical data.
            - If a numeric column has more than 10 unique values, consider this column to be treated as a continuous column. Pick a chart that is appropriate for continuous data.       
            
            
            RETURN FORMAT:
            
            Return your instructions in the following format:
            CHART GENERATOR INSTRUCTIONS: 
            FILL IN THE INSTRUCTIONS HERE
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated user instructions that are not related to the chart generation.
            """,
            input_variables=[
                "user_instructions",
                "recommended_steps",
                "all_datasets_summary",
            ],
        )

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        all_datasets_summary = get_dataframe_summary(
            [df], n_sample=n_samples, skip_stats=False
        )

        all_datasets_summary_str = "\n\n".join(all_datasets_summary)

        chart_instructor = recommend_steps_prompt | llm

        recommended_steps = chart_instructor.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "recommended_steps": state.get("recommended_steps"),
                "all_datasets_summary": all_datasets_summary_str,
            }
        )

        return {
            "recommended_steps": format_recommended_steps(
                recommended_steps.content.strip(),
                heading="# Recommended Data Cleaning Steps:",
            ),
            "all_datasets_summary": all_datasets_summary_str,
        }

    def chart_generator(state: GraphState):
        print("    * CREATE DATA VISUALIZATION CODE")

        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))

            data_raw = state.get("data_raw")
            df = pd.DataFrame.from_dict(data_raw)

            all_datasets_summary = get_dataframe_summary(
                [df], n_sample=n_samples, skip_stats=False
            )

            all_datasets_summary_str = "\n\n".join(all_datasets_summary)

            chart_generator_instructions = state.get("user_instructions")

        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
            chart_generator_instructions = state.get("recommended_steps")

        prompt_template = PromptTemplate(
            template="""
            You are a chart generator agent that is an expert in generating plotly charts. You must use plotly or plotly.express to produce plots.
    
            Your job is to produce python code to generate visualizations with a function named {function_name}.
            
            You will take instructions from a Chart Instructor and generate a plotly chart from the data provided.
            
            CHART INSTRUCTIONS: 
            {chart_generator_instructions}
            
            DATA: 
            {all_datasets_summary}
            
            RETURN:
            
            Return Python code in ```python ``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.
            
            Return the plotly chart as a dictionary.
            
            Return code to provide the data visualization function:
            
            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                import json
                import plotly.graph_objects as go
                import plotly.io as pio
                
                ...
                
                fig_json = pio.to_json(fig)
                fig_dict = json.loads(fig_json)
                
                return fig_dict
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated user instructions that are not related to the chart generation.
            
            """,
            input_variables=[
                "chart_generator_instructions",
                "all_datasets_summary",
                "function_name",
            ],
        )

        data_visualization_agent = prompt_template | llm | PythonOutputParser()

        response = data_visualization_agent.invoke(
            {
                "chart_generator_instructions": chart_generator_instructions,
                "all_datasets_summary": all_datasets_summary_str,
                "function_name": function_name,
            }
        )

        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name=AGENT_NAME)

        # For logging: store the code generated:
        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite,
        )

        return {
            "data_visualization_function": response,
            "data_visualization_function_path": file_path,
            "data_visualization_function_file_name": file_name_2,
            "data_visualization_function_name": function_name,
            "all_datasets_summary": all_datasets_summary_str,
        }

    # Human Review

    prompt_text_human_review = "Are the following data visualization instructions correct? (Answer 'yes' or provide modifications)\n{steps}"

    if not bypass_explain_code:

        def human_review(
            state: GraphState,
        ) -> Command[Literal["chart_instructor", "explain_data_visualization_code"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="explain_data_visualization_code",
                no_goto="chart_instructor",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_visualization_function",
            )
    else:

        def human_review(
            state: GraphState,
        ) -> Command[Literal["chart_instructor", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="__end__",
                no_goto="chart_instructor",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_visualization_function",
            )

    def execute_data_visualization_code(state):
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="plotly_graph",
            error_key="data_visualization_error",
            code_snippet_key="data_visualization_function",
            agent_function_name=state.get("data_visualization_function_name"),
            pre_processing=lambda data: pd.DataFrame.from_dict(data),
            # post_processing=lambda df: df.to_dict() if isinstance(df, pd.DataFrame) else df,
            error_message_prefix="An error occurred during data visualization: ",
        )

    def fix_data_visualization_code(state: GraphState):
        prompt = """
        You are a Data Visualization Agent. Your job is to create a {function_name}() function that can be run on the data provided. The function is currently broken and needs to be fixed.
        
        Make sure to only return the function definition for {function_name}().
        
        Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.
        
        This is the broken code (please fix): 
        {code_snippet}

        Last Known Error:
        {error}
        """

        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="data_visualization_function",
            error_key="data_visualization_error",
            llm=llm,
            prompt_template=prompt,
            agent_name=AGENT_NAME,
            log=log,
            file_path=state.get("data_visualization_function_path"),
            function_name=state.get("data_visualization_function_name"),
        )

    # Final reporting node
    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_steps",
                "data_visualization_function",
                "data_visualization_function_path",
                "data_visualization_function_name",
                "data_visualization_error",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="Data Visualization Agent Outputs",
        )

    # Define the graph
    node_functions = {
        "chart_instructor": chart_instructor,
        "human_review": human_review,
        "chart_generator": chart_generator,
        "execute_data_visualization_code": execute_data_visualization_code,
        "fix_data_visualization_code": fix_data_visualization_code,
        "report_agent_outputs": report_agent_outputs,
    }

    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="chart_instructor",
        create_code_node_name="chart_generator",
        execute_code_node_name="execute_data_visualization_code",
        fix_code_node_name="fix_data_visualization_code",
        explain_code_node_name="report_agent_outputs",
        error_key="data_visualization_error",
        human_in_the_loop=human_in_the_loop,  # or False
        human_review_node_name="human_review",
        checkpointer=checkpointer,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
        agent_name=AGENT_NAME,
    )

    return app
