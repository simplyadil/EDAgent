import operator
import copy
from typing import Annotated, Sequence
from IPython.display import Markdown
import pandas as pd

from langchain_core.messages import BaseMessage, AIMessage
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import START, END, StateGraph
from langgraph.types import Checkpointer


from basic_templates.agent_template import BaseAgent
from tools.eda import (data_describer,
                       data_explainer,
                       data_missing_visualizer,
                       sweetviz_report_generator,
                       dtale_report_generator)
from utils.messages import get_tool_call_names



AGENT_NAME = "exploratory_data_analyst_agent"

EDA_TOOLKIT = [
    data_explainer,
    data_describer,
    data_missing_visualizer,
    sweetviz_report_generator,
    dtale_report_generator,
]

class EDA_Agent(BaseAgent):
    """An agent for exploratory data analysis."""
    def __init__(
            self,
            model,
            create_react_agent_kwargs={},
            invoke_react_agent_kwargs={},
            checkpointer=None,
    ):
        self._params = {
            "model": model,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "invoke_react_agent_kwargs": invoke_react_agent_kwargs,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Creates the compiled state graph for the EDA agent."""

        self.response = None
        return make_eda_tools_agent(**self._params)


    def update_params(self, **kwargs):
        """
        Updates the agent's parameters and rebuilds the compiled graph.
        """
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()


    async def ainvoke_agent(
            self,
            user_instructions = None,
            raw_data = None,
            **kwargs,
    ):
        """ Async runs the agent with the user instructions + the data"""

        response = await self._compiled_graph.ainvoke(
            {"user_instructions": user_instructions,
             "raw_data": raw_data.to_dict() if raw_data is not None else raw_data
            },
            **kwargs,
        )
        self.response = response
        return

    def invoke_agent(
            self,
            user_instructions = None,
            raw_data = None,
            **kwargs,
    ):
        """ Synchronusly Runs the agent with the user instructions + the data"""

        # Debug information
        print(f"EDA Agent - Raw data type: {type(raw_data)}")
        print(f"EDA Agent - Raw data is None: {raw_data is None}")

        if raw_data is not None:
            print(f"EDA Agent - Raw data shape: {raw_data.shape if hasattr(raw_data, 'shape') else 'No shape attribute'}")
            print(f"EDA Agent - Raw data columns: {raw_data.columns.tolist() if hasattr(raw_data, 'columns') else 'No columns attribute'}")

            # Convert to dict safely
            try:
                data_dict = raw_data.to_dict()
                print(f"EDA Agent - Successfully converted data to dict with {len(data_dict)} columns")
            except Exception as e:
                print(f"EDA Agent - Error converting data to dict: {e}")
                data_dict = None
        else:
            data_dict = None

        response = self._compiled_graph.invoke(
            {"user_instructions": user_instructions,
             "data_raw": data_dict
            },
            **kwargs,
        )
        self.response = response
        return

    def get_internal_messages(
            self,
            markdown = False,
    ):
        """
        Returns the internal messages from the agent's response.
        """
        pretty_print = "\n\n".join(
            [
                f"### {msg.type.upper()}\n\nID: {msg.id}\n\nContent:\n\n{msg.content}"
                for msg in self.response["internal_messages"]
            ]
        )
        if markdown:
            return Markdown(pretty_print)
        else:
            return self.response["internal_messages"]


    def get_artifacts(
            self,
            as_dataframe = False):
        """Returns the EDA artifacts from the agent's response."""
        if as_dataframe:
            return pd.DataFrame(self.response["eda_artifacts"])
        else:
            return self.response["eda_artifacts"]

    def get_ai_message(
            self,
            markdown = False,
    ):
        """Returns the AI message from the agent's response."""
        if markdown:
            return Markdown(self.response["messages"][0].content)
        else:
            return self.response["messages"][0].content

    def get_tool_calls(self):
        """Returns the tool calls made by the agent."""
        return self.response["tool_calls"]


def make_eda_tools_agent(
        model,
        create_react_agent_kwargs={},
        invoke_react_agent_kwargs={},
        checkpointer=None,
):
    """Creates an Exploratory Data Analyst Agent that can interact with EDA tools."""

    class GraphState(AgentState):
        internal_messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_raw: dict
        eda_artifacts: dict
        tool_calls: list

    def exploratory_agent(state):
        print(AGENT_NAME)
        print("    * RUN REACT TOOL-CALLING AGENT FOR EDA")

        tool_node = ToolNode(EDA_TOOLKIT)


        eda_agent = create_react_agent(
            model,
            tools=tool_node,
            state_schema=GraphState,
            **create_react_agent_kwargs,
            checkpointer=checkpointer,
        )


        # Create a message that includes instructions to use the data
        user_message = f"""
{state["user_instructions"]}

You have access to a dataset with the following information:
- The dataset has {len(state["data_raw"].keys()) if state["data_raw"] else 0} columns
- Use the available tools to analyze this dataset
- The data is already loaded and available to the tools
"""

        # Create a tools dictionary with the data
        tools_dict = {}
        for tool in EDA_TOOLKIT:
            # Create a wrapper function that passes the data to the tool
            def create_tool_wrapper(tool_func):
                def wrapper(*args, **kwargs):
                    # Add the data to the kwargs
                    kwargs["raw_data"] = state["data_raw"]
                    return tool_func(*args, **kwargs)
                return wrapper

            # Replace the tool function with the wrapper
            tool_copy = copy.deepcopy(tool)
            original_func = tool_copy.func
            tool_copy.func = create_tool_wrapper(original_func)
            tools_dict[tool.name] = tool_copy

        # Add the tools to the config
        config = invoke_react_agent_kwargs.copy() if invoke_react_agent_kwargs else {}
        config["tools"] = tools_dict

        response = eda_agent.invoke(
            {
                "messages": [("user", user_message)],
                "data_raw": state["data_raw"],
            },
            config,
        )
        print("    * POST-PROCESSING EDA RESULTS")

        internal_messages = response["messages"]

        if not internal_messages:
            return {"internal_messages": [], "eda_artifacts": None}

        last_ai_message = AIMessage(internal_messages[-1].content, role=AGENT_NAME)

        last_tool_artifact = None

        if len(internal_messages) > 1:
            last_message = internal_messages[-2]
            if hasattr(last_message, "artifact"):
                last_tool_artifact = last_message.artifact
            elif isinstance(last_message, dict) and "artifact" in last_message:
                last_tool_artifact = last_message["artifact"]

        tool_calls = get_tool_call_names(internal_messages)

        return {
            "messages": [last_ai_message],
            "internal_messages": internal_messages,
            "eda_artifacts": last_tool_artifact,
            "tool_calls": tool_calls,
        }
    workflow = StateGraph(GraphState)
    workflow.add_node("exploratory_agent", exploratory_agent)
    workflow.add_edge(START, "exploratory_agent")
    workflow.add_edge("exploratory_agent", END)

    app = workflow.compile(
        checkpointer=checkpointer,
        name=AGENT_NAME,
    )

    return app

