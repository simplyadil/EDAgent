""" Data Analysis Multi-Agent Agent"""


from langchain_core.messages import BaseMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph


from typing import Annotated, Sequence, TypedDict, Union
import operator

import pandas as pd
import json
from IPython.display import Markdown



from basic_templates.agent_template import BaseAgent
from utils.plotly import plotly_from_dict
from utils.regex import get_generic_summary, remove_consecutive_duplicates


class DataAnalysisAgent(BaseAgent):

    def __init__(
            self,
            model, 
            data_wrangling_agent,
            data_vizualization_agent,
            checkpointer = None,):
        self._params = {
            "model": model,
            "data_wrangling_agent": data_wrangling_agent,
            "data_vizualization_agent": data_vizualization_agent,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None


    def _make_compiled_graph(self):
        self.response = None
        return make_data_analysis_agent(
            model= self._params["model"], 
            data_wrangling_agent= self._params["data_wrangling_agent"],
            data_vizualization_agent= self._params["data_vizualization_agent"],
            checkpointer= self._params["checkpointer"],
        )

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    def invoke_agent(self,
                user_instructions,
                data_raw,
                max_retries=3,
                retry_count=0,
                data_source_type="file",
                **kwargs):

        input_dict = {
            "user_instructions": user_instructions,
            "data_raw": self._convert_data_input(data_raw),
            "max_retries": max_retries,
            "retry_count": retry_count,
            "data_source_type": data_source_type,
        
        }
        # Handle database connection if present
        if isinstance(data_raw, dict) and "connection" in data_raw:
            input_dict["connection"] = data_raw["connection"]
        response = self._compiled_graph.invoke(input_dict, **kwargs)
        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])
        self.response = response
        return response

    async def ainvoke_agent(self,
                        user_instructions,
                        data_raw,
                        max_retries=3,
                        retry_count=0,
                        data_source_type="file",
                        **kwargs):

        input_dict = {
            "user_instructions": user_instructions,
            "data_raw": self._convert_data_input(data_raw),
            "max_retries": max_retries,
            "retry_count": retry_count,
            "data_source_type": data_source_type,
        }
        
        # Handle database connection if present
        if isinstance(data_raw, dict) and "connection" in data_raw:
            input_dict["connection"] = data_raw["connection"]
        
        response = await self._compiled_graph.ainvoke(input_dict, **kwargs)
        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])
        self.response = response
        return response


    def get_data_wrangled(self):
        if self.response.get("plotly_graph"):
            return plotly_from_dict(self.response["plotly_graph"])
        

    def get_data_wrangling_function(self, markdown: bool = False):
        if self.response.get("data_wrangling_function"):
            snippet = self.response.get("data_wrangling_function")
            if markdown:
                return Markdown(f"```python\n{snippet}\n```")
            else:
                return snippet

    def get_data_vizualization_function(self, markdown: bool = False):
        if self.response.get("data_vizualization_function"):
            snippet = self.response.get("data_vizualization_function")
            if markdown:
                return Markdown(f"```python\n{snippet}\n```")
            else:
                return snippet
            

    def get_workflow_summary(self, markdown: bool = False):
        if self.response and self.response.get("messages"):
            agents=[]
            for msg in self.response["messages"]:
                agents.append(msg.role)
            agent_labels = []
            for i, role in enumerate(agents):
                agent_labels.append(f"- **Agent {i+1}:** {role}\n")
            reports = [get_generic_summary(json.loads(msg.content)) for msg in self.response["messages"]]
            summary = "\n\n" + "this is the summary of the workflow" + "\n\n".join(reports)
            return Markdown(summary) if markdown else summary
        

    @staticmethod
    def _convert_data_input(data_raw):
        # Handle database connection case
        if isinstance(data_raw, dict) and "connection" in data_raw:
            return data_raw
        
        # Original conversion logic for other cases
        if isinstance(data_raw, pd.DataFrame):
            return data_raw.to_dict(orient="records")
        elif isinstance(data_raw, dict):
            return [data_raw]
        elif isinstance(data_raw, list):
            return [item.to_dict(orient="records") if isinstance(item, pd.DataFrame) else item for item in data_raw]
        else:
            return [{"Result": data_raw}]
        


def make_data_analysis_agent(
    model,
    data_wrangling_agent,
    data_vizualization_agent,
    checkpointer = None,
):
    llm = model
    routing_preprocessor_prompt = PromptTemplate(
        template="""
        You are an expert in routing decisions for a Pandas Data Manipulation Wrangling Agent, a Charting Visualization Agent, and a Pandas Table Agent. Your job is to tell the agents which actions to perform and determine the correct routing for the incoming user question:
        
        1. Determine what the correct format for a Users Question should be for use with a Pandas Data Wrangling Agent based on the incoming user question. Anything related to data wrangling and manipulation should be passed along. Anything related to data analysis can be handled by the Pandas Agent. Anything that uses Pandas can be passed along. Tables can be returned from this agent. Don't pass along anything about plotting or visualization.
        2. Determine whether or not a chart should be generated or a table should be returned based on the users question.
        3. If a chart is requested, determine the correct format of a Users Question should be used with a Data Visualization Agent. Anything related to plotting and visualization should be passed along.
        
        Use the following criteria on how to route the initial user question:
        
        From the incoming user question, remove any details about the format of the final response as either a Chart or Table and return only the important part of the incoming user question that is relevant for the Pandas Data Wrangling and Transformation agent. This will be the 'user_instructions_data_wrangling'. If 'None' is found, return the original user question.
        
        Next, determine if the user would like a data visualization ('chart') or a 'table' returned with the results of the Data Wrangling Agent. If unknown, not specified or 'None' is found, then select 'table'.  
        
        If a 'chart' is requested, return the 'user_instructions_data_visualization'. If 'None' is found, return None.
        
        Return JSON with 'user_instructions_data_wrangling', 'user_instructions_data_visualization' and 'routing_preprocessor_decision'.
        
        INITIAL_USER_QUESTION: {user_instructions}
        """,
        input_variables=["user_instructions"]
    )

    routing_preprocessor = routing_preprocessor_prompt | llm | JsonOutputParser()

    class PrimaryState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        user_instructions_data_wrangling: str
        user_instructions_data_visualization: str
        routing_preprocessor_decision: str
        data_raw: Union[dict, list]
        data_wrangled: dict
        data_wrangler_function: str
        data_visualization_function: str
        plotly_graph: dict
        plotly_error: str
        max_retries: int
        retry_count: int
        data_source_type: str
        connection: Any

    def preprocess_routing(state: PrimaryState):
        print("---- Data Analysis Agent ----")

        question = state.get("user_instructions")

        #chart routing

        response = routing_preprocessor.invoke({"user_instructions": question})

        return {
            "user_instructions_data_wrangling": response.get("user_instructions_data_wrangling"),
            "user_instructions_data_visualization": response.get("user_instructions_data_visualization"),
            "routing_preprocessor_decision": response.get("routing_preprocessor_decision")
        }
    
    def router_chart_or_table(state: PrimaryState):
        print("---ROUTER: CHART OR TABLE---")
        return "chart" if state.get("routing_preprocessor_decision") == "chart" else "table"
    

    def invoke_data_wrangling_agent(state: PrimaryState):
        response = data_wrangling_agent.invoke({
            "user_instructions": state.get("user_instructions_data_wrangling"),
            "data_raw": state.get("data_raw"),
            "connection": state.get("connection"),
            "data_source_type": state.get("data_source_type", "file"),
            "max_retries": state.get("max_retries"),
            "retry_count": state.get("retry_count"),
        })

        return {
            "messages": response.get("messages"),
            "data_wrangled": response.get("data_wrangled"),
            "data_wrangler_function": response.get("data_wrangler_function"),
            "plotly_error": response.get("data_visualization_error"),
        }
    

    def invoke_data_visualization_agent(state: PrimaryState):
        
        response = data_vizualization_agent.invoke({
            "user_instructions": state.get("user_instructions_data_visualization"),
            "data_raw": state.get("data_wrangled") if state.get("data_wrangled") else state.get("data_raw"),
            "max_retries": state.get("max_retries"),
            "retry_count": state.get("retry_count"),
        })
        
        return {
            "messages": response.get("messages"),
            "data_visualization_function": response.get("data_visualization_function"),
            "plotly_graph": response.get("plotly_graph"),
            "plotly_error": response.get("data_visualization_error"),
        }
    

    def route_printer(state: PrimaryState):
        print("---ROUTE PRINTER---")
        print(f"    Route: {state.get('routing_preprocessor_decision')}")
        print("---END---")
        return {}

    workflow = StateGraph(PrimaryState)

    workflow.add_node("routing_preprocessor", preprocess_routing)
    workflow.add_node("data_wrangling_agent", invoke_data_wrangling_agent)
    workflow.add_node("data_visualization_agent", invoke_data_visualization_agent)
    workflow.add_node("route_printer", route_printer)

    workflow.add_edge(START, "routing_preprocessor")
    workflow.add_edge("routing_preprocessor", "data_wrangling_agent")

    workflow.add_conditional_edges(
        "data_wrangling_agent", 
        router_chart_or_table,
        {
            "chart": "data_visualization_agent",
            "table": "route_printer"
        }
    )

    workflow.add_edge("data_visualization_agent", "route_printer")
    workflow.add_edge("route_printer", END)


    app = workflow.compile(
        checkpointer=checkpointer, 
        name="pandas_data_analyst"
    )

    return app