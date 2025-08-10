"""Data Analysis Copilot"""

import traceback
import streamlit as st
import pandas as pd
import plotly.io as pio
import json
import sys
import os

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.data_analyst_agent import DataAnalysisAgent
from agents.data_visualization_agent import DataVisualizationAgent
from agents.data_wrangling_agent import DataWranglingAgent

from sqlalchemy import create_engine

# * APP INPUTS ----

MODEL_LIST = ["gemini-2.0-flash", "gemini-1.5-pro"]
TITLE = "Pandas Data Analyst AI Copilot"
GOOGLE_API_KEY = "AIzaSyDUa-_8swPWfOVp2avPaRetesKKyRh0cvw"  # <-- Replace with your actual Gemini API key

genai.configure(api_key=GOOGLE_API_KEY)

# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.set_page_config(
    page_title=TITLE,
    page_icon="📊",
)
st.title(TITLE)

st.markdown("""
Welcome to the Pandas Data Analyst AI. Upload a CSV/Excel file or connect to a database, then ask questions about the data.  
The AI agent will analyze your dataset and return either data tables or interactive charts.
""")

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        ##### Bikes Data Set:
        
        -  Show the top 5 bike models by extended sales.
        -  Show the top 5 bike models by extended sales in a bar chart.
        -  Show the top 5 bike models by extended sales in a pie chart.
        -  Make a plot of extended sales by month for each bike model. Use a color to identify the bike models.
        """
    )

# ---------------------------
# Gemini Model Selection
# ---------------------------

model_option = st.sidebar.selectbox("Choose Gemini model", MODEL_LIST, index=0)
llm = ChatGoogleGenerativeAI(model=model_option, google_api_key=GOOGLE_API_KEY)

# ---------------------------
# Data Source Selection
# ---------------------------

data_source = st.sidebar.selectbox("Select data source", ["File", "Database"])

connection = None
df = None

if data_source == "File":
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV or Excel file to get started.")
        st.stop()

elif data_source == "Database":
    st.subheader("Database Connection Details")
    db_host = st.text_input("Host", "localhost")
    db_port = st.text_input("Port", "5432")
    db_user = st.text_input("User", "postgres")
    db_password = st.text_input("Password", type="password")
    db_name = st.text_input("Database Name", "mydb")

    if "connection" not in st.session_state:
        st.session_state.connection = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "tables" not in st.session_state:
        st.session_state.tables = []
    if "selected_table" not in st.session_state:
        st.session_state.selected_table = None

    if st.button("Connect"):
        try:
            conn_str = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            engine = create_engine(conn_str)
            st.session_state.connection = engine.connect()
            # Get list of tables
            st.session_state.tables = pd.read_sql(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public'",
                con=st.session_state.connection
            )["table_name"].tolist()
            st.success("Connected to database. Please select a table.")
        except Exception as e:
            st.error(f"Could not connect to database: {e}")
            st.stop()

    if st.session_state.connection and st.session_state.tables:
        st.session_state.selected_table = st.selectbox(
            "Select a table to load", st.session_state.tables
        )

        if st.button("Load Data") and st.session_state.selected_table:
            try:
                # Load entire table (no LIMIT)
                st.session_state.df = pd.read_sql(
                    f'SELECT * FROM "{st.session_state.selected_table}"',
                    con=st.session_state.connection
                )
                st.subheader(f"Data Preview (Top 5 Rows) — {st.session_state.selected_table}")
                st.dataframe(st.session_state.df.head())  # Preview first 5 rows only
                st.write(f"Loaded DataFrame shape: {st.session_state.df.shape}")
                st.write(st.session_state.df.dtypes)

            except Exception as e:
                st.error(f"Could not load data: {e}")
                st.stop()

    connection = st.session_state.connection
    df = st.session_state.df




# ---------------------------
# Initialize Chat Message History and Storage
# ---------------------------

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

if "plots" not in st.session_state:
    st.session_state.plots = []

if "dataframes" not in st.session_state:
    st.session_state.dataframes = []


def display_chat_history():
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(
                    st.session_state.plots[plot_index], key=f"history_plot_{plot_index}"
                )
            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(
                    st.session_state.dataframes[df_index],
                    key=f"history_dataframe_{df_index}",
                )
            else:
                st.write(msg.content)


# Render current messages from StreamlitChatMessageHistory
display_chat_history()

# ---------------------------
# AI Agent Setup
# ---------------------------

LOG = False

pandas_data_analyst = DataAnalysisAgent(
    model=llm,
    data_wrangling_agent=DataWranglingAgent(
        model=llm,
        log=LOG,
        bypass_recommended_steps=True,
        n_samples=100,
    ),
    data_vizualization_agent=DataVisualizationAgent(
        model=llm,
        n_samples=100,
        log=LOG,
    ),
)

# ---------------------------
# Chat Input and Agent Invocation
# ---------------------------

if question := st.chat_input("Enter your question here:", key="query_input"):
    # Check if data is ready
    if data_source == "File" and df is None:
        st.chat_message("ai").write("Please upload a file first.")
        st.stop()
    if data_source == "Database" and connection is None:
        st.chat_message("ai").write("Please connect to the database first.")
        st.stop()

    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        # Prepare input for the agent
        if data_source == "File":
            data_input = df
        else:
            # Pass dict with connection for DB usage
            data_input = {"connection": connection}

        try:
            pandas_data_analyst.invoke_agent(
                user_instructions=question,
                data_raw=data_input,
            )
            result = pandas_data_analyst.response  # use .response property to get last result
        except Exception as e:
            st.chat_message("ai").write(
                "An error occurred while processing your query. Please try again."
            )
            msgs.add_ai_message(
                "An error occurred while processing your query. Please try again."
            )
            traceback.print_exc()
            st.stop()

        routing = result.get("routing_preprocessor_decision")

        if routing == "chart" and not result.get("plotly_error", False):
            # Process chart result
            plot_data = result.get("plotly_graph")
            if plot_data:
                # Convert dictionary to JSON string if needed
                if isinstance(plot_data, dict):
                    plot_json = json.dumps(plot_data)
                else:
                    plot_json = plot_data
                plot_obj = pio.from_json(plot_json)
                response_text = "Returning the generated chart."
                # Store the chart
                plot_index = len(st.session_state.plots)
                st.session_state.plots.append(plot_obj)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                st.chat_message("ai").write(response_text)
                st.plotly_chart(plot_obj)
            else:
                st.chat_message("ai").write("The agent did not return a valid chart.")
                msgs.add_ai_message("The agent did not return a valid chart.")

        elif routing == "table":
            # Process table result
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                response_text = "Returning the data table."
                # Ensure data_wrangled is a DataFrame
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write(response_text)
                st.dataframe(data_wrangled)
            else:
                st.chat_message("ai").write("No table data was returned by the agent.")
                msgs.add_ai_message("No table data was returned by the agent.")
        else:
            # Fallback if routing decision is unclear or if chart error occurred
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                response_text = (
                    "I apologize. There was an issue with generating the chart. "
                    "Returning the data table instead."
                )
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write(response_text)
                st.dataframe(data_wrangled)
            else:
                response_text = (
                    "An error occurred while processing your query. Please try again."
                )
                msgs.add_ai_message(response_text)
                st.chat_message("ai").write(response_text)
