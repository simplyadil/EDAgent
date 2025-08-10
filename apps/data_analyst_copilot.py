"""Enhanced Data Analysis Copilot with Database Support"""

import streamlit as st
import pandas as pd
import plotly.io as pio
import json
import sys
import os
import sqlalchemy as sql
from sqlalchemy import create_engine, text

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.data_analyst_agent import DataAnalysisAgent
from agents.data_visualization_agent import DataVisualizationAgent
from agents.data_wrangling_agent import DataWranglingAgent

# * APP INPUTS ----

MODEL_LIST = ["gemini-2.0-flash", "gemini-1.5-pro"]
TITLE = "Pandas Data Analyst AI Copilot"
GOOGLE_API_KEY = "AIzaSyDUa-_8swPWfOVp2avPaRetesKKyRh0cvw"

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
Welcome to the Enhanced Pandas Data Analyst AI. Choose to upload a file or connect to a database, then ask questions about your data.  
The AI agent will analyze your dataset and return either data tables or interactive charts.
""")

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        ##### For File Data:
        - Show the top 5 bike models by extended sales.
        - Show the top 5 bike models by extended sales in a bar chart.
        - Make a plot of extended sales by month for each bike model.
        
        ##### For Database Data:
        - Show me the monthly revenue trends from the sales table.
        - Which customers have the highest order values?
        - Create a chart showing product performance by region.
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

st.markdown("## Choose Your Data Source")

data_source = st.radio(
    "Select data source:",
    ["Upload File", "Connect to Database"],
    horizontal=True
)

# Initialize variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'connection' not in st.session_state:
    st.session_state.connection = None
if 'data_source_type' not in st.session_state:
    st.session_state.data_source_type = 'file'

df = st.session_state.df
connection = st.session_state.connection
data_source_type = st.session_state.data_source_type

if data_source == "Upload File":
    # ---------------------------
    # File Upload Section
    # ---------------------------
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
    )
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.session_state.data_source_type = "file"
        st.subheader("Data Preview")
        st.dataframe(df.head())
        data_source_type = "file"
    else:
        st.info("Please upload a CSV or Excel file to get started.")
        st.stop()

else:
    # ---------------------------
    # Database Connection Section  
    # ---------------------------
    
    st.markdown("### Database Connection")
    
    # Database type selection
    db_type = st.selectbox(
        "Database Type:",
        ["PostgreSQL", "SQLite"]
    )
    
    # Connection parameters based on database type
    if db_type == "SQLite":
        db_file = st.text_input("SQLite Database File Path:", value="database.db")
        if st.button("Connect to SQLite"):
            try:
                connection_string = f"sqlite:///{db_file}"
                connection = create_engine(connection_string)
                with connection.connect() as conn:
                    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
                    tables = [row[0] for row in result.fetchall()]
                st.success(f"Connected to SQLite database! Found {len(tables)} tables.")
                st.write("Available tables:", tables)
                st.session_state.connection = connection
                st.session_state.data_source_type = "database"
                data_source_type = "database"
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
                st.stop()
    
    elif db_type == "PostgreSQL":
        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host:", value="localhost")
            database = st.text_input("Database Name:")
            username = st.text_input("Username:")
        with col2:
            port = st.number_input("Port:", value=5432)
            password = st.text_input("Password:", type="password")
        if st.button(f"Connect to {db_type}"):
            try:
                connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                connection = create_engine(connection_string)
                with connection.connect() as conn:
                    result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
                    tables = [row[0] for row in result.fetchall()]
                st.success(f"Connected to {db_type} database! Found {len(tables)} tables.")
                st.write("Available tables:", tables)
                st.session_state.connection = connection
                st.session_state.data_source_type = "database"
                data_source_type = "database"
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
                st.info("Make sure your database is running and credentials are correct.")
                st.stop()
    
    # If we have a database connection, show a preview
    if connection is not None:
        st.markdown("### Database Preview")
        
        # Let user pick a table to preview
        try:
            with connection.connect() as conn:
                # Get table names (this is database-agnostic)
                inspector = sql.inspect(connection)
                tables = inspector.get_table_names()
                
                if tables:
                    preview_table = st.selectbox("Select table to preview:", tables)
                    
                    if preview_table:
                        # Show table preview
                        preview_query = f"SELECT * FROM {preview_table} LIMIT 5"
                        df_preview = pd.read_sql(preview_query, connection)
                        st.dataframe(df_preview)
                        
                        # Show table info
                        info_query = f"SELECT COUNT(*) as row_count FROM {preview_table}"
                        row_count = pd.read_sql(info_query, connection).iloc[0, 0]
                        st.info(f"Table '{preview_table}' has {row_count:,} rows")
                else:
                    st.warning("No tables found in the database.")
                    
        except Exception as e:
            st.warning(f"Could not preview tables: {str(e)}")

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


# Initialize the agent (same agent handles both modes)
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
    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        try:
            if data_source_type == "database":
                # For database queries, pass the connection inside data_raw
                pandas_data_analyst.invoke_agent(
                    user_instructions=question,
                    data_raw={"connection": connection},
                    data_source_type="database"
                )
            else:
                # For file uploads, use the original approach
                pandas_data_analyst.invoke_agent(
                    user_instructions=question,
                    data_raw=df,
                    data_source_type="file"
                )
            result = pandas_data_analyst.get_response()
            
        except Exception as e:
            error_msg = f"An error occurred while processing your query: {str(e)}"
            st.chat_message("ai").write(error_msg)
            msgs.add_ai_message(error_msg)
            st.error(f"Error details: {str(e)}\nPlease check your question and try again. For database queries, ensure table names and columns are correct.")
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
                
                data_source_label = "database" if data_source_type == "database" else "uploaded file"
                response_text = f"Here's your chart based on the {data_source_label} data:"
                
                # Store the chart
                plot_index = len(st.session_state.plots)
                st.session_state.plots.append(plot_obj)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                st.chat_message("ai").write(response_text)
                st.plotly_chart(plot_obj)
            else:
                st.chat_message("ai").write("I couldn't generate a valid chart from your request.")
                msgs.add_ai_message("I couldn't generate a valid chart from your request.")

        elif routing == "table":
            # Process table result
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                data_source_label = "database" if data_source_type == "database" else "uploaded file"
                response_text = f"Here's your data table based on the {data_source_label}:"

                # Always convert to DataFrame for display
                try:
                    display_df = pd.DataFrame(data_wrangled)
                except Exception:
                    display_df = pd.DataFrame({"Result": [data_wrangled]})

                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(display_df)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write(response_text)
                st.dataframe(display_df)
            else:
                st.chat_message("ai").write("No table data was returned. Please try rephrasing your question.")
                msgs.add_ai_message("No table data was returned. Please try rephrasing your question.")
        else:
            # Fallback handling
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
                    "An error occurred while processing your query. Please try again. check if your "
                    f"{'database tables' if data_source_type == 'database' else 'data'} contain the information you're looking for."
                )
                msgs.add_ai_message(response_text)
                st.chat_message("ai").write(response_text)

# ---------------------------
# Sidebar Information
# ---------------------------

with st.sidebar:
    st.markdown("### Current Data Source")
    if data_source_type == "database":
        st.success("Connected to Database")
        if connection:
            st.write("Connection active")
    else:
        st.success("File Upload Mode")
        if df is not None:
            st.write(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
