import pandas as pd
from agents.eda_agent import EDA_Agent
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
import sys

# =============================
# Hardcoded Google API Key (replace with your actual key)
# =============================
GOOGLE_API_KEY = "AIzaSyDUa-_8swPWfOVp2avPaRetesKKyRh0cvw"  # <-- Replace this!
MODEL_NAME = "gemini-2.0-flash"
SAMPLE_CSV = "sample_data/MOROCCO_TRAINING_DATA_WITH_WIND.csv"

# =============================
# Configure Google Generative AI
# =============================
genai.configure(api_key=GOOGLE_API_KEY)

# =============================
# Load sample data
# =============================
if not os.path.exists(SAMPLE_CSV):
    print(f"Sample CSV not found at {SAMPLE_CSV}")
    exit(1)
else:
    print("all good")

df = pd.read_csv(SAMPLE_CSV)
print(f"Loaded data with shape: {df.shape}")

# =============================
# Create the LLM and EDA Agent
# =============================
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY)
eda_agent = EDA_Agent(
    llm,
    invoke_react_agent_kwargs={"recursion_limit": 10},
)

# =============================
# Get user question
# =============================
def get_user_question():
    if len(sys.argv) > 1:
        # Use the first argument as the question
        return " ".join(sys.argv[1:])
    # No argument provided, use default
    return "Describe the dataset."

question = "Describe the dataset using the data_describer tool."

# =============================
# Run the agent
# =============================
print(f"\nRunning EDA Agent with question: {question}\n")
eda_agent.invoke_agent(
    user_instructions=question,
    raw_data=df,
)

# =============================
# Print results
# =============================
ai_message = eda_agent.get_ai_message(markdown=False)
artifacts = eda_agent.get_artifacts(as_dataframe=False)
tool_calls = eda_agent.get_tool_calls()

print("\n=== AI Message ===\n")
print(ai_message)

print("\n=== Tool Calls ===\n")
print(tool_calls)

print("\n=== Artifacts ===\n")
print(artifacts) 