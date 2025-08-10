# EDAgent: Exploratory Data Analysis Agent System

A comprehensive AI-powered data analysis platform built with LangChain and LangGraph for automated exploratory data analysis workflows.

## Overview

EDAgent is a personal project that demonstrates the implementation of sophisticated AI agent architectures for data science applications. The system combines Large Language Models (LLMs) with specialized data analysis tools to create an automated exploratory data analysis pipeline.

## Key Features

- **Multi-Agent Architecture**: Implements specialized agents for EDA, data wrangling, and visualization using LangGraph
- **Tool Integration**: Automated correlation analysis, missing data visualization, and statistical reporting
- **State Management**: Sophisticated workflow orchestration with LangChain's state injection patterns
- **Web Interface**: Streamlit-based application for data upload and analysis
- **Extensible Design**: Modular tool system for easy addition of new analysis capabilities

## Technical Implementation

### Architecture Components

- **Framework**: LangChain/LangGraph for agent orchestration
- **Language Model**: Google Gemini integration for natural language processing
- **Data Analysis**: Pandas, Sweetviz, Dtale, Missingno, Pytimetk
- **Web Framework**: Streamlit for user interface
- **Architecture Pattern**: React agent with tool injection and state management

### Core Technologies

- Python 3.8+
- LangChain/LangGraph
- Google Gemini API
- Streamlit
- Pandas
- Sweetviz
- Dtale
- Missingno
- Pytimetk

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Google Generative AI API key
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd EDAgent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Google API key:
   - Obtain a Google Generative AI API key
   - Update the `GOOGLE_API_KEY` variable in the application files

### Usage

#### Streamlit Web Application

Run the main EDA application:
```bash
streamlit run apps/eda_app.py
```

Features:
- Upload CSV or Excel files
- Use built-in demo data (churn_data.csv)
- Ask natural language questions about your dataset
- View generated visualizations and reports



## Usage Examples

### Natural Language Queries

The system accepts natural language questions such as:

- "Describe the dataset"
- "Analyze missing values in the dataset"
- "Generate a correlation funnel using 'Churn' as the target"
- "Create a Sweetviz report for the dataset"
- "Show me the first 5 rows and describe what they contain"
- "What tools do you have access to?"

### Generated Outputs

The system automatically generates:

- **Summary Statistics**: Comprehensive dataset descriptions and statistical summaries
- **Missing Data Analysis**: Matrix plots, bar charts, and heatmaps for missing value patterns
- **Correlation Analysis**: Correlation funnels with interactive Plotly visualizations
- **Automated Reports**: Sweetviz and Dtale HTML reports for comprehensive EDA
- **Data Visualizations**: Custom charts and plots based on user requests

## Technical Architecture



### State Management

Uses LangGraph's state management with custom `GraphState`:
- Message accumulation with `operator.add`
- Data context through `data_raw` injection
- Artifact tracking for UI rendering
- Tool execution history for transparency

### Tool Integration

Implements LangChain's tool injection pattern:
- `InjectedState` for automatic parameter binding
- Stateless tool design with stateful execution context
- Structured output formats (`content` and `content_and_artifact`)

## Learning Objectives

This project was developed to explore:

- Advanced AI agent architectures and design patterns
- LangChain/LangGraph implementation and best practices
- State management in multi-agent systems
- Tool integration and dependency injection patterns
- Real-world data science workflow automation
- Web application development with Streamlit
- API integration with modern LLM services

## Future Enhancements

- Additional data analysis tools and capabilities
- Enhanced error handling and validation mechanisms
- Performance optimizations for large datasets
- Extended agent capabilities for more complex workflows
- Improved documentation and code examples
- Integration with additional data sources and formats


## License

[MIT](https://choosealicense.com/licenses/mit/)

---

**This project serves as a learning exercise in building sophisticated AI agent systems for data science applications.**
