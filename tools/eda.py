
import os
import tempfile
from tools.dataframe import get_dataframe_summary
from langchain.tools import tool


@tool(response_format="content")
def data_explainer(
        n_sample=30,
        skip_stats=False,
):
    """
    A tool for data explanation. Provides a comprehensive summary of the dataset.

    Args:
        n_sample: Number of rows to display in the sample (default: 30)
        skip_stats: Whether to skip detailed statistics (default: False)

    Returns:
        A detailed explanation of the dataset
    """
    import pandas as pd
    # Try different import paths for get_current_config
    try:
        from langchain.callbacks import get_current_config
    except ImportError:
        try:
            from langchain_core.callbacks import get_current_config
        except ImportError:
            # Fallback if import fails
            def get_current_config():
                return {"configurable": {"raw_data": None}}

    # Get the data from the configurable
    try:
        config = get_current_config()
        raw_data = config.get("configurable", {}).get("raw_data", None)
    except Exception:
        # Fallback if get_current_config fails
        raw_data = None

    if raw_data is None:
        return "I need data to explain. Please provide the data."

    result = get_dataframe_summary(pd.DataFrame(raw_data), n_sample, skip_stats)
    return result


@tool(response_format="content_and_artifact")
def data_describer():
    """
    A tool for data description. Provides summary statistics for the dataset using pandas describe().

    Returns:
        Summary statistics for the dataset
    """
    import pandas as pd
    # Try different import paths for get_current_config
    try:
        from langchain.callbacks import get_current_config
    except ImportError:
        try:
            from langchain_core.callbacks import get_current_config
        except ImportError:
            # Fallback if import fails
            def get_current_config():
                return {"configurable": {"raw_data": None}}

    # Get the data from the configurable
    try:
        config = get_current_config()
        raw_data = config.get("configurable", {}).get("raw_data", None)
    except Exception:
        # Fallback if get_current_config fails
        raw_data = None

    if raw_data is None:
        return "I need data to describe. Please provide the data.", {}

    df = pd.DataFrame(raw_data)
    description_df = df.describe(include="all")
    content = "Summary statistics computed using pandas describe()."
    artifact = {"describe_df": description_df.to_dict()}
    return content, artifact


@tool(response_format="content_and_artifact")
def data_missing_visualizer(
        n_sample=None,
):
    """
    A tool for missing data visualization. Creates visualizations to analyze missing data patterns.

    Args:
        n_sample: Optional number of rows to sample for visualization (default: use all data)

    Returns:
        Visualizations of missing data patterns
    """
    import missingno as msno
    import pandas as pd
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    # Try different import paths for get_current_config
    try:
        from langchain.callbacks import get_current_config
    except ImportError:
        try:
            from langchain_core.callbacks import get_current_config
        except ImportError:
            # Fallback if import fails
            def get_current_config():
                return {"configurable": {"raw_data": None}}

    # Get the data from the configurable
    try:
        config = get_current_config()
        raw_data = config.get("configurable", {}).get("raw_data", None)
    except Exception:
        # Fallback if get_current_config fails
        raw_data = None

    if raw_data is None:
        return "I need data to visualize missing values. Please provide the data.", {}

    df = pd.DataFrame(raw_data)
    if n_sample is not None:
        df = df.sample(min(int(n_sample), len(df)), random_state=42)

    incoded_plot_images = {}

    # create plot, save plot, encode plot
    def create_and_encode_plot(plot_func, plot_name):
        plt.figure(figsize=(10, 6))
        plot_func(df)
        plt.tight_layout()  # Added parentheses to actually call the function
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()  # Added parentheses to actually call the function
        buffer.seek(0)
        plot_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        incoded_plot_images[plot_name] = plot_image

    try:
        create_and_encode_plot(msno.matrix, "matrix_plot")
        create_and_encode_plot(msno.bar, "bar_plot")
        create_and_encode_plot(msno.heatmap, "heatmap_plot")

        content = "Missing data visualizations (matrix, bar, and heatmap) have been generated."
        return content, incoded_plot_images
    except Exception as e:
        return f"Error generating missing data visualizations: {str(e)}", {}


@tool(response_format="content_and_artifact")
def correlation_funnel_generator(
    target=None,
    corr_method="pearson",
    n_bins=4,
    thresh_infreq=0.01,
    name_infreq="-OTHER"
):
    """
    A tool for correlation funnel generation. Creates a correlation funnel visualization.

    Args:
        target: The target column to analyze correlations against (required)
        corr_method: Correlation method to use (default: "pearson")
        n_bins: Number of bins for discretization (default: 4)
        thresh_infreq: Threshold for infrequent values (default: 0.01)
        name_infreq: Suffix for infrequent values (default: "-OTHER")

    Returns:
        Correlation funnel visualization and data
    """
    import pytimetk as tk
    import pandas as pd
    import json
    import plotly.io as pio
    # Try different import paths for get_current_config
    try:
        from langchain.callbacks import get_current_config
    except ImportError:
        try:
            from langchain_core.callbacks import get_current_config
        except ImportError:
            # Fallback if import fails
            def get_current_config():
                return {"configurable": {"raw_data": None}}

    # Get the data from the configurable
    try:
        config = get_current_config()
        raw_data = config.get("configurable", {}).get("raw_data", None)
    except Exception:
        # Fallback if get_current_config fails
        raw_data = None

    if raw_data is None:
        return "I need data to generate a correlation funnel. Please provide the data.", {}

    if target is None:
        return "Please specify a target column for the correlation funnel analysis.", {}

    try:
        df = pd.DataFrame(raw_data)

        # Check if target exists in the dataframe
        if target not in df.columns:
            return f"Target column '{target}' not found in the dataset. Available columns: {', '.join(df.columns.tolist())}", {}

        # Create correlation funnel
        # This is a placeholder - implement the actual correlation funnel logic
        return "Correlation funnel analysis is not fully implemented yet.", {}
    except Exception as e:
        return f"Error generating correlation funnel: {str(e)}", {}


@tool(response_format="content_and_artifact")
def sweetviz_report_generator(
        target=None,
        report_name="sweetviz_report.html",
        open_browser=False,
):
    """
    A tool for generating a comprehensive Sweetviz EDA report.

    Args:
        target: Optional target column for supervised analysis
        report_name: Name of the report file (default: "sweetviz_report.html")
        open_browser: Whether to open the report in a browser (default: False)

    Returns:
        Sweetviz report
    """
    import sweetviz as sv
    import pandas as pd
    # Try different import paths for get_current_config
    try:
        from langchain.callbacks import get_current_config
    except ImportError:
        try:
            from langchain_core.callbacks import get_current_config
        except ImportError:
            # Fallback if import fails
            def get_current_config():
                return {"configurable": {"raw_data": None}}

    # Get the data from the configurable
    try:
        config = get_current_config()
        raw_data = config.get("configurable", {}).get("raw_data", None)
    except Exception:
        # Fallback if get_current_config fails
        raw_data = None

    if raw_data is None:
        return "I need data to generate a Sweetviz report. Please provide the data.", {}

    try:
        df = pd.DataFrame(raw_data)

        # Check if target exists in the dataframe if specified
        if target is not None and target not in df.columns:
            return f"Target column '{target}' not found in the dataset. Available columns: {', '.join(df.columns.tolist())}", {}

        # Create temporary directory for the report
        report_directory = tempfile.mkdtemp()
        print(f"    * Using temporary directory: {report_directory}")

        # Generate the report
        report = sv.analyze(df, target_column=target)

        full_report_path = os.path.join(report_directory, report_name)
        report.show_html(full_report_path, open_browser=open_browser)

        # Read the HTML content
        try:
            with open(full_report_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        except Exception as e:
            html_content = None
            print(f"Error reading HTML content: {e}")

        content = (
            f"Sweetviz EDA report generated and saved as '{os.path.abspath(full_report_path)}'. "
            f"This was saved in a temporary directory."
        )
        artifact = {
            "report_file": os.path.abspath(full_report_path),
            "report_html": html_content,
        }
        return content, artifact
    except Exception as e:
        return f"Error generating Sweetviz report: {str(e)}", {}


@tool(response_format="content_and_artifact")
def dtale_report_generator(
    host="localhost",
    port=40000,
    open_browser=False,
):
    """
    A tool for generating an interactive D-Tale report.

    Args:
        host: Host for the D-Tale server (default: "localhost")
        port: Port for the D-Tale server (default: 40000)
        open_browser: Whether to open the report in a browser (default: False)

    Returns:
        D-Tale interactive report URL
    """
    import dtale
    import pandas as pd
    # Try different import paths for get_current_config
    try:
        from langchain.callbacks import get_current_config
    except ImportError:
        try:
            from langchain_core.callbacks import get_current_config
        except ImportError:
            # Fallback if import fails
            def get_current_config():
                return {"configurable": {"raw_data": None}}

    # Get the data from the configurable
    try:
        config = get_current_config()
        raw_data = config.get("configurable", {}).get("raw_data", None)
    except Exception:
        # Fallback if get_current_config fails
        raw_data = None

    if raw_data is None:
        return "I need data to generate a D-Tale report. Please provide the data.", {}

    try:
        df = pd.DataFrame(raw_data)
        dtale_report = dtale.show(df, host=host, port=port, open_browser=open_browser)

        dtale_url = dtale_report.main_url()
        content = f"D-Tale report generated and available at {dtale_url}"

        artifact = {"dtale_url": dtale_url}
        return content, artifact
    except Exception as e:
        return f"Error generating D-Tale report: {str(e)}", {}