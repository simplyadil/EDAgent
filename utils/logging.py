
import os

def log_ai_function(response: str, file_name: str, log: bool = True, log_path: str = './logs/', overwrite: bool = True):
    """
    Logs the response of an AI function to a file.
    
    Parameters
    ----------
    response : str
        The response of the AI function.
    file_name : str
        The name of the file to save the response to.
    log : bool, optional
        Whether to log the response or not. The default is True.
    log_path : str, optional
        The path to save the log file. The default is './logs/'.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists. The default is True.
        - If True, the file will be overwritten. 
        - If False, a unique file name will be created.
    
    Returns
    -------
    tuple
        The path and name of the log file.    
    """
    
    if log:
        # Ensure the directory exists
        os.makedirs(log_path, exist_ok=True)

        # file_name = 'data_wrangler.py'
        file_path = os.path.join(log_path, file_name)

        if not overwrite:
            # If file already exists and we're NOT overwriting, we create a new name
            if os.path.exists(file_path):
                # Use an incremental suffix (e.g., data_wrangler_1.py, data_wrangler_2.py, etc.)
                # or a time-based suffix if you prefer.
                base_name, ext = os.path.splitext(file_name)
                i = 1
                while True:
                    new_file_name = f"{base_name}_{i}{ext}"
                    new_file_path = os.path.join(log_path, new_file_name)
                    if not os.path.exists(new_file_path):
                        file_path = new_file_path
                        file_name = new_file_name
                        break
                    i += 1

        # Write the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response)

        print(f"      File saved to: {file_path}")
        
        return (file_path, file_name)
    
    else:
        return (None, None)