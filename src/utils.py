# src/utils.py

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Name of the logger.
        level: Logging level.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if the logger already has handlers
    if not logger.handlers:
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create a formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(console_handler)
    
    return logger

# Set up the default logger
logger = setup_logger(__name__)

def save_chart(plt_obj: plt.Figure, filename: str) -> str:
    """
    Save a matplotlib chart to the docs/images directory.
    
    Args:
        plt_obj: Matplotlib figure or pyplot instance.
        filename: Name of the file to save.
        
    Returns:
        Path to the saved chart.
    """
    images_dir = Path("docs/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = images_dir / filename
    plt_obj.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Chart saved to {save_path}")
    
    return str(save_path)

def read_json(file_path: str) -> Dict[str, Any]:
    """
    Read a JSON file into a dictionary.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Dictionary containing the JSON data.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return {}

def write_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """
    Write a dictionary to a JSON file.
    
    Args:
        data: Dictionary to write.
        file_path: Path to the JSON file.
        indent: Number of spaces for indentation.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.info(f"Data written to {file_path}")
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}")
        raise

def read_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file.
        **kwargs: Additional arguments to pass to pandas.read_csv.
        
    Returns:
        DataFrame containing the CSV data.
    """
    try:
        df = pd.read_csv(file_path, **kwargs)
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        return pd.DataFrame()

def write_csv(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Write a pandas DataFrame to a CSV file.
    
    Args:
        df: DataFrame to write.
        file_path: Path to the CSV file.
        **kwargs: Additional arguments to pass to DataFrame.to_csv.
    """
    try:
        df.to_csv(file_path, **kwargs)
        logger.info(f"DataFrame written to {file_path}")
    except Exception as e:
        logger.error(f"Error writing CSV file {file_path}: {e}")
        raise

def merge_predictions(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge new predictions with existing ones.
    Only include new predictions with dates later than the last recorded date.
    
    Args:
        existing: List of existing prediction records.
        new: List of new prediction records.
        
    Returns:
        Merged list of prediction records.
    """
    if existing:
        last_date = max(entry['date'] for entry in existing)
    else:
        last_date = None
    
    merged = existing.copy()
    for entry in new:
        if last_date is None or entry['date'] > last_date:
            merged.append(entry)
    
    return merged

def create_directory(dir_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        dir_path: Path to the directory.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory created (if it didn't exist): {dir_path}")

def get_api_key(key_name: str) -> Optional[str]:
    """
    Get an API key from environment variables.
    
    Args:
        key_name: Name of the environment variable.
        
    Returns:
        API key if available, None otherwise.
    """
    api_key = os.getenv(key_name)
    if not api_key:
        logger.warning(f"Environment variable {key_name} not set.")
    return api_key

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics between true and predicted values.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        Dictionary of metrics.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def timeit(func):
    """
    Decorator to measure the execution time of a function.
    
    Args:
        func: Function to measure.
        
    Returns:
        Wrapped function.
    """
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    
    return wrapper 