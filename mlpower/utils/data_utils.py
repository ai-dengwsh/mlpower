import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
import joblib
import os

def load_data(
    path: str,
    file_type: Optional[str] = None,
    **kwargs
) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
    """
    Load data from various file formats.
    
    Parameters
    ----------
    path : str
        Path to the data file
    file_type : str, optional
        Type of file to load. If None, inferred from file extension.
    **kwargs : dict
        Additional arguments to pass to the loader function
        
    Returns
    -------
    Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]
        Loaded data
    """
    if file_type is None:
        file_type = os.path.splitext(path)[1][1:].lower()
        
    if file_type == "csv":
        return pd.read_csv(path, **kwargs)
    elif file_type == "excel" or file_type in ["xls", "xlsx"]:
        return pd.read_excel(path, **kwargs)
    elif file_type == "json":
        return pd.read_json(path, **kwargs)
    elif file_type == "parquet":
        return pd.read_parquet(path, **kwargs)
    elif file_type == "joblib":
        return joblib.load(path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
        
def save_data(
    data: Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
    path: str,
    file_type: Optional[str] = None,
    **kwargs
):
    """
    Save data to various file formats.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]
        Data to save
    path : str
        Path to save the data
    file_type : str, optional
        Type of file to save. If None, inferred from file extension.
    **kwargs : dict
        Additional arguments to pass to the saver function
    """
    if file_type is None:
        file_type = os.path.splitext(path)[1][1:].lower()
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if isinstance(data, pd.DataFrame):
        if file_type == "csv":
            data.to_csv(path, **kwargs)
        elif file_type == "excel" or file_type in ["xls", "xlsx"]:
            data.to_excel(path, **kwargs)
        elif file_type == "json":
            data.to_json(path, **kwargs)
        elif file_type == "parquet":
            data.to_parquet(path, **kwargs)
        elif file_type == "joblib":
            joblib.dump(data, path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    else:
        if file_type == "joblib":
            joblib.dump(data, path)
        else:
            raise ValueError("Only joblib format is supported for numpy arrays") 