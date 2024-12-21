import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, Dict, List
import os

def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: Optional[int] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot feature importance scores.
    
    Parameters
    ----------
    feature_importance : Dict[str, float]
        Dictionary mapping feature names to their importance scores
    top_n : int, optional
        Number of top features to plot. If None, plots all features.
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    if top_n is not None:
        sorted_features = sorted_features[:top_n]
        
    features, scores = zip(*sorted_features)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance Score")
    plt.title("Feature Importance")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    
def plot_correlation_matrix(
    data: Union[pd.DataFrame, np.ndarray],
    features: Optional[List[str]] = None,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot correlation matrix heatmap.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Data to compute correlations from
    features : list of str, optional
        List of feature names. Required if data is numpy array.
    figsize : tuple, default=(10, 8)
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    if isinstance(data, np.ndarray):
        if features is None:
            features = [f"Feature_{i}" for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=features)
        
    corr_matrix = data.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f"
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close() 