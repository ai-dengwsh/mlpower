import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, Optional, List

class DataCleaner(BaseEstimator, TransformerMixin):
    """
    A comprehensive data cleaning class that handles missing values,
    outliers, and data type conversions.
    """
    
    def __init__(
        self,
        missing_strategy: str = "mean",
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        categorical_encoding: str = "label",
    ):
        """
        Initialize the DataCleaner.
        
        Parameters
        ----------
        missing_strategy : str, default="mean"
            Strategy for handling missing values: "mean", "median", "mode", or "drop"
        outlier_method : str, default="iqr"
            Method for detecting outliers: "iqr" or "zscore"
        outlier_threshold : float, default=1.5
            Threshold for outlier detection
        categorical_encoding : str, default="label"
            Method for encoding categorical variables: "label" or "onehot"
        """
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.categorical_encoding = categorical_encoding
        self._fitted = False
        self._column_stats = {}
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fit the data cleaner by computing necessary statistics.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : None
            Ignored
            
        Returns
        -------
        self : object
            Returns self
        """
        X = self._validate_input(X)
        
        for column in X.columns:
            stats = {}
            if pd.api.types.is_numeric_dtype(X[column]):
                stats["dtype"] = "numeric"
                stats["mean"] = X[column].mean()
                stats["median"] = X[column].median()
                stats["q1"] = X[column].quantile(0.25)
                stats["q3"] = X[column].quantile(0.75)
                stats["iqr"] = stats["q3"] - stats["q1"]
                stats["std"] = X[column].std()
            else:
                stats["dtype"] = "categorical"
                stats["mode"] = X[column].mode()[0]
                stats["unique_values"] = X[column].unique()
                
            self._column_stats[column] = stats
            
        self._fitted = True
        return self
        
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Clean the data by applying the fitted transformations.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_cleaned : pd.DataFrame
            Cleaned data
        """
        if not self._fitted:
            raise ValueError("DataCleaner must be fitted before calling transform")
            
        X = self._validate_input(X)
        X_cleaned = X.copy()
        
        # Handle missing values
        X_cleaned = self._handle_missing_values(X_cleaned)
        
        # Handle outliers
        X_cleaned = self._handle_outliers(X_cleaned)
        
        # Encode categorical variables
        X_cleaned = self._encode_categorical(X_cleaned)
        
        return X_cleaned
        
    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input to pandas DataFrame and validate."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame or numpy array")
        return X
        
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to the specified strategy."""
        for column in X.columns:
            stats = self._column_stats[column]
            if stats["dtype"] == "numeric":
                if self.missing_strategy == "mean":
                    X[column].fillna(stats["mean"], inplace=True)
                elif self.missing_strategy == "median":
                    X[column].fillna(stats["median"], inplace=True)
            else:
                X[column].fillna(stats["mode"], inplace=True)
        return X
        
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers according to the specified method."""
        for column in X.columns:
            stats = self._column_stats[column]
            if stats["dtype"] == "numeric":
                if self.outlier_method == "iqr":
                    lower_bound = stats["q1"] - self.outlier_threshold * stats["iqr"]
                    upper_bound = stats["q3"] + self.outlier_threshold * stats["iqr"]
                    X[column] = X[column].clip(lower_bound, upper_bound)
                elif self.outlier_method == "zscore":
                    z_scores = np.abs((X[column] - stats["mean"]) / stats["std"])
                    X.loc[z_scores > self.outlier_threshold, column] = stats["median"]
        return X
        
    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables according to the specified method."""
        for column in X.columns:
            stats = self._column_stats[column]
            if stats["dtype"] == "categorical":
                if self.categorical_encoding == "label":
                    X[column] = pd.Categorical(X[column]).codes
                elif self.categorical_encoding == "onehot":
                    dummies = pd.get_dummies(X[column], prefix=column)
                    X = pd.concat([X.drop(column, axis=1), dummies], axis=1)
        return X 