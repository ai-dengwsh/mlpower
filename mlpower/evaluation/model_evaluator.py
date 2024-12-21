import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, List, Dict, Any
import os

class ModelEvaluator:
    """
    A comprehensive model evaluation class that provides various
    metrics and visualizations for model performance analysis.
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        average: str = "weighted"
    ):
        """
        Initialize ModelEvaluator.
        
        Parameters
        ----------
        metrics : list of str, optional
            List of metrics to compute. If None, uses all available metrics.
        average : str, default="weighted"
            Averaging strategy for multiclass metrics
        """
        self.metrics = metrics or [
            "accuracy", "precision", "recall", "f1", "roc_auc"
        ]
        self.average = average
        
    def evaluate(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        X_train: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_train: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model using various metrics.
        
        Parameters
        ----------
        model : object
            Fitted model with predict and predict_proba methods
        X : array-like of shape (n_samples, n_features)
            Test data
        y : array-like of shape (n_samples,)
            True labels for X
        X_train : array-like of shape (n_samples, n_features), optional
            Training data for learning curve analysis
        y_train : array-like of shape (n_samples,), optional
            Training labels for learning curve analysis
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing evaluation metrics
        """
        X = self._validate_input(X)
        y = np.array(y)
        
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        results = {}
        
        # Calculate metrics
        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy_score(y, y_pred)
            
        if "precision" in self.metrics:
            results["precision"] = precision_score(
                y, y_pred, average=self.average
            )
            
        if "recall" in self.metrics:
            results["recall"] = recall_score(
                y, y_pred, average=self.average
            )
            
        if "f1" in self.metrics:
            results["f1"] = f1_score(
                y, y_pred, average=self.average
            )
            
        if "roc_auc" in self.metrics:
            if y_pred_proba.shape[1] == 2:  # Binary classification
                results["roc_auc"] = roc_auc_score(
                    y, y_pred_proba[:, 1]
                )
            else:  # Multiclass
                results["roc_auc"] = roc_auc_score(
                    y, y_pred_proba, multi_class="ovr"
                )
                
        return results
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels
        y_pred : array-like of shape (n_samples,)
            Predicted labels
        labels : list of str, optional
            List of label names
        save_path : str, optional
            Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close()
        
    def plot_learning_curve(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        cv: int = 5,
        n_jobs: int = -1,
        save_path: Optional[str] = None
    ):
        """
        Plot learning curve.
        
        Parameters
        ----------
        model : object
            Unfitted model
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Training labels
        cv : int, default=5
            Number of cross-validation folds
        n_jobs : int, default=-1
            Number of jobs to run in parallel
        save_path : str, optional
            Path to save the plot
        """
        X = self._validate_input(X)
        y = np.array(y)
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            train_sizes, train_mean,
            label="Training score",
            color="blue"
        )
        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color="blue"
        )
        plt.plot(
            train_sizes, test_mean,
            label="Cross-validation score",
            color="red"
        )
        plt.fill_between(
            train_sizes,
            test_mean - test_std,
            test_mean + test_std,
            alpha=0.1,
            color="red"
        )
        
        plt.title("Learning Curve")
        plt.xlabel("Training Examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close()
        
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> str:
        """
        Get detailed classification report.
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels
        y_pred : array-like of shape (n_samples,)
            Predicted labels
        labels : list of str, optional
            List of label names
            
        Returns
        -------
        str
            Classification report as string
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=labels
        )
        
    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input to pandas DataFrame and validate."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame or numpy array")
        return X 