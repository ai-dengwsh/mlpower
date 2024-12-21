import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from typing import Union, Optional, List, Dict, Any
import joblib
import os

class AutoML(BaseEstimator, ClassifierMixin):
    """
    Automated Machine Learning class that automatically selects
    and optimizes the best model for a given dataset.
    """
    
    def __init__(
        self,
        models: Optional[List[str]] = None,
        cv_folds: int = 5,
        scoring: str = "accuracy",
        random_state: int = 42
    ):
        """
        Initialize AutoML.
        
        Parameters
        ----------
        models : list of str, optional
            List of models to try. If None, uses all available models.
        cv_folds : int, default=5
            Number of cross-validation folds
        scoring : str, default="accuracy"
            Scoring metric for model selection
        random_state : int, default=42
            Random state for reproducibility
        """
        self.models = models or ["rf", "gb", "svm", "lr", "mlp"]
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self._fitted = False
        self._best_model = None
        self._best_params = None
        self._model_scores = {}
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray):
        """
        Fit the AutoML model by trying different models and selecting the best one.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self
        """
        X = self._validate_input(X)
        y = np.array(y)
        
        best_score = float("-inf")
        
        for model_name in self.models:
            # Get model and default parameters
            model, params = self._get_model_config(model_name)
            
            # Evaluate model with cross-validation
            scores = cross_val_score(
                model, X, y,
                cv=self.cv_folds,
                scoring=self.scoring
            )
            
            mean_score = np.mean(scores)
            self._model_scores[model_name] = {
                "mean_score": mean_score,
                "std_score": np.std(scores)
            }
            
            # Update best model if current one is better
            if mean_score > best_score:
                best_score = mean_score
                self._best_model = model
                self._best_params = params
                
        # Fit the best model on all data
        self._best_model.fit(X, y)
        self._fitted = True
        return self
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict using the best fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred : np.ndarray
            Predicted values
        """
        if not self._fitted:
            raise ValueError("AutoML must be fitted before calling predict")
            
        X = self._validate_input(X)
        return self._best_model.predict(X)
        
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities using the best fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred_proba : np.ndarray
            Predicted class probabilities
        """
        if not self._fitted:
            raise ValueError("AutoML must be fitted before calling predict_proba")
            
        X = self._validate_input(X)
        return self._best_model.predict_proba(X)
        
    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input to pandas DataFrame and validate."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame or numpy array")
        return X
        
    def _get_model_config(self, model_name: str) -> tuple:
        """Get model instance and default parameters for a given model name."""
        configs = {
            "rf": (
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state
                ),
                {"n_estimators": 100}
            ),
            "gb": (
                GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=self.random_state
                ),
                {"n_estimators": 100}
            ),
            "svm": (
                SVC(
                    probability=True,
                    random_state=self.random_state
                ),
                {"kernel": "rbf"}
            ),
            "lr": (
                LogisticRegression(
                    random_state=self.random_state
                ),
                {"C": 1.0}
            ),
            "mlp": (
                MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    random_state=self.random_state
                ),
                {"hidden_layer_sizes": (100, 50)}
            )
        }
        
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")
            
        return configs[model_name]
        
    def get_model_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Get the cross-validation scores for all tried models.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping model names to their scores
        """
        if not self._fitted:
            raise ValueError("AutoML must be fitted before getting model scores")
        return self._model_scores
        
    def get_best_model(self) -> tuple:
        """
        Get the best model and its parameters.
        
        Returns
        -------
        tuple
            (best_model, best_parameters)
        """
        if not self._fitted:
            raise ValueError("AutoML must be fitted before getting best model")
        return self._best_model, self._best_params
        
    def save_model(self, path: str):
        """
        Save the fitted model to disk.
        
        Parameters
        ----------
        path : str
            Path to save the model
        """
        if not self._fitted:
            raise ValueError("AutoML must be fitted before saving")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        
    @classmethod
    def load_model(cls, path: str) -> "AutoML":
        """
        Load a fitted model from disk.
        
        Parameters
        ----------
        path : str
            Path to the saved model
            
        Returns
        -------
        AutoML
            Loaded model
        """
        return joblib.load(path) 