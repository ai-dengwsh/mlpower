import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from typing import Union, Optional, List, Dict

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    A comprehensive feature selection class that combines multiple
    feature selection methods to identify the most important features.
    """
    
    def __init__(
        self,
        n_features: Optional[int] = None,
        method: str = "combined",
        threshold: float = 0.05,
        scoring_methods: Optional[List[str]] = None
    ):
        """
        Initialize the FeatureSelector.
        
        Parameters
        ----------
        n_features : int, optional
            Number of features to select. If None, uses threshold.
        method : str, default="combined"
            Feature selection method: "combined", "statistical", "mutual_info", or "importance"
        threshold : float, default=0.05
            Threshold for feature importance when n_features is None
        scoring_methods : list of str, optional
            List of scoring methods to use for combined selection
        """
        self.n_features = n_features
        self.method = method
        self.threshold = threshold
        self.scoring_methods = scoring_methods or ["statistical", "mutual_info", "importance"]
        self._fitted = False
        self._feature_scores = {}
        self._selected_features = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y):
        """
        Fit the feature selector by computing feature importance scores.
        
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
        
        if self.method == "combined":
            self._compute_combined_scores(X, y)
        elif self.method == "statistical":
            self._compute_statistical_scores(X, y)
        elif self.method == "mutual_info":
            self._compute_mutual_info_scores(X, y)
        elif self.method == "importance":
            self._compute_importance_scores(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        self._select_features()
        self._fitted = True
        return self
        
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Transform the data by selecting the most important features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_selected : pd.DataFrame
            Data with selected features
        """
        if not self._fitted:
            raise ValueError("FeatureSelector must be fitted before calling transform")
            
        X = self._validate_input(X)
        return X[self._selected_features]
        
    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input to pandas DataFrame and validate."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame or numpy array")
        return X
        
    def _compute_statistical_scores(self, X: pd.DataFrame, y: np.ndarray):
        """Compute statistical (ANOVA F-value) scores for features."""
        selector = SelectKBest(score_func=f_classif)
        selector.fit(X, y)
        self._feature_scores["statistical"] = dict(zip(X.columns, selector.scores_))
        
    def _compute_mutual_info_scores(self, X: pd.DataFrame, y: np.ndarray):
        """Compute mutual information scores for features."""
        selector = SelectKBest(score_func=mutual_info_classif)
        selector.fit(X, y)
        self._feature_scores["mutual_info"] = dict(zip(X.columns, selector.scores_))
        
    def _compute_importance_scores(self, X: pd.DataFrame, y: np.ndarray):
        """Compute feature importance scores using Random Forest."""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        self._feature_scores["importance"] = dict(zip(X.columns, rf.feature_importances_))
        
    def _compute_combined_scores(self, X: pd.DataFrame, y: np.ndarray):
        """Compute combined scores using multiple methods."""
        for method in self.scoring_methods:
            if method == "statistical":
                self._compute_statistical_scores(X, y)
            elif method == "mutual_info":
                self._compute_mutual_info_scores(X, y)
            elif method == "importance":
                self._compute_importance_scores(X, y)
                
        # Normalize and combine scores
        combined_scores = {}
        for feature in X.columns:
            scores = []
            for method in self.scoring_methods:
                score = self._feature_scores[method][feature]
                normalized_score = score / np.sum(list(self._feature_scores[method].values()))
                scores.append(normalized_score)
            combined_scores[feature] = np.mean(scores)
            
        self._feature_scores["combined"] = combined_scores
        
    def _select_features(self):
        """Select features based on scores and criteria."""
        scores = self._feature_scores[self.method]
        
        if self.n_features is not None:
            # Select top n features
            sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            self._selected_features = [f[0] for f in sorted_features[:self.n_features]]
        else:
            # Select features above threshold
            max_score = max(scores.values())
            threshold_score = max_score * self.threshold
            self._selected_features = [f for f, s in scores.items() if s >= threshold_score]
            
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance scores for all features.
        
        Returns
        -------
        Dict[str, float]
            Dictionary mapping feature names to their importance scores
        """
        if not self._fitted:
            raise ValueError("FeatureSelector must be fitted before getting feature importance")
        return self._feature_scores[self.method] 