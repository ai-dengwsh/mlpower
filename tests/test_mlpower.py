import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from mlpower.preprocessing import DataCleaner
from mlpower.feature_engineering import FeatureSelector
from mlpower.models import AutoML
from mlpower.evaluation import ModelEvaluator

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    return X, y

def test_data_cleaner(sample_data):
    """Test DataCleaner functionality."""
    X, _ = sample_data
    
    # Add missing values and outliers
    X.iloc[0:5, 0] = np.nan
    X.iloc[0, 1] = 100
    
    cleaner = DataCleaner()
    X_clean = cleaner.fit_transform(X)
    
    # Check if missing values are handled
    assert not X_clean.isnull().any().any()
    
    # Check if outliers are handled
    assert X_clean.iloc[0, 1] != 100

def test_feature_selector(sample_data):
    """Test FeatureSelector functionality."""
    X, y = sample_data
    n_features = 5
    
    selector = FeatureSelector(n_features=n_features)
    X_selected = selector.fit_transform(X, y)
    
    # Check if correct number of features are selected
    assert X_selected.shape[1] == n_features
    
    # Check if feature importance scores are available
    importance = selector.get_feature_importance()
    assert len(importance) == X.shape[1]

def test_auto_ml(sample_data):
    """Test AutoML functionality."""
    X, y = sample_data
    
    model = AutoML(models=["rf"])
    model.fit(X, y)
    
    # Check predictions
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    
    # Check probabilities
    y_pred_proba = model.predict_proba(X)
    assert y_pred_proba.shape == (len(y), 2)
    
    # Check model scores
    scores = model.get_model_scores()
    assert "rf" in scores

def test_model_evaluator(sample_data):
    """Test ModelEvaluator functionality."""
    X, y = sample_data
    
    # Train a simple model
    model = AutoML(models=["rf"])
    model.fit(X, y)
    
    evaluator = ModelEvaluator()
    
    # Test evaluation metrics
    metrics = evaluator.evaluate(model, X, y)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    
    # Test classification report
    y_pred = model.predict(X)
    report = evaluator.get_classification_report(y, y_pred)
    assert isinstance(report, str)

def test_end_to_end(sample_data):
    """Test complete workflow."""
    X, y = sample_data
    
    # Add some missing values and outliers
    X.iloc[0:5, 0] = np.nan
    X.iloc[0, 1] = 100
    
    # Data cleaning
    cleaner = DataCleaner()
    X_clean = cleaner.fit_transform(X)
    
    # Feature selection
    selector = FeatureSelector(n_features=5)
    X_selected = selector.fit_transform(X_clean, y)
    
    # Model training
    model = AutoML(models=["rf"])
    model.fit(X_selected, y)
    
    # Model evaluation
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_selected, y)
    
    # Check if workflow completes successfully
    assert len(metrics) > 0
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0 