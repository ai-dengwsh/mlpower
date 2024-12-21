"""
MLPower - A powerful machine learning library
"""

__version__ = "0.1.0"

from mlpower.preprocessing import DataCleaner
from mlpower.feature_engineering import FeatureSelector
from mlpower.models import AutoML
from mlpower.evaluation import ModelEvaluator

__all__ = [
    "DataCleaner",
    "FeatureSelector",
    "AutoML",
    "ModelEvaluator",
] 