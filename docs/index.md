# MLPower Documentation

MLPower is a comprehensive machine learning library that simplifies the machine learning workflow with powerful tools for data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents

1. [Installation](#installation)
2. [Core Components](#core-components)
3. [Quick Start Guide](#quick-start-guide)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Contributing](#contributing)

## Installation

```bash
pip install mlpower
```

## Core Components

### 1. Data Preprocessing (`mlpower.preprocessing`)

The `DataCleaner` class provides comprehensive data cleaning capabilities:

```python
from mlpower.preprocessing import DataCleaner

cleaner = DataCleaner(
    missing_strategy="mean",  # Options: "mean", "median", "mode"
    outlier_method="iqr",     # Options: "iqr", "zscore"
    outlier_threshold=1.5,
    categorical_encoding="label"  # Options: "label", "onehot"
)

X_clean = cleaner.fit_transform(X)
```

### 2. Feature Engineering (`mlpower.feature_engineering`)

The `FeatureSelector` class implements multiple feature selection methods:

```python
from mlpower.feature_engineering import FeatureSelector

selector = FeatureSelector(
    n_features=10,           # Number of features to select
    method="combined",       # Options: "combined", "statistical", "mutual_info", "importance"
    threshold=0.05,
    scoring_methods=["statistical", "mutual_info", "importance"]
)

X_selected = selector.fit_transform(X, y)
feature_importance = selector.get_feature_importance()
```

### 3. Automated Machine Learning (`mlpower.models`)

The `AutoML` class automates model selection and training:

```python
from mlpower.models import AutoML

model = AutoML(
    models=["rf", "gb", "svm", "lr", "mlp"],  # Available models
    cv_folds=5,
    scoring="accuracy"
)

model.fit(X, y)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### 4. Model Evaluation (`mlpower.evaluation`)

The `ModelEvaluator` class provides comprehensive model evaluation tools:

```python
from mlpower.evaluation import ModelEvaluator

evaluator = ModelEvaluator(
    metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
    average="weighted"
)

# Get evaluation metrics
metrics = evaluator.evaluate(model, X_test, y_test)

# Plot confusion matrix
evaluator.plot_confusion_matrix(y_test, y_pred, save_path="confusion_matrix.png")

# Plot learning curve
evaluator.plot_learning_curve(model, X, y, save_path="learning_curve.png")

# Get detailed classification report
report = evaluator.get_classification_report(y_test, y_pred)
```

### 5. Utility Functions (`mlpower.utils`)

Various utility functions for data handling and visualization:

```python
from mlpower.utils import load_data, save_data, plot_feature_importance, plot_correlation_matrix

# Load and save data
data = load_data("data.csv")
save_data(data, "processed_data.csv")

# Visualizations
plot_feature_importance(feature_importance, top_n=10)
plot_correlation_matrix(X, save_path="correlation_matrix.png")
```

## Examples

### Complete Workflow Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from mlpower.preprocessing import DataCleaner
from mlpower.feature_engineering import FeatureSelector
from mlpower.models import AutoML
from mlpower.evaluation import ModelEvaluator

# Load data
X = pd.read_csv("data.csv")
y = X.pop("target")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Data preprocessing
cleaner = DataCleaner()
X_train_clean = cleaner.fit_transform(X_train)
X_test_clean = cleaner.transform(X_test)

# Feature selection
selector = FeatureSelector(n_features=10)
X_train_selected = selector.fit_transform(X_train_clean, y_train)
X_test_selected = selector.transform(X_test_clean)

# Model training
model = AutoML()
model.fit(X_train_selected, y_train)

# Model evaluation
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, X_test_selected, y_test)
print("Model Performance:", metrics)

# Save model
model.save_model("trained_model.joblib")
```

## API Reference

### DataCleaner

```python
class DataCleaner(BaseEstimator, TransformerMixin):
    """
    A comprehensive data cleaning class that handles missing values,
    outliers, and data type conversions.
    
    Parameters
    ----------
    missing_strategy : str, default="mean"
        Strategy for handling missing values: "mean", "median", "mode"
    outlier_method : str, default="iqr"
        Method for detecting outliers: "iqr" or "zscore"
    outlier_threshold : float, default=1.5
        Threshold for outlier detection
    categorical_encoding : str, default="label"
        Method for encoding categorical variables: "label" or "onehot"
    """
```

### FeatureSelector

```python
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    A comprehensive feature selection class that combines multiple
    feature selection methods.
    
    Parameters
    ----------
    n_features : int, optional
        Number of features to select
    method : str, default="combined"
        Feature selection method: "combined", "statistical", "mutual_info", "importance"
    threshold : float, default=0.05
        Threshold for feature importance
    scoring_methods : list of str, optional
        List of scoring methods to use for combined selection
    """
```

### AutoML

```python
class AutoML(BaseEstimator, ClassifierMixin):
    """
    Automated Machine Learning class that automatically selects
    and optimizes the best model.
    
    Parameters
    ----------
    models : list of str, optional
        List of models to try: "rf", "gb", "svm", "lr", "mlp"
    cv_folds : int, default=5
        Number of cross-validation folds
    scoring : str, default="accuracy"
        Scoring metric for model selection
    random_state : int, default=42
        Random state for reproducibility
    """
```

### ModelEvaluator

```python
class ModelEvaluator:
    """
    A comprehensive model evaluation class that provides various
    metrics and visualizations.
    
    Parameters
    ----------
    metrics : list of str, optional
        List of metrics to compute
    average : str, default="weighted"
        Averaging strategy for multiclass metrics
    """
```

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

Please ensure your code follows our coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the LICENSE file for details. 