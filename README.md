# MLPower

MLPower is a powerful and user-friendly machine learning library that provides advanced features for data preprocessing, feature engineering, model training, and evaluation.

## Features

- **Preprocessing**: Advanced data cleaning and preprocessing tools
- **Feature Engineering**: Automated feature selection and engineering
- **Models**: Implementation of various machine learning algorithms
- **Evaluation**: Comprehensive model evaluation and comparison tools
- **Utils**: Utility functions for common machine learning tasks

## Installation

```bash
pip install mlpower
```

## Quick Start

```python
from mlpower.preprocessing import DataCleaner
from mlpower.feature_engineering import FeatureSelector
from mlpower.models import AutoML
from mlpower.evaluation import ModelEvaluator

# Load and preprocess data
cleaner = DataCleaner()
X_clean = cleaner.fit_transform(X)

# Feature engineering
selector = FeatureSelector()
X_selected = selector.fit_transform(X_clean, y)

# Train model
model = AutoML()
model.fit(X_selected, y)

# Evaluate
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, X_selected, y)
```

## Version

v1.0.0

## Documentation

For detailed documentation, please visit [https://findshan1.github.io/mlpower/](https://findshan1.github.io/mlpower/)

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

