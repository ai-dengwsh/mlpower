import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from mlpower.preprocessing import DataCleaner
from mlpower.feature_engineering import FeatureSelector
from mlpower.models import AutoML
from mlpower.evaluation import ModelEvaluator
from mlpower.utils import plot_feature_importance, plot_correlation_matrix

# Generate sample data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

# Convert to DataFrame
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)

# Add some missing values and outliers for demonstration
X.iloc[10:20, 0] = np.nan
X.iloc[0, 0] = 100

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Data cleaning
cleaner = DataCleaner(
    missing_strategy="mean",
    outlier_method="iqr",
    outlier_threshold=1.5
)
X_train_clean = cleaner.fit_transform(X_train)
X_test_clean = cleaner.transform(X_test)

# Feature selection
selector = FeatureSelector(n_features=10)
X_train_selected = selector.fit_transform(X_train_clean, y_train)
X_test_selected = selector.transform(X_test_clean)

# Plot feature importance
feature_importance = selector.get_feature_importance()
plot_feature_importance(
    feature_importance,
    top_n=10,
    save_path="feature_importance.png"
)

# Plot correlation matrix
plot_correlation_matrix(
    X_train_selected,
    save_path="correlation_matrix.png"
)

# Train model
model = AutoML(
    models=["rf", "gb", "lr"],
    cv_folds=5,
    scoring="accuracy"
)
model.fit(X_train_selected, y_train)

# Evaluate model
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, X_test_selected, y_test)
print("\nModel Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Plot confusion matrix
y_pred = model.predict(X_test_selected)
evaluator.plot_confusion_matrix(
    y_test,
    y_pred,
    save_path="confusion_matrix.png"
)

# Plot learning curve
evaluator.plot_learning_curve(
    model._best_model,
    X_train_selected,
    y_train,
    save_path="learning_curve.png"
)

# Get classification report
report = evaluator.get_classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# Save model
model.save_model("trained_model.joblib") 