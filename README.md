# Dimemsionality reduction

This repository contains a Python class for performing dimensionality reduction experiments, particularly in the context of machine learning. The class provides a comprehensive set of methods for applying various dimensionality reduction techniques and evaluating the performance of machine learning models.

## DimensionalityReduction Overview

### Overview
The `DimensionalityReduction` class is designed to facilitate dimensionality reduction experiments, particularly in the context of machine learning. It provides a comprehensive set of methods for applying various dimensionality reduction techniques and evaluating the performance of machine learning models.

### Key Features
- **Supported Dimensionality Reduction Techniques:**
  - Principal Component Analysis (PCA)
  - Linear Discriminant Analysis (LDA)
  - Kernel Principal Component Analysis (KernelPCA)
  - Isomap
  - Locally Linear Embedding (LLE)
  - Spectral Embedding
  - t-Distributed Stochastic Neighbor Embedding (t-SNE)

- **Data Splitting and Scaling:**
  - The class supports splitting data into training and testing sets.
  - Optional feature scaling using `StandardScaler`.

- **Model Training and Evaluation:**
  - Train and evaluate supervised and unsupervised machine learning models.
  - Evaluation metrics include accuracy for supervised models and adjusted Rand score for unsupervised models.

- **Multiple Model Training:**
  - Train multiple models with or without dimensionality reduction.
  - Evaluate the performance of each model over multiple runs.

### Usage Example

```python
# Sample usage of the DimensionalityReduction class

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# ... (import other models as needed)

# Sample data
X_data = ...  # Your feature data
y_data = ...  # Your target variable

# Define models to be trained
models_to_train = {
    'RandomForest': (RandomForestClassifier(), 'supervised'),
    'SVM': (SVC(), 'supervised'),
    # ... add more models as needed
}

# Create DimensionalityReduction object
dim_reduction = DimensionalityReduction(X_data, y_data)

# Run multiple training iterations
results_df = dim_reduction.run_multiple_training(
    models=models_to_train,
    use_reduction=True,
    reduction_method='PCA',
    n_runs=5
)

# Display results
print(results_df)
```

### Dependencies
- `scikit-learn` for machine learning and dimensionality reduction techniques.
- `pandas` for handling data in tabular form.
- `numpy` for numerical operations.
- `tqdm` for displaying progress bars during training iterations.

### Notes
- The class provides flexibility in choosing dimensionality reduction methods and machine learning models.
- Experiment with different models and reduction methods to find the most suitable combination for your specific dataset and task.