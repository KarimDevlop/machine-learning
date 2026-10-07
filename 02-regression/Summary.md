## Car Price Prediction — Summary

### Topic notebooks
The notebooks are numbered in lesson order. Each has a setup cell that imports
the shared dataset, train/validation/test splits, and helper functions from
[common.py](./common.py). Notebook kernels remain separate, so run the setup
cell in each notebook; variables created later in one notebook do not
automatically appear in another.

1. [Data preparation and exploration](./01_data_preparation_and_exploration.ipynb)
2. [Validation framework](./02_validation_framework.ipynb)
3. [Linear regression](./03_linear_regression.ipynb)
4. [Baseline model and RMSE](./04_baseline_model_and_rmse.ipynb)
5. [Validation and feature engineering](./05_validation_and_feature_engineering.ipynb)
6. [Regularization and model tuning](./06_regularization_and_tuning.ipynb)
7. [Using the model](./07_using_the_model.ipynb)

### Goal
- Predict car prices: **MSRP**.
- Use numerical and categorical car features.

### Data Preparation and EDA
- Standardize column names and strings:
  - Lowercase.
  - Replace spaces with underscores.
- Explore distributions and missing values.
- Price has a **long tail** → apply `np.log1p()`.
- Fill missing numerical values with `0`.

### Validation Framework
- Split data into **train**, **validation**, and **test**.
- Train on training data.
- Compare models using validation data.
- Evaluate the final model on test data.

### Linear Regression
- Predict with: `y_pred = w0 + X.dot(w)`.
- Learn the bias and weights using the **normal equation**.
- Implement training with NumPy.
- Start with five numerical features as a **baseline**.
- Use `prepare_X()` for consistent feature preparation.

### Evaluation
- Measure prediction error with **RMSE**.
- Lower RMSE → better predictions.
- Evaluate log-price predictions against log-price targets.

### Feature Engineering
- Create car age: `age = 2017 - year`.
- Encode categories using **one-hot encoding**.
- Adding `age` significantly improves predictions.

### Regularization
- Many categorical features can cause numerical instability.
- Add `r` to the diagonal: `XᵀX + rI`.
- Stabilize weights and improve predictions.

### Final Model
- Compare regularization values on validation data.
- Lesson choice: `r = 0.001`.
- Combine training and validation data.
- Retrain the model.
- Convert predictions to prices with `np.expm1()`.

### Next
- Explore further experiments and complete homework.
- Learn classification using **scikit-learn**.
