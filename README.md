# Smart Diab — Diabetes Progression Prediction

Smart Diab is a machine learning project that compares multiple regression models to predict diabetes disease progression scores using the built-in diabetes dataset from scikit-learn.

## Project Overview

The project trains and evaluates six machine learning models and compares their performance using cross-validation and test-set metrics.

## Models Used

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Multi-Layer Perceptron (MLP)

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python diabetes_progression.py
```

## Outputs

The project generates model comparison charts, prediction and residual plots, feature importance visualizations, risk-band comparison results, and a metrics JSON file.

## Dataset

The project uses the diabetes dataset included in scikit-learn.

## Important Disclaimer

This project is for educational purposes only. The predicted progression score and derived risk bands are not a medical diagnosis and are not clinically validated.

## Future Improvements

- Build an interactive web application.
- Add more model evaluation visualizations.
- Improve experiment tracking and reporting.
