# Smart Diab — Quick Presentation Guide

## 30-second explanation
“Smart Diab is an educational machine-learning project. It uses the diabetes dataset included with scikit-learn, containing 442 samples and 10 input features. I compared six regression models using five-fold cross-validation and a separate test set. The program creates model-comparison charts, prediction-error plots, feature-importance analysis, and a JSON results file. It predicts a progression score, not a medical diagnosis.”

## Important results from this run
- Data: 442 samples; 10 input features; 0 missing values reported.
- Split: 353 training samples and 89 test samples.
- Selected using cross-validation: Lasso (mean CV R² about 0.456).
- Highest test R² in this run: Gradient Boosting (about 0.482).
- Score-band accuracy: about 60.67%; within one band: 97.75%.

## If asked “What is R²?”
“R-squared measures how well the model explains variation in the target score. A value closer to 1 is generally better, but it should be considered together with error measures and validation results.”

## If asked “Why did you compare models?”
“Different algorithms learn patterns differently. Cross-validation helps compare them more reliably than relying on one training result.”

## If asked “What is feature importance?”
“I used permutation importance. The method shuffles one feature at a time and checks how much the model score drops. A larger drop suggests that the model relies more on that feature for this dataset.”

## If asked “Is this a real diabetes risk detector?”
“No. It predicts a progression score from a small educational dataset. The Low/Medium/High bands are based on score percentiles and are not clinically validated risk categories. It must not be used for diagnosis or treatment.”

## Which images to show
1. `results/02_model_comparison.png` — model performance.
2. `results/04_feature_importance.png` — which columns matter to the selected model.
3. `results/03_predictions_and_residuals.png` — prediction quality and errors.
4. `results/05_risk_band_confusion.png` — band classification summary.
