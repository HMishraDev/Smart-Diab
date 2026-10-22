"""
Diabetes Progression Predictor with Risk Bands

What this script does:
1. Loads the diabetes dataset that comes with scikit-learn (442 patients).
2. Trains a few regression models to predict a disease progression score.
3. Compares them using cross-validation and a held-out test set.
4. Turns the predicted score into a Low / Medium / High risk band.
5. Shows which patient measurements matter most for the prediction.

Run:  python diabetes_progression.py
All charts and numbers are saved in the "results" folder.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # no display needed, we only save the charts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
OUT_DIR = "results"
BAND_NAMES = ["Low", "Medium", "High"]

# Short descriptions of the columns (from the scikit-learn dataset notes)
FEATURE_INFO = {
    "age": "Age",
    "sex": "Sex",
    "bmi": "Body mass index",
    "bp": "Average blood pressure",
    "s1": "Total cholesterol",
    "s2": "LDL cholesterol",
    "s3": "HDL cholesterol",
    "s4": "Total / HDL ratio",
    "s5": "Log of serum triglycerides",
    "s6": "Blood sugar level",
}


def load_data():
    data = load_diabetes(as_frame=True)
    df = data.frame
    print(f"Dataset: {df.shape[0]} patients, {df.shape[1] - 1} features")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(
        "Target (progression score) -> "
        f"min {df['target'].min():.0f}, max {df['target'].max():.0f}, "
        f"mean {df['target'].mean():.1f}"
    )
    return df


def plot_correlation(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation between features and target")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/01_correlation.png", dpi=150)
    plt.close(fig)
    return corr["target"].drop("target").sort_values(ascending=False)


def get_models():
    # The features in this dataset are already centred and scaled,
    # so we do not need a separate scaling step.
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
        ),
        # Small neural network: 2 hidden layers (16 and 8 neurons).
        # Neural networks train better when the target is scaled too,
        # so the target is standardised and converted back for predictions.
        "Neural Network (MLP)": TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(16, 8),
                alpha=0.01,
                early_stopping=True,
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            transformer=StandardScaler(),
        ),
    }


def compare_models(models, X_train, X_test, y_train, y_test):
    rows = []
    for name, model in models.items():
        # 5-fold cross validation on the training data only
        cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "Model": name,
                "CV R2 (mean)": cv_r2.mean(),
                "CV R2 (std)": cv_r2.std(),
                "Test R2": r2_score(y_test, pred),
                "Test MAE": mean_absolute_error(y_test, pred),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            }
        )
    table = pd.DataFrame(rows).set_index("Model")
    return table


def plot_model_comparison(table):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(table))
    width = 0.38
    ax.bar(x - width / 2, table["CV R2 (mean)"], width, label="Cross-validation R2")
    ax.bar(x + width / 2, table["Test R2"], width, label="Test R2")
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=20, ha="right")
    ax.set_ylabel("R2 score")
    ax.set_title("Model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/02_model_comparison.png", dpi=150)
    plt.close(fig)


def plot_predictions(y_test, pred, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(y_test, pred, alpha=0.7)
    low, high = y_test.min(), y_test.max()
    axes[0].plot([low, high], [low, high], "r--", label="Perfect prediction")
    axes[0].set_xlabel("Actual score")
    axes[0].set_ylabel("Predicted score")
    axes[0].set_title(f"Actual vs predicted ({model_name})")
    axes[0].legend()

    residuals = y_test - pred
    axes[1].scatter(pred, residuals, alpha=0.7)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Predicted score")
    axes[1].set_ylabel("Error (actual - predicted)")
    axes[1].set_title("Residual plot")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/03_predictions_and_residuals.png", dpi=150)
    plt.close(fig)


def feature_importance(model, X_test, y_test):
    # Permutation importance: shuffle one column at a time and see
    # how much the score drops. A big drop means the feature matters.
    result = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="r2"
    )
    imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([f"{c} ({FEATURE_INFO[c]})" for c in imp.index], imp.values)
    ax.set_xlabel("Drop in R2 when the feature is shuffled")
    ax.set_title("Which measurements matter most")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/04_feature_importance.png", dpi=150)
    plt.close(fig)
    return imp.sort_values(ascending=False)


def to_band(values, cutoffs):
    # 0 = Low, 1 = Medium, 2 = High
    return np.digitize(values, cutoffs)


def risk_band_analysis(y_train, y_test, pred):
    # Cut-offs come from the training data only (bottom third, middle, top third)
    cutoffs = np.percentile(y_train, [100 / 3, 200 / 3])
    true_band = to_band(y_test, cutoffs)
    pred_band = to_band(pred, cutoffs)

    band_accuracy = float((true_band == pred_band).mean())
    within_one = float((np.abs(true_band - pred_band) <= 1).mean())
    high_recall = float(recall_score(true_band == 2, pred_band == 2))
    cm = confusion_matrix(true_band, pred_band, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(BAND_NAMES)
    ax.set_yticklabels(BAND_NAMES)
    ax.set_xlabel("Predicted band")
    ax.set_ylabel("Actual band")
    ax.set_title("Risk band confusion matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/05_risk_band_confusion.png", dpi=150)
    plt.close(fig)

    return cutoffs, {
        "band_accuracy": band_accuracy,
        "within_one_band": within_one,
        "high_risk_recall": high_recall,
        "confusion_matrix": cm.tolist(),
    }


def predict_patients(model, X_test, y_test, cutoffs, count=3):
    print("\nSample predictions from the test set")
    print("-" * 60)
    samples = X_test.iloc[:count]
    predicted = model.predict(samples)
    for i, (idx, row_pred) in enumerate(zip(samples.index, predicted)):
        band = BAND_NAMES[int(to_band(row_pred, cutoffs))]
        actual_band = BAND_NAMES[int(to_band(y_test.loc[idx], cutoffs))]
        print(
            f"Patient {i + 1}: predicted {row_pred:6.1f} ({band} risk) | "
            f"actual {y_test.loc[idx]:6.1f} ({actual_band} risk)"
        )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data()
    target_corr = plot_correlation(df)
    print("\nFeature correlation with target:")
    print(target_corr.round(2).to_string())

    X = df.drop(columns="target")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    models = get_models()
    table = compare_models(models, X_train, X_test, y_train, y_test)
    print("\nModel comparison")
    print(table.round(3).to_string())
    plot_model_comparison(table)

    best_name = table["CV R2 (mean)"].idxmax()
    best_model = models[best_name]
    pred = best_model.predict(X_test)
    print(f"\nBest model (by cross-validation): {best_name}")
    plot_predictions(y_test, pred, best_name)

    importance = feature_importance(best_model, X_test, y_test)
    print("\nFeature importance (permutation):")
    print(importance.round(3).to_string())

    cutoffs, band_results = risk_band_analysis(y_train, y_test, pred)
    print(f"\nRisk band cut-offs: Low < {cutoffs[0]:.0f} <= Medium < {cutoffs[1]:.0f} <= High")
    print(f"Band accuracy:            {band_results['band_accuracy']:.2%}")
    print(f"Within one band:          {band_results['within_one_band']:.2%}")
    print(f"High-risk recall:         {band_results['high_risk_recall']:.2%}")

    predict_patients(best_model, X_test, y_test, cutoffs)

    summary = {
        "rows": int(df.shape[0]),
        "features": int(df.shape[1] - 1),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "best_model": best_name,
        "model_table": table.round(4).reset_index().to_dict(orient="records"),
        "target_correlation": target_corr.round(3).to_dict(),
        "feature_importance": importance.round(4).to_dict(),
        "risk_cutoffs": [round(float(c), 1) for c in cutoffs],
        "risk_band_results": band_results,
    }
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Charts and metrics.json saved in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
