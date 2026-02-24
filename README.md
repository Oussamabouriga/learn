```

"""
XGBoost Regressor end-to-end pipeline for tabular data (numeric + categorical)
with:
- Train/test split
- OneHotEncoding for categorical columns
- Optional target imbalance handling for regression (sample weights)
- XGBoost training
- Full evaluation metrics (regression)
- Prediction on new rows using original categorical values (pipeline handles encoding)

WHY THIS APPROACH?
------------------
- You want to predict a score (e.g., satisfaction note 0..10) => regression.
- Your data has categorical columns => OneHotEncoder is a safe and common approach.
- Your target is imbalanced (e.g., many 10s) => we use sample weights so rare target ranges matter more.
- Using a sklearn Pipeline + ColumnTransformer means:
  * no leakage
  * easy train/test
  * same preprocessing is applied during prediction

REQUIREMENTS
------------
pip install pandas numpy scikit-learn xgboost matplotlib

USAGE (quick)
-------------
result = train_xgb_regression_pipeline(
    df=df,
    target_col="evaluate_note",
    test_size=0.2,
    random_state=42,
    use_target_balance_weights=True
)

# Predict on new data (list of dicts OR DataFrame)
new_rows = [
    {
        "age": 34,
        "delai_declaration": 120,
        "gender": "Homme",
        "channel": "Phone",
        "city": "Sousse"
    }
]
preds = predict_new(result["pipeline"], new_rows)
print(preds)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,  # sklearn >=1.4 ; fallback handled below if missing
    r2_score,
    median_absolute_error,
    max_error,
    explained_variance_score
)
from sklearn.inspection import permutation_importance

# XGBoost
from xgboost import XGBRegressor

# Optional plotting
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Helper: robust RMSE for sklearn versions without root_mean_squared_error
# -------------------------------------------------------------------
def _rmse(y_true, y_pred) -> float:
    try:
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# -------------------------------------------------------------------
# 1) Split columns (numeric / categorical)
# -------------------------------------------------------------------
def split_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Split dataframe into X, y and detect numeric/categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str
        Target column name (regression target).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    numeric_cols : list[str]
    categorical_cols : list[str]
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    data = df.copy()
    X = data.drop(columns=[target_col])
    y = pd.to_numeric(data[target_col], errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    return X, y, numeric_cols, categorical_cols


# -------------------------------------------------------------------
# 2) (Optional) Stratified split for regression via target bins
# -------------------------------------------------------------------
def make_y_bins_for_stratification(y: pd.Series, n_bins: int = 10) -> pd.Series:
    """
    Create bins from y for a more balanced train/test split in regression.
    Useful when target is concentrated (e.g., many 10s).

    We use qcut when possible (quantile bins), fallback to cut.
    """
    y = pd.Series(y).reset_index(drop=True)

    # If too few unique values, qcut may fail; fallback safely
    try:
        bins = pd.qcut(y, q=min(n_bins, y.nunique()), duplicates="drop")
    except Exception:
        bins = pd.cut(y, bins=min(n_bins, max(2, y.nunique())))
    return bins.astype(str)


# -------------------------------------------------------------------
# 3) Regression imbalance handling (sample weights)
# -------------------------------------------------------------------
def compute_regression_sample_weights(
    y: pd.Series,
    method: str = "inverse_freq_bins",
    n_bins: int = 10,
    clip_min: float = 0.5,
    clip_max: float = 5.0,
    power: float = 1.0
) -> np.ndarray:
    """
    Compute sample weights for imbalanced regression targets.

    Why?
    ----
    If 70% of your target is near 10, the model may learn to predict near 10 too often.
    Weights increase the importance of rare target ranges.

    Methods
    -------
    - inverse_freq_bins (recommended simple method):
      Bin target values, then weight each sample inversely proportional to its bin frequency.

    Parameters
    ----------
    y : pd.Series
    method : str
    n_bins : int
        Number of bins used on y for frequency estimation.
    clip_min, clip_max : float
        Limit weights to avoid instability (very large weights can hurt training).
    power : float
        Weight strength. 1.0 = full inverse frequency, 0.5 = softer.

    Returns
    -------
    weights : np.ndarray
    """
    y = pd.Series(y).reset_index(drop=True)

    if method != "inverse_freq_bins":
        raise ValueError("Currently supported method: 'inverse_freq_bins'")

    # Bin target for frequency estimation
    try:
        y_bins = pd.qcut(y, q=min(n_bins, y.nunique()), duplicates="drop")
    except Exception:
        y_bins = pd.cut(y, bins=min(n_bins, max(2, y.nunique())))

    freq = y_bins.value_counts(dropna=False)
    weights = y_bins.map(lambda b: 1.0 / freq[b] if pd.notna(b) else 1.0).astype(float).values

    # Normalize around 1.0 for stable training
    weights = weights / np.mean(weights)

    # Optional strength control
    weights = np.power(weights, power)

    # Clip for stability
    weights = np.clip(weights, clip_min, clip_max)

    return weights


# -------------------------------------------------------------------
# 4) Preprocessing pipeline (numeric + categorical)
# -------------------------------------------------------------------
def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Build preprocessing:
    - Numeric: median imputation
    - Categorical: most frequent imputation + OneHotEncoder(handle_unknown='ignore')

    handle_unknown='ignore' is critical so prediction works on new categories gracefully.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor


# -------------------------------------------------------------------
# 5) Build XGBoost model + hyperparameter documentation
# -------------------------------------------------------------------
def build_xgb_regressor(
    random_state: int = 42,
    **xgb_params
) -> XGBRegressor:
    """
    Build XGBRegressor with useful defaults.
    You can override any parameter via xgb_params.

    IMPORTANT HYPERPARAMETERS (simple explanations)
    -----------------------------------------------
    Core learning:
    - n_estimators:
        Number of trees (boosting rounds).
        More trees = more learning capacity (but can overfit if too high).
    - learning_rate (eta):
        How much each new tree changes the model.
        Small = slower but often more stable; large = faster but riskier.
    - max_depth:
        Max depth of each tree.
        Small depth = simpler patterns; large depth = more complex patterns / overfitting risk.
    - min_child_weight:
        Minimum "weight" needed in a child node to split.
        Higher = more conservative splits (reduces overfitting).
    - gamma:
        Minimum loss reduction required to make a split.
        Higher = fewer splits (more conservative).

    Sampling (regularization by randomness):
    - subsample:
        Fraction of training rows used per tree.
        <1.0 can reduce overfitting.
    - colsample_bytree:
        Fraction of features used per tree.
        <1.0 can reduce overfitting and improve robustness.
    - colsample_bylevel / colsample_bynode:
        Similar, but applied per tree level/node (advanced tuning).

    Regularization:
    - reg_alpha:
        L1 regularization (promotes simpler model).
    - reg_lambda:
        L2 regularization (stabilizes weights).
    - max_delta_step:
        Usually more useful in classification; can constrain updates.

    Tree construction:
    - tree_method:
        "hist" is usually fast and good.
    - grow_policy:
        "depthwise" (default) or "lossguide" (advanced).
    - max_leaves:
        Useful with lossguide to limit leaves.

    Objectives & metrics:
    - objective:
        For regression common choices:
        * "reg:squarederror" (MSE-style)
        * "reg:absoluteerror" (MAE-style; newer versions)
        * "reg:pseudohubererror" (robust to outliers)
    - eval_metric:
        Metric monitored during training; examples: "rmse", "mae".

    Others:
    - random_state:
        Reproducibility.
    - n_jobs:
        Number of CPU threads.
    - early_stopping_rounds:
        Used only if you pass validation set to fit (manual fit).
    """
    default_params = dict(
        # Objective (regression)
        objective="reg:squarederror",
        eval_metric="rmse",

        # Core boosting
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        gamma=0.0,

        # Sampling
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,

        # Regularization
        reg_alpha=0.0,
        reg_lambda=1.0,

        # Tree / speed
        tree_method="hist",
        grow_policy="depthwise",
        max_leaves=0,  # 0 means unused in depthwise

        # Misc
        n_jobs=-1,
        random_state=random_state,
        verbosity=0,
    )

    default_params.update(xgb_params)
    return XGBRegressor(**default_params)


# -------------------------------------------------------------------
# 6) Evaluation metrics (many metrics) + explanation
# -------------------------------------------------------------------
def regression_metrics_report(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Compute many regression metrics.

    HOW TO READ THEM (important)
    ----------------------------
    Lower is better:
    - MAE  (Mean Absolute Error): average absolute error in target units.
      Example: MAE=0.80 means prediction is off by ~0.8 points on average.
    - MSE  (Mean Squared Error): squares errors, penalizes large errors more.
    - RMSE (Root Mean Squared Error): same unit as target, more sensitive to big mistakes.
    - MedAE (Median Absolute Error): typical error (robust to outliers).
    - Max Error: worst single prediction error.
    - MAPE / sMAPE: percentage errors (can be unstable near zero target).
      (For scores 0..10, if many zeros exist, use with caution.)

    Higher is better:
    - R² (R-squared): proportion of variance explained.
      1.0 = perfect, 0 = similar to predicting the mean, <0 = worse than mean baseline.
    - Explained Variance: similar spirit to R² (higher better).

    Notes:
    - No single metric is enough. For satisfaction score 0..10, MAE + RMSE + R² is a strong combo.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = _rmse(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    mxerr = max_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    # MAPE / sMAPE (safe implementations)
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0
    smape = np.mean(
        2.0 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    ) * 100.0

    metrics = pd.DataFrame({
        "metric": [
            "MAE",
            "MSE",
            "RMSE",
            "MedianAE",
            "MaxError",
            "R2",
            "ExplainedVariance",
            "MAPE_%",
            "sMAPE_%"
        ],
        "value": [
            mae, mse, rmse, medae, mxerr, r2, evs, mape, smape
        ],
        "better_if": [
            "lower",
            "lower",
            "lower",
            "lower",
            "lower",
            "higher",
            "higher",
            "lower",
            "lower"
        ],
        "what_it_means": [
            "Average absolute error (same unit as target)",
            "Squared error (penalizes big errors strongly)",
            "Error in target unit, sensitive to big errors",
            "Median absolute error (robust typical error)",
            "Worst single prediction error",
            "Variance explained (1 best, 0 mean baseline)",
            "How much variance is captured",
            "Average percentage error (careful near zero)",
            "Symmetric percentage error"
        ]
    })

    return metrics


# -------------------------------------------------------------------
# 7) Optional: custom cross-validation for regression (with sample weights)
# -------------------------------------------------------------------
def cross_validate_xgb_regression(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    random_state: int = 42,
    xgb_params: Optional[Dict[str, Any]] = None,
    use_target_balance_weights: bool = True
) -> pd.DataFrame:
    """
    Simple KFold cross-validation (manual) so we can pass sample weights.

    Why manual CV?
    --------------
    sklearn cross_val_score doesn't easily pass fold-specific sample weights through pipeline fit.
    Here we do it manually.

    Returns a dataframe with fold metrics.
    """
    from sklearn.model_selection import KFold

    xgb_params = xgb_params or {}
    X, y, num_cols, cat_cols = split_feature_types(df, target_col)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx].copy(), y.iloc[va_idx].copy()

        preprocessor = build_preprocessor(num_cols, cat_cols)
        model = build_xgb_regressor(random_state=random_state + fold_idx, **xgb_params)

        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        fit_kwargs = {}
        if use_target_balance_weights:
            sw = compute_regression_sample_weights(y_tr, n_bins=10)
            fit_kwargs["model__sample_weight"] = sw

        pipe.fit(X_tr, y_tr, **fit_kwargs)
        pred_va = pipe.predict(X_va)

        fold_metrics = regression_metrics_report(y_va, pred_va).set_index("metric")["value"].to_dict()
        fold_metrics["fold"] = fold_idx
        rows.append(fold_metrics)

    cv_df = pd.DataFrame(rows)

    # Reorder columns
    order = ["fold", "MAE", "RMSE", "R2", "MedianAE", "MSE", "MaxError", "ExplainedVariance", "MAPE_%", "sMAPE_%"]
    cv_df = cv_df[[c for c in order if c in cv_df.columns]]

    return cv_df


# -------------------------------------------------------------------
# 8) Main train function
# -------------------------------------------------------------------
def train_xgb_regression_pipeline(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_regression_split: bool = True,
    use_target_balance_weights: bool = True,
    weight_n_bins: int = 10,
    weight_clip_min: float = 0.5,
    weight_clip_max: float = 5.0,
    weight_power: float = 1.0,
    xgb_params: Optional[Dict[str, Any]] = None,
    run_cv: bool = True,
    cv_folds: int = 5,
    make_plots: bool = True
) -> Dict[str, Any]:
    """
    Train + evaluate XGBoost regressor pipeline on tabular data.

    Returns a dictionary with:
    - pipeline
    - train/test data
    - predictions
    - metrics_train / metrics_test
    - cv_results (optional)
    - feature names + importances (gain/weight from xgb if possible)
    - permutation importance on test set
    """

    xgb_params = xgb_params or {}

    # Split feature types
    X, y, numeric_cols, categorical_cols = split_feature_types(df, target_col)

    # Stratified split for regression (via bins)
    stratify_vals = None
    if stratify_regression_split:
        try:
            stratify_vals = make_y_bins_for_stratification(y, n_bins=10)
        except Exception:
            stratify_vals = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_vals if stratify_vals is not None else None
    )

    # Build pipeline
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = build_xgb_regressor(random_state=random_state, **xgb_params)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # Sample weights for imbalanced regression target (TRAIN ONLY)
    fit_kwargs = {}
    train_sample_weights = None
    if use_target_balance_weights:
        train_sample_weights = compute_regression_sample_weights(
            y_train,
            n_bins=weight_n_bins,
            clip_min=weight_clip_min,
            clip_max=weight_clip_max,
            power=weight_power
        )
        fit_kwargs["model__sample_weight"] = train_sample_weights

    # Fit
    pipeline.fit(X_train, y_train, **fit_kwargs)

    # Predict
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    # Metrics
    metrics_train = regression_metrics_report(y_train, pred_train)
    metrics_test = regression_metrics_report(y_test, pred_test)

    # Cross-validation (optional)
    cv_results = None
    if run_cv:
        try:
            cv_results = cross_validate_xgb_regression(
                df=df,
                target_col=target_col,
                n_splits=cv_folds,
                random_state=random_state,
                xgb_params=xgb_params,
                use_target_balance_weights=use_target_balance_weights
            )
        except Exception as e:
            print(f"[WARN] CV skipped due to error: {e}")

    # Feature names after preprocessing (for explainability / SHAP compatibility later)
    feature_names = []
    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    except Exception:
        pass

    # Built-in XGBoost feature importance (model-level, not per-row explanation)
    xgb_feature_importance = None
    try:
        booster = pipeline.named_steps["model"]
        imp = booster.feature_importances_
        if feature_names and len(feature_names) == len(imp):
            xgb_feature_importance = (
                pd.DataFrame({"feature": feature_names, "importance_gain": imp})
                .sort_values("importance_gain", ascending=False)
                .reset_index(drop=True)
            )
        else:
            xgb_feature_importance = pd.DataFrame({
                "feature_index": np.arange(len(imp)),
                "importance_gain": imp
            }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    except Exception:
        pass

    # Permutation importance on test set (slower but model-agnostic)
    perm_importance_df = None
    try:
        pi = permutation_importance(
            pipeline, X_test, y_test,
            n_repeats=5,
            random_state=random_state,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )
        if feature_names and len(feature_names) == len(pi.importances_mean):
            perm_importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance_mean": pi.importances_mean,
                "importance_std": pi.importances_std
            }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] Permutation importance skipped: {e}")

    # Optional plots
    if make_plots:
        _plot_regression_diagnostics(y_test, pred_test, title_prefix="Test")
        if xgb_feature_importance is not None:
            _plot_top_feature_importance(xgb_feature_importance, top_n=15)

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "cv_results": cv_results,
        "feature_names": feature_names,
        "xgb_feature_importance": xgb_feature_importance,
        "permutation_importance": perm_importance_df,
        "train_sample_weights": train_sample_weights,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


# -------------------------------------------------------------------
# 9) Diagnostics plots
# -------------------------------------------------------------------
def _plot_regression_diagnostics(y_true, y_pred, title_prefix="Test"):
    """Simple diagnostic plots."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    # Scatter: Actual vs Predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.4)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Réel")
    plt.ylabel("Prédit")
    plt.title(f"{title_prefix} - Réel vs Prédit")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Residual histogram
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel("Résidu (réel - prédit)")
    plt.ylabel("Fréquence")
    plt.title(f"{title_prefix} - Distribution des résidus")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_top_feature_importance(importance_df: pd.DataFrame, top_n: int = 15):
    """Bar plot of top feature importances."""
    top = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance_gain"])
    plt.xlabel("Importance")
    plt.ylabel("Variable")
    plt.title(f"Top {top_n} variables (importance XGBoost)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 10) Predict new rows (list / dataframe) with original categorical values
# -------------------------------------------------------------------
def predict_new(pipeline: Pipeline, new_data: Any, clip_to_range: Optional[Tuple[float, float]] = (0, 10)) -> np.ndarray:
    """
    Predict on new rows.
    You can pass:
    - list of dicts
    - dict (single row)
    - pd.DataFrame

    Example
    -------
    new_rows = [
        {"age": 33, "delai": 120, "gender": "Homme", "status": "open"},
        {"age": 50, "delai": 300, "gender": "Femme", "status": "closed"},
    ]
    preds = predict_new(pipe, new_rows)

    NOTE
    ----
    You pass categorical values exactly as in your dataframe (strings/categories).
    The pipeline's OneHotEncoder(handle_unknown='ignore') handles encoding automatically.
    """
    if isinstance(new_data, dict):
        X_new = pd.DataFrame([new_data])
    elif isinstance(new_data, list):
        X_new = pd.DataFrame(new_data)
    elif isinstance(new_data, pd.DataFrame):
        X_new = new_data.copy()
    else:
        raise ValueError("new_data must be dict, list[dict], or pandas DataFrame.")

    preds = pipeline.predict(X_new)

    if clip_to_range is not None:
        preds = np.clip(preds, clip_to_range[0], clip_to_range[1])

    return preds


# -------------------------------------------------------------------
# 11) Optional SHAP example (if shap is installed)
# -------------------------------------------------------------------
def explain_with_shap(
    trained_result: Dict[str, Any],
    sample_size: int = 200
):
    """
    Optional SHAP explanation.
    Install first: pip install shap

    This gives per-feature contribution explanations for XGBoost predictions.
    """
    try:
        import shap
    except ImportError:
        print("SHAP is not installed. Run: pip install shap")
        return None

    pipe = trained_result["pipeline"]
    X_test = trained_result["X_test"]

    # Transform data with same preprocessing
    X_test_transformed = pipe.named_steps["preprocessor"].transform(X_test)
    model = pipe.named_steps["model"]

    # Sample for faster plots
    if len(X_test) > sample_size:
        idx = np.random.RandomState(42).choice(len(X_test), size=sample_size, replace=False)
        Xs = X_test_transformed[idx]
    else:
        Xs = X_test_transformed

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    feature_names = trained_result.get("feature_names", None)
    shap.summary_plot(shap_values, Xs, feature_names=feature_names, show=True)

    return shap_values


# -------------------------------------------------------------------
# 12) EXAMPLE RUN (EDIT target_col and xgb_params)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example toy data (replace with your real df)
    np.random.seed(42)
    n = 1000
    df_example = pd.DataFrame({
        "age": np.random.randint(18, 80, size=n),
        "delai_declaration": np.random.randint(0, 5000, size=n),
        "channel": np.random.choice(["Phone", "App", "Agency"], size=n),
        "status": np.random.choice(["open", "closed", "pending"], size=n),
        "city": np.random.choice(["Sousse", "Tunis", "Sfax"], size=n),
        "pieces_count": np.random.randint(0, 8, size=n),
    })

    # Imbalanced target 0..10 (many 10s)
    base = 10 - 0.0015 * df_example["delai_declaration"] + 0.02 * (df_example["age"] > 60)
    noise = np.random.normal(0, 0.8, size=n)
    y = np.clip(base + noise, 0, 10)

    # Force imbalance (many high scores)
    mask = np.random.rand(n) < 0.55
    y[mask] = np.clip(y[mask] + np.random.uniform(0.7, 1.6, size=mask.sum()), 0, 10)
    df_example["evaluate_note"] = np.round(y, 1)

    # XGBoost parameters (you can tune)
    xgb_params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.5,
        "gamma": 0.0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
    }

    result = train_xgb_regression_pipeline(
        df=df_example,
        target_col="evaluate_note",
        test_size=0.2,
        random_state=42,
        stratify_regression_split=True,          # better split for imbalanced target
        use_target_balance_weights=True,         # helps rare target ranges
        weight_n_bins=10,
        weight_clip_min=0.5,
        weight_clip_max=4.0,
        weight_power=1.0,
        xgb_params=xgb_params,
        run_cv=True,
        cv_folds=5,
        make_plots=True
    )

    print("\n=== TEST METRICS ===")
    print(result["metrics_test"].to_string(index=False))

    if result["cv_results"] is not None:
        print("\n=== CROSS-VALIDATION (fold metrics) ===")
        print(result["cv_results"])
        print("\n=== CV mean ===")
        print(result["cv_results"].mean(numeric_only=True))

    if result["xgb_feature_importance"] is not None:
        print("\n=== TOP FEATURE IMPORTANCE (XGBoost) ===")
        print(result["xgb_feature_importance"].head(10))

    # Predict on new rows (categorical values passed exactly like in dataframe)
    new_rows = [
        {
            "age": 34,
            "delai_declaration": 120,
            "channel": "Phone",
            "status": "open",
            "city": "Sousse",
            "pieces_count": 2,
        },
        {
            "age": 58,
            "delai_declaration": 2500,
            "channel": "Agency",
            "status": "pending",
            "city": "Tunis",
            "pieces_count": 5,
        }
    ]
    preds = predict_new(result["pipeline"], new_rows, clip_to_range=(0, 10))
    print("\n=== PREDICTIONS ON NEW ROWS ===")
    for row, p in zip(new_rows, preds):
        print(f"Input={row} -> predicted_note={p:.2f}")

What to focus on first (practical)
	•	Keep these metrics in your slide/report: MAE, RMSE, R²
	•	For imbalance in regression: use the included sample weights (use_target_balance_weights=True)
	•	For explainability: use SHAP (optional function included) + built-in feature importance
	•	For production prediction: always use the returned pipeline so OneHotEncoding is applied automatically to new categorical values

If you want, I can also give you a CatBoost version of the same pipeline (much simpler for categorical columns, no one-hot needed in many cases).

```
