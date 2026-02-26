```
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


# ============================================================
# Helper: get "score" for ROC (prefer predict_proba, fallback decision_function)
# ============================================================
def _get_model_scores(model, X):
    """
    Returns scores for ROC/AUC.
    - If predict_proba exists: returns proba for positive class (binary) or full proba (multiclass)
    - Else if decision_function exists: returns decision scores
    - Else: returns model.predict(X) (last-resort; may be wrong for ROC)
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


# ============================================================
# 1) Single model: plot ROC curve for Train/Test (your choice)
# ============================================================
def plot_roc_auc_single(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name="Model",
    which="test",  # "train", "test", "both"
    pos_label=1,
    title=None,
):
    """
    Plot ROC curve(s) and show AUC for one model.

    Parameters
    ----------
    model : fitted estimator
    X_train, y_train, X_test, y_test : arrays or DataFrames
    model_name : str
    which : {"train","test","both"}
    pos_label : int/str
        Positive class label for binary ROC
    title : str or None
    """
    which = which.lower().strip()
    if which not in {"train", "test", "both"}:
        raise ValueError("which must be one of: 'train', 'test', 'both'")

    fig, ax = plt.subplots(figsize=(7, 6))

    def _plot_split(X, y, split_name):
        scores = _get_model_scores(model, X)

        # Binary classification: use positive class probability if available
        if isinstance(scores, np.ndarray) and scores.ndim == 2 and scores.shape[1] >= 2:
            y_score = scores[:, 1]
        else:
            y_score = np.asarray(scores).reshape(-1)

        fpr, tpr, _ = roc_curve(y, y_score, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} ({split_name}) AUC={roc_auc:.3f}")
        return roc_auc

    auc_train = None
    auc_test = None

    if which in {"train", "both"}:
        auc_train = _plot_split(X_train, y_train, "Train")
    if which in {"test", "both"}:
        auc_test = _plot_split(X_test, y_test, "Test")

    # Diagonal line (random)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if title is None:
        title = f"ROC Curve — {model_name}"
    ax.set_title(title, pad=14)

    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return {"auc_train": auc_train, "auc_test": auc_test}


# ============================================================
# 2) Multiple models: plot ROC curves for Train/Test/Both
# ============================================================
def plot_roc_auc_models(
    models,  # list like: [("Name1", model1), ("Name2", model2), ...]
    X_train,
    y_train,
    X_test,
    y_test,
    which="test",  # "train", "test", "both"
    pos_label=1,
    title=None,
):
    """
    Plot ROC curve(s) for a list of models and return a table of AUCs.

    Parameters
    ----------
    models : list[tuple[str, fitted_estimator]]
    which : {"train","test","both"}
    """
    which = which.lower().strip()
    if which not in {"train", "test", "both"}:
        raise ValueError("which must be one of: 'train', 'test', 'both'")

    fig, ax = plt.subplots(figsize=(8, 7))

    results = []

    def _plot_one(model, name, X, y, split_name):
        scores = _get_model_scores(model, X)

        # Binary classification: use positive class probability if available
        if isinstance(scores, np.ndarray) and scores.ndim == 2 and scores.shape[1] >= 2:
            y_score = scores[:, 1]
        else:
            y_score = np.asarray(scores).reshape(-1)

        fpr, tpr, _ = roc_curve(y, y_score, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} ({split_name}) AUC={roc_auc:.3f}")
        return roc_auc

    for name, model in models:
        auc_train = None
        auc_test = None

        if which in {"train", "both"}:
            auc_train = _plot_one(model, name, X_train, y_train, "Train")
        if which in {"test", "both"}:
            auc_test = _plot_one(model, name, X_test, y_test, "Test")

        results.append({
            "model": name,
            "auc_train": auc_train,
            "auc_test": auc_test
        })

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if title is None:
        title = f"ROC Curves — {which.upper()}"
    ax.set_title(title, pad=14)

    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return results

# Single model: test only
plot_roc_auc_single(
    model=clf,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_name="LogReg",
    which="test",
    pos_label=1
)

# Multiple models: both train and test
models = [
    ("LogReg", logreg),
    ("XGBoost", xgb_clf),
    ("CatBoost", cb_clf),
]
auc_table = plot_roc_auc_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    which="both"
)
auc_table
```
