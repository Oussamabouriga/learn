```
Below is a full, practical documentation (same style as before) for:
	1.	Encoding categorical + numeric data for ML training (on a pandas DataFrame)
	2.	Handling imbalanced target data (classification) with strong, real-world techniques + parameters explained

⸻

Part A — Encoding & Preprocessing for ML (DataFrame → model-ready)

1) Goal: build a clean X matrix for scikit-learn

Most ML models require a numeric matrix:
	•	categorical → encoded (one-hot / ordinal / target encoding)
	•	numeric → impute missing + optionally scale
	•	keep everything in a single pipeline to avoid leakage

⸻

2) Split features by type (pandas)

import numpy as np
import pandas as pd

def split_columns(df, target_col):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return X, y, num_cols, cat_cols


⸻

3) Recommended approach: ColumnTransformer + Pipeline

This is the best practice because it:
	•	applies transforms only on train folds (no leakage)
	•	keeps consistent columns between train/test
	•	works with CV + hyperparameter search cleanly

3.1 Minimal robust preprocessing (impute + one-hot)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def make_preprocessor(num_cols, cat_cols, scale_numeric=False):
    # numeric pipeline
    num_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        # useful for linear models / SVM / kNN / neural nets
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(steps=num_steps)

    # categorical pipeline
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",  # ignore other columns
        verbose_feature_names_out=False
    )
    return preprocessor

Key parameters explained

SimpleImputer
	•	strategy="median": robust to outliers for numeric
	•	strategy="most_frequent": common baseline for categorical

OneHotEncoder
	•	handle_unknown="ignore": unseen category at test time won’t crash; it becomes all zeros
	•	sparse_output=False: returns dense array (easier to inspect). For big data, set True.

StandardScaler
	•	transforms numeric columns to zero-mean/unit-variance
	•	not needed for tree-based models, but important for distance/linear models

⸻

4) Fit preprocessing on a DataFrame + train a model (end-to-end)

Example with a classifier (works same for regression models too):

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("data.csv")
TARGET = "target"

X, y, num_cols, cat_cols = split_columns(df, TARGET)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = make_preprocessor(num_cols, cat_cols, scale_numeric=True)

model = LogisticRegression(max_iter=2000)

clf = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", model)
])

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print(classification_report(y_test, pred))


⸻

5) Other encoding strategies (when to use)

5.1 Ordinal Encoding (ordered categories)

Use when categories are truly ordered, e.g. ["low","medium","high"].

from sklearn.preprocessing import OrdinalEncoder

cat_pipe_ordinal = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ord", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

Parameters
	•	handle_unknown="use_encoded_value": allows unseen categories
	•	unknown_value=-1: unseen categories mapped to -1 (safe)

Warning: don’t use ordinal encoding for non-ordered categories: it injects fake numeric meaning.

⸻

5.2 Target Encoding (high-cardinality categories)

Useful when you have columns like:
	•	city (thousands of values)
	•	customer_id (very high)

One-hot would explode the feature space. Target encoding replaces each category by the mean target (with smoothing).
Best done with a dedicated library (and done inside CV to avoid leakage).

Recommended lib: category_encoders

pip install category_encoders

import category_encoders as ce
from sklearn.pipeline import Pipeline

cat_pipe_target = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("te", ce.TargetEncoder(
        cols=cat_cols,
        smoothing=10.0,
        min_samples_leaf=20
    ))
])

Parameters
	•	smoothing: higher = more shrinkage to global mean (reduces overfit)
	•	min_samples_leaf: categories with few samples get stronger shrinkage

⸻

6) Get feature names after encoding (very useful)

prep = clf.named_steps["prep"]
feature_names = prep.get_feature_names_out()
print(len(feature_names))
print(feature_names[:30])


⸻

Part B — Handling Imbalanced Target Data (Classification)

Imbalanced target = one class is rare (fraud, churn, disease, defect).
If you evaluate only with accuracy, you can get misleading results.

1) Correct evaluation for imbalanced classification

1.1 Use these metrics (recommended)
	•	Precision: among predicted positives, how many are correct?
	•	Recall: among true positives, how many did we catch?
	•	F1: balance precision & recall
	•	ROC-AUC: ranking quality (can be optimistic in extreme imbalance)
	•	PR-AUC (Average Precision): often better for rare positives
	•	Balanced Accuracy: average recall across classes

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

proba = clf.predict_proba(X_test)[:, 1]
pred = clf.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

print("ROC-AUC:", roc_auc_score(y_test, proba))
print("PR-AUC :", average_precision_score(y_test, proba))


⸻

2) Technique 1: Use class weights (strong baseline)

Many models support weighting minority class higher.

Logistic Regression / SVM / Trees:

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

Parameters
	•	class_weight="balanced": weights inversely proportional to class frequency
	•	or custom: class_weight={0:1, 1:5}

Pros
	•	No synthetic data
	•	Works very well as baseline

Cons
	•	May reduce precision if pushed too hard toward recall

⸻

3) Technique 2: Resampling (SMOTE / under-sampling) with imblearn

Install:

pip install imbalanced-learn

IMPORTANT: resample inside pipeline only

Otherwise you leak information into test/CV.

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

preprocessor = make_preprocessor(num_cols, cat_cols, scale_numeric=False)

smote = SMOTE(
    sampling_strategy=0.5,
    k_neighbors=5,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    class_weight=None
)

pipe = ImbPipeline(steps=[
    ("prep", preprocessor),
    ("smote", smote),
    ("model", model)
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print(classification_report(y_test, pred))

SMOTE parameters explained
	•	sampling_strategy
	•	float like 0.5: after resampling, minority count becomes 0.5 × majority
	•	"auto": balance fully (minority = majority)
	•	dict {minority_label: desired_count}
	•	k_neighbors
	•	number of neighbors used to create synthetic samples
	•	small minority class? lower it (e.g. 3)
	•	random_state: reproducibility

Pros
	•	Can improve recall significantly

Cons
	•	Can create unrealistic synthetic points, especially with messy categorical variables
(for mixed numeric/categorical, prefer SMOTENC)

⸻

4) Technique 3: SMOTENC for mixed data (categorical + numeric)

If you have categorical features and you one-hot too early, SMOTE can behave poorly.
Use SMOTENC with categorical indices (before one-hot).

This requires a slightly different preprocessing approach (ordinal encode categories first).
Here’s a clean template:

from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTENC

# ordinal encode cats (temporary numeric codes)
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

pre_smote_prep = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ],
    remainder="drop"
)

# indices of categorical columns AFTER transformer:
cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

smote_nc = SMOTENC(
    categorical_features=cat_indices,
    sampling_strategy=0.5,
    k_neighbors=5,
    random_state=42
)

model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)

pipe = ImbPipeline(steps=[
    ("prep", pre_smote_prep),
    ("smote", smote_nc),
    ("model", model)
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print(classification_report(y_test, pred))

SMOTENC parameters explained
	•	categorical_features: list of indices for categorical columns
	•	others same as SMOTE

⸻

5) Technique 4: Under-sampling (when you have huge data)

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(
    sampling_strategy=0.8,
    random_state=42
)

pipe = ImbPipeline(steps=[
    ("prep", preprocessor),
    ("under", rus),
    ("model", LogisticRegression(max_iter=2000, class_weight=None))
])

Parameters
	•	sampling_strategy=0.8: minority becomes 0.8× majority (after under-sampling majority)

Pros
	•	Fast
	•	Avoids synthetic points

Cons
	•	You throw away data (can hurt performance)

⸻

6) Technique 5: Threshold tuning (very important)

Most models output probabilities. Default threshold is 0.5, which is often wrong for imbalance.

from sklearn.metrics import precision_recall_curve

proba = pipe.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, proba)

# Example: choose threshold achieving recall >= 0.85 (business constraint)
target_recall = 0.85
idx = np.where(recall >= target_recall)[0][-1]
best_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

pred_custom = (proba >= best_threshold).astype(int)

print("Chosen threshold:", best_threshold)
print(classification_report(y_test, pred_custom))

Why this matters
	•	You can explicitly choose recall vs precision tradeoff based on business needs.

⸻

7) Recommended “best practice” recipe (real-world)
	1.	Split with stratify=y
	2.	Use a pipeline (ColumnTransformer + model)
	3.	Start with class_weight="balanced" baseline
	4.	Evaluate with PR-AUC, F1, recall/precision
	5.	Add SMOTE/SMOTENC if needed
	6.	Tune threshold to hit business targets (e.g., recall ≥ 90%)
	7.	Validate with stratified CV

⸻

8) Stratified Cross-Validation for imbalanced classification

from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    pipe, X, y,
    cv=skf,
    scoring="average_precision",  # PR-AUC
    n_jobs=-1
)
print("CV PR-AUC mean:", scores.mean(), "std:", scores.std())


⸻

If you tell me:
	•	your target column name,
	•	which column(s) are categorical,
	•	and whether your task is classification or regression,

I can give you a single final ready-to-run script tailored to your DataFrame (best preprocessing + best imbalance strategy).

```
