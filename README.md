```
Got it — here is the same documentation you pasted, but remade cleanly, expanded, and practical, with:
	•	No custom helper functions (everything written inline so you understand)
	•	Encoding categorical + numeric data (OneHot + Ordinal + Target encoding)
	•	How to apply on a DataFrame
	•	Imbalanced target (classification) with real-world techniques
	•	Every technique’s main hyperparameters explained
	•	Evaluation methods explained (what they show + how to interpret)
	•	Predict from a dictionary at the end (you type inputs manually)

⸻

Part A — Encoding & Preprocessing for ML (DataFrame → model-ready)

1) Goal: Build a clean X matrix for scikit-learn

Most ML models need:
	•	Numeric matrix input
	•	No missing values (most models)
	•	Same feature columns in train and test

So we do:
	•	Categorical → encoded
	•	Numeric → impute missing (and optionally scale)
	•	Use a Pipeline so transforms are learned only on training (avoid leakage)

⸻

2) Load Data + choose target

import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

TARGET = "target"  # change this to your target column name

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

print(df.shape)
print(df.head())


⸻

3) Detect numeric vs categorical columns (pandas)

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)


⸻

4) Best practice: ColumnTransformer + Pipeline

Why this is best:
	•	transforms are fitted only on training folds (no leakage)
	•	consistent encoded columns train/test
	•	works in CV + hyperparameter search

⸻

5) Minimal robust preprocessing (Impute + OneHot)

5.1 Imports

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

5.2 Build the preprocessing pipelines (inline)

✅ Numeric pipeline:
	•	fill missing with median
	•	scaling optional (useful for linear/SVM/kNN; not needed for trees)

scale_numeric = True  # set False if you use tree models (RandomForest, XGBoost, etc.)

numeric_steps = [
    ("imputer", SimpleImputer(strategy="median"))
]
if scale_numeric:
    numeric_steps.append(("scaler", StandardScaler()))

num_pipe = Pipeline(steps=numeric_steps)

✅ Categorical pipeline (OneHotEncoding):
	•	fill missing with most frequent
	•	one-hot encode
	•	ignore unknown categories at test time

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

✅ Combine them:

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)


⸻

6) Key hyperparameters explained (Preprocessing)

SimpleImputer(strategy=...)

Numeric
	•	median: robust to outliers (recommended baseline)
	•	mean: ok if no strong outliers
	•	most_frequent: rarely used for numeric

Categorical
	•	most_frequent: best simple default
	•	constant + fill_value="missing": forces a “missing category” bucket (useful sometimes)

⸻

OneHotEncoder(...)
	•	handle_unknown="ignore"
If you later predict with a category never seen during training → model still works.
	•	sparse_output=False
Easier to inspect (dense array). For huge data → set True to reduce memory.
	•	drop="first"
Drops one column per categorical variable (avoids collinearity). Useful for linear models; usually not needed for trees.

⸻

StandardScaler()
	•	rescales numeric columns to mean 0, std 1
	•	needed for:
	•	logistic regression
	•	SVM
	•	kNN
	•	neural nets
	•	not needed for:
	•	RandomForest
	•	GradientBoostedTrees

⸻

7) Train/test split + model training (end-to-end)

We’ll show with Logistic Regression as example (classic for classification).

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # IMPORTANT for imbalanced classification
)

model = LogisticRegression(max_iter=2000)

clf = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", model)
])

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print(classification_report(y_test, pred))


⸻

8) Get feature names after encoding

This is super useful to debug OneHotEncoder columns.

prep = clf.named_steps["prep"]
feature_names = prep.get_feature_names_out()
print("Total encoded features:", len(feature_names))
print(feature_names[:40])


⸻

9) Other encoding strategies (when to use)

9.1 Ordinal Encoding (ordered categories)

Use ONLY if your categories are truly ordered:
low < medium < high

from sklearn.preprocessing import OrdinalEncoder

cat_pipe_ordinal = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ord", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

preprocessor_ordinal = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe_ordinal, cat_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

OrdinalEncoder params
	•	handle_unknown="use_encoded_value": prevents crash on new categories
	•	unknown_value=-1: unseen categories mapped to -1

⚠️ Warning: if categories are not ordered, ordinal encoding injects fake numeric meaning.

⸻

9.2 Target Encoding (high-cardinality categories)

Use for: city, merchant_id, customer_id (many unique values)
One-hot may create thousands of columns.

Install:

pip install category_encoders

import category_encoders as ce

cat_pipe_target = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("te", ce.TargetEncoder(
        smoothing=10.0,
        min_samples_leaf=20
    ))
])

preprocessor_target = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe_target, cat_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

TargetEncoder params
	•	smoothing: higher = more shrinkage to global mean (reduces overfit)
	•	min_samples_leaf: small categories get stronger smoothing

⚠️ Must be used in a pipeline + CV to avoid leakage.

⸻

⸻

Part B — Handling Imbalanced Target Data (Classification)

Imbalanced classification = rare positive class (fraud, churn, defect).
Accuracy becomes misleading. We need better metrics and techniques.

⸻

1) Correct evaluation metrics for imbalanced classification

1.1 What each metric tells you
	•	Precision: when model predicts positive, how often it’s correct
→ High precision = few false alarms
	•	Recall: among real positives, how many were detected
→ High recall = fewer missed positives
	•	F1: balance precision + recall
→ Useful when you want both
	•	ROC-AUC: ranking ability (can look good even with extreme imbalance)
	•	PR-AUC (Average Precision): best for rare positives
	•	Balanced Accuracy: average recall across classes

1.2 Code

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
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

2) Technique 1: Class weights (strong baseline)

Works well for LogisticRegression, SVM, trees.

model_w = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

clf_w = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", model_w)
])

clf_w.fit(X_train, y_train)
pred_w = clf_w.predict(X_test)

print(classification_report(y_test, pred_w))

Class weight params
	•	class_weight="balanced"
weights are inverse to class frequency
	•	or manual: class_weight={0:1, 1:5}

Pros: no synthetic data, simple, effective baseline
Cons: can increase recall but reduce precision

⸻

3) Technique 2: SMOTE oversampling (numeric-friendly)

Install:

pip install imbalanced-learn

Important: do it inside pipeline only.

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

smote = SMOTE(
    sampling_strategy=0.5,
    k_neighbors=5,
    random_state=42
)

rf_clf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

pipe_smote = ImbPipeline(steps=[
    ("prep", preprocessor),
    ("smote", smote),
    ("model", rf_clf)
])

pipe_smote.fit(X_train, y_train)
pred_smote = pipe_smote.predict(X_test)

print(classification_report(y_test, pred_smote))

SMOTE params explained
	•	sampling_strategy
	•	0.5: minority becomes 50% of majority
	•	"auto": full balance
	•	dict: exact per-class counts
	•	k_neighbors
	•	how many neighbors used to create synthetic samples
	•	small minority count → reduce to 3
	•	random_state: reproducibility

Pros: improves recall
Cons: synthetic samples can be unrealistic, especially for categorical variables

⸻

4) Technique 3: SMOTENC (mixed numeric + categorical)

Use when you have categorical features and you want oversampling before one-hot.

from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTENC

# encode categorical as ordinal first (temporary)
cat_pipe_ord_for_smote = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

pre_smote_prep = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", cat_pipe_ord_for_smote, cat_cols)
    ],
    remainder="drop"
)

cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

smote_nc = SMOTENC(
    categorical_features=cat_indices,
    sampling_strategy=0.5,
    k_neighbors=5,
    random_state=42
)

pipe_smote_nc = ImbPipeline(steps=[
    ("prep", pre_smote_prep),
    ("smote", smote_nc),
    ("model", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
])

pipe_smote_nc.fit(X_train, y_train)
pred_nc = pipe_smote_nc.predict(X_test)

print(classification_report(y_test, pred_nc))

SMOTENC params
	•	categorical_features: indices of categorical columns in transformed matrix
	•	others like SMOTE

⸻

5) Technique 4: Under-sampling (fast for huge datasets)

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(
    sampling_strategy=0.8,
    random_state=42
)

pipe_under = ImbPipeline(steps=[
    ("prep", preprocessor),
    ("under", rus),
    ("model", LogisticRegression(max_iter=2000))
])

pipe_under.fit(X_train, y_train)
pred_under = pipe_under.predict(X_test)

print(classification_report(y_test, pred_under))

UnderSampler params
	•	sampling_strategy=0.8: minority becomes 0.8× majority after sampling

Pros: fast, avoids synthetic points
Cons: discards data (can hurt performance)

⸻

6) Technique 5: Threshold tuning (VERY important)

Default threshold = 0.5 is often wrong in imbalance.

from sklearn.metrics import precision_recall_curve

proba = pipe_smote.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, proba)

target_recall = 0.85
idx = np.where(recall >= target_recall)[0][-1]
best_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

pred_custom = (proba >= best_threshold).astype(int)

print("Best threshold:", best_threshold)
print(classification_report(y_test, pred_custom))

What it shows
	•	You choose a threshold to meet business goal (e.g., recall ≥ 85%)
	•	It directly controls FP vs FN tradeoff

⸻

7) Best-practice recipe (real-world)
	1.	Split with stratify=y
	2.	Use pipeline (preprocessing + model)
	3.	Start with class_weight="balanced"
	4.	Evaluate with PR-AUC + F1 + recall/precision
	5.	Add SMOTE/SMOTENC if needed
	6.	Tune threshold
	7.	Validate with Stratified CV

⸻

8) Stratified cross-validation for imbalanced classification

from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    pipe_smote,  # or clf_w, pipe_under, etc.
    X, y,
    cv=skf,
    scoring="average_precision",
    n_jobs=-1
)

print("CV PR-AUC mean:", scores.mean())
print("CV PR-AUC std :", scores.std())


⸻

Extra: Predict by typing a dictionary (manual test)

Works best with OneHotEncoder pipeline (handle_unknown="ignore").

sample = {
    # Put your real column names here:
    "age": 29,
    "city": "Tunis",
    "subscription_type": "premium",
    "rides_last_month": 10
}

sample_df = pd.DataFrame([sample])

pred = clf.predict(sample_df)[0]
proba = clf.predict_proba(sample_df)[0, 1]

print("Predicted class:", pred)
print("Predicted probability of positive class:", proba)


⸻

If you want, I can “lock” this into one single script that you copy/paste and run, but I need only:
	•	target column name
	•	are you predicting binary class (0/1) or multi-class?


```
