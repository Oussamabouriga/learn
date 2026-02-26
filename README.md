```
Below is a complete, detailed explanation of the last ensemble code (classification + regression + Bayesian optimization + SHAP). I’ll use the English method names exactly like in code, explain why we did each step, and give alternatives.

⸻

0) What we’re building (the idea)

Method name (high level)

Two-Stage (Hierarchical) Ensemble
	•	Stage 1: XGBoost Classification predicts a bucket/class of satisfaction.
	•	Stage 2: XGBoost Regression predicts the exact note inside that class range.

Why do this?
	•	Your note target (0–10) is not uniform (lots of 10s etc.).
	•	A regressor alone may “average” too much.
	•	A classifier can separate the problem into meaningful zones (bad/neutral/good).
	•	Then a regressor specialized per zone can better predict within that zone.

⸻

1) Data & inputs (assumptions)

Method names
	•	Encoded Feature Matrix: X_train_encoded, X_test_encoded
(already numeric, prepared with your transformation pipeline)
	•	Target: y_train, y_test (0..10)

Why?
	•	XGBoost needs numeric input.
	•	Keeping the same encoding ensures consistent features for all models.

Alternative:
	•	Use CatBoost and pass raw categorical columns directly (no encoding).
	•	Use a sklearn Pipeline to avoid any manual alignment.

⸻

2) Class mapping: turning the note into categories

Method name in code

to_class(y) (custom mapping function)

Mapping:
	•	0–2 → extrêmement mauvais
	•	3–6 → mauvais
	•	7–8 → neutre
	•	9 → bien
	•	10 → très bien

Why?
	•	You want to trigger actions mainly when the note is “bad”.
	•	The boundary zone (7.x) is critical for decision-making.
	•	This creates business-friendly classes.

Why rounding (np.rint) before mapping?
	•	Because your note can be float (e.g., 7.8).
	•	Classes like “9” and “10” are exact labels; rounding makes mapping stable.

Alternatives:
	•	Don’t round; define continuous bins directly:
	•	0..2.999, 3..6.999, 7..8.999, etc.
	•	Use ordinal classification (classes ordered).
	•	Use soft labels for borderline notes.

⸻

3) Handling imbalance in classification

Method name in code

Inverse Class Frequency Sample Weighting
	•	clf_weights = 1 / freq(class) (then normalized + clipped)
	•	passed to .fit(..., sample_weight=clf_weights)

Why?
	•	If most examples are class “très bien”, classifier will learn to predict “très bien” too often.
	•	Sample weights force the classifier to care about minority classes.

Alternatives:
	•	scale_pos_weight (only for binary classification).
	•	SMOTE / oversampling (less recommended for tree methods sometimes, but possible).
	•	Use objective metrics like macro F1 instead of accuracy (for model selection).

⸻

4) Bayesian optimization (Optuna) for the whole ensemble

Method name

Bayesian Hyperparameter Optimization (Optuna)
Optuna’s default sampler is TPE (Tree-structured Parzen Estimator).

Why Bayesian instead of Random/Grid?
	•	Random Search wastes trials.
	•	Grid Search explodes combinatorially.
	•	Bayesian/TPE uses previous results to propose better configurations sooner.

What Optuna optimizes in this code

A single scalar objective combining:
	•	MAE_note_CV (end-to-end note error)
	•	class_accuracy_CV (classification quality)

Objective:
objective = MAE_CV + lambda_acc*(1 - Accuracy_CV)

Why this objective?
	•	The final product output is a note, so MAE is the primary objective.
	•	But if classifier is weak, regressors will be used on wrong class ⇒ note is wrong.
	•	lambda_acc balances importance of classification vs regression.

Alternatives:
	1.	Multi-objective optimization:
	•	minimize MAE, minimize RMSE, maximize accuracy, maximize R² simultaneously.
	2.	Use a different penalty:
	•	MAE + penalty_for_confusion(mauvais predicted as neutre)
	3.	Optimize business metric:
	•	maximize recall on “mauvais” class (critical for action triggering).

⸻

5) Cross-validation

Method name

KFold Cross-Validation (KFold(n_splits=5))

Why?
	•	Gives a more stable estimate than a single train/validation split.
	•	Reduces chance of picking hyperparameters that work “by luck”.

Where we use it:
	•	Inside Optuna objective: we train and validate across folds.

Alternatives:
	•	StratifiedKFold (better for classification imbalance)
For your stage 1 classification, it can be better than KFold.
	•	Use fewer folds for speed (3 folds) then confirm with 5 folds.

⸻

6) Stage 1 model: XGBoost Classifier

Method name

XGBClassifier
Key code options:
	•	objective="multi:softprob"
	•	num_class=5
	•	eval_metric="mlogloss"
	•	tree_method="hist"

What it does:
	•	Outputs class probabilities (soft probabilities).
	•	We then use .predict() to choose class.

Why multi:softprob?
	•	It gives probabilities for each class, not just labels.
	•	Useful if later you want a confidence score.

Alternative objectives:
	•	multi:softmax gives direct class labels (no probabilities).
	•	Use calibration (Platt/Isotonic) if probabilities must be reliable.

⸻

7) Stage 2 models: “one regressor per class”

Method name

Per-Class Regressors (a form of Mixture of Experts)

We train 5 regressors:
	•	One regressor trained only on rows of a class.

Why?
	•	Patterns for low notes may differ from patterns for high notes.
	•	A regressor specialized on the region often fits better.

We also clip predictions to class ranges:
	•	Example: if predicted class = 0, regressor output is clipped to [0,2]

Why clipping?
	•	It enforces business logic: predicted note must remain consistent with predicted class.

Alternative:
	•	Train one regressor only, and use class as a feature.
	•	Use probabilistic mixture:
	•	weighted sum of regressor outputs using classifier probabilities:
prediction = Σ p(class=k) * reg_k(x)
This often improves stability at class boundaries.
	•	Use a single ordinal regression model instead.

⸻

8) “Fallback regressor” when too few samples in a class

Method in code

If class has < 20 samples, we do:
	•	predict = middle of class range

Why?
	•	Training a regressor with too few samples is unstable and overfits.
	•	This ensures the pipeline never crashes.

Alternatives:
	•	Lower threshold from 20 to 10 if you have small dataset.
	•	Use a global regressor as fallback.
	•	Merge rare classes.

⸻

9) Final training (best hyperparameters) and evaluation

After Optuna:
	•	train classifier on full train
	•	train each regressor on full train per class
	•	evaluate on test

Metrics shown:
	•	classification accuracy (5 classes)
	•	regression MAE/RMSE/R² on final note
	•	Accuracy@±1 (tolerance metric)

Why keep both classification and regression metrics?
	•	Because the ensemble quality depends on both.

Alternatives:
	•	Use macro F1 for classification.
	•	Evaluate note MAE per class (more diagnostic).
	•	Use cost-weighted errors (bad predictions cost more).

⸻

10) Feature importance + SHAP

Methods
	•	XGBoost built-in feature importance:
	•	model.feature_importances_
	•	SHAP (TreeExplainer):
	•	global importance (mean(|SHAP|))
	•	local explanation for example row

Why SHAP?
	•	It explains why a prediction is high/low.
	•	Useful for your report and for business trust.

Why two SHAP types?
	•	Global: which features matter overall.
	•	Local: why this specific example got this score/class.

Alternatives:
	•	Permutation importance (model-agnostic, slower).
	•	Partial dependence plots.
	•	LIME (local approximation).

⸻

11) Example row prediction

We:
	•	predict class using classifier
	•	predict note using the regressor for that class
	•	show which class was chosen and the final note

Why?
	•	It validates that the pipeline works end-to-end.
	•	It shows interpretability for a real case.

⸻

Recommended improvements (if you want it more robust)
	1.	Use StratifiedKFold on the classification labels for CV splits
Better distribution of classes in each fold.
	2.	Use soft gating (probability mixture)
Instead of taking only predicted class:
	•	get probabilities from classifier
	•	predict note from all regressors
	•	take weighted sum
This improves “boundary” cases like 7.8.
	3.	Optimize business cost:
	•	penalize predicting “neutre” when true is “mauvais” more than other errors.
	4.	Add calibration:
	•	if you want the classifier’s probabilities to be trusted.

⸻

If you want, I can rewrite the ensemble in the more advanced version:
	•	soft mixture of regressors (probability-weighted)
	•	StratifiedKFold
	•	Optuna objective that explicitly penalizes “mauvais misclassified as neutre”.

```
