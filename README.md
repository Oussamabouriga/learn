```
Here’s a “course-style” explanation of ensemble models: what they are, how they work, main families (bagging / boosting / stacking), when to use them, and how to evaluate them. I’ll keep it clear, but complete.

⸻

1) What is an ensemble model?

Definition

An ensemble model combines several models to produce one final prediction.

Instead of trusting one model, you “ask multiple models” and combine their answers.

Why it works

Ensembles reduce:
	•	Variance (models that overfit / unstable)
	•	Bias (models that are too simple)
	•	And often give better generalization than a single model.

Key idea:
	•	Different models make different errors
	•	Combining them can cancel errors out.

⸻

2) The 3 big families of ensembles

A) Bagging (Bootstrap Aggregating)

How it works
	1.	Create many random training subsets using bootstrap sampling (sample with replacement).
	2.	Train one model on each subset (often trees).
	3.	Combine predictions:

	•	Regression: average
	•	Classification: majority vote / average probabilities

Example
	•	Random Forest is the classic bagging model.

Why it helps
	•	Reduces variance
	•	Good when your base model is unstable (decision trees)

When to use
	•	You want a strong baseline
	•	You want stability
	•	You have noise and want robust predictions

⸻

B) Boosting

How it works (intuitive)

Boosting trains models sequentially:
	1.	Train model #1.
	2.	Look at its errors.
	3.	Train model #2 to focus more on the examples that were hard.
	4.	Repeat many rounds.

Final prediction is a weighted sum of all weak models.

Example
	•	XGBoost, LightGBM, CatBoost are boosting algorithms.

Why it helps
	•	Reduces bias and improves accuracy
	•	Learns complex patterns
	•	Often state-of-the-art for tabular data

When to use
	•	You want top performance on tabular features
	•	Nonlinear relations
	•	Mixed numeric/categorical features (CatBoost is excellent here)

⸻

C) Stacking (Stacked Generalization)

How it works

Stacking combines different model types:
	1.	Train several “base models” (e.g., XGBoost, RandomForest, Linear).
	2.	Generate predictions from them on validation data.
	3.	Train a “meta-model” that learns how to combine their predictions.

Meta-model learns:
“when should I trust model A vs model B?”

Why it helps
	•	Can outperform single boosting in some cases
	•	Very flexible (mix models with different strengths)

When to use
	•	You have enough data
	•	You want to combine different model families
	•	You can manage complexity

⸻

3) How ensembles combine predictions

Regression

Common aggregation:
	•	Mean: average predictions
	•	Weighted mean: more weight on better models

Classification

Common aggregation:
	•	Majority vote
	•	Average probability (better if you want probabilities)

⸻

4) A special ensemble: “Mixture of Experts” / “Two-stage models”

This is exactly what you built.

Principle

You don’t just average models, you route the example to the right expert.

It has:
	•	a gating model (router / selector)
	•	multiple expert models (specialists)

How it works
	1.	Gating model decides which region / class the example belongs to.
	2.	Expert model for that region predicts final output.

In your system:
	•	Stage 1: classifier predicts {bad / neutral / good}
	•	Stage 2: regressor predicts note inside predicted class

Why this is useful:
	•	Different “zones” behave differently
	•	One model might not capture them all well
	•	Each expert focuses on its subset

Common improvement:
	•	“Soft routing”: use probabilities to combine experts instead of choosing only one.

⸻

5) Why ensemble models often perform better

Error decomposition intuition

A model’s generalization error comes from:
	•	Bias (systematic error)
	•	Variance (instability)
	•	Noise (irreducible)

Ensembles help:
	•	Bagging → reduces variance
	•	Boosting → reduces bias (often)
	•	Stacking → reduces both if done well
	•	Mixture-of-experts → handles heterogeneous patterns (different regimes)

⸻

6) Main risks and limitations

A) Complexity

Ensembles are harder to:
	•	train
	•	debug
	•	explain
	•	deploy

B) Leakage risk (stacking)

If the meta-model sees predictions generated from training data without proper CV, it can overfit.

C) Interpretability

You need tools like:
	•	SHAP
	•	permutation importance
	•	local explanation per stage

D) Wrong routing (mixture-of-experts)

If the gating classifier misclassifies, the wrong expert regressor is used → error may increase.

Solution:
	•	soft routing (probability-weighted experts)
	•	penalty for critical misroutes (bad predicted as neutral)

⸻

7) Which ensemble should you choose? (quick guide)
	•	Want stable baseline, easy to use → Random Forest (bagging)
	•	Want best performance on tabular → XGBoost / CatBoost (boosting)
	•	Want to combine diverse models → Stacking
	•	Your data has different “regimes” (bad vs good behaves differently) → Mixture of Experts / Two-stage ensemble

For your satisfaction note case (0..10 with imbalance):
	•	Boosting works very well.
	•	Your two-stage ensemble is a good business-aligned approach.

⸻

8) How to evaluate ensembles properly

For regression note prediction:
	•	MAE, RMSE, R²
	•	Accuracy@±t (great for business)
	•	Evaluate per segment:
	•	per class (bad/neutral/good)
	•	per delay buckets
	•	per customer type

For the two-stage ensemble:
	1.	Stage 1 performance: confusion matrix + recall on “mauvais”
	2.	Final performance: MAE/RMSE + Accuracy@±t

⸻

9) Mini glossary (English → meaning)
	•	Bagging: train many models in parallel on bootstrapped samples
	•	Boosting: train models sequentially to correct mistakes
	•	Stacking: meta-model learns to combine base models
	•	Mixture of Experts: gating model routes input to specialized experts
	•	Hard gating: pick one expert
	•	Soft gating: combine experts using probabilities

⸻

If you want, I can turn this into:
	•	a 5-slide French version for presentation,
	•	or a hands-on “practical course” with diagrams + recommended default settings for RandomForest/XGBoost/CatBoost/Stacking.

```
