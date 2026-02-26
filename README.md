```
XGBoost (régression) — ce qu’on a utilisé, quoi changer, et à quoi ça sert

Modèle 1 — Baseline XGBoost (sans pondération)
Hyperparamètres typiques qu’on a fixés :
	•	objective="reg:squarederror" : fonction de perte (erreur quadratique).
	•	eval_metric="rmse" : métrique suivie pendant l’entraînement.
	•	n_estimators : nombre d’arbres (itérations).
	•	learning_rate : taille du pas (à quel point chaque arbre corrige).
	•	max_depth : profondeur max des arbres (complexité).
	•	min_child_weight : minimum d’exemples/poids dans une feuille (évite splits trop spécifiques).
	•	gamma : seuil minimal d’amélioration pour faire un split (plus grand = plus conservateur).
	•	subsample : % de lignes utilisées par arbre (régularise).
	•	colsample_bytree : % de variables utilisées par arbre (régularise).
	•	reg_alpha : régularisation L1 (sparsité).
	•	reg_lambda : régularisation L2 (stabilité).
	•	tree_method="hist" : méthode rapide de construction d’arbres.
	•	missing=np.nan : traitement des valeurs manquantes.
	•	n_jobs=-1 : multi-cœurs.
	•	random_state : reproductibilité.
	•	verbosity : logs.

Ce que tu peux changer facilement :
	•	n_estimators, learning_rate, max_depth (impact le plus fort)
	•	subsample, colsample_bytree (surapprentissage / robustesse)
	•	reg_alpha, reg_lambda, gamma (régularisation)

⸻

Modèle 2 — XGBoost avec “Sample Weighting” (déséquilibre de cible)
Même modèle que baseline + une technique :
	•	sample_weight passé dans fit(...)

Technique utilisée : Inverse Frequency Binning + Sample Weights
	•	On découpe la cible en tranches (bins)
	•	Poids = 1 / fréquence de la tranche
	•	(option) normalisation et clipping des poids

Ce que tu peux changer :
	•	n_bins (nombre de tranches)
	•	clip_min, clip_max (évite des poids trop extrêmes)
	•	“force” de pondération (par ex racine carrée de l’inverse fréquence au lieu de l’inverse)

⸻

Modèle 3 — XGBoost Random Search (pondéré)
Optimisation utilisée : RandomizedSearchCV
	•	param_distributions : listes (ou distributions) testées
	•	n_iter : nombre d’essais (plus grand = plus de chances de trouver mieux, mais plus long)
	•	cv=KFold(n_splits=5) : validation croisée
	•	scoring="neg_root_mean_squared_error" : score de sélection
	•	refit=True : réentraîne le meilleur modèle sur tout le train

Ce que tu peux changer :
	•	n_iter : 20–50 (rapide), 80–150 (plus sérieux)
	•	n_splits : 3 (vite), 5 (standard)
	•	l’espace param_distributions (plus large = mieux mais plus long)

⸻

Modèle 4 — XGBoost Small Grid Search (pondéré)
Optimisation utilisée : GridSearchCV ciblé
	•	param_grid : petite grille autour des meilleurs paramètres
	•	cv=KFold(n_splits=5)
	•	scoring idem
	•	refit=True

Ce que tu peux changer :
	•	la largeur autour des meilleurs params (±1 depth, ±0.01 lr, etc.)
	•	n_splits
	•	la taille de la grille (plus grande = explosion du temps)

⸻

Modèle 5 — XGBoost Bayesian Optimization (Optuna) (pondéré)
Optimisation utilisée : Optuna (TPE Bayesian-like)
	•	n_trials : nombre d’essais (paramètre principal)
	•	distributions trial.suggest_* : bornes et type (log, int, float)
	•	KFold(n_splits=5) intégré dans l’objectif
	•	objectif : minimiser RMSE

Ce que tu peux changer :
	•	n_trials : 30–60 (bon), 80–200 (si budget)
	•	n_splits : 3 pour accélérer, puis 5 pour confirmer
	•	bornes des hyperparamètres (resserrer si tu as déjà une bonne zone)

⸻

Fonction de perte XGBoost (objective) — ce qu’on a utilisé + alternatives

Utilisée :
	•	reg:squarederror : optimise MSE (classique), bon point de départ.

Autres objectives utiles :
	•	reg:absoluteerror : optimise MAE, plus robuste aux outliers (si dispo selon version xgboost)
	•	reg:pseudohubererror : compromis entre MAE et MSE, robuste aux gros écarts
	•	count:poisson : si cible type “compte” (pas ton cas)

⸻

Hyperparamètres XGBoost — explication simple (les plus importants)
	•	n_estimators : nombre d’arbres. Trop élevé = risque d’overfit si pas régularisé.
	•	learning_rate : “force” de correction par arbre. Petit = plus stable.
	•	max_depth : complexité des arbres. Grand = risque d’overfit.
	•	min_child_weight : empêche les splits sur peu d’exemples.
	•	gamma : empêche les splits inutiles (plus grand = plus strict).
	•	subsample : réduit overfit en utilisant un sous-échantillon de lignes.
	•	colsample_bytree : réduit overfit en utilisant un sous-échantillon de variables.
	•	reg_alpha : L1 (met certains poids à zéro, simplifie).
	•	reg_lambda : L2 (stabilise).
	•	tree_method="hist" : accélère.
	•	max_delta_step : limite l’amplitude des mises à jour (souvent peu important en régression).
	•	grow_policy / max_leaves : structure des arbres (plus avancé, utile sur grands datasets).

⸻

CatBoost — modèles, hyperparamètres et optimisation

Pourquoi CatBoost change la transformation ?
	•	CatBoost prend directement les colonnes catégorielles via cat_features
	•	Donc pas besoin de One-Hot / Frequency / Target encoding (sauf cas spécifiques)

⸻

Modèle 1 — CatBoost Baseline

Hyperparamètres typiques :
	•	loss_function="RMSE" : perte / objectif
	•	iterations : nombre d’arbres (comme n_estimators)
	•	learning_rate : pas d’apprentissage
	•	depth : profondeur des arbres
	•	l2_leaf_reg : régularisation L2 (important)
	•	random_strength : ajoute de l’aléatoire pour réduire l’overfit
	•	bagging_temperature : contrôle le bagging (diversité)
	•	bootstrap_type="Bernoulli" + subsample : sous-échantillonnage des lignes
	•	colsample_bylevel : sous-échantillonnage de variables par niveau
	•	early_stopping_rounds : stop si pas d’amélioration
	•	allow_writing_files=False : pas de fichiers CatBoost

Ce que tu peux changer facilement :
	•	iterations, learning_rate, depth
	•	l2_leaf_reg
	•	subsample, bagging_temperature
	•	random_strength

⸻

Modèle 2 — CatBoost avec Sample Weighting (déséquilibre de cible)

Même modèle + une technique :
	•	Pool(..., weight=weights_train)
=> pondère l’entraînement pour mieux apprendre les zones rares du target

Ce que tu peux changer :
	•	n_bins pour la création des poids
	•	clip_min, clip_max
	•	la stratégie de pondération (plus douce si besoin)

⸻

Modèle 3 — CatBoost Random Search (pondéré)

On teste plusieurs configs au hasard (comme XGBoost), mais souvent via :
	•	ParameterSampler + CV manuelle (parce que CatBoost + weights + Pool)

Ce que tu peux changer :
	•	n_iter : nombre d’essais
	•	n_splits : folds CV
	•	l’espace de paramètres testé

⸻

Modèle 4 — CatBoost Small Grid Search (pondéré)

Grid ciblée autour du meilleur Random Search
Ce que tu peux changer :
	•	la taille de la grille (très important)
	•	n_splits (3 vs 5)

⸻

Modèle 5 — CatBoost Bayesian (Optuna) (pondéré)

Même logique que pour XGBoost :
	•	n_trials
	•	bornes suggest_*
	•	KFold CV
	•	objective = RMSE moyen

Ce que tu peux changer :
	•	n_trials : paramètre principal
	•	n_splits : accélérer
	•	resserrer les bornes selon ce que tu as trouvé avec random/grid

⸻

Fonction de perte CatBoost — utilisée + alternatives

Utilisée :
	•	loss_function="RMSE"

Alternatives utiles :
	•	loss_function="MAE" : plus robuste aux outliers
	•	loss_function="Huber" : compromis MAE/RMSE
	•	loss_function="Quantile" : si tu veux prédire une quantile (ex: 0.9 pour pessimiste)
	•	loss_function="LogCosh" : robuste et stable

⸻

Techniques d’optimisation — résumé et quoi régler

Random Search
	•	Paramètres à régler :
	•	n_iter (le plus important)
	•	cv folds (3 ou 5)
	•	l’espace de recherche

Quand l’utiliser :
	•	début de tuning, budget limité

Grid Search (Small)
	•	Paramètres à régler :
	•	taille de grille (petite)
	•	folds CV

Quand l’utiliser :
	•	après random, pour affiner

Bayesian (Optuna)
	•	Paramètres à régler :
	•	n_trials (principal)
	•	bornes de recherche
	•	folds CV

Quand l’utiliser :
	•	quand tu veux optimiser fort avec moins d’essais qu’une grille

⸻

Recommandation pratique (très simple)

Pour ton cas (note 0–10, dataset déséquilibré) :
	1.	Baseline
	2.	Sample Weighting (déséquilibre)
	3.	Random Search (20–50 essais)
	4.	Bayesian Optuna (40–80 essais) si tu veux le meilleur
	5.	Small Grid uniquement si tu veux un réglage fin autour des meilleurs

⸻

Si tu veux, je peux te produire une table “slide-ready” :
	•	colonne 1: modèle / technique
	•	colonne 2: hyperparamètres utilisés
	•	colonne 3: paramètres que tu peux ajuster (n_trials, n_iter, n_splits, etc.)
	•	colonne 4: objectif/metric (RMSE/MAE)

```
