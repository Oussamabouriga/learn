```
Bayesian hyperparameter optimization (idée générale)

Quand on optimise des hyperparamètres, on cherche la meilleure combinaison (ex: max_depth, learning_rate, etc.) sans tester toutes les possibilités.

Bayesian optimization fait ça de manière “intelligente” :
	1.	On teste une configuration
	2.	On observe le score (ex: RMSE moyen en K-Fold)
	3.	On apprend une relation “paramètres → performance” à partir des essais déjà faits
	4.	On choisit le prochain essai en équilibrant :
	•	Exploitation : tester là où ça a l’air bon
	•	Exploration : tester là où on est encore incertain

Résultat : on trouve souvent de meilleurs hyperparamètres avec moins d’essais que Grid Search.

⸻

Pourquoi Optuna est “Bayesian” dans la pratique ?

Optuna utilise par défaut un algorithme appelé TPE (Tree-structured Parzen Estimator).
Ce n’est pas un “GP Bayesian” classique (Gaussian Process) mais le principe est le même : utiliser l’historique pour proposer mieux.

TPE en simple
	•	Optuna garde toutes les tentatives passées (params, score).
	•	Il sépare les essais en deux groupes :
	•	bons essais (scores faibles si on minimise RMSE)
	•	mauvais essais
	•	Il modélise “où se trouvent les bons” vs “où se trouvent les mauvais”.
	•	Il propose de nouveaux paramètres plus probables d’être bons.

⸻

Comment Optuna fonctionne concrètement (dans ton code)

1) trial

Chaque essai est un objet trial qui sert à :
	•	tirer des hyperparamètres dans un espace donné
	•	enregistrer le résultat

Exemple :
	•	trial.suggest_int("max_depth", 3, 10)
	•	trial.suggest_float("learning_rate", 0.01, 0.2, log=True)

2) objective(trial)

C’est la fonction qui répond à la question :

“Si je prends ces hyperparamètres, quelle est la performance du modèle ?”

Dans ton cas, objective(trial) fait :
	•	construit un modèle avec les hyperparamètres proposés
	•	fait une validation croisée K-Fold
	•	calcule un score moyen (ex: RMSE moyen)
	•	retourne ce score à Optuna

3) study.optimize(objective, n_trials=...)

Optuna répète :
	•	proposer une config
	•	exécuter objective
	•	stocker résultat
	•	proposer mieux

4) study.best_params

A la fin, Optuna te donne :
	•	la meilleure config trouvée
	•	le meilleur score CV

⸻

Pourquoi c’est mieux que Random Search ?

Random Search
	•	teste au hasard
	•	ne “retient” pas vraiment ce qui marche
	•	plus tu veux de qualité, plus tu dois augmenter n_iter

Bayesian / Optuna
	•	apprend au fur et à mesure
	•	évite de perdre des essais sur des zones nulles
	•	converge plus vite vers une bonne zone

⸻

Hyperparamètres Optuna les plus importants (ce que TU peux régler)

1) n_trials (le plus important)
	•	nombre d’essais
	•	plus grand = meilleure chance de trouver mieux, mais plus long

Recommandation :
	•	rapide : 20–30
	•	correct : 40–80
	•	sérieux : 100–200

2) n_splits de K-Fold
	•	3 folds = plus rapide mais plus bruité
	•	5 folds = standard
	•	10 folds = très stable mais lourd

Bon compromis :
	•	tuning : 3 folds
	•	validation finale : 5 folds

3) Les bornes de recherche (très important)

Si tu donnes des bornes trop larges, Optuna explore trop.
Si tu donnes des bornes bien choisies, Optuna converge vite.

Exemple bon :
	•	learning_rate: 0.01 → 0.15 (log)
	•	max_depth: 3 → 10

4) Le type de sampling
	•	suggest_float(..., log=True) : très important pour des paramètres comme learning_rate, reg_lambda
car la bonne valeur peut être 0.01 ou 0.1 et il faut explorer en échelle logarithmique.

⸻

“Exploitation vs Exploration” (simple)

Optuna essaie de trouver un compromis :
	•	Exploitation : “j’ai vu que depth=6 marche bien, je reste autour”
	•	Exploration : “je ne sais pas si depth=9 peut faire mieux, j’essaie”

Si tu fais peu de n_trials, il explore moins, donc le résultat peut être moins bon.

⸻

Comment l’utiliser correctement dans ton cas (XGBoost/CatBoost + weights)

Étape 1 (tuning)
	•	KFold 3 splits
	•	n_trials=40
	•	objectif : RMSE CV

Étape 2 (validation finale)
	•	on prend best_params
	•	on entraîne sur tout le train (avec weights)
	•	on évalue sur test + SHAP

⸻

Pièges fréquents (à éviter)

1) Sur-tuning CV (overfit sur la CV)

Si tu fais trop d’essais, tu peux optimiser “trop” sur la CV et perdre en test.
Solution :
	•	garder un vrai test set jamais touché
	•	réduire l’espace de recherche
	•	limiter n_trials raisonnablement

2) Recherche trop large

Solution :
	•	resserrer les bornes après un premier run

3) Temps énorme

Solution :
	•	réduire n_trials, n_splits, iterations/n_estimators

⸻

Phrase simple pour slide

Bayesian Optimization (Optuna) : méthode qui teste des hyperparamètres de manière intelligente en apprenant des essais précédents, afin de trouver plus vite une bonne configuration qu’une recherche aléatoire ou exhaustive.

Si tu veux, je peux aussi te faire un mini schéma “Process Optuna” en 5 étapes pour la présentation.

```
