```

	•	1) Transformation des données : Préparation des données (nettoyage, encodage, transformations, séparation train/test, etc.) pour obtenir un jeu exploitable par les modèles.
	•	2) Modèle de base : Entraînement d’un premier XGBoost Regressor avec validation croisée K-Fold pour établir une référence de performance.
	•	3) Modèle ajusté pour déséquilibre : Entraînement d’un modèle avec ajustement du déséquilibre de la cible (poids des exemples) et évaluation avec K-Fold pour mieux prendre en compte les cas rares.
	•	4) Optimisation par Random Search : Recherche aléatoire d’hyperparamètres avec validation croisée K-Fold pour identifier rapidement de meilleures configurations.
	•	5) Optimisation par Grid Search : Recherche ciblée et systématique autour des meilleurs paramètres (issus du Random Search) avec K-Fold pour affiner le modèle final.

```
