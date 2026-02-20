```

XGBoost (régression)
	•	Prédit une note de satisfaction en apprenant étape par étape à réduire ses erreurs. (arbres + boosting, minimise une loss)
	•	Fonctionne bien même si certaines informations manquent. (gestion des valeurs manquantes / imputation simple)
	•	On vérifie qu’il est fiable et on comprend ses décisions. (cross-validation, SHAP)

CatBoost (régression)
	•	Même idée : prédire la note et s’améliorer progressivement avec ses erreurs. (boosting d’arbres, loss)
	•	Très pratique quand on a beaucoup de champs “texte/catégorie” (type, statut, canal…). (cat features, encodage intégré)
	•	On peut aussi le tester proprement et expliquer ce qui influence la note. (cross-validation, importance/contributions)

```
