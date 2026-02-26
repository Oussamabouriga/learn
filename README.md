```
1) Métriques (comment les lire, quand c’est bon/mauvais, et comment comparer)

MAE — Mean Absolute Error
	•	Ce que ça mesure : l’erreur moyenne en points de note (même unité que 0–10).
	•	Lecture : “en moyenne, je me trompe de X points”.
	•	Bon / mauvais :
	•	Bon si MAE faible (ex: 0.8 meilleur que 1.6).
	•	Mauvais si MAE élevé.
	•	Pour comparer : excellent metric principale car très interprétable.

RMSE — Root Mean Squared Error
	•	Ce que ça mesure : erreur en points aussi, mais pénalise fortement les grosses erreurs.
	•	Lecture : si RMSE >> MAE, ça veut dire que le modèle fait parfois de grosses erreurs.
	•	Bon / mauvais :
	•	Bon si faible.
	•	Mauvais si élevé, surtout s’il monte alors que MAE reste stable.
	•	Pour comparer : utile pour juger la stabilité et éviter un modèle qui fait quelques erreurs graves.

R² — R-squared (coefficient de détermination)
	•	Ce que ça mesure : proportion de la variance de la cible expliquée par le modèle.
	•	Lecture simple :
	•	1.0 = parfait
	•	0.0 = pas mieux que prédire la moyenne
	•	< 0 = pire que prédire la moyenne
	•	Bon / mauvais :
	•	Bon si proche de 1
	•	Mauvais si proche de 0 ou négatif
	•	Pour comparer : utile comme indicateur global, mais à lire avec MAE/RMSE.

MedianAE — Median Absolute Error
	•	Ce que ça mesure : l’erreur “typique” (médiane), moins sensible aux outliers.
	•	Lecture : “la moitié des prédictions ont une erreur ≤ X”.
	•	Bon / mauvais :
	•	Bon si faible
	•	Pour comparer : utile si tu suspectes des valeurs extrêmes.

MaxError — Maximum Error
	•	Ce que ça mesure : la pire erreur observée.
	•	Lecture : “dans le pire cas, je me suis trompé de X points”.
	•	Bon / mauvais :
	•	Bon si faible, mais peut être instable (1 seul cas peut tout changer).
	•	Pour comparer : utile si tu veux limiter les pires cas, mais pas suffisant seul.

Explained Variance — Explained Variance Score
	•	Ce que ça mesure : proche de R², mesure la part de variance captée.
	•	Lecture : plus proche de 1 = mieux.
	•	Pour comparer : similaire à R², souvent redondant.

MSE — Mean Squared Error
	•	Ce que ça mesure : erreur quadratique moyenne (pas en unité “points”).
	•	Lecture : difficile à interpréter directement, préférer RMSE.
	•	Pour comparer : utile pour optimisation mais moins lisible.

MAPE — Mean Absolute Percentage Error
	•	Ce que ça mesure : erreur en % (|y - ŷ| / y).
	•	Problème : si y = 0 (possible sur 0–10), MAPE devient énorme ou instable.
	•	Conclusion : à éviter si la cible peut être 0.

sMAPE — symmetric MAPE
	•	Ce que ça mesure : % symétrique, plus stable que MAPE.
	•	Bon / mauvais : plus faible = mieux.
	•	Pour comparer : utile, mais MAE/RMSE restent souvent plus fiables sur note 0–10.

Accuracy@±t — tolerance accuracy (régression)
	•	Méthode : % de prédictions avec |y - ŷ| ≤ t (ex: t=0.5 ou 1).
	•	Lecture : “X% des prédictions sont à moins de 1 point”.
	•	Bon / mauvais :
	•	Bon si élevé (ex: 70% > 50%)
	•	Pour comparer : metric très “métier”, parfaite pour expliquer aux non-tech.

⸻

Comment comparer des modèles correctement

Tu compares toujours sur le même test set :
	•		1.	Regarde MAE (erreur moyenne compréhensible)
	•		2.	Regarde RMSE (grosses erreurs)
	•		3.	Regarde Accuracy@±1 (utilisable métier)
	•		4.	Utilise R² comme soutien (pas seul)
	•		5.	Compare Train vs Test :
	•	train beaucoup meilleur que test → surapprentissage possible

⸻

2) Méthodes de transformation des données (nom en anglais + explication + quand utiliser)

Data Cleaning
	•	But : corriger données incohérentes, types incorrects, valeurs manquantes.
	•	Quand : toujours en premier.

Missing Values Handling
	•	But : gérer les valeurs manquantes (np.nan).
	•	Exemple : convertir des “0 métier” en NaN si 0 signifie “non renseigné”.
	•	Quand : quand tu sais que 0 ne représente pas une vraie valeur.

Train/Test Split
	•	But : séparer données pour évaluer la généralisation.
	•	Important : le split doit être fait avant les transformations “apprises” (target encoding, scaling appris, etc.).

⸻

One-Hot Encoding
	•	But : convertir une catégorie en colonnes binaires.
	•	Quand : variables catégorielles avec peu de catégories (faible cardinalité).
	•	Avantage : simple, robuste.
	•	Limite : explose le nombre de colonnes si trop de catégories.

Frequency / Count Encoding
	•	But : remplacer chaque catégorie par sa fréquence d’apparition.
	•	Quand : variables à forte cardinalité (ex: modèle téléphone, marque).
	•	Avantage : compact, rapide.
	•	Limite : perd l’information “qualitative” (deux catégories différentes peuvent avoir même fréquence).

Target Encoding
	•	But : remplacer une catégorie par une statistique liée à la cible (ex: moyenne de la note par catégorie).
	•	Quand : forte cardinalité, mais il faut une relation “catégorie → cible”.
	•	Risque : data leakage si mal fait.
	•	Règle : calculer sur train uniquement, idéalement avec CV/regularization.

⸻

Log Transformation (log1p)
	•	But : réduire l’effet des grosses valeurs (skewness), stabiliser la variance.
	•	Utile pour : variables comme delai, montant, ancienneté très asymétriques.
	•	Quand : si distribution très “longue” (beaucoup petits, quelques très grands).
	•	log1p permet log(0+1) donc safe si valeurs ≥ 0.

⸻

Data Scaling (pour variables numériques)

Important : pour XGBoost / CatBoost, le scaling n’est souvent pas obligatoire, mais il peut aider selon les features.

StandardScaler
	•	Met moyenne=0, écart-type=1.
	•	Quand : modèles sensibles à l’échelle (linéaires, SVM, KNN).
	•	Pour XGBoost/CatBoost : pas essentiel.

MinMaxScaler
	•	Met les valeurs dans [0,1].
	•	Quand : réseaux de neurones / KNN / méthodes distance.
	•	Pas indispensable pour XGBoost/CatBoost.

RobustScaler
	•	Utilise médiane et IQR (robuste aux outliers).
	•	Quand : beaucoup d’outliers.

⸻

“Scaling delai” (cas spécifique)

Pour les variables de type delai très larges (0 → 5000 minutes / ou plus) :
	•	Option 1 : Log Transformation (log1p) (souvent le meilleur)
	•	Option 2 : Clipping / Winsorizing (couper les extrêmes)
	•	Option 3 : Binning (transformer en classes d’intervalles) si tu veux une logique “tranches”.

⸻

3) Méthode utilisée pour le déséquilibre de la cible (nom anglais + comment ça marche)

Sample Weighting for Imbalanced Regression

(ou : Target-Bin Inverse Frequency Weighting)
	•	Problème : la cible est concentrée (ex: beaucoup de 10), donc le modèle apprend surtout cette zone.
	•	Méthode :
	1.	On découpe la cible en tranches (bins)
	2.	On calcule la fréquence de chaque tranche
	3.	On donne un poids = 1 / fréquence
	4.	On entraîne le modèle avec sample_weight (XGBoost) ou Pool(weight=...) (CatBoost)
	•	Effet : les tranches rares comptent plus dans la loss pendant l’apprentissage.

Quand l’utiliser :
	•	quand tu veux mieux traiter des cas rares (ex: mauvaises notes)
	•	quand le modèle “prédit trop souvent la moyenne / les notes hautes”

Attention :
	•	si les poids sont trop agressifs, la performance globale peut baisser
	•	il faut contrôler n_bins et clipper les poids

⸻

Si tu veux, je peux te transformer tout ça en slides ultra courtes (1 ligne par métrique + 1 ligne par méthode) pour que ce soit directement utilisable en présentation.

```
