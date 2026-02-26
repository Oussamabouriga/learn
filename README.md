```
## Data transformation methods (cheat sheet)

| Method (English) | À quoi ça sert (mots simples) | Quand l’utiliser |
|---|---|---|
| **Data Cleaning** | Mettre les données “propres” : bons types, valeurs cohérentes, suppression/correction des erreurs | Toujours au début |
| **Missing Values Handling** | Gérer les valeurs manquantes : remplacer les valeurs vides et convertir certains “0 métier” en manquant | Quand il y a des champs vides ou des 0 qui ne veulent pas dire “0” |
| **Train/Test Split** | Séparer les données pour tester le modèle sur des cas jamais vus (éviter de tricher) | Avant d’apprendre des transformations ou d’entraîner le modèle |
| **One-Hot Encoding** | Transformer une catégorie en colonnes 0/1 (ex: Android → 1, iOS → 0) | Catégories peu nombreuses (faible cardinalité) |
| **Frequency/Count Encoding** | Remplacer une catégorie par sa fréquence (ex: “Google” apparaît 12%) | Catégories très nombreuses (forte cardinalité) |
| **Target Encoding** | Remplacer une catégorie par une valeur liée à la note (ex: moyenne de note pour “Google”) | Forte cardinalité + relation forte avec la note (attention à la fuite de données) |
| **Log Transformation (`log1p`)** | Réduire l’effet des très grandes valeurs (ex: délais très élevés) | Variables très asymétriques (beaucoup de petits, quelques très grands) |
| **Data Scaling (StandardScaler / MinMaxScaler / RobustScaler)** | Mettre les variables sur une même échelle (éviter qu’une variable domine juste parce qu’elle est grande) | Utile pour modèles sensibles à l’échelle; souvent optionnel pour XGBoost/CatBoost |

---

## Metrics (how to read them)

| Metric (English) | À quoi ça sert (mots simples) | Comment la lire | Bon / Mauvais |
|---|---|---|---|
| **MAE (Mean Absolute Error)** | Erreur moyenne en points de note | “En moyenne, je me trompe de **X** points” | Bon si faible / Mauvais si élevé |
| **RMSE (Root Mean Squared Error)** | Mesure l’erreur en donnant plus de poids aux grosses erreurs | Si RMSE >> MAE → le modèle fait parfois de grosses erreurs | Bon si faible / Mauvais si élevé |
| **R² (R-squared)** | Indique si le modèle explique vraiment la note | 1 = parfait, 0 = comme prédire la moyenne, <0 = pire que la moyenne | Bon si proche de 1 / Mauvais si proche de 0 ou <0 |
| **Accuracy@±t (Tolerance Accuracy)** | % de prédictions “assez proches” de la vraie note | “X% des prédictions sont à moins de **t** point(s)” (ex: t=1) | Bon si élevé / Mauvais si faible |
| **MedianAE (Median Absolute Error)** | Erreur typique (moins sensible aux valeurs extrêmes) | “La moitié des erreurs sont ≤ **X** points” | Bon si faible / Mauvais si élevé |
| **MaxError (Maximum Error)** | La pire erreur faite par le modèle | “Dans le pire cas, je me trompe de **X** points” | Bon si faible / Mauvais si très élevé (dépend d’un seul cas) |
| **MAPE / sMAPE** | Erreur en pourcentage | Utile si la cible n’est jamais proche de 0 | Bon si faible, mais à éviter si la note peut être 0 (instable) |

**Phrase courte pour comparer les modèles :**  
Pour une note 0–10, on compare surtout **MAE + RMSE + Accuracy@±1**, et on utilise **R²** comme indicateur global.
```
