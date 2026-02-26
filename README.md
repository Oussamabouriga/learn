```
Voici un tableau simple et clair (prêt à copier dans une slide) :

Méthode (English)	À quoi ça sert (mots simples)	Quand l’utiliser
Data Cleaning	Mettre les données “propres” : bons types, valeurs cohérentes, suppression/correction des erreurs	Toujours au début
Missing Values Handling	Gérer les valeurs manquantes : remplacer les valeurs vides et convertir certains “0 métier” en manquant	Quand il y a des champs vides ou des 0 qui ne veulent pas dire “0”
Train/Test Split	Séparer les données pour tester le modèle sur des cas jamais vus (éviter de tricher)	Avant d’apprendre des transformations ou d’entraîner le modèle
One-Hot Encoding	Transformer une catégorie en colonnes 0/1 (ex: Android → 1, iOS → 0)	Catégories peu nombreuses (faible cardinalité)
Frequency/Count Encoding	Remplacer une catégorie par sa fréquence (ex: “Google” apparaît 12%)	Catégories très nombreuses (forte cardinalité)
Target Encoding	Remplacer une catégorie par une valeur liée à la note (ex: moyenne de note pour “Google”)	Forte cardinalité + relation forte avec la note (attention à la fuite de données)
Log Transformation (log1p)	Réduire l’effet des très grandes valeurs (ex: délais très élevés)	Variables très asymétriques (beaucoup de petits, quelques très grands)
Data Scaling (StandardScaler / MinMaxScaler / RobustScaler)	Mettre les variables sur une même échelle (éviter qu’une variable domine juste parce qu’elle est grande)	Utile pour modèles sensibles à l’échelle; souvent optionnel pour XGBoost/CatBoost

Si tu veux, je peux faire le même tableau pour les métriques (MAE, RMSE, R², Accuracy@±1) avec “comment lire” + “bon/mauvais”.

```
