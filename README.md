```
	•	One-Hot Encoding : conversion des variables catégorielles à faible cardinalité en colonnes binaires.
	•	Frequency / Count Encoding : remplacement des catégories par leur fréquence d’apparition (utile pour les variables à forte cardinalité).
	•	Target Encoding (si utilisé) : encodage des catégories via la cible, avec précaution pour éviter la fuite de données.
	•	Missing Values Handling : gestion des valeurs manquantes / incohérentes (ex. conversion de certains 0 métier en NaN).
	•	Log Transformation (log1p) (si nécessaire) : réduction de l’asymétrie de certaines variables numériques.
	•	Data Scaling (si appliqué sur variables numériques) : mise à l’échelle des variables (ex. StandardScaler, MinMaxScaler, RobustScaler) pour harmoniser les ordres de grandeur.
	•	Train/Test Split : séparation des données avant les transformations apprises pour éviter le data leakage.
```
