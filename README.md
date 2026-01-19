# Artificial Intelligence (AI) – Algorithm Overview

## Artificial Intelligence (AI)

### Machine Learning (ML)

#### Supervised Learning

##### Classification (Discrete Labels)

- **Linear Models**
  - Logistic Regression  
  - Linear Discriminant Analysis (LDA)

- **Tree-Based Models**
  - Decision Tree  
  - Random Forest  
  - XGBoost  
  - LightGBM  
  - CatBoost  

- **Distance-Based**
  - K-Nearest Neighbors (KNN)

- **Margin-Based**
  - Support Vector Machine (SVM)

- **Probabilistic**
  - Naive Bayes

##### Regression (Continuous Values)

- Linear Regression  
- Ridge / Lasso Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- XGBoost Regressor  
- LightGBM Regressor  
- CatBoost Regressor  

---

#### Unsupervised Learning

##### Clustering
- K-Means  
- DBSCAN  
- Hierarchical Clustering  

##### Dimensionality Reduction
- PCA (Principal Component Analysis)  
- t-SNE  
- UMAP  

##### Anomaly Detection
- Isolation Forest  
- One-Class SVM  

---

#### Reinforcement Learning

- Q-Learning  
- Deep Q-Network (DQN)  
- Policy Gradient  

---

### Deep Learning (DL)

- Feedforward Neural Networks (MLP)  
- Convolutional Neural Networks (CNN – Images)  
- Recurrent Neural Networks (RNN / LSTM – Sequences)  
- Transformers  
- Tabular Deep Learning
  - TabNet  
  - NODE (Neural Oblivious Decision Trees)



Bien noté — sans émojis.

Voici la version propre et professionnelle du plan en français.


---

Plan de Projet ML sur 7 Jours

Prédiction de la satisfaction client (0–10) + Explainable AI

Objectif
Prédire la satisfaction client à partir de données structurées et expliquer chaque prédiction avec des méthodes d’Explainable AI.


---

Jour 1 – Compréhension et nettoyage des données

Tâches

Charger le jeu de données

Comprendre la signification de chaque variable

Identifier les valeurs manquantes, les outliers et les erreurs de format

Corriger les problèmes de qualité des données


Livrables

Jeu de données propre

Dictionnaire des variables



---

Jour 2 – Analyse exploratoire (EDA)

Tâches

Étudier la distribution de la satisfaction (0 à 10)

Visualiser les variables principales

Calculer les corrélations

Calculer l’information mutuelle


Objectif
Identifier les variables fortement et faiblement liées à la satisfaction.

Livrables

Liste des variables influentes

Premiers insights analytiques



---

Jour 3 – Feature engineering

Tâches

Encoder les variables catégorielles

Créer de nouvelles variables pertinentes

Supprimer les variables inutiles (ID, bruit, redondances)


Livrables

Jeu de données prêt pour la modélisation



---

Jour 4 – Modèles de base (baseline)

Tâches

Entraîner une régression linéaire

Entraîner un Random Forest

Évaluer les performances avec MAE et RMSE


Objectif
Obtenir une référence de performance pour comparer les modèles avancés.

Livrables

Résultats de base

Modèle simple fonctionnel



---

Jour 5 – Modèle avancé (Boosting)

Tâches

Entraîner CatBoost, XGBoost ou LightGBM

Ajuster les principaux hyperparamètres

Comparer les performances avec la baseline


Livrables

Meilleur modèle sélectionné

Performance optimisée



---

Jour 6 – Explainable AI (SHAP)

Tâches

Calculer les valeurs SHAP

Analyser l’importance globale des variables

Expliquer des prédictions individuelles

Interpréter l’impact des features


Livrables

Modèle explicable

Compréhension claire des décisions



---

Jour 7 – Rapport et présentation

Tâches

Synthétiser les résultats

Rédiger un rapport clair

Préparer une présentation

Formuler des recommandations business


Livrables

Projet finalisé

Support de présentation prêt



---

1. Data Understanding & Cleaning

Missing Value Handling
Replace missing values using mean, median, mode, or special labels. This prevents models from failing or learning wrong patterns.

Outlier Detection
Identify extreme values using boxplots or Z-scores. Outliers can distort model learning and reduce accuracy.

Data Type Correction
Convert columns to correct formats (numeric, categorical, datetime). This ensures algorithms interpret the data properly.


---

2. Exploratory Data Analysis (EDA)

Distribution Analysis
Visualize the target (0–10) with histograms or bar charts. This shows whether satisfaction is balanced or biased.

Correlation Analysis
Measure linear relationships between features and the target. Strong correlations suggest important variables.

Group Comparisons
Compare satisfaction across categories (e.g., service type, region). This reveals which groups are more or less satisfied.


---

3. Feature Importance (Before Training)

Mutual Information
Measures any dependency between features and the target, even non-linear. It ranks features by how informative they are.

ANOVA / Statistical Tests
Tests whether satisfaction differs significantly between groups. Useful for categorical variables.

Variance Threshold
Removes features with almost no variation. Such features carry little information.

Redundancy Analysis
Finds highly correlated features. Removing duplicates simplifies the model.


---

4. Feature Engineering

Categorical Encoding
Convert text categories into numbers (One-Hot, Target Encoding). Models need numeric inputs to learn.

Feature Creation
Create new variables from existing ones. This can reveal hidden patterns.

Feature Scaling
Normalize numeric features. Some models perform better with scaled inputs.


---

5. Baseline Models

Linear Regression
Models a simple linear relationship between features and satisfaction. It provides a fast reference performance.

Decision Tree
Splits data into rules based on feature values. Easy to interpret but can overfit.

Random Forest
Combines many trees for better stability. More accurate than a single tree.


---

6. Advanced Models (Boosting)

XGBoost
Uses sequential trees to correct previous errors. Very accurate for structured data.

LightGBM
Optimized for speed and large datasets. Efficient and scalable.

CatBoost
Handles categorical features automatically. Requires less preprocessing.


---

7. Model Evaluation

MAE (Mean Absolute Error)
Measures average prediction error in satisfaction points. Easy to interpret.

RMSE (Root Mean Squared Error)
Penalizes large errors more strongly. Highlights big mistakes.

R² Score
Measures how well the model explains the data. Higher is better.

Cross-Validation
Tests the model on multiple data splits. Ensures stability and reliability.


---

8. Explainable AI (After Training)

SHAP Values
Explain how each feature influences a prediction. Shows both global and local effects.

Permutation Importance
Shuffles features to see performance drop. Identifies truly important variables.

Partial Dependence Plots (PDP)
Shows how satisfaction changes when a feature changes. Helps with business interpretation.

ICE Plots
Shows feature effects per individual. Reveals different behavior patterns.


---

9. Error Analysis

Residual Analysis
Studies prediction errors. Helps detect systematic mistakes.

Segment Analysis
Checks performance across groups. Identifies bias or weak areas.


---

10. Business Interpretation

Impact Translation
Convert feature effects into business insights. Makes results actionable.

Recommendation Formulation
Suggest actions based on model findings. Helps decision-makers.


---

11. Final Reporting

Technical Report
Documents methods, results, and metrics. Ensures reproducibility.

Presentation
Summarizes key insights visually. Communicates results clearly.


---

Final Summary

This pipeline gives you:

Strong feature analysis

Reliable models

Transparent explanations

Business-ready insights


