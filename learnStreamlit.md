
# 🧩 Rapport Technique

## Gestion optimisée des données dans un tableau de bord Streamlit avec cache et détection de changements

---

### 🎯 **Objectif du système**

L’objectif est d’optimiser un tableau de bord **Streamlit** connecté à une base de données **PostgreSQL**, afin de :

* **Éviter les requêtes inutiles** vers la base de données à chaque rafraîchissement de l’application ;
* **Mettre à jour automatiquement** les données du tableau de bord **uniquement** lorsqu’un changement est détecté dans la base ;
* **Améliorer les performances** grâce au **cache** et à la **gestion d’état** (`st.cache_data` et `st.session_state`).

---

## 🧠 1. Les trois composantes principales

1. **Le cache Streamlit (`st.cache_data`)**

   * Permet d’enregistrer en mémoire locale (ou sur disque) le résultat d’une fonction qui interroge la base de données.
   * Évite d’exécuter la requête complète tant que les paramètres ou l’état ne changent pas.
   * Idéal pour manipuler des volumes de données importants.

2. **La gestion d’état (`st.session_state`)**

   * Conserve des variables entre les exécutions de l’application (par exemple : les dernières données chargées, le dernier horodatage de modification, etc.).
   * Garantit la cohérence des données affichées sans recharger inutilement la base.

3. **La détection de changements**

   * Mécanisme qui identifie si la base de données a été modifiée depuis le dernier chargement.
   * Deux approches sont possibles :

     * Par **Checksum (MAX(updated_at))** ;
     * Par **LISTEN/NOTIFY**.

---

## ⚙️ 2. Méthode 1 : Détection par *Checksum (MAX(updated_at))*

### Principe

Cette méthode repose sur l’existence d’une colonne `updated_at` dans la table PostgreSQL.
Chaque enregistrement possède un horodatage de sa dernière modification.

Une fonction Python (`get_checksum()` par exemple) exécute une petite requête :

> *« Quelle est la date de dernière modification dans cette table ? »*

Cette date est ensuite transformée en **empreinte (checksum)**.
Lorsqu’on relance l’application :

* Si le nouveau checksum est identique à celui stocké dans `st.session_state` → aucune modification → le cache est réutilisé.
* Si le checksum est différent → les données ont changé → la fonction de récupération (`get_data()`) recharge les nouvelles valeurs depuis la base et met à jour le cache.

### Avantages

* Très simple à mettre en place ;
* Ne nécessite aucun droit particulier sur la base de données (lecture seule suffisante) ;
* Très rapide même sur des millions de lignes, car seule une petite requête est exécutée.

### Inconvénients

* Nécessite la présence d’une colonne `updated_at` mise à jour automatiquement via un trigger SQL ;
* La détection se fait à intervalles réguliers ou lors des rafraîchissements de l’application (pas en temps réel).

---

## ⚙️ 3. Méthode 2 : Détection par *LISTEN / NOTIFY*

### Principe

Cette approche utilise un **mécanisme natif de PostgreSQL** permettant à la base de **notifier directement l’application** lorsqu’un changement se produit.

* Une **fonction et un trigger** sont créés dans la base : à chaque `INSERT`, `UPDATE` ou `DELETE`, PostgreSQL exécute un `NOTIFY`.
* L’application Streamlit (ou un script Python parallèle) ouvre une connexion en mode “écoute” via la commande `LISTEN`.
* Dès qu’une notification arrive, elle indique qu’une modification vient d’avoir lieu : l’application peut alors vider son cache et recharger les données fraîches.

### Avantages

* Détection **instantanée et automatique** des changements ;
* Aucun besoin de colonne `updated_at` ;
* Parfait pour des tableaux de bord en **temps réel**.

### Inconvénients

* Nécessite des **droits de création de trigger** dans la base (rôle administrateur ou propriétaire de la table) ;
* Mise en œuvre plus complexe (gestion d’un thread d’écoute asynchrone dans l’application).

---

## 🧩 4. Interaction avec le cache et la gestion d’état

Quelle que soit la méthode choisie, le flux logique reste le même :

1. **Initialisation**

   * Lors du premier lancement, la fonction `get_data()` interroge la base complète et enregistre les résultats en cache (`st.cache_data`).
   * Le checksum (ou signal LISTEN) est sauvegardé dans `st.session_state`.

2. **Vérification à chaque rafraîchissement**

   * La fonction `get_checksum()` (ou le listener PostgreSQL) compare l’état actuel avec celui stocké dans `st.session_state`.

3. **Décision**

   * Si aucun changement → les données du cache sont utilisées instantanément.
   * Si changement détecté → le cache est invalidé, la base est relue, et `st.session_state` est mis à jour.

4. **Affichage**

   * L’application Streamlit affiche toujours les données les plus récentes, sans surcharge inutile sur la base de données.

---

## 📊 5. Synthèse comparative

| Critère                         | Méthode Checksum (MAX(updated_at))                                 | Méthode LISTEN / NOTIFY                     |
| ------------------------------- | ------------------------------------------------------------------ | ------------------------------------------- |
| **Principe**                    | Vérification périodique d’un horodatage maximal                    | Notification directe envoyée par PostgreSQL |
| **Besoins techniques**          | Colonne `updated_at` + trigger SQL simple                          | Fonction + trigger `NOTIFY` dans la base    |
| **Permissions nécessaires**     | Lecture seule                                                      | Droit de création de trigger                |
| **Complexité de mise en place** | Faible                                                             | Moyenne à élevée                            |
| **Performance**                 | Excellente (requête légère)                                        | Instantanée (événement en temps réel)       |
| **Cas d’usage idéal**           | Tableaux de bord périodiques (rafraîchissement manuel ou planifié) | Tableaux de bord dynamiques en temps réel   |

---

## 🚀 6. Conclusion

L’intégration du **cache** et de la **gestion d’état** dans Streamlit permet de créer des tableaux de bord **performants et réactifs**, même sur de grands volumes de données.
Le choix de la méthode dépend du niveau de contrôle sur la base et du besoin en réactivité :

* **Checksum (`MAX(updated_at)`)** : solution simple, universelle, sans privilèges spéciaux — idéale pour les environnements en lecture seule.
* **LISTEN / NOTIFY** : solution plus avancée, en temps réel, adaptée aux systèmes maîtrisant la base PostgreSQL et disposant des droits nécessaires.

Dans les deux cas, l’application gagne en efficacité :
👉 Moins d’appels à la base, moins de latence, et une expérience utilisateur fluide et optimisée.

