--- a/home/unikerp/Documents/recommandation_articles/README.md
+++ b/home/unikerp/Documents/recommandation_articles/README.md
@@ -1,31 +1,63 @@
-
-
 # 📰 Recommandation d'Articles
 
-Ce projet est une application de recommandation d'articles développée en Python.  
-Elle utilise des bibliothèques comme **NumPy** et **Pandas** pour le traitement des données.
+Ce projet implémente un système de recommandation d'articles simple en Python. Il propose deux approches de recommandation :
+
+1.  **Basée sur la Similarité Utilisateur :** Recommande des articles populaires auprès d'utilisateurs similaires (basée sur la similarité cosinus de leurs achats).
+2.  **Basée sur les Attributs et l'Historique :** Recommande à un utilisateur B des articles achetés par un utilisateur A, uniquement si A et B partagent les mêmes attributs (pays, religion) et si B n'a pas déjà acheté ces articles.
+
+Le projet utilise **Pandas** pour la manipulation des données, **NumPy** pour les calculs numériques, **Scikit-learn** pour le calcul de similarité et **Matplotlib** pour la visualisation.
+
+## ✨ Fonctionnalités
+
+*   Chargement des données d'achats depuis un fichier CSV.
+*   Nettoyage et prétraitement des données (gestion des dates, valeurs manquantes).
+*   Création d'une matrice utilisateur-article.
+*   Calcul de la similarité cosinus entre utilisateurs.
+*   Génération de recommandations basées sur la similarité (filtrant les articles déjà achetés).
+*   Génération de recommandations basées sur des règles d'attributs et l'historique d'achat.
+*   Visualisation de la distribution des achats par catégorie.
+*   Tests unitaires pour les fonctions de recommandation.
 
 ## 📁 Structure du projet
 
-L'application est organisée dans un dossier `src` et le fichier principal est `main.py`.
+```
+recommandation_articles/
+├── data/
+│   └── purchases.csv         # Fichier contenant les données d'achats et profils
+├── notebooks/                # (Optionnel) Pour les explorations et tests
+├── src/
+│   ├── __init__.py
+│   ├── data_processing.py    # Fonctions de chargement et nettoyage
+│   ├── main.py               # Point d'entrée principal du script
+│   ├── recommendation.py     # Fonctions de génération des recommandations
+│   ├── similarity.py         # (Si vous avez une fonction custom, sinon inutile)
+│   └── visualization.py      # Fonction de visualisation
+├── tests/
+│   ├── __init__.py           # Rend le dossier 'tests' découvrable
+│   └── test_recommendation.py # Tests unitaires pour les recommandations
+├── README.md                 # Ce fichier
+└── requirements.txt          # Dépendances du projet
+```
 
-## 🛠️ Installation et Execusion
+## ⚙️ Prérequis
 
-Avant de lancer l'application, assurez-vous d'avoir **Python 3** installé.  
-Ensuite, installez les dépendances nécessaires :
+*   Python 3.x
+*   pip (gestionnaire de paquets Python)
+
+## 🛠️ Installation
+
+1.  Clonez ce dépôt (si applicable) ou assurez-vous d'avoir tous les fichiers.
+2.  Naviguez jusqu'au répertoire racine du projet (`/home/unikerp/Documents/recommandation_articles`).
+3.  Installez les dépendances via le fichier `requirements.txt` :
 
 ```bash
-pip3 install numpy
-pip3 install pandas
+pip install -r requirements.txt
+```
 
-## 🛠️ Execusion
+## ▶️ Utilisation
 
 ```bash
 python3 -m src.main
+```
 
-## 🛠️ Test unitaire 
+Le script affichera dans la console :
+*   Les recommandations basées sur la similarité pour l'utilisateur configuré dans `main.py`.
+*   Les recommandations basées sur les attributs pour les utilisateurs configurés dans `main.py`.
+*   Les statistiques d'achat par catégorie.
+*   Une fenêtre affichant un graphique de la distribution des achats.
+
+## 🧪 Tests Unitaires
+
+Le projet inclut des tests unitaires pour vérifier le bon fonctionnement des fonctions de recommandation. Pour les exécuter, naviguez jusqu'à la racine du projet et lancez :
+
 ```bash
  python3 -m unittest discover tests
+```
 
+Si tous les tests passent, vous devriez voir un message indiquant `OK`.
+
+*(Note : Assurez-vous d'avoir créé un fichier `__init__.py` vide dans le répertoire `tests` pour que la découverte des tests fonctionne correctement.)*
+
+## 💾 Données
+
+Le script s'attend à trouver un fichier `data/purchases.csv` contenant au minimum les colonnes : `user_id`, `article_id`, `date_achat`, `prix`, `categorie`, `ville`, `pays`, `sexe`, `age`, `religion`.
