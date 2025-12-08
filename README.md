# Configuration des fichiers
Les datasets de MIMIC ne sont pas loadés sur GitHub compte tenu de leur taille énorme.
Le fichier **Projet.ipynb** contient l'essentiel du workflow:
    - Dataframes bruts et dataframe final créés avec des queries de MIMIC via DuckDB
    - Visualisation de la distribution des données
    - Nettoyage des données 
        - Visualisation des valeurs manquantes et imputation multivariée
        - Visualisation des valeurs aberrantes et élimination des valeurs physiologiquement impossibles
    - Séparation du df_final en train/val/test
    - Entraînement des modèles:
        - Logistic Regression
        - RandomForest
    - Tuning des hyperparamètres et sélection des features les plus importantes avec 5-fold CV
    - Évaluation sur données test

Le fichier **table1.ipynb** contient les queries et un brouillon de la table des caractéristiques des patients utilisés dans le rapport.

Les **fichiers python** suivants contiennent les fonctions qui supportent la réalisation de chacun des étapes ci-dessus. Elles sont importées au fur et à mesure dans le notebook:
    - nettoyage_donnees.py (à noter que la majorité des fonctions pandas pour créer des df ne sont pas utilisées au final: elles ont été écrites mais par la suite on a réalisé que le traitement des datasets énormes comme MIMIC avec pandas fait crasher l'ordi.. (-_-)' donc switch à DuckDB)
    - models.py
    - feature_selection.py

