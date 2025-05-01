import pandas as pd
from src.data_processing import load_data, clean_data, create_user_item_matrix, load_data_for_surprise
# Importation des fonctions de recommandation existantes et nouvelles
from src.recommendation import (
    generate_recommendations,
    recommend_based_on_user_attributes,
    train_svd_model,
    generate_svd_recommendations
)
from src.evaluation import evaluate_model # Importation de la fonction d'évaluation
from src.visualization import plot_purchase_distribution
import logging # Optionnel
from surprise import SVD # Importation de l'algorithme SVD
import pickle # Pour sauvegarder/charger le modèle

# Configuration simple du logging (optionnel)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. Chargement et nettoyage des données
    # Assurez-vous que 'purchases.csv' contient 'user_id', 'article_id', 'prix', 'date_achat', 'categorie', 'religion', 'pays'
    # Ou chargez et fusionnez un fichier de profil utilisateur ici.
    logging.info("Chargement et nettoyage des données...")
    file_path = 'data/purchases.csv' # Chemin vers le fichier de données
    df = load_data(file_path)
    if df is None:
        logging.error("Échec du chargement des données initiales. Arrêt du script.")
        return
    df = clean_data(df)
    logging.info("Données chargées et nettoyées.")

    # 2. Création de la matrice utilisateur-article
    # (Toujours utile pour la méthode basée sur la similarité)
    logging.info("Création de la matrice utilisateur-article...")
    user_item_matrix = create_user_item_matrix(df)
    logging.info("Matrice utilisateur-article créée.")
    
    # --- Recommandation basée sur la similarité (Ancienne méthode) ---
    logging.info("Génération des recommandations basées sur la similarité...")
    user_id_similarity = 1 # Utilisateur pour la recommandation par similarité
    n_recommendations = 3  # Nombre de recommandations à générer
    recommendations_similarity = generate_recommendations(user_item_matrix, user_id_similarity, n_recommendations)

    print(f"\n--- Recommandations basées sur la similarité pour l'utilisateur {user_id_similarity} ---")
    print("----------------------------------------")
    print(recommendations_similarity)

    # --- Recommandation basée sur les attributs et achats (Nouvelle méthode) ---
    logging.info("Génération des recommandations basées sur les attributs utilisateurs...")
    user1_id_ref = 1  # Utilisateur de référence (France, Catholique)
    user2_id_target = 11 # Utilisateur cible (France, Catholique)
    recommendations_attributes = recommend_based_on_user_attributes(df, user1_id_ref, user2_id_target)

    print(f"\n--- Recommandations pour l'utilisateur {user2_id_target} basées sur l'utilisateur {user1_id_ref} (même religion/pays) ---")
    print("--------------------------------------------------------------------------------")
    if recommendations_attributes:
        print(recommendations_attributes)
    else:
        print("Aucune recommandation possible selon les critères définis (profils différents ou articles déjà achetés).")

    # --- Évaluation et Recommandation avec Surprise/SVD (Nouvelle méthode) ---
    logging.info("Préparation des données pour Surprise...")
    data_surprise = load_data_for_surprise(file_path)

    if data_surprise:
        # 3.a Évaluation du modèle SVD
        logging.info("Évaluation du modèle SVD...")
        svd_model_instance = SVD() # Créer une instance pour l'évaluation
        avg_precision, avg_recall = evaluate_model(data_surprise, svd_model_instance, k=10)
        print(f"\n--- Évaluation du modèle SVD (Cross-Validation) ---")
        print("-------------------------------------------------")
        print(f"Precision@10 moyenne: {avg_precision:.4f}")
        print(f"Recall@10 moyenne:    {avg_recall:.4f}")

        # 3.b Entraînement du modèle SVD sur l'ensemble des données pour la prédiction
        logging.info("Entraînement du modèle SVD final sur toutes les données...")
        svd_model_final = train_svd_model(data_surprise)

        if svd_model_final:
            # --- Sauvegarde du modèle SVD entraîné ---
            model_filename = 'svd_model.pkl'
            logging.info(f"Sauvegarde du modèle SVD final dans {model_filename}...")
            with open(model_filename, 'wb') as f:
                pickle.dump(svd_model_final, f)
            logging.info("Modèle sauvegardé.")
            # -----------------------------------------
            # 3.c Génération de recommandations SVD pour un utilisateur spécifique
            user_id_svd = 1 # Utilisateur pour la recommandation SVD
            n_reco_svd = 5 # Nombre de recommandations SVD à générer

            # Obtenir la liste de tous les articles uniques
            all_items = df['article_id'].unique()
            # Obtenir les articles déjà achetés par l'utilisateur cible
            items_bought = df[df['user_id'] == user_id_svd]['article_id'].unique()
            # Créer la liste des articles à prédire (tous sauf ceux déjà achetés)
            items_to_predict = [item for item in all_items if item not in items_bought]

            logging.info(f"Génération des recommandations SVD pour l'utilisateur {user_id_svd}...")
            recommendations_svd = generate_svd_recommendations(svd_model_final, user_id_svd, items_to_predict, n_reco_svd)

            print(f"\n--- Recommandations SVD pour l'utilisateur {user_id_svd} ---")
            print("-----------------------------------------")
            print(recommendations_svd)
        else:
            logging.error("Échec de l'entraînement du modèle SVD final.")
    else:
        logging.error("Échec du chargement des données pour Surprise.")

    print("\nStatistiques d'achat :")
    print("----------------------------------------")
    print(df.groupby('categorie')['article_id'].count())
    
    # 4. Visualisation des résultats
    logging.info("Affichage des visualisations...")
    plot_purchase_distribution(df)
    logging.info("Script terminé.")

if __name__ == "__main__":
    main()
