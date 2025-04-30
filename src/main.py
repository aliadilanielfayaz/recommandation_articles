import pandas as pd
from src.data_processing import load_data, clean_data, create_user_item_matrix
# Importation des deux fonctions de recommandation
from src.recommendation import generate_recommendations, recommend_based_on_user_attributes
from src.visualization import plot_purchase_distribution
import logging # Optionnel

# Configuration simple du logging (optionnel)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. Chargement et nettoyage des données
    # Assurez-vous que 'purchases.csv' contient 'user_id', 'article_id', 'prix', 'date_achat', 'categorie', 'religion', 'pays'
    # Ou chargez et fusionnez un fichier de profil utilisateur ici.
    logging.info("Chargement et nettoyage des données...")
    df = load_data('data/purchases.csv')
    df = clean_data(df)
    logging.info("Données chargées et nettoyées.")
    
    # 2. Création de la matrice utilisateur-article
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

    print("\nStatistiques d'achat :")
    print("----------------------------------------")
    print(df.groupby('categorie')['article_id'].count())
    
    # 4. Visualisation des résultats
    logging.info("Affichage des visualisations...")
    plot_purchase_distribution(df)
    logging.info("Script terminé.")

if __name__ == "__main__":
    main()
