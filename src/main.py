import pandas as pd
from src.data_processing import load_data, clean_data, create_user_item_matrix
from src.recommendation import generate_recommendations
from src.visualization import plot_purchase_distribution

def main():
    # 1. Chargement et nettoyage des données
    df = load_data('data/purchases.csv')
    df = clean_data(df)
    
    # 2. Création de la matrice utilisateur-article
    user_item_matrix = create_user_item_matrix(df)
    
    # 3. Génération des recommandations pour un utilisateur spécifique
    user_id = 1
    n_recommendations = 3  # Nombre de recommandations à générer
    recommendations = generate_recommendations(user_item_matrix, user_id, n_recommendations)
    
    print(f"\nRecommandations pour l'utilisateur {user_id} :")
    print("----------------------------------------")
    print(recommendations)
    print("\nStatistiques d'achat :")
    print("----------------------------------------")
    print(df.groupby('categorie')['article_id'].count())
    
    # 4. Visualisation des résultats
    plot_purchase_distribution(df)

if __name__ == "__main__":
    main()
