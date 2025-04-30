import pandas as pd
import numpy as np
from .similarity import cosine_similarity
import logging # Optionnel: pour afficher des messages d'information

def generate_recommendations(user_item_matrix, user_id, n_recommendations=10):
    """
    Générer des recommandations d'articles pour un utilisateur donné.
    
    Parameters:
    -----------
    user_item_matrix : DataFrame
        Matrice utilisateur-article
    user_id : int
        ID de l'utilisateur
    n_recommendations : int, optional (default=10)
        Nombre de recommandations à générer
    """
    user_vector = user_item_matrix.loc[user_id].values
    similarity_scores = user_item_matrix.dot(user_vector) / np.linalg.norm(user_item_matrix, axis=1)
    
    similar_users = similarity_scores.sort_values(ascending=False)
    recommendations = user_item_matrix.loc[similar_users.index].sum().sort_values(ascending=False)
    
    return recommendations.head(n_recommendations)

def recommend_based_on_user_attributes(df, user1_id, user2_id):
    """
    Recommander des articles à user2 basés sur les achats de user1,
    sous conditions de religion et pays identiques, et que user2 n'ait
    jamais acheté ces articles.

    Parameters:
    -----------
    df : DataFrame
        Le dataframe contenant les achats et les profils utilisateurs
        (doit inclure 'user_id', 'article_id', 'religion', 'pays').
    user1_id : int
        L'ID de l'utilisateur de référence.
    user2_id : int
        L'ID de l'utilisateur pour qui générer les recommandations.

    Returns:
    --------
    list
        Une liste d'article_id recommandés pour user2.
        Retourne une liste vide si les conditions ne sont pas remplies
        ou si aucune recommandation n'est possible.
    """
    user1_data = df[df['user_id'] == user1_id]
    user2_data = df[df['user_id'] == user2_id]

    if user1_data.empty or user2_data.empty:
        logging.warning(f"User1 ({user1_id}) ou User2 ({user2_id}) non trouvé dans les données.")
        return []

    # Extraire les profils (on prend la première ligne, supposant que le profil est constant)
    user1_profile = user1_data.iloc[0]
    user2_profile = user2_data.iloc[0]

    # Condition 1: Même religion et même pays
    if not (user1_profile['religion'] == user2_profile['religion'] and user1_profile['pays'] == user2_profile['pays']):
        logging.info(f"Condition de profil non remplie entre User {user1_id} et User {user2_id}.")
        return []

    # Condition 2 & 3: Articles achetés par user1 mais pas par user2
    articles_user1 = set(user1_data['article_id'].unique())
    articles_user2 = set(user2_data['article_id'].unique())

    recommendations = list(articles_user1 - articles_user2)
    return recommendations
