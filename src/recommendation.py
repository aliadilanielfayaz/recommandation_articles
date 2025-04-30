import pandas as pd
import numpy as np
# Utilisation de sklearn pour la similarité cosinus, plus standard
from sklearn.metrics.pairwise import cosine_similarity
import logging # Optionnel: pour afficher des messages d'information

# Configuration simple du logging (optionnel, peut être mis ailleurs)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_recommendations(user_item_matrix, user_id, n_recommendations=10):
    """
    Génère des recommandations pour un utilisateur basé sur la similarité cosinus
    avec d'autres utilisateurs.

    Parameters:
    -----------
    user_item_matrix : DataFrame
        Matrice utilisateur-article (utilisateurs en index, articles en colonnes).
    user_id : int
        L'ID de l'utilisateur pour qui générer les recommandations.
    n_recommendations : int, optional
        Le nombre maximum de recommandations à retourner (par défaut 10).

    Returns:
    --------
    pd.Series
        Une série contenant les article_id recommandés et leurs scores.
        Ne contient pas les articles déjà achetés par user_id.
    """
    try:
        # --- CORRECTION IMPORTANTE : Vérifier si user_id existe ---
        if user_id not in user_item_matrix.index:
            logging.warning(f"User ID {user_id} non trouvé dans la matrice utilisateur-article.")
            return pd.Series(dtype=float) # Retourne une série vide

        # Calculer la similarité cosinus entre tous les utilisateurs
        # S'assurer que la matrice ne contient pas de NaN qui pourrait poser problème
        user_item_matrix_filled = user_item_matrix.fillna(0)
        user_similarity = cosine_similarity(user_item_matrix_filled)
        user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

        # Trouver les utilisateurs les plus similaires (excluant l'utilisateur lui-même)
        k_similar_users = 10
        num_users = len(user_similarity_df[user_id])
        actual_k = min(k_similar_users, num_users - 1)

        if actual_k <= 0:
             logging.warning(f"Pas assez d'autres utilisateurs pour trouver des voisins similaires pour User ID {user_id}.")
             return pd.Series(dtype=float)

        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:actual_k+1]

        if similar_users.empty or similar_users.sum() == 0:
             logging.warning(f"Aucun utilisateur similaire trouvé pour User ID {user_id}.")
             return pd.Series(dtype=float)

        # Calculer les scores de recommandation
        recommendations = user_item_matrix_filled.loc[similar_users.index].sum(axis=0).sort_values(ascending=False)

        # --- CORRECTION IMPORTANTE : Filtrer les articles déjà achetés ---
        items_bought_by_user = user_item_matrix_filled.loc[user_id][user_item_matrix_filled.loc[user_id] > 0].index
        filtered_recommendations = recommendations.drop(items_bought_by_user, errors='ignore')

        # Retourner les N meilleures recommandations filtrées
        return filtered_recommendations.head(n_recommendations)

    except Exception as e:
        logging.error(f"Erreur lors de la génération de recommandations pour User ID {user_id}: {e}")
        return pd.Series(dtype=float)


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
    try:
        user1_data = df[df['user_id'] == user1_id]
        user2_data = df[df['user_id'] == user2_id]

        if user1_data.empty or user2_data.empty:
            logging.warning(f"User1 ({user1_id}) ou User2 ({user2_id}) non trouvé dans les données.")
            return []

        required_cols = ['religion', 'pays']
        if not all(col in user1_data.columns for col in required_cols) or \
           not all(col in user2_data.columns for col in required_cols):
            logging.error("Les colonnes 'religion' ou 'pays' sont manquantes dans le DataFrame.")
            return []

        user1_profile = user1_data.iloc[0]
        user2_profile = user2_data.iloc[0]

        if not (user1_profile['religion'] == user2_profile['religion'] and user1_profile['pays'] == user2_profile['pays']):
            logging.info(f"Condition de profil non remplie entre User {user1_id} et User {user2_id}.")
            return []

        if 'article_id' not in df.columns:
             logging.error("La colonne 'article_id' est manquante dans le DataFrame.")
             return []
        articles_user1 = set(user1_data['article_id'].unique())
        articles_user2 = set(user2_data['article_id'].unique())

        recommendations = list(articles_user1 - articles_user2)
        logging.info(f"Recommandations basées sur attributs pour {user2_id} depuis {user1_id}: {recommendations}")
        return recommendations

    except Exception as e:
        logging.error(f"Erreur dans recommend_based_on_user_attributes entre {user1_id} et {user2_id}: {e}")
        return []

