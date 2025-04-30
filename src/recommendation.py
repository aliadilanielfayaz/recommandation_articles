import pandas as pd
import numpy as np
from .similarity import cosine_similarity

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
