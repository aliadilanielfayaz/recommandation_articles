import numpy as np

def cosine_similarity(user1, user2):
    """
    Calculer la similarité cosinus entre deux utilisateurs.
    """
    dot_product = np.dot(user1, user2)
    norm_user1 = np.linalg.norm(user1)
    norm_user2 = np.linalg.norm(user2)
    return dot_product / (norm_user1 * norm_user2)

