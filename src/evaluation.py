from collections import defaultdict
import logging
import numpy as np # Importation manquante de NumPy

from surprise import Dataset
from surprise.model_selection import KFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def precision_recall_at_k(predictions, k=10, threshold=1):
    """
    Calcule Precision@k et Recall@k à partir des prédictions Surprise.
    Basé sur les exemples de la documentation Surprise.

    Parameters:
    -----------
    predictions : list of Prediction objects
        La liste des prédictions retournées par un algorithme Surprise sur un jeu de test.
    k : int, optional
        Le nombre de recommandations à considérer (par défaut 10).
    threshold : float, optional
        Le seuil de note à partir duquel un article est considéré comme pertinent (par défaut 1,
        car nous utilisons des notes implicites de 1).

    Returns:
    --------
    tuple
        Un tuple contenant (precision@k, recall@k).
    """
    # Mapper les prédictions aux utilisateurs
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Trier les articles prédits pour l'utilisateur
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Nombre d'articles pertinents recommandés dans le top k
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@k et Recall@k pour cet utilisateur
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    # Moyenne sur tous les utilisateurs
    avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    avg_recall = sum(rec for rec in recalls.values()) / len(recalls)

    return avg_precision, avg_recall

def evaluate_model(data, model_algo, k=10, n_splits=5):
    """
    Évalue un algorithme Surprise en utilisant la validation croisée
    et calcule Precision@k et Recall@k.

    Parameters:
    -----------
    data : surprise.dataset.DatasetAutoFolds
        Le jeu de données chargé par Surprise.
    model_algo : surprise algorithm instance
        L'algorithme Surprise à évaluer (ex: SVD()).
    k : int, optional
        Le nombre de recommandations pour Precision@k et Recall@k (par défaut 10).
    n_splits : int, optional
        Le nombre de folds pour la validation croisée (par défaut 5).

    Returns:
    --------
    tuple
        Un tuple contenant (average_precision_at_k, average_recall_at_k).
    """
    kf = KFold(n_splits=n_splits)
    precisions = []
    recalls = []

    logging.info(f"Début de l'évaluation avec {n_splits}-fold cross-validation...")
    for fold, (trainset, testset) in enumerate(kf.split(data)):
        logging.info(f"  Fold {fold+1}/{n_splits}...")
        model_algo.fit(trainset)
        predictions = model_algo.test(testset)
        precision, recall = precision_recall_at_k(predictions, k=k)
        precisions.append(precision)
        recalls.append(recall)
        logging.info(f"    Precision@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}")

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    logging.info(f"Évaluation terminée. Moyenne Precision@{k}: {avg_precision:.4f}, Moyenne Recall@{k}: {avg_recall:.4f}")

    return avg_precision, avg_recall