from surprise import Dataset, Reader # Importation pour Surprise
import logging # Pour afficher des messages

# Configuration du logging (peut être fait dans main.py aussi)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd

def load_data(file_path):
    """
    Charger les données à partir d'un fichier CSV.
    """
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Nettoyer les données : traiter les valeurs manquantes, formater les dates, etc.
    """
    df['date_achat'] = pd.to_datetime(df['date_achat'])
    df.fillna(0, inplace=True)
    return df

def create_user_item_matrix(df):
    """
    Créer la matrice utilisateur-article.
    """
    return df.pivot_table(index='user_id', columns='article_id', values='prix', aggfunc='sum', fill_value=0)




def load_data_for_surprise(file_path):
    """
    Charge les données depuis un fichier CSV et les prépare pour la bibliothèque Surprise.
    Assign une note implicite de 1 à chaque interaction.

    Parameters:
    -----------
    file_path : str
        Le chemin vers le fichier CSV contenant les achats.
        Le fichier doit contenir au moins 'user_id' et 'article_id'.

    Returns:
    --------
    surprise.dataset.DatasetAutoFolds
        Un objet Dataset Surprise prêt à être utilisé pour l'entraînement et l'évaluation.
        Retourne None en cas d'erreur.
    """
    try:
        logging.info(f"Chargement des données pour Surprise depuis : {file_path}")
        df = pd.read_csv(file_path)

        # Vérifier la présence des colonnes nécessaires
        if 'user_id' not in df.columns or 'article_id' not in df.columns:
            logging.error("Le fichier CSV doit contenir les colonnes 'user_id' et 'article_id'.")
            return None

        # Sélectionner les colonnes et ajouter une note implicite de 1
        df_surprise = df[['user_id', 'article_id']].copy()
        df_surprise['rating'] = 1

        # Définir le lecteur avec l'échelle de notation (ici, juste 1)
        reader = Reader(rating_scale=(1, 1))

        # Charger le jeu de données depuis le DataFrame
        data = Dataset.load_from_df(df_surprise, reader)
        logging.info("Données chargées avec succès pour Surprise.")
        return data

    except FileNotFoundError:
        logging.error(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
        return None
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données pour Surprise : {e}")
        return None

