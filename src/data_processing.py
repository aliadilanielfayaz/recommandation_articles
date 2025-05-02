from surprise import Dataset, Reader # Importation pour Surprise
import logging # Pour afficher des messages

# Configuration du logging (peut être fait dans main.py aussi)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd

import sqlite3 # Pour interagir avec SQLite

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


## FOR SURPRISE

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


## FOR SQLITE

# --- Fonctions pour lire depuis SQLite ---

DB_FILE_PATH = 'data/reco_data.db' # Chemin vers la base de données SQLite
TABLE_NAME = 'purchases' # Nom de la table

def load_data_from_sqlite(db_path=DB_FILE_PATH, table_name=TABLE_NAME):
    """
    Charge toutes les données depuis une table SQLite spécifiée.

    Parameters:
    -----------
    db_path : str, optional
        Le chemin vers le fichier de base de données SQLite.
    table_name : str, optional
        Le nom de la table à lire.

    Returns:
    --------
    DataFrame
        Le DataFrame chargé depuis la base de données ou None en cas d'erreur.
    """
    try:
        logging.info(f"Connexion à la base de données SQLite : {db_path}")
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name}"
        logging.info(f"Lecture des données depuis la table '{table_name}'...")
        df = pd.read_sql_query(query, conn)
        conn.close()
        logging.info(f"Données chargées avec succès depuis SQLite. {len(df)} lignes lues.")
        # Convertir la colonne date_achat (stockée en texte) en datetime
        if 'date_achat' in df.columns:
             df['date_achat'] = pd.to_datetime(df['date_achat'], errors='coerce')
        return df
    except sqlite3.Error as e:
        logging.error(f"Erreur SQLite lors de la lecture de la table '{table_name}': {e}")
        return None
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données depuis SQLite : {e}")
        return None

def load_data_for_surprise_from_sqlite(db_path=DB_FILE_PATH, table_name=TABLE_NAME):
    """
    Charge les données nécessaires pour Surprise depuis une table SQLite.
    Assign une note implicite de 1 à chaque interaction.

    Parameters:
    -----------
    db_path : str, optional
        Le chemin vers le fichier de base de données SQLite.
    table_name : str, optional
        Le nom de la table contenant 'user_id' et 'article_id'.

    Returns:
    --------
    surprise.dataset.DatasetAutoFolds
        Un objet Dataset Surprise prêt à être utilisé, ou None en cas d'erreur.
    """
    df_base = load_data_from_sqlite(db_path, table_name) # Réutilise la fonction précédente
    if df_base is None:
        logging.error("Impossible de charger les données de base depuis SQLite pour Surprise.")
        return None
    if 'user_id' not in df_base.columns or 'article_id' not in df_base.columns:
        logging.error("Les colonnes 'user_id' ou 'article_id' sont manquantes dans les données SQLite.")
        return None

    df_surprise = df_base[['user_id', 'article_id']].copy()
    df_surprise['rating'] = 1
    reader = Reader(rating_scale=(1, 1))
    data = Dataset.load_from_df(df_surprise, reader)
    logging.info("Données pour Surprise chargées avec succès depuis SQLite.")
    return data
