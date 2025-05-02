import sqlite3
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemins relatifs à la racine du projet
CSV_FILE_PATH = 'data/purchases.csv'
DB_FILE_PATH = 'data/reco_data.db'
TABLE_NAME = 'purchases'

def create_and_populate_db():
    """
    Lit les données depuis le fichier CSV et les insère dans une table SQLite.
    La table est supprimée et recréée à chaque exécution.
    """
    if not os.path.exists(CSV_FILE_PATH):
        logging.error(f"Le fichier CSV source '{CSV_FILE_PATH}' n'a pas été trouvé.")
        return

    # S'assurer que le répertoire data existe
    os.makedirs(os.path.dirname(DB_FILE_PATH), exist_ok=True)

    try:
        logging.info(f"Lecture du fichier CSV : {CSV_FILE_PATH}")
        df = pd.read_csv(CSV_FILE_PATH)

        # Convertir les colonnes de date en string pour SQLite (ou gérer autrement si besoin)
        if 'date_achat' in df.columns:
             df['date_achat'] = pd.to_datetime(df['date_achat']).dt.strftime('%Y-%m-%d %H:%M:%S')

        logging.info(f"Connexion à la base de données SQLite : {DB_FILE_PATH}")
        conn = sqlite3.connect(DB_FILE_PATH)
        cursor = conn.cursor()

        # Supprimer la table si elle existe déjà pour éviter les doublons
        logging.info(f"Suppression de la table '{TABLE_NAME}' si elle existe...")
        cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

        logging.info(f"Insertion des données dans la table '{TABLE_NAME}'...")
        # Utiliser pandas.to_sql pour créer la table et insérer les données
        # index=False pour ne pas écrire l'index du DataFrame comme colonne SQL
        # if_exists='replace' recrée la table à chaque fois
        df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)

        logging.info(f"Données insérées avec succès. {len(df)} lignes ajoutées.")

        # Vérification (optionnel)
        count = cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        logging.info(f"Vérification : La table '{TABLE_NAME}' contient {count} lignes.")

        conn.commit()
        conn.close()
        logging.info("Connexion à la base de données fermée.")

    except pd.errors.EmptyDataError:
        logging.error(f"Le fichier CSV '{CSV_FILE_PATH}' est vide.")
    except sqlite3.Error as e:
        logging.error(f"Erreur SQLite : {e}")
    except Exception as e:
        logging.error(f"Une erreur inattendue est survenue : {e}")

if __name__ == "__main__":
    create_and_populate_db()
