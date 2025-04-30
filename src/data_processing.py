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
