import matplotlib.pyplot as plt
import pandas as pd # Importation manquante de Pandas
import plotly.express as px
import logging

def plot_purchase_distribution(df):
    """
    Afficher plusieurs visualisations des données d'achat.
    """
    if df is None or 'categorie' not in df.columns:
        logging.warning("DataFrame invalide ou colonne 'categorie' manquante pour plot_purchase_distribution.")
        return

    # Créer une figure avec plusieurs sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Distribution des achats par catégorie
    category_counts = df.groupby('categorie')['article_id'].count()
    category_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Nombre d\'achats par catégorie')
    ax1.set_xlabel('Catégorie')
    ax1.set_ylabel('Nombre d\'achats')

    # 2. Distribution des dépenses par utilisateur
    user_spending = df.groupby('user_id')['prix'].sum()
    user_spending.plot(kind='bar', ax=ax2)
    ax2.set_title('Dépenses totales par utilisateur')
    ax2.set_xlabel('ID Utilisateur')
    ax2.set_ylabel('Dépenses totales (€)')

    # Ajuster la mise en page
    plt.tight_layout()
    plt.show()


def plot_interactive_user_summary(df):
    """
    Affiche un nuage de points interactif des utilisateurs montrant
    le nombre d'achats vs le montant total dépensé, coloré par pays.

    Parameters:
    -----------
    df : DataFrame
        Le DataFrame contenant les achats et les profils utilisateurs.
        Doit inclure 'user_id', 'article_id', 'prix', 'pays', 'religion', 'age'.
    """
    if df is None or not all(col in df.columns for col in ['user_id', 'article_id', 'prix', 'pays', 'religion', 'age']):
        logging.warning("DataFrame invalide ou colonnes manquantes pour plot_interactive_user_summary.")
        return

    logging.info("Préparation des données pour le graphique interactif des utilisateurs...")
    # Agréger les données par utilisateur
    user_summary = df.groupby('user_id').agg(
        n_purchases=('article_id', 'count'),
        total_spent=('prix', 'sum')
    ).reset_index()

    # Obtenir les informations de profil uniques par utilisateur
    user_profiles = df[['user_id', 'pays', 'religion', 'age']].drop_duplicates(subset=['user_id'])

    # Fusionner les agrégats et les profils
    user_data_interactive = pd.merge(user_summary, user_profiles, on='user_id', how='left')

    logging.info("Création du graphique interactif...")
    # Créer le scatter plot interactif
    fig = px.scatter(user_data_interactive,
                     x='n_purchases',
                     y='total_spent',
                     color='pays',  # Colorer par pays
                     hover_data=['user_id', 'pays', 'religion', 'age', 'n_purchases', 'total_spent'], # Infos au survol
                     title="Résumé Interactif des Utilisateurs (Achats vs Dépenses)",
                     labels={'n_purchases': "Nombre d'Achats", 'total_spent': "Montant Total Dépensé (€)"})

    fig.show() # Afficher le graphique interactif
