import matplotlib.pyplot as plt

def plot_purchase_distribution(df):
    """
    Afficher plusieurs visualisations des données d'achat.
    """
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
