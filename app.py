from fastapi import FastAPI, HTTPException
import logging

import pickle
import os
import pandas as pd
from typing import List # Pour le type hinting de la réponse

# Imports depuis votre projet src (nécessite que src soit dans PYTHONPATH ou structure adaptée)
# Assurez-vous que le répertoire parent de 'src' est dans PYTHONPATH ou lancez uvicorn depuis la racine
from src.recommendation import generate_svd_recommendations

# Import de la fonction de chargement SQLite
from src.data_processing import load_data_from_sqlite 

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API de Recommandation d'Articles",
    description="Sert des recommandations d'articles basées sur un modèle SVD pré-entraîné.",
    version="0.1.0"
)

# --- Variables globales pour le modèle et les données ---
MODEL_PATH = "svd_model.pkl"
# DATA_PATH = "data/purchases.csv" # Chemin vers les données d'achat CSV (Non utilisé)
model = None
purchase_df = None
all_items = set()

@app.on_event("startup")
async def load_resources():
    """Charge le modèle et les données nécessaires au démarrage."""
    global model
    global purchase_df
    global all_items

    # Charger le modèle
    if os.path.exists(MODEL_PATH):
        logging.info(f"Chargement du modèle depuis {MODEL_PATH}...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logging.info("Modèle chargé avec succès.")
    else:
        logging.error(f"Le fichier du modèle {MODEL_PATH} n'a pas été trouvé !")
        # Idéalement, l'application ne devrait pas démarrer ou signaler une erreur critique
        # Pour l'instant, on continue mais le modèle sera None
        model = None # Assurer que model est None si chargement échoue

    # --- Charger les données d'achat depuis SQLite ---
    logging.info("Chargement des données d'achat depuis SQLite...")
    purchase_df = load_data_from_sqlite() # Utilise les chemins par défaut
    if purchase_df is not None:
        if 'article_id' in purchase_df.columns:
            all_items = set(purchase_df['article_id'].unique())
            logging.info(f"Données d'achat chargées depuis SQLite. {len(all_items)} articles uniques trouvés.")
        else:
            logging.error("La colonne 'article_id' est manquante dans les données SQLite.")
            purchase_df = None # Marquer comme non chargé
    else:
        logging.error("Échec du chargement des données d'achat depuis SQLite.")
        purchase_df = None # Assurer que purchase_df est None


@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API de Recommandation !"}

@app.get("/recommend/{user_id}", response_model=List[str])
async def get_recommendations(user_id: int, n: int = 5):
    """
    Génère des recommandations SVD pour un utilisateur donné.
    """
    logging.info(f"Requête de recommandation reçue pour user_id: {user_id}, n={n}")
    if model is None:
        logging.error("Le modèle SVD n'est pas chargé.")
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    if purchase_df is None:
        logging.error("Les données d'achat ne sont pas chargées.")
        raise HTTPException(status_code=503, detail="Données d'achat non disponibles")

    # Obtenir les articles déjà achetés par l'utilisateur
    items_bought_by_user = set(purchase_df[purchase_df['user_id'] == user_id]['article_id'].unique())
    # Déterminer les articles à prédire (tous sauf ceux déjà achetés)
    items_to_predict = list(all_items - items_bought_by_user)

    if not items_to_predict:
        logging.info(f"L'utilisateur {user_id} a déjà acheté tous les articles ou aucun article à prédire.")
        return []

    # Générer les recommandations
    recommendations = generate_svd_recommendations(model, user_id, items_to_predict, n_recommendations=n)

    logging.info(f"Recommandations pour {user_id}: {recommendations}")
    return recommendations
