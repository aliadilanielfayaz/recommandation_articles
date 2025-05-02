# Utiliser une image Python officielle comme image de base
# 'slim' est une version plus légère
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier d'abord le fichier des dépendances pour profiter du cache Docker
COPY requirements.txt .

# Installer les dépendances
# --no-cache-dir réduit la taille de l'image
# Mettre à jour pip et installer build-essential pour les dépendances C de surprise
RUN pip install --no-cache-dir --upgrade pip \
    && apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier le code source de l'application (API et logique métier)
COPY ./src ./src
# Copier les données nécessaires à l'API (pour filtrer les achats)
COPY ./data ./data 
# Copier le modèle pré-entraîné
COPY svd_model.pkl . 
# Copier le fichier principal de l'API
COPY app.py . 

# Exposer le port sur lequel l'API écoutera
EXPOSE 8000

# Commande pour lancer l'application FastAPI avec Uvicorn
# --host 0.0.0.0 est important pour que l'API soit accessible depuis l'extérieur du conteneur
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
