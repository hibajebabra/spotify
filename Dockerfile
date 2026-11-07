# ============================================================
# Dockerfile — Spotify Music Intelligence Platform
# ============================================================
# Usage :
#   docker build -t spotify-intelligence .
#   docker run -p 8501:8501 spotify-intelligence
# ============================================================

FROM python:3.13-slim

# Métadonnées
LABEL description="Spotify Music Intelligence Platform"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1

# Répertoire de travail
WORKDIR /app

# Copier uniquement requirements.txt en premier (optimise le cache Docker)
COPY requirements.txt .

# Installer les dépendances
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copier le code source
COPY src/        ./src/
COPY app.py      .
COPY .gitignore  .

# Créer les dossiers de données (vides au démarrage)
RUN mkdir -p data/raw data/processed models reports/figures

# Port Streamlit
EXPOSE 8501

# Point d'entrée : dashboard Streamlit
# Note : le dataset et les modèles doivent être montés en volume ou générés d'abord
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

