"""
config.py — Configuration centrale du projet
=============================================
Rôle : centraliser tous les chemins, constantes et paramètres
afin de ne jamais coder des valeurs "en dur" dans le code métier.

Si tu changes l'emplacement d'un fichier, tu ne modifies QUE ce fichier.
"""

from pathlib import Path

# ── Racine du projet ──────────────────────────────────────────────────────────
# Path(__file__) = c:/Users/HIBA/Desktop/spotify/src/config.py
# .parent        = c:/Users/HIBA/Desktop/spotify/src/
# .parent.parent = c:/Users/HIBA/Desktop/spotify/          ← racine du projet
ROOT_DIR = Path(__file__).parent.parent

# ── Dossiers de données ───────────────────────────────────────────────────────
DATA_DIR       = ROOT_DIR / "data"
RAW_DATA_DIR   = DATA_DIR / "raw"
PROC_DATA_DIR  = DATA_DIR / "processed"

# ── Fichiers de données ───────────────────────────────────────────────────────
RAW_CSV_PATH        = RAW_DATA_DIR  / "dataset.csv"
PROCESSED_PARQUET   = PROC_DATA_DIR / "tracks_clean.parquet"
FEATURE_COLS_PATH   = PROC_DATA_DIR / "feature_columns.json"   # colonnes du modèle

# ── Dossiers modèles et rapports ──────────────────────────────────────────────
MODELS_DIR   = ROOT_DIR / "models"
REPORTS_DIR  = ROOT_DIR / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures"

# ── Fichiers modèles sauvegardés ─────────────────────────────────────────────
REGRESSION_MODEL_PATH  = MODELS_DIR / "best_regressor.pkl"
CLUSTERING_MODEL_PATH  = MODELS_DIR / "kmeans_pipeline.pkl"

# ── Colonne cible (variable à prédire) ───────────────────────────────────────
TARGET_COL = "popularity"

# ── Identifiant unique des morceaux ──────────────────────────────────────────
ID_COL = "track_id"

# ── Caractéristiques audio numériques utilisées dans les modèles ─────────────
# Ces noms DOIVENT correspondre exactement aux colonnes du dataset réel.
# On les définit ici pour que le formulaire Streamlit soit toujours aligné
# avec le pipeline d'entraînement.
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# ── Paramètres de séparation train / validation / test ───────────────────────
TEST_SIZE       = 0.15   # 15% pour le test final
VAL_SIZE        = 0.15   # 15% pour la validation (sur le reste)
RANDOM_STATE    = 42     # graine aléatoire pour la reproductibilité

# ── Paramètres MLflow ────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_REGRESSION  = "spotify_regression"
MLFLOW_EXPERIMENT_CLUSTERING  = "spotify_clustering"

# ── Paramètres K-Means ────────────────────────────────────────────────────────
KMEANS_K_RANGE = range(2, 11)   # on teste K de 2 à 10

# ── Création automatique des dossiers manquants ───────────────────────────────
# Garantit que les dossiers existent avant toute écriture de fichier.
for _dir in [RAW_DATA_DIR, PROC_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import sys
    # Forcer l'encodage UTF-8 sur Windows pour afficher les accents correctement
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # Test rapide : affiche tous les chemins configurés
    print("=== Configuration du projet ===")
    print(f"Racine          : {ROOT_DIR}")
    print(f"Donnees brutes  : {RAW_DATA_DIR}")
    print(f"Donnees traitees: {PROC_DATA_DIR}")
    print(f"Modeles         : {MODELS_DIR}")
    print(f"Rapports        : {REPORTS_DIR}")
    print(f"CSV brut        : {RAW_CSV_PATH}")
    print(f"Parquet propre  : {PROCESSED_PARQUET}")
    print(f"Caracteristiques: {AUDIO_FEATURES}")
    print("\n[OK] Configuration chargee correctement.")
