"""
train_clustering.py — Clustering musical avec K-Means
=====================================================
Rôle : Regrouper les morceaux en clusters homogènes selon leurs caractéristiques
       audio, sans utiliser la cible (popularity) ni les genres.

Méthode :
  - K-Means avec sélection du K optimal via le score silhouette + courbe d'inertie.
  - StandardScaler obligatoire (K-Means est sensible aux échelles).
  - Évaluation de K = 2 à 10.

Limites importantes (à garder en tête) :
  - K-Means suppose des clusters sphériques de taille similaire.
  - Les clusters ne correspondent PAS nécessairement aux genres musicaux officiels.
  - Le score silhouette est indicatif — il faut interpréter les clusters avec les données.

Usage :
  python -m src.train_clustering
"""

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    AUDIO_FEATURES,
    CLUSTERING_MODEL_PATH,
    FIGURES_DIR,
    KMEANS_K_RANGE,
    MLFLOW_EXPERIMENT_CLUSTERING,
    MODELS_DIR,
    PROCESSED_PARQUET,
    RANDOM_STATE,
)

# ── Configuration des logs ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Variables utilisées pour le clustering
# On exclut duration_ms (trop variable, domine les distances)
# et on se concentre sur les caractéristiques perceptuelles
CLUSTERING_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
]


def load_data() -> pd.DataFrame:
    """Charge le dataset prétraité."""
    if not PROCESSED_PARQUET.exists():
        raise FileNotFoundError(
            f"❌ Données non trouvées : {PROCESSED_PARQUET}\n"
            "→ Lance d'abord : python -m src.preprocessing"
        )
    df = pd.read_parquet(PROCESSED_PARQUET)
    logger.info(f"Dataset chargé : {df.shape[0]:,} lignes")
    return df


def select_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne et retourne uniquement les colonnes utiles pour le clustering."""
    available = [c for c in CLUSTERING_FEATURES if c in df.columns]
    missing   = set(CLUSTERING_FEATURES) - set(available)

    if missing:
        logger.warning(f"Features absentes (ignorées) : {missing}")

    logger.info(f"Features retenues pour le clustering : {available}")
    return df[available].copy()


def evaluate_k_range(X_scaled: np.ndarray, k_range) -> tuple[list, list]:
    """
    Évalue K-Means pour plusieurs valeurs de K.

    Retourne :
      - inertias      : liste d'inertie (somme des distances² aux centres)
      - silhouettes   : liste de scores silhouette (-1 à 1, plus grand = mieux)

    Conseil d'interprétation :
      - Inertie : chercher le "coude" où la réduction ralentit.
      - Silhouette : chercher le maximum.
      - Si les deux critères divergent, préférer le score silhouette.
    """
    inertias    = []
    silhouettes = []

    for k in k_range:
        logger.info(f"  Test K={k}...")
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)

        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels, sample_size=min(10_000, len(X_scaled)),
                               random_state=RANDOM_STATE)
        silhouettes.append(sil)
        logger.info(f"    Inertie={km.inertia_:.1f}  |  Silhouette={sil:.4f}")

    return inertias, silhouettes


def plot_elbow_and_silhouette(k_range, inertias: list, silhouettes: list) -> Path:
    """
    Génère un graphique avec deux sous-plots :
      - Gauche : courbe d'inertie (méthode du coude).
      - Droite : score silhouette par K.

    Sauvegarde la figure dans reports/figures/.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sélection du nombre optimal de clusters K", fontsize=14, fontweight="bold")

    k_values = list(k_range)

    # Courbe d'inertie
    axes[0].plot(k_values, inertias, marker="o", color="#1DB954", linewidth=2)
    axes[0].set_title("Méthode du coude (Inertie)")
    axes[0].set_xlabel("Nombre de clusters K")
    axes[0].set_ylabel("Inertie")
    axes[0].grid(True, alpha=0.3)

    # Score silhouette
    best_k_idx = int(np.argmax(silhouettes))
    axes[1].plot(k_values, silhouettes, marker="o", color="#1DB954", linewidth=2)
    axes[1].axvline(x=k_values[best_k_idx], color="red", linestyle="--",
                    label=f"Meilleur K={k_values[best_k_idx]}")
    axes[1].set_title("Score Silhouette")
    axes[1].set_xlabel("Nombre de clusters K")
    axes[1].set_ylabel("Score Silhouette")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "kmeans_selection.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Graphique sauvegardé : {path}")
    return path


def build_clustering_pipeline(k: int) -> Pipeline:
    """
    Construit le pipeline de clustering :
      1. SimpleImputer : gère les NaN résiduels.
      2. StandardScaler : indispensable pour K-Means (même échelle pour toutes les features).
      3. KMeans : algorithme de clustering.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("kmeans",  KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)),
    ])


def interpret_clusters(df_with_clusters: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Calcule le profil moyen de chaque cluster pour faciliter l'interprétation.
    Retourne un DataFrame avec les statistiques par cluster.
    """
    profile = (
        df_with_clusters
        .groupby("cluster")[features]
        .mean()
        .round(3)
    )
    profile["n_tracks"] = df_with_clusters.groupby("cluster").size()

    return profile


def run_clustering_pipeline() -> tuple[pd.DataFrame, Pipeline, int]:
    """
    Pipeline complète de clustering.
    Retourne le DataFrame avec labels, le pipeline entraîné, et le K optimal.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_CLUSTERING)

    # ── 1. Chargement ─────────────────────────────────────────────────────────
    df = load_data()
    X  = select_clustering_features(df)
    features_used = list(X.columns)

    # ── 2. Pré-traitement pour l'évaluation de K ──────────────────────────────
    # On applique imputer + scaler manuellement pour évaluer les K
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled  = scaler.fit_transform(X_imputed)

    # ── 3. Évaluation de K ────────────────────────────────────────────────────
    logger.info(f"\n── Évaluation de K = {list(KMEANS_K_RANGE)} ──")
    inertias, silhouettes = evaluate_k_range(X_scaled, KMEANS_K_RANGE)

    # ── 4. Sélection du K optimal (max silhouette) ────────────────────────────
    k_values    = list(KMEANS_K_RANGE)
    best_k_idx  = int(np.argmax(silhouettes))
    best_k      = k_values[best_k_idx]
    best_sil    = silhouettes[best_k_idx]

    logger.info(f"\n🏆 K optimal sélectionné : K={best_k} (Silhouette={best_sil:.4f})")

    # ── 5. Graphique de sélection ─────────────────────────────────────────────
    plot_path = plot_elbow_and_silhouette(KMEANS_K_RANGE, inertias, silhouettes)

    # ── 6. Entraînement du pipeline final ─────────────────────────────────────
    logger.info(f"\n── Entraînement du pipeline final avec K={best_k} ──")
    clustering_pipe = build_clustering_pipeline(best_k)
    labels = clustering_pipe.fit_predict(X)

    # Ajout des labels au DataFrame original
    df_result = df.copy()
    df_result["cluster"] = labels

    # Score silhouette final
    X_final_scaled = clustering_pipe.named_steps["scaler"].transform(
        clustering_pipe.named_steps["imputer"].transform(X)
    )
    final_sil = silhouette_score(
        X_final_scaled, labels,
        sample_size=min(10_000, len(labels)),
        random_state=RANDOM_STATE,
    )

    # ── 7. Interprétation des clusters ────────────────────────────────────────
    logger.info("\n── Profil moyen des clusters ──")
    profile = interpret_clusters(df_result, features_used)
    print("\n" + profile.to_string())

    # ── 8. MLflow logging ─────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"KMeans_K{best_k}"):
        mlflow.log_param("k_optimal",      best_k)
        mlflow.log_param("features",       features_used)
        mlflow.log_param("k_range",        list(KMEANS_K_RANGE))
        mlflow.log_metric("silhouette_score", final_sil)
        mlflow.log_metric("inertia", float(clustering_pipe.named_steps["kmeans"].inertia_))
        mlflow.sklearn.log_model(
            clustering_pipe,
            artifact_path="clustering_pipeline",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
        mlflow.log_artifact(str(plot_path))

    logger.info(f"Score silhouette final : {final_sil:.4f}")

    # ── 9. Sauvegarde du pipeline ─────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLUSTERING_MODEL_PATH, "wb") as f:
        pickle.dump(clustering_pipe, f)
    logger.info(f"✅ Pipeline sauvegardé : {CLUSTERING_MODEL_PATH}")

    return df_result, clustering_pipe, best_k


if __name__ == "__main__":
    df_clustered, pipe, k = run_clustering_pipeline()

    print(f"\n✅ Clustering terminé. K={k} clusters créés.")
    print(f"   Distribution des clusters :")
    print(df_clustered["cluster"].value_counts().sort_index().to_string())
    print(f"\n   MLflow UI : mlflow ui --port 5000")
