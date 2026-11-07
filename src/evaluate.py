"""
evaluate.py — Évaluation et interprétabilité des modèles
=========================================================
Rôle : Analyser en profondeur les performances du meilleur modèle de régression.

Ce module produit :
  1. Distribution des erreurs de prédiction.
  2. Graphique valeurs réelles vs. prédites.
  3. Importance des features par permutation.
  4. Performance par genre (si effectifs suffisants).
  5. Identification des cas où le modèle se trompe le plus.

Note épistémologique : l'importance d'une variable dans un modèle ne prouve
pas qu'elle cause la popularité. Corrélation ≠ causalité.

Usage :
  python -m src.evaluate
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    AUDIO_FEATURES,
    FEATURE_COLS_PATH,
    FIGURES_DIR,
    ID_COL,
    PROCESSED_PARQUET,
    RANDOM_STATE,
    REGRESSION_MODEL_PATH,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)

# ── Configuration des logs ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Style des graphiques
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.facecolor"] = "#1a1a2e"
plt.rcParams["axes.facecolor"]   = "#16213e"
plt.rcParams["text.color"]       = "white"
plt.rcParams["axes.labelcolor"]  = "white"
plt.rcParams["xtick.color"]      = "white"
plt.rcParams["ytick.color"]      = "white"

MIN_GENRE_SAMPLES = 50  # Effectif minimum pour analyser un genre


def load_model_and_data() -> tuple:
    """Charge le modèle sauvegardé et le dataset prétraité."""
    if not REGRESSION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"❌ Modèle introuvable : {REGRESSION_MODEL_PATH}\n"
            "→ Lance d'abord : python -m src.train_regression"
        )
    if not PROCESSED_PARQUET.exists():
        raise FileNotFoundError(
            f"❌ Données introuvables : {PROCESSED_PARQUET}\n"
            "→ Lance d'abord : python -m src.preprocessing"
        )

    with open(REGRESSION_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df = pd.read_parquet(PROCESSED_PARQUET)

    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in AUDIO_FEATURES if c in df.columns]

    logger.info(f"Modèle chargé : {REGRESSION_MODEL_PATH}")
    logger.info(f"Dataset : {df.shape[0]:,} lignes")
    return model, df, feature_cols


def get_test_set(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Reproduit le même split test que dans train_regression.py."""
    X = df[feature_cols]
    y = df[TARGET_COL]
    groups = df[ID_COL]

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    _, test_idx = next(gss.split(X, y, groups=groups))

    return X.iloc[test_idx], y.iloc[test_idx], df.iloc[test_idx]


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray) -> Path:
    """
    Graphique de la distribution des erreurs (résidus).
    Un bon modèle a des résidus centrés sur 0 et approximativement normaux.
    """
    residuals = y_true.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Distribution des erreurs de prédiction", color="white",
                 fontsize=14, fontweight="bold")

    # Distribution des résidus
    axes[0].hist(residuals, bins=50, color="#1DB954", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2, label="Erreur = 0")
    axes[0].set_title("Distribution des résidus", color="white")
    axes[0].set_xlabel("Résidu (réel - prédit)")
    axes[0].set_ylabel("Fréquence")
    axes[0].legend()
    axes[0].set_facecolor("#16213e")
    axes[0].tick_params(colors="white")

    # Résidus vs valeurs prédites
    axes[1].scatter(y_pred, residuals, alpha=0.3, color="#1DB954", s=5)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_title("Résidus vs. Valeurs prédites", color="white")
    axes[1].set_xlabel("Popularité prédite")
    axes[1].set_ylabel("Résidu")
    axes[1].set_facecolor("#16213e")
    axes[1].tick_params(colors="white")

    plt.tight_layout()
    path = FIGURES_DIR / "residuals.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Graphique sauvegardé : {path}")
    return path


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray) -> Path:
    """
    Graphique valeurs réelles vs. prédites.
    La ligne diagonale rouge représente la prédiction parfaite.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.scatter(y_true, y_pred, alpha=0.2, color="#1DB954", s=5)

    # Ligne parfaite
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Prédiction parfaite")

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    ax.set_title(f"Valeurs réelles vs. prédites\nMAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.3f}",
                 color="white", fontsize=12)
    ax.set_xlabel("Popularité réelle", color="white")
    ax.set_ylabel("Popularité prédite", color="white")
    ax.legend(facecolor="#16213e", labelcolor="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = FIGURES_DIR / "actual_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Graphique sauvegardé : {path}")
    return path


def compute_permutation_importance(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """
    Calcule l'importance par permutation.

    Principe : pour chaque feature, on mélange aléatoirement ses valeurs
    et on mesure la dégradation de performance. Une grande dégradation
    indique que la feature est importante.

    Avantage vs. importance basée sur les arbres : fonctionne pour tout modèle.
    Limite : peut être biaisé si des features sont corrélées entre elles.

    RAPPEL : importance ≠ causalité.
    """
    logger.info("Calcul de l'importance par permutation (peut prendre ~1 minute)...")

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature":    X_test.columns,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=False)

    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame) -> Path:
    """Graphique de l'importance des features par permutation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    colors = ["#1DB954" if v > 0 else "#e74c3c" for v in importance_df["importance"]]
    bars = ax.barh(importance_df["feature"], importance_df["importance"],
                   xerr=importance_df["std"], color=colors, alpha=0.85,
                   error_kw={"color": "white", "capsize": 3})

    ax.axvline(0, color="white", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title(
        "Importance des variables par permutation\n"
        "(une valeur négative = la feature nuit légèrement au modèle quand permutée)",
        color="white", fontsize=12
    )
    ax.set_xlabel("Variation de MAE (plus élevé = plus important)", color="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = FIGURES_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Graphique sauvegardé : {path}")
    return path


def evaluate_by_genre(
    df_test: pd.DataFrame, y_pred: np.ndarray, min_samples: int = MIN_GENRE_SAMPLES
) -> Optional[pd.DataFrame]:
    """
    Évalue les performances du modèle par genre musical.
    N'analyse que les genres avec au moins min_samples observations dans le test set.

    Limite : avec peu d'observations par genre dans le test set,
    les métriques sont peu fiables statistiquement.
    """
    if "track_genre" not in df_test.columns:
        logger.warning("Colonne 'track_genre' absente — analyse par genre ignorée.")
        return None

    df_eval = df_test.copy()
    df_eval["predicted"] = y_pred
    df_eval["residual"]  = df_eval[TARGET_COL] - y_pred

    genre_counts = df_eval["track_genre"].value_counts()
    valid_genres  = genre_counts[genre_counts >= min_samples].index

    if len(valid_genres) == 0:
        logger.warning(f"Aucun genre avec ≥ {min_samples} observations dans le test set.")
        return None

    rows = []
    for genre in valid_genres:
        mask = df_eval["track_genre"] == genre
        yt   = df_eval.loc[mask, TARGET_COL]
        yp   = df_eval.loc[mask, "predicted"]
        rows.append({
            "genre":    genre,
            "n":        int(mask.sum()),
            "MAE":      round(mean_absolute_error(yt, yp), 2),
            "RMSE":     round(np.sqrt(mean_squared_error(yt, yp)), 2),
            "R²":       round(r2_score(yt, yp), 3),
        })

    genre_df = pd.DataFrame(rows).sort_values("MAE")
    logger.info(f"\nPerformance par genre ({len(valid_genres)} genres avec ≥{min_samples} obs.) :")
    logger.info(f"\n{genre_df.to_string(index=False)}")

    return genre_df


def run_evaluation() -> None:
    """Pipeline complète d'évaluation."""
    logger.info("=" * 60)
    logger.info("ÉVALUATION DU MODÈLE — DÉBUT")
    logger.info("=" * 60)

    # 1. Chargement
    model, df, feature_cols = load_model_and_data()

    # 2. Extraction du test set
    X_test, y_test, df_test = get_test_set(df, feature_cols)
    logger.info(f"Test set : {len(X_test):,} observations")

    # 3. Prédictions
    y_pred = model.predict(X_test)

    # 4. Métriques globales
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MÉTRIQUES FINALES (Test set)")
    print("=" * 50)
    print(f"  MAE  : {mae:.3f}  (erreur moyenne en points de popularité)")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  R²   : {r2:.4f}  (variance expliquée)")
    print("=" * 50)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Graphiques
    logger.info("\n── Génération des graphiques ──")
    plot_residuals(y_test, y_pred)
    plot_actual_vs_predicted(y_test, y_pred)

    # 6. Importance par permutation
    logger.info("\n── Importance des features ──")
    importance_df = compute_permutation_importance(model, X_test, y_test)
    print("\n" + importance_df.to_string(index=False))
    plot_feature_importance(importance_df)

    # 7. Performance par genre
    logger.info("\n── Performance par genre ──")
    genre_df = evaluate_by_genre(df_test, y_pred)

    logger.info("\n" + "=" * 60)
    logger.info("ÉVALUATION TERMINÉE")
    logger.info("=" * 60)
    print(f"\nFigures sauvegardées dans : {FIGURES_DIR}")


if __name__ == "__main__":
    run_evaluation()
