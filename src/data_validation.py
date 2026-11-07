"""
data_validation.py — Validation de la qualité des données brutes
=================================================================
Rôle : Vérifier que le CSV téléchargé est conforme avant tout traitement.

Ce module répond aux questions :
  - Le fichier existe-t-il ?
  - Les colonnes attendues sont-elles présentes ?
  - Y a-t-il des valeurs manquantes et où ?
  - Y a-t-il des doublons (par ligne et par track_id) ?
  - Les plages des caractéristiques audio sont-elles raisonnables ?
  - Les types de colonnes sont-ils corrects ?

Usage en ligne de commande :
  python -m src.data_validation
"""

import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Import de la configuration centrale
from src.config import RAW_CSV_PATH, AUDIO_FEATURES, ID_COL, TARGET_COL

# ── Configuration des logs ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Schéma attendu ────────────────────────────────────────────────────────────
# Colonnes minimales requises pour que le projet fonctionne.
# Si une colonne est absente, la validation échoue immédiatement.
REQUIRED_COLUMNS = {
    "track_id",
    "track_name",
    "artists",
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "track_genre",
}

# Plages théoriques des caractéristiques audio Spotify
# Source : documentation de l'API Spotify
AUDIO_FEATURE_RANGES = {
    "danceability":     (0.0, 1.0),
    "energy":           (0.0, 1.0),
    "loudness":         (-60.0, 5.0),   # en dB, généralement négatif
    "speechiness":      (0.0, 1.0),
    "acousticness":     (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness":         (0.0, 1.0),
    "valence":          (0.0, 1.0),
    "tempo":            (0.0, 300.0),   # en BPM
    "duration_ms":      (0, 10_000_000), # < ~3h
    "popularity":       (0, 100),
}


def load_raw_data(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """
    Charge le CSV brut et retourne un DataFrame.
    Lève une erreur explicite si le fichier n'existe pas.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"❌ Fichier introuvable : {path}\n"
            "→ Télécharge le dataset depuis Kaggle et place-le dans data/raw/dataset.csv\n"
            "→ Commande Kaggle API : kaggle datasets download maharshipandya/spotify-tracks-dataset -p data/raw --unzip"
        )

    logger.info(f"Chargement du fichier : {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


def check_required_columns(df: pd.DataFrame) -> dict:
    """
    Vérifie que toutes les colonnes requises sont présentes.

    Retourne un dictionnaire avec :
      - 'present'  : colonnes requises présentes
      - 'missing'  : colonnes requises absentes (bloquant)
      - 'extra'    : colonnes supplémentaires non attendues (info)
    """
    actual = set(df.columns)
    missing = REQUIRED_COLUMNS - actual
    extra   = actual - REQUIRED_COLUMNS

    if missing:
        logger.error(f"Colonnes manquantes : {missing}")
    else:
        logger.info("✅ Toutes les colonnes requises sont présentes.")

    if extra:
        logger.info(f"Colonnes supplémentaires (non bloquant) : {extra}")

    return {
        "present": list(REQUIRED_COLUMNS & actual),
        "missing": list(missing),
        "extra":   list(extra),
    }


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le nombre et le pourcentage de valeurs manquantes par colonne.
    Retourne un DataFrame trié par % manquant décroissant.
    """
    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df) * 100).round(2)

    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct":   missing_pct,
    }).sort_values("missing_pct", ascending=False)

    # On filtre pour n'afficher que les colonnes avec au moins une valeur manquante
    report = report[report["missing_count"] > 0]

    if report.empty:
        logger.info("✅ Aucune valeur manquante détectée.")
    else:
        logger.warning(f"⚠️  {len(report)} colonnes ont des valeurs manquantes :")
        logger.warning(f"\n{report.to_string()}")

    return report


def check_duplicates(df: pd.DataFrame) -> dict:
    """
    Vérifie deux types de doublons :
      1. Lignes entièrement identiques.
      2. Mêmes track_id avec des informations différentes (ex: genres différents).
         → Ce n'est PAS nécessairement une erreur dans ce dataset.
    """
    # 1. Doublons complets (toutes colonnes identiques)
    full_dups = df.duplicated().sum()
    logger.info(f"Doublons complets : {full_dups:,}")

    # 2. track_id répétés
    if ID_COL in df.columns:
        id_counts   = df[ID_COL].value_counts()
        repeated_ids = (id_counts > 1).sum()
        max_repeats  = id_counts.max()

        logger.info(
            f"track_id uniques : {df[ID_COL].nunique():,} / {len(df):,} lignes"
        )
        logger.info(
            f"track_id répétés (plusieurs genres) : {repeated_ids:,} "
            f"(maximum {max_repeats} occurrences pour un même morceau)"
        )
        logger.info(
            "→ Note : un même morceau peut apparaître dans plusieurs genres. "
            "Ce n'est pas une erreur mais doit être pris en compte lors du split train/test."
        )
    else:
        repeated_ids = None
        max_repeats  = None
        logger.warning(f"Colonne '{ID_COL}' introuvable — vérification des doublons par ID ignorée.")

    return {
        "full_duplicates":   int(full_dups),
        "repeated_track_ids": repeated_ids,
        "max_id_occurrences": max_repeats,
    }


def check_audio_feature_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie que les valeurs des caractéristiques audio sont dans les plages attendues.

    Pour chaque caractéristique, calcule le nombre de valeurs hors plage.
    Attention : des valeurs hors plage peuvent indiquer des erreurs de collecte
    ou des cas légitimes (ex: loudness légèrement positif pour du contenu normalisé).
    On les signale sans les supprimer.
    """
    results = []

    for col, (lo, hi) in AUDIO_FEATURE_RANGES.items():
        if col not in df.columns:
            continue

        series = df[col].dropna()
        out_of_range = ((series < lo) | (series > hi)).sum()
        pct = round(out_of_range / len(series) * 100, 3) if len(series) > 0 else 0.0

        results.append({
            "feature":      col,
            "expected_min": lo,
            "expected_max": hi,
            "actual_min":   round(float(series.min()), 4) if len(series) > 0 else None,
            "actual_max":   round(float(series.max()), 4) if len(series) > 0 else None,
            "out_of_range_count": int(out_of_range),
            "out_of_range_pct":   pct,
        })

    report = pd.DataFrame(results)

    outliers = report[report["out_of_range_count"] > 0]
    if outliers.empty:
        logger.info("✅ Toutes les valeurs audio sont dans les plages attendues.")
    else:
        logger.warning(
            f"⚠️  {len(outliers)} caractéristiques ont des valeurs hors plage :\n"
            f"{outliers[['feature','out_of_range_count','out_of_range_pct']].to_string(index=False)}"
        )

    return report


def check_dtypes(df: pd.DataFrame) -> dict:
    """
    Affiche les types réels des colonnes et identifie les cas suspects
    (ex: une colonne numérique stockée en texte).
    """
    dtype_info = df.dtypes.astype(str).to_dict()

    # Colonnes numériques attendues
    expected_numeric = set(AUDIO_FEATURES) | {TARGET_COL}
    actually_numeric = set(df.select_dtypes(include=[np.number]).columns)

    not_numeric = expected_numeric - actually_numeric
    if not_numeric:
        logger.warning(
            f"⚠️  Ces colonnes devraient être numériques mais ne le sont pas : {not_numeric}"
        )
    else:
        logger.info("✅ Toutes les colonnes numériques attendues ont le bon type.")

    return dtype_info


def run_full_validation(df: pd.DataFrame) -> dict:
    """
    Exécute toutes les vérifications dans l'ordre et retourne un rapport complet.
    """
    logger.info("=" * 60)
    logger.info("VALIDATION DES DONNÉES — DÉBUT")
    logger.info("=" * 60)

    report = {}

    # 1. Colonnes
    logger.info("\n── 1. Vérification des colonnes ──")
    report["columns"] = check_required_columns(df)

    # Si des colonnes critiques manquent, on arrête ici
    if report["columns"]["missing"]:
        logger.error(
            f"BLOQUANT : colonnes absentes {report['columns']['missing']}. "
            "Vérifiez la source du dataset."
        )
        return report

    # 2. Types
    logger.info("\n── 2. Types des colonnes ──")
    report["dtypes"] = check_dtypes(df)

    # 3. Valeurs manquantes
    logger.info("\n── 3. Valeurs manquantes ──")
    mv_df = check_missing_values(df)
    report["missing_values"] = mv_df.to_dict() if not mv_df.empty else {}

    # 4. Doublons
    logger.info("\n── 4. Doublons ──")
    report["duplicates"] = check_duplicates(df)

    # 5. Plages des caractéristiques audio
    logger.info("\n── 5. Plages des caractéristiques audio ──")
    ranges_df = check_audio_feature_ranges(df)
    report["feature_ranges"] = ranges_df.to_dict(orient="records")

    # 6. Statistiques générales
    logger.info("\n── 6. Statistiques générales ──")
    logger.info(f"  Genres uniques       : {df['track_genre'].nunique() if 'track_genre' in df.columns else 'N/A'}")
    logger.info(f"  Artistes uniques     : {df['artists'].nunique() if 'artists' in df.columns else 'N/A'}")
    logger.info(f"  Popularité — moy     : {df[TARGET_COL].mean():.1f}" if TARGET_COL in df.columns else "")
    logger.info(f"  Popularité — médiane : {df[TARGET_COL].median():.1f}" if TARGET_COL in df.columns else "")
    logger.info(f"  Popularité — std     : {df[TARGET_COL].std():.1f}" if TARGET_COL in df.columns else "")

    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION DES DONNÉES — TERMINÉE")
    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    # ── Exécution directe du script ──────────────────────────────────────────
    try:
        df = load_raw_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Aperçu du dataset
    print("\n📋 Aperçu des premières lignes :")
    print(df.head(3).to_string())

    print("\n📐 Dimensions :")
    print(f"  {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    print("\n🏷️  Colonnes disponibles :")
    for col in df.columns:
        print(f"  • {col} ({df[col].dtype})")

    # Validation complète
    report = run_full_validation(df)

    print("\n✅ Validation terminée. Consultez les logs ci-dessus pour le détail.")
