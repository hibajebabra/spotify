"""
preprocessing.py — Pipeline de préparation des données
=======================================================
Rôle : Nettoyer, enrichir et exporter les données vers un fichier Parquet.

Ce module :
  1. Charge et valide le CSV brut.
  2. Supprime les lignes entièrement dupliquées.
  3. Gère les valeurs manquantes.
  4. Ajoute une colonne indicatrice de valeurs aberrantes (sans les supprimer).
  5. Convertit les types si nécessaire.
  6. Sauvegarde la liste des colonnes du modèle (pour aligner le formulaire Streamlit).
  7. Exporte les données nettoyées en Parquet.

Pourquoi Parquet ?
  → Format binaire compressé, 10x plus rapide à lire que CSV pour les analyses.
  → Conserve les types de données (pas de ré-inférence au chargement).
  → Standard industriel pour les pipelines Data Engineering.

Usage :
  python -m src.preprocessing
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    RAW_CSV_PATH,
    PROCESSED_PARQUET,
    FEATURE_COLS_PATH,
    AUDIO_FEATURES,
    TARGET_COL,
    ID_COL,
    PROC_DATA_DIR,
)
from src.data_validation import load_raw_data, run_full_validation

# ── Configuration des logs ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def remove_full_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes entièrement identiques (toutes colonnes confondues).
    Les track_id répétés avec des genres différents ne sont PAS supprimés ici —
    c'est un comportement attendu du dataset Spotify.
    """
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)

    if n_removed > 0:
        logger.info(f"  {n_removed:,} lignes entièrement dupliquées supprimées.")
    else:
        logger.info("  Aucun doublon complet détecté.")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratégie de gestion des valeurs manquantes :

    - Pour les colonnes texte (track_name, artists, track_genre) :
        → Remplace NaN par la chaîne "Unknown" pour ne pas perdre la ligne.
    - Pour les colonnes numériques audio :
        → Remplace NaN par la médiane de la colonne.
        → La médiane est plus robuste que la moyenne face aux valeurs aberrantes.
    - Pour la cible (popularity) :
        → Supprime les lignes sans valeur cible (on ne peut pas entraîner sur du vide).

    Pourquoi ne pas supprimer les lignes avec NaN ?
    → Sur un large dataset, les valeurs manquantes sont souvent localisées
       (quelques colonnes seulement). Supprimer toute la ligne serait excessif.
    """
    df = df.copy()

    # 1. Cible manquante → on supprime car on ne peut pas entraîner sans elle
    if TARGET_COL in df.columns:
        n_before = len(df)
        df = df.dropna(subset=[TARGET_COL])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.warning(f"  {n_dropped:,} lignes supprimées (popularité manquante).")

    # 2. Colonnes texte → remplacement par "Unknown"
    text_cols = ["track_name", "artists", "track_genre"]
    for col in text_cols:
        if col in df.columns and df[col].isnull().any():
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna("Unknown")
            logger.info(f"  '{col}' : {n_missing:,} valeurs NaN remplacées par 'Unknown'.")

    # 3. Colonnes numériques audio → remplacement par la médiane
    for col in AUDIO_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            n_missing  = df[col].isnull().sum()
            df[col]    = df[col].fillna(median_val)
            logger.info(
                f"  '{col}' : {n_missing:,} NaN remplacés par la médiane ({median_val:.4f})."
            )

    return df


def detect_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 3.0) -> pd.Series:
    """
    Détecte les valeurs aberrantes avec la méthode IQR (Interquartile Range).

    Une observation est marquée comme aberrante si au moins une de ses
    caractéristiques audio sort de la plage [Q1 - factor*IQR, Q3 + factor*IQR].

    On utilise factor=3.0 (plus permissif que le classique 1.5) car la musique
    a une grande diversité naturelle (silence complet vs. métal très fort).

    IMPORTANT : on ne supprime PAS ces observations — on ajoute une colonne
    indicatrice 'is_outlier' pour permettre une analyse ultérieure.
    """
    mask = pd.Series(False, index=df.index)

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lo = q1 - factor * iqr
        hi = q3 + factor * iqr

        col_mask = (df[col] < lo) | (df[col] > hi)
        mask = mask | col_mask

    n_outliers = mask.sum()
    pct = round(n_outliers / len(df) * 100, 2)
    logger.info(f"  Valeurs aberrantes détectées (IQR, factor={factor}) : {n_outliers:,} ({pct}%)")

    return mask


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute quelques variables dérivées qui peuvent enrichir l'analyse.

    Note : ces variables sont des transformations des variables existantes,
    pas des fuites de données (elles n'utilisent pas la cible).

    - 'duration_min' : durée en minutes (plus lisible que les millisecondes).
    - 'energy_x_dance' : interaction entre énergie et danceability.
    - 'loudness_norm' : loudness normalisée entre 0 et 1 pour faciliter la comparaison.
    """
    df = df.copy()

    if "duration_ms" in df.columns:
        df["duration_min"] = (df["duration_ms"] / 60_000).round(3)

    if "energy" in df.columns and "danceability" in df.columns:
        df["energy_x_dance"] = df["energy"] * df["danceability"]

    if "loudness" in df.columns:
        lo = df["loudness"].min()
        hi = df["loudness"].max()
        if hi > lo:
            df["loudness_norm"] = (df["loudness"] - lo) / (hi - lo)

    logger.info("  Variables dérivées ajoutées : duration_min, energy_x_dance, loudness_norm")

    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes qui devraient être numériques mais ne le sont pas.
    Remplace les erreurs de conversion par NaN plutôt que de lever une exception.

    Compatible pandas 2.x et 3.x : on ne vérifie pas uniquement dtype==object
    car pandas 3.x utilise StringDtype pour les colonnes texte.
    """
    df = df.copy()

    for col in AUDIO_FEATURES + [TARGET_COL]:
        if col not in df.columns:
            continue
        # On tente la conversion si la colonne n'est pas déjà purement numérique
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"  '{col}' n'est pas numerique — tentative de conversion.")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_feature_columns(df: pd.DataFrame) -> None:
    """
    Sauvegarde la liste des colonnes utilisées comme variables d'entrée du modèle.
    Ce fichier JSON sera lu par app.py (Streamlit) pour aligner le formulaire
    avec le pipeline d'entraînement.

    → C'est un mécanisme de sécurité : si tu ajoutes ou supprimes une variable,
       le formulaire Streamlit se mettra automatiquement à jour.
    """
    # Colonnes effectivement présentes dans le DataFrame parmi les features audio
    available_features = [col for col in AUDIO_FEATURES if col in df.columns]

    PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(available_features, f, indent=2)

    logger.info(f"  Colonnes du modèle sauvegardées : {FEATURE_COLS_PATH}")
    logger.info(f"  Variables : {available_features}")


def run_preprocessing_pipeline(csv_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """
    Pipeline complète de préparation des données.
    Retourne le DataFrame nettoyé et le sauvegarde en Parquet.
    """
    logger.info("=" * 60)
    logger.info("PREPROCESSING — DÉBUT")
    logger.info("=" * 60)

    # ── Étape 1 : Chargement ──────────────────────────────────────────────────
    logger.info("\n── Étape 1 : Chargement ──")
    df = load_raw_data(csv_path)
    logger.info(f"  Dimensions initiales : {df.shape}")

    # ── Étape 2 : Validation ──────────────────────────────────────────────────
    logger.info("\n── Étape 2 : Validation du schéma ──")
    report = run_full_validation(df)
    if report["columns"]["missing"]:
        raise ValueError(
            f"Colonnes manquantes dans le dataset : {report['columns']['missing']}"
        )

    # ── Étape 3 : Correction des types ────────────────────────────────────────
    logger.info("\n── Étape 3 : Correction des types ──")
    df = fix_dtypes(df)

    # ── Étape 4 : Doublons complets ───────────────────────────────────────────
    logger.info("\n── Étape 4 : Suppression des doublons complets ──")
    df = remove_full_duplicates(df)

    # ── Étape 5 : Valeurs manquantes ──────────────────────────────────────────
    logger.info("\n── Étape 5 : Gestion des valeurs manquantes ──")
    df = handle_missing_values(df)

    # ── Étape 6 : Détection des valeurs aberrantes ────────────────────────────
    logger.info("\n── Étape 6 : Détection des valeurs aberrantes (IQR) ──")
    audio_cols_present = [c for c in AUDIO_FEATURES if c in df.columns]
    outlier_mask = detect_outliers_iqr(df, audio_cols_present)
    df["is_outlier"] = outlier_mask.astype(int)  # 0 = normal, 1 = aberrant

    # ── Étape 7 : Feature engineering ────────────────────────────────────────
    logger.info("\n── Étape 7 : Feature engineering ──")
    df = add_engineered_features(df)

    # ── Étape 8 : Sauvegarde des colonnes du modèle ───────────────────────────
    logger.info("\n── Étape 8 : Sauvegarde de la liste des features ──")
    save_feature_columns(df)

    # ── Étape 9 : Export Parquet ──────────────────────────────────────────────
    logger.info("\n── Étape 9 : Export en Parquet ──")
    PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PARQUET, index=False)
    logger.info(f"  ✅ Données sauvegardées : {PROCESSED_PARQUET}")
    logger.info(f"  Dimensions finales : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    # ── Résumé final ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING — TERMINÉ")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    df_clean = run_preprocessing_pipeline()

    print("\n📊 Aperçu du dataset nettoyé :")
    print(df_clean.head(3).to_string())

    print("\n📐 Dimensions finales :")
    print(f"  {df_clean.shape[0]:,} lignes × {df_clean.shape[1]} colonnes")

    print("\n🏷️  Colonnes disponibles :")
    for col in df_clean.columns:
        print(f"  • {col} ({df_clean[col].dtype})")

    print(f"\n🎯 Popularité — min: {df_clean['popularity'].min()}, "
          f"max: {df_clean['popularity'].max()}, "
          f"moyenne: {df_clean['popularity'].mean():.1f}")

    outlier_pct = df_clean["is_outlier"].mean() * 100
    print(f"⚠️  Observations aberrantes : {outlier_pct:.1f}%")
