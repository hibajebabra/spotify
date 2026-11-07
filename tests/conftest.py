"""
conftest.py — Fixtures partagées pour tous les tests Pytest
===========================================================
Ce fichier est automatiquement chargé par Pytest avant les tests.
Les fixtures définies ici sont disponibles dans tous les fichiers de test.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_dataframe():
    """
    Crée un DataFrame minimal qui simule le dataset Spotify brut.
    Utilisé pour tester les fonctions sans dépendre du fichier CSV réel.
    """
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "track_id":          [f"id_{i:04d}" for i in range(n)],
        "track_name":        [f"Song {i}" for i in range(n)],
        "artists":           [f"Artist {i % 20}" for i in range(n)],
        "album_name":        [f"Album {i % 30}" for i in range(n)],
        "popularity":        np.random.randint(0, 101, n).astype(float),
        "duration_ms":       np.random.randint(120_000, 360_000, n).astype(float),
        "danceability":      np.random.uniform(0.0, 1.0, n),
        "energy":            np.random.uniform(0.0, 1.0, n),
        "loudness":          np.random.uniform(-40.0, 0.0, n),
        "speechiness":       np.random.uniform(0.0, 0.5, n),
        "acousticness":      np.random.uniform(0.0, 1.0, n),
        "instrumentalness":  np.random.uniform(0.0, 1.0, n),
        "liveness":          np.random.uniform(0.0, 0.5, n),
        "valence":           np.random.uniform(0.0, 1.0, n),
        "tempo":             np.random.uniform(60.0, 200.0, n),
        "track_genre":       np.random.choice(["pop", "rock", "jazz", "classical", "hip-hop"], n),
        "explicit":          np.random.choice([True, False], n),
        "key":               np.random.randint(0, 12, n),
        "mode":              np.random.randint(0, 2, n),
        "time_signature":    np.random.choice([3, 4], n),
    })

    return df


@pytest.fixture
def sample_dataframe_with_missing(sample_raw_dataframe):
    """DataFrame avec quelques valeurs manquantes pour tester le preprocessing."""
    df = sample_raw_dataframe.copy()
    # Introduire des NaN dans différentes colonnes
    df.loc[0:4,   "danceability"] = np.nan
    df.loc[5:7,   "track_genre"]  = np.nan
    df.loc[10:12, "loudness"]     = np.nan
    return df


@pytest.fixture
def sample_dataframe_with_duplicates(sample_raw_dataframe):
    """DataFrame avec des lignes dupliquées."""
    df = sample_raw_dataframe.copy()
    # Dupliquer les 5 premières lignes
    duplicates = df.head(5).copy()
    return pd.concat([df, duplicates], ignore_index=True)


@pytest.fixture
def feature_columns():
    """Liste des features audio utilisées dans les modèles."""
    return [
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


@pytest.fixture
def tmp_feature_cols_file(tmp_path, feature_columns):
    """Fichier JSON temporaire avec les colonnes du modèle."""
    path = tmp_path / "feature_columns.json"
    with open(path, "w") as f:
        json.dump(feature_columns, f)
    return path
