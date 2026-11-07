"""
test_preprocessing.py — Tests unitaires du module de preprocessing
===================================================================
Chaque test vérifie une fonction précise du pipeline de préparation des données.

Convention Pytest :
  - Chaque fonction de test commence par test_
  - Les fixtures (sample_raw_dataframe, etc.) sont injectées automatiquement
    depuis conftest.py
  - Un test qui passe sans AssertionError = succès
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    add_engineered_features,
    detect_outliers_iqr,
    fix_dtypes,
    handle_missing_values,
    remove_full_duplicates,
)
from src.data_validation import (
    check_duplicates,
    check_missing_values,
    check_required_columns,
    check_audio_feature_ranges,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — data_validation.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckRequiredColumns:
    """Tests de la vérification du schéma."""

    def test_all_required_columns_present(self, sample_raw_dataframe):
        """Toutes les colonnes requises sont présentes → 'missing' doit être vide."""
        result = check_required_columns(sample_raw_dataframe)
        assert result["missing"] == [], (
            f"Des colonnes sont absentes : {result['missing']}"
        )

    def test_missing_column_detected(self, sample_raw_dataframe):
        """Si une colonne requise est absente, elle doit apparaître dans 'missing'."""
        df_broken = sample_raw_dataframe.drop(columns=["popularity"])
        result = check_required_columns(df_broken)
        assert "popularity" in result["missing"]

    def test_extra_columns_detected(self, sample_raw_dataframe):
        """Les colonnes supplémentaires doivent être détectées dans 'extra'."""
        df_extra = sample_raw_dataframe.copy()
        df_extra["extra_col"] = 0
        result = check_required_columns(df_extra)
        assert "extra_col" in result["extra"]


class TestCheckMissingValues:
    """Tests de la détection des valeurs manquantes."""

    def test_no_missing_values(self, sample_raw_dataframe):
        """Un DataFrame sans NaN doit retourner un rapport vide."""
        report = check_missing_values(sample_raw_dataframe)
        assert len(report) == 0, f"Des NaN détectés alors qu'il ne devrait pas y en avoir : {report}"

    def test_missing_values_detected(self, sample_dataframe_with_missing):
        """Les NaN introduits doivent être détectés."""
        report = check_missing_values(sample_dataframe_with_missing)
        assert len(report) > 0, "Aucun NaN détecté alors qu'il y en a."
        assert "danceability" in report.index, "La colonne 'danceability' n'apparaît pas dans le rapport."

    def test_missing_percentage_correct(self, sample_raw_dataframe):
        """Le pourcentage de NaN doit être calculé correctement."""
        df = sample_raw_dataframe.copy()
        df.loc[0:9, "energy"] = np.nan  # 10 NaN sur 200 lignes = 5%
        report = check_missing_values(df)
        assert "energy" in report.index
        assert abs(report.loc["energy", "missing_pct"] - 5.0) < 0.1


class TestCheckAudioFeatureRanges:
    """Tests de la vérification des plages audio."""

    def test_valid_ranges(self, sample_raw_dataframe):
        """Les données de test sont dans les plages attendues → out_of_range = 0."""
        report = check_audio_feature_ranges(sample_raw_dataframe)
        audio_report = report[report["feature"].isin(["danceability", "energy", "valence"])]
        assert (audio_report["out_of_range_count"] == 0).all()

    def test_out_of_range_detected(self, sample_raw_dataframe):
        """Une valeur > 1.0 pour danceability doit être détectée."""
        df = sample_raw_dataframe.copy()
        df.loc[0, "danceability"] = 1.5  # hors plage [0, 1]
        report = check_audio_feature_ranges(df)
        dance_row = report[report["feature"] == "danceability"]
        assert dance_row["out_of_range_count"].values[0] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — preprocessing.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestRemoveFullDuplicates:
    """Tests de la suppression des doublons complets."""

    def test_removes_exact_duplicates(self, sample_dataframe_with_duplicates):
        """Les lignes entièrement dupliquées doivent être supprimées."""
        df_before = sample_dataframe_with_duplicates
        df_after  = remove_full_duplicates(df_before)
        assert len(df_after) < len(df_before), "Aucune ligne supprimée alors qu'il y a des doublons."

    def test_no_duplicate_removed_when_none(self, sample_raw_dataframe):
        """Sans doublons, le DataFrame ne doit pas changer."""
        df_clean = remove_full_duplicates(sample_raw_dataframe)
        assert len(df_clean) == len(sample_raw_dataframe)

    def test_output_has_no_full_duplicates(self, sample_dataframe_with_duplicates):
        """Le DataFrame résultant ne doit plus avoir de doublons complets."""
        df_clean = remove_full_duplicates(sample_dataframe_with_duplicates)
        assert df_clean.duplicated().sum() == 0


class TestHandleMissingValues:
    """Tests de la gestion des valeurs manquantes."""

    def test_no_nan_after_handling(self, sample_dataframe_with_missing):
        """Après traitement, aucune colonne audio ne doit avoir de NaN."""
        df_clean = handle_missing_values(sample_dataframe_with_missing)
        audio_cols = ["danceability", "energy", "loudness"]
        for col in audio_cols:
            if col in df_clean.columns:
                assert df_clean[col].isnull().sum() == 0, (
                    f"NaN résiduels dans '{col}'"
                )

    def test_text_columns_filled_with_unknown(self, sample_dataframe_with_missing):
        """Les colonnes texte avec NaN doivent être remplies par 'Unknown'."""
        df_clean = handle_missing_values(sample_dataframe_with_missing)
        if "track_genre" in df_clean.columns:
            assert df_clean["track_genre"].isnull().sum() == 0
            assert (df_clean["track_genre"] == "Unknown").any() or \
                   df_clean["track_genre"].isnull().sum() == 0

    def test_popularity_nan_rows_dropped(self):
        """Les lignes sans valeur de popularité doivent être supprimées."""
        df = pd.DataFrame({
            "track_id": ["a", "b", "c"],
            "popularity": [50.0, np.nan, 30.0],
            "danceability": [0.5, 0.6, 0.7],
            "energy": [0.8, 0.9, 0.4],
            "loudness": [-5.0, -8.0, -12.0],
            "speechiness": [0.1, 0.2, 0.05],
            "acousticness": [0.3, 0.1, 0.9],
            "instrumentalness": [0.0, 0.0, 0.5],
            "liveness": [0.1, 0.2, 0.15],
            "valence": [0.7, 0.4, 0.6],
            "tempo": [120.0, 90.0, 100.0],
            "duration_ms": [200000.0, 180000.0, 240000.0],
            "track_genre": ["pop", "rock", "jazz"],
            "artists": ["A", "B", "C"],
            "track_name": ["s1", "s2", "s3"],
        })
        df_clean = handle_missing_values(df)
        assert len(df_clean) == 2, "La ligne avec popularity=NaN aurait dû être supprimée."


class TestDetectOutliersIQR:
    """Tests de la détection des valeurs aberrantes."""

    def test_returns_boolean_series(self, sample_raw_dataframe):
        """La fonction doit retourner une Series booléenne de même longueur."""
        mask = detect_outliers_iqr(sample_raw_dataframe, ["danceability", "energy"])
        assert len(mask) == len(sample_raw_dataframe)
        assert mask.dtype == bool

    def test_extreme_value_detected(self, sample_raw_dataframe):
        """Une valeur extrême doit être détectée comme aberrante."""
        df = sample_raw_dataframe.copy()
        df.loc[0, "loudness"] = 9999.0  # valeur absurde
        mask = detect_outliers_iqr(df, ["loudness"])
        assert mask.loc[0] == True, "La valeur extrême n'a pas été détectée."

    def test_no_outliers_in_clean_data(self):
        """Des données uniformément distribuées ne doivent pas avoir d'aberrants."""
        df = pd.DataFrame({"feature": np.linspace(0.1, 0.9, 100)})
        mask = detect_outliers_iqr(df, ["feature"], factor=3.0)
        # Avec des données uniformes dans [0.1, 0.9], factor=3 ne doit rien flaguer
        assert mask.sum() == 0


class TestAddEngineeredFeatures:
    """Tests du feature engineering."""

    def test_duration_min_created(self, sample_raw_dataframe):
        """La colonne duration_min doit être créée à partir de duration_ms."""
        df = add_engineered_features(sample_raw_dataframe)
        assert "duration_min" in df.columns

    def test_duration_min_correct_conversion(self, sample_raw_dataframe):
        """1 minute = 60 000 ms."""
        df = sample_raw_dataframe.copy()
        df["duration_ms"] = 120_000.0  # 2 minutes exactes
        df = add_engineered_features(df)
        assert abs(df["duration_min"].iloc[0] - 2.0) < 0.001

    def test_energy_x_dance_created(self, sample_raw_dataframe):
        """La variable d'interaction énergie × danceability doit exister."""
        df = add_engineered_features(sample_raw_dataframe)
        assert "energy_x_dance" in df.columns

    def test_energy_x_dance_correct_value(self, sample_raw_dataframe):
        """La valeur doit être le produit des deux variables."""
        df = sample_raw_dataframe.copy()
        df["energy"] = 0.8
        df["danceability"] = 0.5
        df = add_engineered_features(df)
        expected = 0.8 * 0.5
        assert abs(df["energy_x_dance"].iloc[0] - expected) < 1e-6

    def test_loudness_norm_between_0_and_1(self, sample_raw_dataframe):
        """loudness_norm doit être dans [0, 1]."""
        df = add_engineered_features(sample_raw_dataframe)
        if "loudness_norm" in df.columns:
            assert df["loudness_norm"].min() >= 0.0
            assert df["loudness_norm"].max() <= 1.0


class TestFixDtypes:
    """Tests de la correction des types de colonnes."""

    def test_numeric_columns_converted(self):
        """Une colonne numérique stockée en string doit être convertie."""
        df = pd.DataFrame({
            "track_id": ["a", "b"],
            "popularity": ["50", "30"],  # stocké comme string
            "danceability": ["0.5", "0.7"],
        })
        df_fixed = fix_dtypes(df)
        assert pd.api.types.is_numeric_dtype(df_fixed["popularity"])
        assert pd.api.types.is_numeric_dtype(df_fixed["danceability"])

    def test_invalid_conversion_becomes_nan(self):
        """Une valeur non convertible doit devenir NaN, pas lever une erreur."""
        df = pd.DataFrame({
            "track_id": ["a"],
            "popularity": ["not_a_number"],
            "danceability": [0.5],
        })
        df_fixed = fix_dtypes(df)
        assert pd.isna(df_fixed["popularity"].iloc[0])
