"""
test_predictions.py — Tests unitaires des pipelines de modèles
==============================================================
Ces tests vérifient que les pipelines sklearn peuvent :
  - Être instanciés et entraînés sur des données synthétiques.
  - Produire des prédictions du bon type et de la bonne forme.
  - Retourner des métriques cohérentes.
  - Fonctionner correctement après sauvegarde / rechargement pickle.

Note : on ne teste PAS les performances absolues du modèle ici —
cela dépend du dataset réel. On teste la STRUCTURE et la ROBUSTESSE du pipeline.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.train_regression import (
    build_pipeline,
    compute_metrics,
    get_feature_columns,
)
from src.train_clustering import build_clustering_pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures locales
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_X_y(feature_columns):
    """
    Génère des données X (features) et y (cible) synthétiques pour entraîner
    les modèles dans les tests sans dépendre du CSV réel.
    """
    np.random.seed(42)
    n = 500

    X = pd.DataFrame({
        "danceability":      np.random.uniform(0.0, 1.0, n),
        "energy":            np.random.uniform(0.0, 1.0, n),
        "loudness":          np.random.uniform(-40.0, 0.0, n),
        "speechiness":       np.random.uniform(0.0, 0.5, n),
        "acousticness":      np.random.uniform(0.0, 1.0, n),
        "instrumentalness":  np.random.uniform(0.0, 1.0, n),
        "liveness":          np.random.uniform(0.0, 0.5, n),
        "valence":           np.random.uniform(0.0, 1.0, n),
        "tempo":             np.random.uniform(60.0, 200.0, n),
        "duration_ms":       np.random.uniform(120_000, 360_000, n),
    })[feature_columns]

    # Cible synthétique : combinaison linéaire + bruit (n'a pas de sens musical)
    y = (
        30
        + 20 * X["energy"]
        + 15 * X["danceability"]
        - 0.5 * np.abs(X["loudness"])
        + np.random.normal(0, 10, n)
    ).clip(0, 100)
    y = pd.Series(y.values, name="popularity")

    return X, y


@pytest.fixture
def train_test_split_data(synthetic_X_y):
    """Split 80/20 des données synthétiques."""
    X, y = synthetic_X_y
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — build_pipeline et compute_metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildPipeline:
    """Tests de la construction des pipelines sklearn."""

    def test_pipeline_with_linear_regression(self, train_test_split_data):
        """Un pipeline LinearRegression doit s'entraîner et prédire."""
        X_train, X_test, y_train, y_test = train_test_split_data
        pipe = build_pipeline(LinearRegression(), use_scaler=True)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        assert preds.shape == (len(X_test),), "Le vecteur de prédictions a la mauvaise forme."
        assert not np.any(np.isnan(preds)), "Des NaN dans les prédictions."

    def test_pipeline_with_random_forest(self, train_test_split_data):
        """Un pipeline RandomForest doit s'entraîner et prédire."""
        X_train, X_test, y_train, y_test = train_test_split_data
        pipe = build_pipeline(
            RandomForestRegressor(n_estimators=10, random_state=42),
            use_scaler=False,
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        assert len(preds) == len(X_test)
        assert np.all(np.isfinite(preds)), "Des valeurs infinies dans les prédictions."

    def test_pipeline_with_dummy_baseline(self, train_test_split_data):
        """Le DummyRegressor doit prévoir exactement la moyenne."""
        X_train, X_test, y_train, y_test = train_test_split_data
        pipe = build_pipeline(DummyRegressor(strategy="mean"), use_scaler=False)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        expected = y_train.mean()
        assert np.allclose(preds, expected, atol=0.001), (
            f"DummyRegressor ne prédit pas la moyenne : {preds[:3]} ≠ {expected}"
        )

    def test_pipeline_handles_nan_input(self, feature_columns):
        """Le pipeline doit gérer les NaN en entrée (grâce au SimpleImputer)."""
        X_train = pd.DataFrame(np.random.rand(100, len(feature_columns)), columns=feature_columns)
        y_train = pd.Series(np.random.uniform(0, 100, 100))

        X_test = pd.DataFrame(np.random.rand(10, len(feature_columns)), columns=feature_columns)
        # Introduire des NaN dans X_test
        X_test.iloc[0, 0] = np.nan
        X_test.iloc[1, 2] = np.nan

        pipe = build_pipeline(LinearRegression(), use_scaler=True)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        assert len(preds) == 10
        assert not np.any(np.isnan(preds)), "Des NaN en sortie malgré le SimpleImputer."


class TestComputeMetrics:
    """Tests du calcul des métriques d'évaluation."""

    def test_perfect_predictions(self):
        """Prédictions parfaites → MAE=0, RMSE=0, R²=1."""
        y_true = pd.Series([10.0, 50.0, 80.0, 30.0])
        y_pred = np.array([10.0, 50.0, 80.0, 30.0])
        metrics = compute_metrics(y_true, y_pred)

        assert metrics["MAE"]  == 0.0
        assert metrics["RMSE"] == 0.0
        assert metrics["R2"]   == 1.0

    def test_metrics_are_floats(self, synthetic_X_y):
        """Les métriques doivent être des flottants, pas des erreurs."""
        X, y = synthetic_X_y
        pipe = build_pipeline(DummyRegressor(), use_scaler=False)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        metrics = compute_metrics(y, preds)

        assert isinstance(metrics["MAE"],  float)
        assert isinstance(metrics["RMSE"], float)
        assert isinstance(metrics["R2"],   float)

    def test_r2_dummy_near_zero(self, synthetic_X_y):
        """Un DummyRegressor doit avoir un R² proche de 0."""
        X, y = synthetic_X_y
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = build_pipeline(DummyRegressor(strategy="mean"), use_scaler=False)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        metrics = compute_metrics(y_te, preds)

        assert -0.1 <= metrics["R2"] <= 0.05, (
            f"R² du DummyRegressor devrait être ~0, obtenu : {metrics['R2']}"
        )

    def test_mae_non_negative(self, train_test_split_data):
        """La MAE doit toujours être positive ou nulle."""
        X_train, X_test, y_train, y_test = train_test_split_data
        pipe = build_pipeline(LinearRegression(), use_scaler=True)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = compute_metrics(y_test, preds)

        assert metrics["MAE"] >= 0.0
        assert metrics["RMSE"] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Sauvegarde et rechargement des modèles
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelPersistence:
    """Tests de la sauvegarde et du rechargement des pipelines."""

    def test_save_and_reload_pipeline(self, train_test_split_data):
        """Un pipeline sauvegardé en pickle doit donner les mêmes prédictions."""
        X_train, X_test, y_train, y_test = train_test_split_data
        pipe = build_pipeline(LinearRegression(), use_scaler=True)
        pipe.fit(X_train, y_train)
        preds_before = pipe.predict(X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pkl"
            with open(path, "wb") as f:
                pickle.dump(pipe, f)

            with open(path, "rb") as f:
                pipe_loaded = pickle.load(f)

        preds_after = pipe_loaded.predict(X_test)
        np.testing.assert_array_almost_equal(
            preds_before, preds_after,
            decimal=6,
            err_msg="Les prédictions avant et après rechargement diffèrent.",
        )

    def test_reloaded_model_correct_feature_count(self, train_test_split_data, feature_columns):
        """Le modèle rechargé doit accepter le bon nombre de features."""
        X_train, X_test, y_train, y_test = train_test_split_data
        pipe = build_pipeline(RandomForestRegressor(n_estimators=5, random_state=42))
        pipe.fit(X_train, y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rf_model.pkl"
            with open(path, "wb") as f:
                pickle.dump(pipe, f)
            with open(path, "rb") as f:
                pipe_loaded = pickle.load(f)

        preds = pipe_loaded.predict(X_test)
        assert len(preds) == len(X_test)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Pipeline de clustering
# ═══════════════════════════════════════════════════════════════════════════════

class TestClusteringPipeline:
    """Tests du pipeline K-Means."""

    def test_clustering_pipeline_fits(self, synthetic_X_y):
        """Le pipeline de clustering doit s'ajuster sans erreur."""
        X, _ = synthetic_X_y
        pipe = build_clustering_pipeline(k=3)
        labels = pipe.fit_predict(X)
        assert len(labels) == len(X)

    def test_labels_are_integers(self, synthetic_X_y):
        """Les labels doivent être des entiers entre 0 et K-1."""
        X, _ = synthetic_X_y
        k = 4
        pipe = build_clustering_pipeline(k=k)
        labels = pipe.fit_predict(X)

        assert labels.dtype in [np.int32, np.int64]
        assert set(labels).issubset(set(range(k))), (
            f"Labels inattendus : {set(labels)}"
        )

    def test_k_clusters_produced(self, synthetic_X_y):
        """Le nombre de clusters produits doit correspondre à K."""
        X, _ = synthetic_X_y
        k = 5
        pipe = build_clustering_pipeline(k=k)
        labels = pipe.fit_predict(X)

        unique_labels = len(np.unique(labels))
        assert unique_labels == k, (
            f"Attendu {k} clusters, obtenu {unique_labels}."
        )
