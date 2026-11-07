"""
train_regression.py — Entraînement des modèles de régression
=============================================================
Rôle : Prédire le score de popularité (0-100) d'un morceau à partir de
       ses caractéristiques audio.

Modèles comparés :
  1. DummyRegressor (baseline — prédit toujours la moyenne)
  2. LinearRegression (modèle simple et interprétable)
  3. RandomForestRegressor (ensemble d'arbres de décision)
  4. HistGradientBoostingRegressor (boosting rapide, gère les NaN nativement)

Stratégie de séparation :
  → GroupShuffleSplit par track_id pour éviter qu'un même morceau apparaisse
    dans plusieurs ensembles (fuite de données).

Toutes les expériences sont loggées dans MLflow.

Usage :
  python -m src.train_regression
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    AUDIO_FEATURES,
    FEATURE_COLS_PATH,
    MLFLOW_EXPERIMENT_REGRESSION,
    MODELS_DIR,
    PROCESSED_PARQUET,
    RANDOM_STATE,
    REGRESSION_MODEL_PATH,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
    ID_COL,
)

# ── Configuration des logs ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_processed_data() -> pd.DataFrame:
    """Charge le dataset nettoyé depuis le fichier Parquet."""
    if not PROCESSED_PARQUET.exists():
        raise FileNotFoundError(
            f"❌ Données non trouvées : {PROCESSED_PARQUET}\n"
            "→ Lance d'abord : python -m src.preprocessing"
        )
    df = pd.read_parquet(PROCESSED_PARQUET)
    logger.info(f"Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


def get_feature_columns() -> list[str]:
    """
    Charge la liste des features depuis le JSON généré par preprocessing.py.
    Garantit que l'entraînement et le formulaire Streamlit utilisent
    exactement les mêmes colonnes.
    """
    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH) as f:
            cols = json.load(f)
        logger.info(f"Features chargées depuis {FEATURE_COLS_PATH} : {cols}")
        return cols

    # Fallback : utiliser AUDIO_FEATURES directement
    logger.warning("feature_columns.json introuvable — utilisation de AUDIO_FEATURES par défaut.")
    return AUDIO_FEATURES


def split_by_groups(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,  pd.Series,  pd.Series]:
    """
    Sépare le dataset en train / validation / test en respectant les groupes
    définis par track_id.

    Pourquoi GroupShuffleSplit ?
    → Un même morceau peut apparaître plusieurs fois dans le dataset (genres différents).
    → Si on faisait un split aléatoire, le même morceau se retrouverait à la fois
      dans train et test — le modèle aurait "vu" ces données, faussant les métriques.
    → GroupShuffleSplit garantit qu'aucun track_id n'est partagé entre les ensembles.

    Retourne : X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df[feature_cols]
    y = df[TARGET_COL]
    groups = df[ID_COL]  # identifiant de groupe

    # 1er split : on isole le test set (15% des groupes)
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))

    X_tv, y_tv   = X.iloc[train_val_idx], y.iloc[train_val_idx]
    groups_tv    = groups.iloc[train_val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # 2e split : on isole la validation (15% du reste)
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_size_adjusted, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(gss_val.split(X_tv, y_tv, groups=groups_tv))

    X_train, y_train = X_tv.iloc[train_idx], y_tv.iloc[train_idx]
    X_val,   y_val   = X_tv.iloc[val_idx],   y_tv.iloc[val_idx]

    logger.info(
        f"Séparation des données :\n"
        f"  Train : {len(X_train):,} lignes ({len(X_train)/len(df)*100:.1f}%)\n"
        f"  Val   : {len(X_val):,} lignes ({len(X_val)/len(df)*100:.1f}%)\n"
        f"  Test  : {len(X_test):,} lignes ({len(X_test)/len(df)*100:.1f}%)"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Calcule les métriques d'évaluation d'un modèle de régression.

    MAE  : erreur absolue moyenne (même unité que la cible — points de popularité)
    RMSE : racine de l'erreur quadratique moyenne (pénalise plus les grandes erreurs)
    R²   : proportion de variance expliquée (1.0 = parfait, 0.0 = pas mieux que la moyenne)
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def build_pipeline(model, use_scaler: bool = True) -> Pipeline:
    """
    Construit un pipeline Scikit-learn avec :
    1. SimpleImputer  : remplace les NaN résiduels par la médiane.
    2. StandardScaler : normalise les features (moyenne=0, std=1).
    3. Le modèle      : estimateur passé en paramètre.

    Note : HistGradientBoosting gère nativement les NaN, mais on garde
           l'imputer pour uniformiser les pipelines.
    Note : DummyRegressor et HistGradientBoosting n'ont pas besoin de scaler,
           mais ça ne nuit pas de l'inclure pour la cohérence.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))

    return Pipeline(steps)


def train_and_evaluate_all(
    X_train, X_val, X_test, y_train, y_val, y_test,
    feature_cols: list[str],
) -> tuple[dict, object]:
    """
    Entraîne et évalue tous les modèles.
    Logue chaque expérience dans MLflow.
    Retourne les résultats et le meilleur modèle (selon MAE sur validation).
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_REGRESSION)

    # Définition des modèles à comparer
    # On utilise des hyperparamètres raisonnables sans optimisation complexe
    # pour cette phase — l'optimisation vient après la baseline.
    model_configs = {
        "DummyRegressor": {
            "model": DummyRegressor(strategy="mean"),
            "use_scaler": False,
            "params": {"strategy": "mean"},
        },
        "LinearRegression": {
            "model": LinearRegression(),
            "use_scaler": True,
            "params": {},
        },
        "RandomForest": {
            "model": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "use_scaler": False,  # les arbres ne nécessitent pas de scaling
            "params": {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 5},
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            ),
            "use_scaler": False,
            "params": {"max_iter": 200, "max_depth": 6, "learning_rate": 0.05},
        },
    }

    results = {}
    best_model_name  = None
    best_mae         = float("inf")
    best_pipeline    = None

    for name, config in model_configs.items():
        logger.info(f"\n── Entraînement : {name} ──")

        with mlflow.start_run(run_name=name):
            # Log des paramètres
            mlflow.log_params(config["params"])
            mlflow.log_param("features", feature_cols)
            mlflow.log_param("train_size", len(X_train))

            # Construction et entraînement du pipeline
            pipe = build_pipeline(config["model"], config["use_scaler"])
            pipe.fit(X_train, y_train)

            # Évaluation sur validation
            y_val_pred = pipe.predict(X_val)
            val_metrics = compute_metrics(y_val, y_val_pred)

            # Évaluation sur test (réservé pour la comparaison finale)
            y_test_pred = pipe.predict(X_test)
            test_metrics = compute_metrics(y_test, y_test_pred)

            # Log des métriques dans MLflow
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            # Sauvegarde du modèle dans MLflow
            # On force la sérialisation pickle (au lieu de skops) pour éviter
            # l'erreur "UntrustedTypesFoundException: numpy.dtype" qui apparaît
            # avec les versions récentes de mlflow + skops sur Windows.
            mlflow.sklearn.log_model(
                pipe,
                artifact_path=f"model_{name}",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )

            results[name] = {
                "validation": val_metrics,
                "test":       test_metrics,
            }

            logger.info(f"  Validation → MAE: {val_metrics['MAE']:.3f} | "
                        f"RMSE: {val_metrics['RMSE']:.3f} | R²: {val_metrics['R2']:.3f}")
            logger.info(f"  Test       → MAE: {test_metrics['MAE']:.3f} | "
                        f"RMSE: {test_metrics['RMSE']:.3f} | R²: {test_metrics['R2']:.3f}")

            # On sélectionne le meilleur modèle selon la MAE sur validation
            if val_metrics["MAE"] < best_mae:
                best_mae        = val_metrics["MAE"]
                best_model_name = name
                best_pipeline   = pipe

    logger.info(f"\n🏆 Meilleur modèle (MAE validation) : {best_model_name} (MAE={best_mae:.3f})")
    return results, best_pipeline, best_model_name


def save_best_model(pipeline, model_name: str) -> None:
    """Sauvegarde le meilleur modèle en fichier pickle pour le dashboard Streamlit."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGRESSION_MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"✅ Modèle '{model_name}' sauvegardé : {REGRESSION_MODEL_PATH}")


def print_comparison_table(results: dict) -> None:
    """Affiche un tableau comparatif des modèles."""
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Modèle":        model_name,
            "Val MAE":       metrics["validation"]["MAE"],
            "Val RMSE":      metrics["validation"]["RMSE"],
            "Val R²":        metrics["validation"]["R2"],
            "Test MAE":      metrics["test"]["MAE"],
            "Test RMSE":     metrics["test"]["RMSE"],
            "Test R²":       metrics["test"]["R2"],
        })

    df_results = pd.DataFrame(rows).sort_values("Val MAE")
    print("\n" + "=" * 70)
    print("COMPARAISON DES MODÈLES")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70)
    print("\nNote : les métriques 'Test' sont utilisées uniquement pour la")
    print("comparaison finale. Le modèle a été sélectionné sur 'Val MAE'.")


if __name__ == "__main__":
    # ── 1. Chargement ─────────────────────────────────────────────────────────
    df = load_processed_data()
    feature_cols = get_feature_columns()

    # ── 2. Vérification que les features existent dans le DataFrame ───────────
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Features manquantes dans le dataset : {missing_features}")

    # ── 3. Séparation des données ─────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_by_groups(df, feature_cols)

    # ── 4. Entraînement et évaluation ─────────────────────────────────────────
    results, best_pipeline, best_name = train_and_evaluate_all(
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    )

    # ── 5. Sauvegarde du meilleur modèle ──────────────────────────────────────
    save_best_model(best_pipeline, best_name)

    # ── 6. Tableau comparatif ─────────────────────────────────────────────────
    print_comparison_table(results)

    print(f"\n✅ Entraînement terminé. MLflow UI : mlflow ui --port 5000")
    print(f"   Meilleur modèle sauvegardé : {REGRESSION_MODEL_PATH}")
