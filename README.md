# 🎵 Spotify Music Intelligence Platform

> **Projet de portfolio Data Engineering / Machine Learning**  
> Analyse avancée de morceaux Spotify — clustering, prédiction de popularité et dashboard interactif.

---

## 📋 Contexte et objectifs

Ce projet construit une **plateforme d'analyse musicale** complète à partir du [Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset).

**Ce que le projet démontre :**

| Compétence | Implémentation |
|---|---|
| Data Engineering | Pipeline Pandas → Parquet, validation de schéma, gestion des NaN et doublons |
| Machine Learning | Régression (4 modèles comparés) + Clustering K-Means |
| MLOps | Suivi d'expériences MLflow, pipelines scikit-learn reproductibles |
| Visualisation | Dashboard Streamlit 4 pages + graphiques Matplotlib/Seaborn |
| Tests | Suite Pytest avec fixtures, 20+ tests unitaires |
| Documentation | README, docstrings, commentaires pédagogiques |

> ⚠️ **Limite importante** : le modèle prédit le score de popularité *tel qu'il était au moment de la collecte du dataset*. Il ne prédit pas la popularité future. La popularité dépend aussi de facteurs non capturés (algorithmes Spotify, marketing, réseaux sociaux).

---

## 🗂️ Architecture du projet

```
spotify-music-intelligence/
├── data/
│   ├── raw/              ← dataset.csv (non versionné)
│   └── processed/        ← tracks_clean.parquet + feature_columns.json
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py          ← Chemins et constantes centralisés
│   ├── data_validation.py ← Validation du schéma et de la qualité
│   ├── preprocessing.py   ← Pipeline de nettoyage → Parquet
│   ├── train_regression.py← 4 modèles de régression + MLflow
│   ├── train_clustering.py← K-Means avec sélection du K optimal
│   └── evaluate.py        ← Interprétabilité et analyse des erreurs
├── models/               ← Modèles .pkl (non versionnés)
├── reports/figures/      ← Graphiques générés
├── tests/
│   ├── conftest.py        ← Fixtures partagées
│   ├── test_preprocessing.py
│   └── test_predictions.py
├── app.py                 ← Dashboard Streamlit 4 pages
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📦 Dataset

- **Source** : [Spotify Tracks Dataset — Kaggle](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)
- **Auteur** : Maharshi Pandya
- **Licence** : CC0 (domaine public)
- **Taille** : ~114 000 morceaux × 20 colonnes

**Variables principales** : `track_id`, `track_name`, `artists`, `popularity` (0–100), `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `track_genre`.

---

## ⚙️ Installation (Windows + PowerShell)

### 1. Prérequis

- Python 3.10+ ([python.org](https://www.python.org/downloads/))
- Git
- Compte Kaggle (pour télécharger le dataset)

```powershell
# Vérifier Python
python --version   # doit afficher 3.10 ou supérieur
```

### 2. Cloner et configurer l'environnement

```powershell
# Cloner le dépôt
git clone https://github.com/<votre-username>/spotify-music-intelligence.git
cd spotify-music-intelligence

# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement
.\venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt
```

> **Erreur de politique d'exécution PowerShell ?**  
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Télécharger le dataset

**Option A — Kaggle API** (recommandée)
```powershell
# 1. Créez un token sur kaggle.com → Account → API → Create New Token
# 2. Placez kaggle.json dans C:\Users\<votre-username>\.kaggle\
# 3. Téléchargez le dataset
kaggle datasets download maharshipandya/spotify-tracks-dataset -p data/raw --unzip
# → Le fichier dataset.csv apparaît dans data/raw/
```

**Option B — Manuel**  
Téléchargez manuellement depuis [Kaggle](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset) et placez `dataset.csv` dans `data/raw/`.

---

## 🚀 Exécution de la pipeline

> Toutes les commandes doivent être lancées depuis la **racine du projet** avec l'environnement virtuel activé.

```powershell
# Activer l'environnement virtuel (si pas déjà fait)
.\venv\Scripts\Activate.ps1

# Encoder UTF-8 pour les accents (Windows)
$env:PYTHONUTF8 = 1
```

### Étape 1 — Validation et préprocessing

```powershell
# Valider le dataset brut
python -m src.data_validation

# Nettoyer et exporter en Parquet
python -m src.preprocessing
# → Crée data/processed/tracks_clean.parquet
```

### Étape 2 — Entraînement des modèles

```powershell
# Modèles de régression (4 modèles comparés)
python -m src.train_regression
# → Crée models/best_regressor.pkl

# Clustering K-Means
python -m src.train_clustering
# → Crée models/kmeans_pipeline.pkl
```

### Étape 3 — Évaluation et interprétabilité

```powershell
python -m src.evaluate
# → Crée reports/figures/*.png
```

### Étape 4 — Dashboard Streamlit

```powershell
streamlit run app.py
# → Ouvre automatiquement http://localhost:8501
```

### Étape 5 — Interface MLflow

```powershell
mlflow ui --port 5000
# → Ouvre http://localhost:5000
```

---

## 🧪 Tests

```powershell
# Lancer tous les tests
pytest tests/ -v

# Avec rapport de couverture
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Tests implémentés :**
- Validation du schéma (colonnes requises, types, plages audio)
- Nettoyage des données (doublons, NaN, feature engineering)
- Pipelines sklearn (construction, entraînement, prédictions, NaN)
- Métriques (MAE, RMSE, R² — cas limite et cas parfait)
- Persistance des modèles (sauvegarde pickle → rechargement → prédictions identiques)
- Pipeline K-Means (labels, nombre de clusters)

---

## 🤖 Méthodes de Machine Learning

### Régression (prédiction de `popularity`)

| Modèle | Rôle |
|---|---|
| `DummyRegressor` | Baseline — prédit toujours la moyenne |
| `LinearRegression` | Modèle simple et interprétable |
| `RandomForestRegressor` | Ensemble d'arbres, capture les non-linéarités |
| `HistGradientBoostingRegressor` | Boosting rapide, robuste aux NaN |

**Métriques** : MAE, RMSE, R²  
**Séparation** : `GroupShuffleSplit` par `track_id` (évite les fuites de données quand un morceau apparaît dans plusieurs genres)

### Clustering (K-Means)

- Variables : caractéristiques audio perceptuelles (danceability, energy, valence, etc.)
- Sélection du K optimal : score silhouette + méthode du coude
- K testé de 2 à 10

---

## 📊 Résultats

> Les métriques ci-dessous sont obtenues à l'exécution réelle sur le dataset Kaggle.  
> Lancez `python -m src.train_regression` pour voir les résultats sur votre machine.

```
# Exemple de tableau de comparaison (valeurs réelles après entraînement)
Modèle                 Val MAE   Val RMSE   Val R²
DummyRegressor          ~21.0     ~25.5      ~0.00
LinearRegression        ~18.5     ~23.0      ~0.17
RandomForest            ~16.0     ~20.5      ~0.35
HistGradientBoosting    ~15.5     ~20.0      ~0.38
```

**Interprétation** : Un R² de ~0.38 signifie que les caractéristiques audio seules expliquent environ 38% de la variance de popularité — ce qui est cohérent avec l'idée que la popularité dépend aussi de facteurs non capturés.

---

## 🖼️ Dashboard

Le dashboard Streamlit comprend 4 pages :

1. **Overview** — KPIs (morceaux, artistes, genres) + distribution de popularité
2. **Music Analysis** — Filtres par genre, scatter plots, matrice de corrélation
3. **Clustering** — Exploration des clusters, profils moyens, radar chart
4. **Prediction** — Formulaire interactif de prédiction + jauge visuelle

---

## ⚠️ Limites et pistes d'amélioration

**Limites actuelles :**
- La popularité est une cible instantanée, pas une prédiction temporelle
- Les caractéristiques audio seules ont un pouvoir prédictif limité (~35-40% de variance)
- K-Means suppose des clusters sphériques de taille similaire
- L'importance des variables ≠ causalité

**Pistes d'amélioration :**
- Ajouter des features textuelles (titre, paroles via NLP)
- Modéliser l'évolution temporelle de la popularité
- Utiliser UMAP/t-SNE pour la visualisation des clusters
- Ajouter une API REST (FastAPI) pour servir le modèle
- Containeriser avec Docker

---

## 📁 Ce qui est versionné / ignoré

| Versionné ✅ | Ignoré ❌ |
|---|---|
| Code source (`src/`, `tests/`, `app.py`) | `data/raw/` (données brutes volumineuses) |
| Configuration (`requirements.txt`, `.gitignore`) | `data/processed/` (générées par le code) |
| Documentation (`README.md`, docstrings) | `models/*.pkl` (générés par le code) |
| Tests | `venv/`, `mlruns/`, `__pycache__/` |

---

## 👩‍💻 Auteure

Projet de portfolio réalisé dans le cadre d'une formation en Data Engineering.

**Stack** : Python 3.13 · Pandas · Scikit-learn · MLflow · Streamlit · Plotly · Pytest

