# ============================================================
# run_pipeline.ps1 — Script PowerShell d'automatisation
# ============================================================
# Lance toutes les étapes de la pipeline dans l'ordre.
# Usage :
#   .\run_pipeline.ps1
#   .\run_pipeline.ps1 -SkipPreprocessing   (si Parquet déjà généré)
#
# Prérequis :
#   - venv activé OU Python dans le PATH
#   - dataset.csv dans data/raw/
# ============================================================

param(
    [switch]$SkipPreprocessing = $false,
    [switch]$SkipTraining      = $false,
    [switch]$SkipTests         = $false
)

# Encodage UTF-8 pour Windows
$env:PYTHONUTF8 = "1"

# Couleurs pour l'affichage
function Write-Step { param($msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok   { param($msg) Write-Host "  ✅ $msg" -ForegroundColor Green }
function Write-Fail { param($msg) Write-Host "  ❌ $msg" -ForegroundColor Red; exit 1 }

Write-Host "============================================" -ForegroundColor Magenta
Write-Host "  🎵 Spotify Music Intelligence Pipeline    " -ForegroundColor Magenta
Write-Host "============================================" -ForegroundColor Magenta

# --- Détection du Python ---
Write-Step "Vérification de l'environnement Python"

$PYTHON = $null
if (Test-Path ".\venv\Scripts\python.exe") {
    $PYTHON = ".\venv\Scripts\python.exe"
    Write-Ok "venv détecté : $PYTHON"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PYTHON = "python"
    Write-Ok "Python système détecté"
} else {
    Write-Fail "Python introuvable. Installez Python 3.10+ ou créez le venv."
}

# --- Vérification du dataset ---
Write-Step "Vérification du dataset"
if (-not (Test-Path "data\raw\dataset.csv")) {
    Write-Host "  ⚠️  data\raw\dataset.csv introuvable." -ForegroundColor Yellow
    Write-Host "  → Téléchargez le dataset depuis :" -ForegroundColor Yellow
    Write-Host "    https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset" -ForegroundColor Yellow
    Write-Host "  → Ou utilisez : kaggle datasets download maharshipandya/spotify-tracks-dataset -p data/raw --unzip" -ForegroundColor Yellow
    exit 1
} else {
    Write-Ok "dataset.csv trouvé"
}

# --- Tests (optionnel) ---
if (-not $SkipTests) {
    Write-Step "Lancement des tests Pytest"
    & $PYTHON -m pytest tests/ -v --tb=short
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Des tests ont échoué. Corrigez les erreurs avant de continuer."
    }
    Write-Ok "Tous les tests passent"
}

# --- Preprocessing ---
if (-not $SkipPreprocessing) {
    Write-Step "Étape 1 : Validation et préprocessing"
    & $PYTHON -m src.preprocessing
    if ($LASTEXITCODE -ne 0) { Write-Fail "Preprocessing échoué" }
    Write-Ok "Données prétraitées → data/processed/tracks_clean.parquet"
} else {
    Write-Host "  ⏭️  Preprocessing ignoré (--SkipPreprocessing)" -ForegroundColor Yellow
}

# --- Entraînement ---
if (-not $SkipTraining) {
    Write-Step "Étape 2 : Entraînement de la régression"
    & $PYTHON -m src.train_regression
    if ($LASTEXITCODE -ne 0) { Write-Fail "Entraînement régression échoué" }
    Write-Ok "Meilleur modèle → models/best_regressor.pkl"

    Write-Step "Étape 3 : Clustering K-Means"
    & $PYTHON -m src.train_clustering
    if ($LASTEXITCODE -ne 0) { Write-Fail "Clustering échoué" }
    Write-Ok "Pipeline K-Means → models/kmeans_pipeline.pkl"

    Write-Step "Étape 4 : Évaluation et interprétabilité"
    & $PYTHON -m src.evaluate
    if ($LASTEXITCODE -ne 0) { Write-Fail "Évaluation échouée" }
    Write-Ok "Figures → reports/figures/"
} else {
    Write-Host "  ⏭️  Entraînement ignoré (--SkipTraining)" -ForegroundColor Yellow
}

# --- Résumé ---
Write-Host "`n============================================" -ForegroundColor Magenta
Write-Host "  ✅ Pipeline terminée avec succès !         " -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "Lancer le dashboard :"
Write-Host "  streamlit run app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "Lancer MLflow UI :"
Write-Host "  mlflow ui --port 5000" -ForegroundColor Cyan

