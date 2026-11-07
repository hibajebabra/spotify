"""
app.py — Dashboard Streamlit : Spotify Music Intelligence Platform
==================================================================
Interface multi-pages avec 4 sections :
  Page 1 — Overview         : KPIs et distribution de popularité
  Page 2 — Music Analysis   : Exploration et corrélations
  Page 3 — Clustering       : Visualisation des clusters
  Page 4 — Prediction       : Formulaire de prédiction de popularité

Usage :
  streamlit run app.py

Note : Le dataset doit être prétraité (python -m src.preprocessing)
       et les modèles entraînés avant de lancer le dashboard.
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Music Intelligence",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Chemins (reproduits depuis config pour éviter l'import lourd au démarrage) ─
ROOT_DIR            = Path(__file__).parent
PROCESSED_PARQUET   = ROOT_DIR / "data" / "processed" / "tracks_clean.parquet"
REGRESSION_MODEL    = ROOT_DIR / "models" / "best_regressor.pkl"
CLUSTERING_MODEL    = ROOT_DIR / "models" / "kmeans_pipeline.pkl"
FEATURE_COLS_PATH   = ROOT_DIR / "data" / "processed" / "feature_columns.json"
FIGURES_DIR         = ROOT_DIR / "reports" / "figures"

# ── Thème couleurs ────────────────────────────────────────────────────────────
SPOTIFY_GREEN  = "#1DB954"
DARK_BG        = "#121212"
CARD_BG        = "#1e1e2e"
ACCENT         = "#7c3aed"
TEXT_PRIMARY   = "#ffffff"
TEXT_SECONDARY = "#a0a0b0"

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fond principal */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid #1DB954;
    }

    /* Titre principal */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1DB954, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }

    /* Cartes KPI */
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #1DB954;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.15);
        transition: transform 0.2s;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1DB954;
        margin: 0.3rem 0;
    }

    .kpi-label {
        font-size: 0.85rem;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Sections */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1DB954;
        border-left: 4px solid #1DB954;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Badges de genre */
    .genre-badge {
        display: inline-block;
        background: rgba(29, 185, 84, 0.15);
        border: 1px solid #1DB954;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        color: #1DB954;
        margin: 0.2rem;
    }

    /* Bouton de prédiction */
    .stButton > button {
        background: linear-gradient(135deg, #1DB954, #17a84a);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.4);
        transition: all 0.3s;
    }

    /* Résultat de prédiction */
    .prediction-box {
        background: linear-gradient(135deg, #1e3a2e, #2a4a3e);
        border: 2px solid #1DB954;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .prediction-score {
        font-size: 4rem;
        font-weight: 900;
        color: #1DB954;
        text-shadow: 0 0 20px rgba(29, 185, 84, 0.5);
    }

    /* Onglets */
    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }

    /* Limite avertissement */
    .warning-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #fca5a5;
        font-size: 0.85rem;
    }

    /* Info box */
    .info-box {
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid #7c3aed;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #c4b5fd;
        font-size: 0.85rem;
    }

    /* Masquer le footer Streamlit */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires (cache pour la performance)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Charge le dataset prétraité depuis Parquet (mis en cache par Streamlit)."""
    if not PROCESSED_PARQUET.exists():
        return None
    return pd.read_parquet(PROCESSED_PARQUET)


@st.cache_resource
def load_regression_model():
    """Charge le modèle de régression (mis en cache, chargé une seule fois)."""
    if not REGRESSION_MODEL.exists():
        return None
    with open(REGRESSION_MODEL, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_clustering_model():
    """Charge le pipeline K-Means."""
    if not CLUSTERING_MODEL.exists():
        return None
    with open(CLUSTERING_MODEL, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_feature_columns() -> list[str]:
    """Charge la liste des colonnes du modèle."""
    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH) as f:
            return json.load(f)
    return [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness",
        "valence", "tempo", "duration_ms",
    ]


def make_plotly_layout(title: str = "", height: int = 400) -> dict:
    """Retourne un dictionnaire de layout Plotly avec le thème sombre."""
    return dict(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,30,46,0.8)",
        font=dict(color="#ffffff", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#333355", zerolinecolor="#333355"),
        yaxis=dict(gridcolor="#333355", zerolinecolor="#333355"),
        colorway=[SPOTIFY_GREEN, "#7c3aed", "#f59e0b", "#ef4444", "#06b6d4"],
        margin=dict(t=50, b=40, l=40, r=20),
    )


def popularity_to_label(score: float) -> tuple[str, str]:
    """Convertit un score de popularité en label et couleur."""
    if score >= 75:
        return "🔥 Très populaire", "#1DB954"
    elif score >= 50:
        return "📈 Populaire", "#f59e0b"
    elif score >= 25:
        return "📊 Modéré", "#7c3aed"
    else:
        return "📉 Peu populaire", "#ef4444"


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Navigation
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 1rem 0;'>
            <div style='font-size:2.5rem;'>🎵</div>
            <div style='font-size:1.1rem; font-weight:700; color:#1DB954;'>
                Music Intelligence
            </div>
            <div style='font-size:0.75rem; color:#a0a0b0; margin-top:0.2rem;'>
                Spotify Analytics Platform
            </div>
        </div>
        <hr style='border-color:#1DB954; margin: 0.5rem 0 1rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=["🏠 Overview", "🎸 Music Analysis", "🔮 Clustering", "🎯 Prediction"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#333355; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
        <div style='font-size:0.75rem; color:#a0a0b0; text-align:center;'>
            <div>Dataset : Kaggle Spotify Tracks</div>
            <div style='margin-top:0.3rem;'>Modèles : scikit-learn + MLflow</div>
            <div style='margin-top:0.3rem; color:#1DB954;'>🎓 Data Engineering Portfolio</div>
        </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Chargement des données
# ══════════════════════════════════════════════════════════════════════════════

df = load_data()

if df is None:
    st.markdown("""
        <div class='main-title'>🎵 Spotify Music Intelligence</div>
    """, unsafe_allow_html=True)
    st.error("⚠️ Données non disponibles.")
    st.info("""
    **Pour démarrer le dashboard :**
    ```powershell
    # 1. Placer le dataset dans data/raw/dataset.csv
    # 2. Lancer le preprocessing
    $env:PYTHONUTF8=1; .\\venv\\Scripts\\python.exe -m src.preprocessing

    # 3. (Optionnel) Entraîner les modèles
    $env:PYTHONUTF8=1; .\\venv\\Scripts\\python.exe -m src.train_regression
    $env:PYTHONUTF8=1; .\\venv\\Scripts\\python.exe -m src.train_clustering
    ```
    """)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown('<div class="main-title">🎵 Spotify Music Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#a0a0b0; font-size:1rem; margin-bottom:2rem;">'
        'Analyse approfondie de la popularité et des caractéristiques musicales'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── KPIs ──
    st.markdown('<div class="section-header">📊 Indicateurs clés</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Morceaux</div>
                <div class="kpi-value">{df['track_id'].nunique():,}</div>
                <div style="color:#a0a0b0; font-size:0.8rem;">{len(df):,} entrées totales</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        n_artists = df["artists"].nunique() if "artists" in df.columns else 0
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Artistes</div>
                <div class="kpi-value">{n_artists:,}</div>
                <div style="color:#a0a0b0; font-size:0.8rem;">uniques</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_pop = df["popularity"].mean()
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Popularité moyenne</div>
                <div class="kpi-value">{avg_pop:.1f}</div>
                <div style="color:#a0a0b0; font-size:0.8rem;">sur 100</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        n_genres = df["track_genre"].nunique() if "track_genre" in df.columns else 0
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Genres</div>
                <div class="kpi-value">{n_genres}</div>
                <div style="color:#a0a0b0; font-size:0.8rem;">genres musicaux</div>
            </div>
        """, unsafe_allow_html=True)

    # ── Distribution de popularité ──
    st.markdown('<div class="section-header">📈 Distribution de la popularité</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        fig_dist = px.histogram(
            df, x="popularity", nbins=50,
            color_discrete_sequence=[SPOTIFY_GREEN],
        )
        fig_dist.update_traces(opacity=0.85)
        fig_dist.update_layout(
            **make_plotly_layout("Distribution du score de popularité", height=380),
            xaxis_title="Score de popularité (0–100)",
            yaxis_title="Nombre de morceaux",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        st.markdown('<div style="padding-top:1rem;">', unsafe_allow_html=True)
        desc = df["popularity"].describe()
        stats_data = {
            "Minimum":  f"{desc['min']:.0f}",
            "1er quartile": f"{desc['25%']:.1f}",
            "Médiane":  f"{desc['50%']:.1f}",
            "Moyenne":  f"{desc['mean']:.1f}",
            "3e quartile": f"{desc['75%']:.1f}",
            "Maximum":  f"{desc['max']:.0f}",
            "Écart-type": f"{desc['std']:.1f}",
        }
        for label, value in stats_data.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:0.4rem 0;border-bottom:1px solid #333355;">'
                f'<span style="color:#a0a0b0;">{label}</span>'
                f'<span style="color:#1DB954;font-weight:600;">{value}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("""
            <div class="info-box" style="margin-top:1rem;">
                ℹ️ La popularité est fortement asymétrique.
                Beaucoup de morceaux ont une popularité faible à nulle.
            </div>
        """, unsafe_allow_html=True)

    # ── Top genres ──
    if "track_genre" in df.columns:
        st.markdown('<div class="section-header">🎸 Top 15 genres par nombre de morceaux</div>', unsafe_allow_html=True)

        genre_counts = df.groupby("track_genre")["track_id"].nunique().sort_values(ascending=False).head(15)
        fig_genres = px.bar(
            x=genre_counts.values,
            y=genre_counts.index,
            orientation="h",
            color=genre_counts.values,
            color_continuous_scale=[[0, "#333355"], [1, SPOTIFY_GREEN]],
        )
        fig_genres.update_layout(
            **make_plotly_layout("", height=450),
            xaxis_title="Nombre de morceaux uniques",
            yaxis_title="Genre",
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_genres, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MUSIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎸 Music Analysis":
    st.markdown('<div class="main-title">🎸 Music Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#a0a0b0; margin-bottom:2rem;">Exploration des caractéristiques audio et relations entre variables</div>',
        unsafe_allow_html=True,
    )

    # ── Filtres ──
    st.markdown('<div class="section-header">🔍 Filtres</div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        if "track_genre" in df.columns:
            genres = sorted(df["track_genre"].unique())
            selected_genres = st.multiselect(
                "Genres musicaux",
                options=genres,
                default=genres[:5] if len(genres) >= 5 else genres,
            )
        else:
            selected_genres = []

    with col_f2:
        pop_range = st.slider(
            "Plage de popularité",
            min_value=0, max_value=100,
            value=(0, 100),
        )

    # Filtrage
    df_filtered = df.copy()
    if selected_genres and "track_genre" in df.columns:
        df_filtered = df_filtered[df_filtered["track_genre"].isin(selected_genres)]
    df_filtered = df_filtered[
        df_filtered["popularity"].between(pop_range[0], pop_range[1])
    ]

    st.info(f"**{len(df_filtered):,}** morceaux correspondent aux filtres.")

    if len(df_filtered) == 0:
        st.warning("Aucun morceau ne correspond aux filtres sélectionnés.")
        st.stop()

    # ── Popularité vs. Caractéristiques ──
    st.markdown('<div class="section-header">📊 Popularité vs. Caractéristiques audio</div>', unsafe_allow_html=True)

    audio_cols = [c for c in ["danceability", "energy", "valence", "acousticness",
                               "tempo", "loudness", "speechiness", "instrumentalness"] if c in df.columns]

    col_x, col_c = st.columns([3, 1])
    with col_x:
        x_feature = st.selectbox("Axe X — Caractéristique audio", options=audio_cols, index=0)
    with col_c:
        color_by = st.selectbox("Couleur par", options=["track_genre", "popularity"], index=0)

    # Échantillon pour la performance (max 5000 points)
    df_sample = df_filtered.sample(min(5000, len(df_filtered)), random_state=42)

    fig_scatter = px.scatter(
        df_sample, x=x_feature, y="popularity",
        color=color_by if color_by in df_sample.columns else None,
        opacity=0.4, size_max=8,
        color_continuous_scale="Viridis" if color_by == "popularity" else None,
    )
    fig_scatter.update_layout(
        **make_plotly_layout(f"Popularité vs. {x_feature}", height=450),
        xaxis_title=x_feature,
        yaxis_title="Popularité",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("""
        <div class="warning-box">
            ⚠️ Une corrélation visible sur ce graphique ne prouve pas un lien de causalité.
            Le score de popularité dépend aussi de facteurs non capturés ici
            (algorithmes de recommandation, marketing, etc.).
        </div>
    """, unsafe_allow_html=True)

    # ── Matrice de corrélation ──
    st.markdown('<div class="section-header">🔗 Matrice de corrélation</div>', unsafe_allow_html=True)

    num_cols = [c for c in ["popularity", "danceability", "energy", "loudness",
                             "speechiness", "acousticness", "instrumentalness",
                             "liveness", "valence", "tempo"] if c in df_filtered.columns]

    corr_matrix = df_filtered[num_cols].corr().round(3)

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=[
            [0.0, "#ef4444"], [0.5, "#1e1e2e"], [1.0, "#1DB954"]
        ],
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        hovertemplate="%{x} vs %{y}<br>Corrélation : %{z:.3f}<extra></extra>",
    ))
    fig_corr.update_layout(
        **make_plotly_layout("Matrice de corrélation de Pearson", height=500),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Popularité par genre ──
    if "track_genre" in df_filtered.columns and len(selected_genres) > 1:
        st.markdown('<div class="section-header">🎵 Popularité moyenne par genre</div>', unsafe_allow_html=True)

        genre_pop = (
            df_filtered.groupby("track_genre")["popularity"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        genre_pop.columns = ["genre", "mean_pop", "std_pop", "count"]
        genre_pop = genre_pop[genre_pop["count"] >= 10]

        fig_genre_pop = go.Figure()
        fig_genre_pop.add_trace(go.Bar(
            x=genre_pop["genre"],
            y=genre_pop["mean_pop"],
            error_y=dict(type="data", array=genre_pop["std_pop"], color="rgba(255,255,255,0.4)"),
            marker_color=SPOTIFY_GREEN,
            opacity=0.85,
            hovertemplate="Genre: %{x}<br>Popularité moy: %{y:.1f}<extra></extra>",
        ))
        fig_genre_pop.update_layout(
            **make_plotly_layout("", height=400),
            xaxis_title="Genre",
            yaxis_title="Popularité moyenne",
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig_genre_pop, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Clustering":
    st.markdown('<div class="main-title">🔮 Clustering Musical</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#a0a0b0; margin-bottom:2rem;">'
        'Groupes de morceaux similaires identifiés par K-Means'
        '</div>',
        unsafe_allow_html=True,
    )

    clustering_model = load_clustering_model()

    if clustering_model is None:
        st.warning("Le modèle de clustering n'a pas encore été entraîné.")
        st.info("""
        **Pour entraîner le clustering :**
        ```powershell
        $env:PYTHONUTF8=1; .\\venv\\Scripts\\python.exe -m src.train_clustering
        ```
        """)

        # Afficher quand même les graphiques de sélection si disponibles
        elbow_path = FIGURES_DIR / "kmeans_selection.png"
        if elbow_path.exists():
            st.markdown('<div class="section-header">📊 Sélection du K optimal</div>', unsafe_allow_html=True)
            st.image(str(elbow_path), use_column_width=True)
    else:
        # Appliquer le clustering
        clustering_features = [
            c for c in ["danceability", "energy", "speechiness", "acousticness",
                         "instrumentalness", "liveness", "valence", "tempo", "loudness"]
            if c in df.columns
        ]

        df_clust = df[clustering_features].copy()
        labels = clustering_model.predict(df_clust)
        df_vis = df.copy()
        df_vis["cluster"] = labels.astype(str)

        k = len(np.unique(labels))

        st.markdown(f"""
            <div class="kpi-card" style="max-width:300px; margin:0 auto 2rem auto;">
                <div class="kpi-label">Nombre de clusters</div>
                <div class="kpi-value">K = {k}</div>
                <div style="color:#a0a0b0; font-size:0.8rem;">sélectionné par score silhouette</div>
            </div>
        """, unsafe_allow_html=True)

        # ── Scatter 2D ──
        st.markdown('<div class="section-header">🗺️ Visualisation 2D des clusters</div>', unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            cx = st.selectbox("Axe X", options=clustering_features, index=0)
        with col_c2:
            cy = st.selectbox("Axe Y", options=clustering_features, index=1)

        df_sample = df_vis.sample(min(5000, len(df_vis)), random_state=42)
        color_seq = [SPOTIFY_GREEN, "#7c3aed", "#f59e0b", "#ef4444", "#06b6d4",
                     "#f97316", "#ec4899", "#14b8a6", "#6366f1", "#84cc16"]

        fig_2d = px.scatter(
            df_sample, x=cx, y=cy,
            color="cluster",
            opacity=0.6,
            color_discrete_sequence=color_seq[:k],
            hover_data=["track_name", "artists", "popularity"] if "track_name" in df_sample.columns else None,
        )
        fig_2d.update_layout(**make_plotly_layout(f"{cx} vs {cy} — Clusters", height=500))
        st.plotly_chart(fig_2d, use_container_width=True)

        # ── Profil des clusters ──
        st.markdown('<div class="section-header">📋 Profil moyen des clusters</div>', unsafe_allow_html=True)

        profile = (
            df_vis.groupby("cluster")[clustering_features + ["popularity"]]
            .mean()
            .round(3)
            .reset_index()
        )
        profile["n_tracks"] = df_vis["cluster"].value_counts().sort_index().values

        st.dataframe(
            profile.style
                .background_gradient(subset=clustering_features, cmap="RdYlGn")
                .format({c: "{:.3f}" for c in clustering_features + ["popularity"]})
                .set_properties(**{"background-color": "#1e1e2e", "color": "white"}),
            use_container_width=True,
        )

        # ── Radar chart par cluster ──
        st.markdown('<div class="section-header">🕸️ Radar des caractéristiques par cluster</div>', unsafe_allow_html=True)

        radar_features = [c for c in ["danceability", "energy", "valence",
                                       "acousticness", "liveness", "speechiness"] if c in df.columns]

        fig_radar = go.Figure()
        for i, row in profile.iterrows():
            cluster_id = row["cluster"]
            values = [row[f] for f in radar_features]
            values_closed = values + [values[0]]
            features_closed = radar_features + [radar_features[0]]

            fig_radar.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=features_closed,
                fill="toself",
                name=f"Cluster {cluster_id}",
                opacity=0.7,
                line_color=color_seq[int(cluster_id) % len(color_seq)],
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], color="white"),
                angularaxis=dict(color="white"),
                bgcolor="rgba(30,30,46,0.8)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=500,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("""
            <div class="warning-box">
                ⚠️ Les clusters ne correspondent pas nécessairement aux genres musicaux officiels.
                K-Means regroupe les morceaux selon leurs similarités dans l'espace des
                caractéristiques audio — des morceaux de genres différents peuvent partager
                des propriétés acoustiques similaires.
            </div>
        """, unsafe_allow_html=True)

        st.image(str(FIGURES_DIR / "kmeans_selection.png"), use_column_width=True) if (FIGURES_DIR / "kmeans_selection.png").exists() else None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Prediction":
    st.markdown('<div class="main-title">🎯 Prédiction de popularité</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#a0a0b0; margin-bottom:1rem;">'
        'Estimez le score de popularité d\'un morceau à partir de ses caractéristiques audio'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
        <div class="warning-box" style="margin-bottom:1.5rem;">
            ⚠️ <strong>Limite importante</strong> : Ce modèle prédit le score de popularité
            tel qu'il était au moment de la collecte du dataset.
            Il ne prédit pas la popularité future et ne prend pas en compte
            les facteurs extra-musicaux (algorithmes Spotify, marketing, réseaux sociaux).
        </div>
    """, unsafe_allow_html=True)

    regression_model = load_regression_model()
    feature_cols     = load_feature_columns()

    if regression_model is None:
        st.error("Le modèle de régression n'a pas encore été entraîné.")
        st.info("""
        **Pour entraîner le modèle :**
        ```powershell
        $env:PYTHONUTF8=1; .\\venv\\Scripts\\python.exe -m src.train_regression
        ```
        """)
    else:
        # ── Formulaire ──
        st.markdown('<div class="section-header">🎛️ Paramètres du morceau</div>', unsafe_allow_html=True)

        # Limites et valeurs par défaut issues du dataset réel
        feature_config = {
            "danceability":     {"label": "Danceability",     "min": 0.0, "max": 1.0, "default": 0.6,    "step": 0.01, "help": "Aptitude au dansage (0=très peu, 1=très dansant)"},
            "energy":           {"label": "Energy",           "min": 0.0, "max": 1.0, "default": 0.7,    "step": 0.01, "help": "Intensité et activité perçue (0=calme, 1=très énergique)"},
            "loudness":         {"label": "Loudness (dB)",    "min": -60.0,"max": 5.0,"default": -8.0,   "step": 0.1,  "help": "Volume moyen en décibels (généralement entre -60 et 0)"},
            "speechiness":      {"label": "Speechiness",      "min": 0.0, "max": 1.0, "default": 0.05,   "step": 0.01, "help": "Proportion de paroles (0=instrumental, >0.66=parlé)"},
            "acousticness":     {"label": "Acousticness",     "min": 0.0, "max": 1.0, "default": 0.2,    "step": 0.01, "help": "Probabilité que le morceau soit acoustique"},
            "instrumentalness": {"label": "Instrumentalness", "min": 0.0, "max": 1.0, "default": 0.01,   "step": 0.01, "help": "Probabilité qu'il n'y ait pas de voix"},
            "liveness":         {"label": "Liveness",         "min": 0.0, "max": 1.0, "default": 0.12,   "step": 0.01, "help": "Probabilité que le morceau soit enregistré en live"},
            "valence":          {"label": "Valence",          "min": 0.0, "max": 1.0, "default": 0.5,    "step": 0.01, "help": "Positivité musicale (0=triste/sombre, 1=joyeux/euphorique)"},
            "tempo":            {"label": "Tempo (BPM)",      "min": 40.0,"max": 250.0,"default": 120.0, "step": 1.0,  "help": "Vitesse en battements par minute"},
            "duration_ms":      {"label": "Durée (ms)",       "min": 30000,"max":900000,"default":210000, "step": 1000, "help": "Durée en millisecondes (210000 ≈ 3min30)"},
        }

        input_values = {}
        cols = st.columns(2)

        for i, col in enumerate(feature_cols):
            if col not in feature_config:
                continue
            cfg = feature_config[col]
            with cols[i % 2]:
                value = st.slider(
                    label=f"{cfg['label']} ({col})",
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(cfg["default"]),
                    step=float(cfg["step"]),
                    help=cfg["help"],
                    key=f"slider_{col}",
                )
                input_values[col] = value

        # Durée en format lisible
        if "duration_ms" in input_values:
            dur_min = int(input_values["duration_ms"] // 60000)
            dur_sec = int((input_values["duration_ms"] % 60000) / 1000)
            st.caption(f"⏱️ Durée : {dur_min}:{dur_sec:02d}")

        # ── Prédiction ──
        st.markdown('<div class="section-header">🚀 Résultat de la prédiction</div>', unsafe_allow_html=True)

        if st.button("🎵 Prédire le score de popularité", key="predict_btn"):
            # Construction du DataFrame d'entrée avec exactement les mêmes colonnes
            # et le même ordre que le pipeline d'entraînement
            X_input = pd.DataFrame([{col: input_values.get(col, 0.0) for col in feature_cols}])

            try:
                score = float(regression_model.predict(X_input)[0])
                score_clipped = max(0.0, min(100.0, score))
                label, color = popularity_to_label(score_clipped)

                st.markdown(f"""
                    <div class="prediction-box">
                        <div style="color:#a0a0b0; font-size:0.9rem; margin-bottom:0.5rem;">
                            Score de popularité estimé
                        </div>
                        <div class="prediction-score" style="color:{color};">
                            {score_clipped:.1f}
                        </div>
                        <div style="font-size:1.5rem; margin-top:0.5rem;">{label}</div>
                        <div style="color:#a0a0b0; font-size:0.8rem; margin-top:1rem;">
                            sur une échelle de 0 (inconnu) à 100 (très populaire)
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Jauge visuelle
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score_clipped,
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "white"},
                        "bar": {"color": color},
                        "bgcolor": "#1e1e2e",
                        "bordercolor": "#333355",
                        "steps": [
                            {"range": [0, 25],  "color": "rgba(239,68,68,0.2)"},
                            {"range": [25, 50], "color": "rgba(124,58,237,0.2)"},
                            {"range": [50, 75], "color": "rgba(245,158,11,0.2)"},
                            {"range": [75, 100],"color": "rgba(29,185,84,0.2)"},
                        ],
                    },
                    number={"suffix": "/100", "font": {"color": color, "size": 36}},
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    height=300,
                    margin=dict(t=20, b=20, l=40, r=40),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Comparaison avec la distribution du dataset
                pop_pct = (df["popularity"] < score_clipped).mean() * 100
                st.info(
                    f"📊 Ce score est supérieur à **{pop_pct:.0f}%** des morceaux du dataset."
                )

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

        # ── Informations sur le modèle ──
        with st.expander("ℹ️ Informations sur le modèle et ses limites"):
            st.markdown("""
            **Modèle sélectionné** : le meilleur selon la MAE sur le set de validation.

            **Variables utilisées** : caractéristiques audio extraites par l'API Spotify.

            **Ce que le modèle NE capture pas** :
            - L'identité de l'artiste et sa notoriété existante
            - Les algorithmes de recommandation Spotify
            - La date de sortie et le contexte marketing
            - Les tendances culturelles et virales (TikTok, etc.)
            - Les paroles et leur contenu

            **Interprétation** : Le score prédit est une estimation statistique basée
            sur des patterns dans des données historiques. Il ne constitue pas une
            garantie de succès commercial.

            **Pour voir les métriques exactes** : Consultez les graphiques dans
            `reports/figures/` ou lancez l'interface MLflow :
            ```
            mlflow ui --port 5000
            ```
            """)
