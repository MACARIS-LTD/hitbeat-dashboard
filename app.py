import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# ── Configuración de página ───────────────────────────────────
st.set_page_config(
    page_title="HitBeat — Predictor de Alcance Musical",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos personalizados ────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1DB954, #F5C842, #A855F7, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #334155;
    }
    .nivel-alto  { color: #ef4444; font-weight: 700; }
    .nivel-medio { color: #f59e0b; font-weight: 700; }
    .nivel-bajo  { color: #22c55e; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e293b;
        border-radius: 8px;
        padding: 8px 20px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Carga de datos y modelo ───────────────────────────────────
@st.cache_data
def cargar_datos():
    ruta = os.path.join(os.path.dirname(__file__), "data", "hitbeat_dashboard_data.csv")
    df = pd.read_csv(ruta)
    return df

@st.cache_resource
def cargar_modelo():
    ruta = os.path.join(os.path.dirname(__file__), "data", "modelo_balada.pkl")
    with open(ruta, "rb") as f:
        return pickle.load(f)

df    = cargar_datos()
modelo = cargar_modelo()

FEATURES = [
    "yt_channel_subscribers_log", "channel_age_years",
    "is_vevo", "is_licensed_content",
    "lastfm_artist_listeners", "lastfm_artist_playcount",
    "lastfm_plays_per_listener", "lastfm_top_tag_score",
    "lastfm_top10_listeners_mean", "lastfm_similar_match_mean",
    "territorio_top_1", "territorio_top_2", "n_territorios_top",
    "career_stage", "label_type", "has_featuring", "n_collaborators",
    "release_month",
    "duracion_total_s", "tempo_bpm",
    "energia_rms_mean", "energia_rms_std",
    "brillo_centroide_mean", "brillo_centroide_std",
    "ancho_banda_hz", "caida_espectral_rolloff", "tasa_cruces_cero",
    "bailabilidad_essentia", "complejidad_dinamica_essentia",
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
]

NIVEL_COLORS = {"Alto": "#ef4444", "Medio": "#f59e0b", "Bajo": "#22c55e"}
NIVEL_EMOJIS = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}

# ── Header ────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 9])
with col_title:
    st.markdown('<p class="main-title">🎵 HitBeat</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predicción de alcance musical en YouTube · Música latina en español · IPN ESCOM</p>', unsafe_allow_html=True)

st.divider()

# ── Navegación principal ──────────────────────────────────────
tabs = st.tabs(["🎯 Predecir canción nueva", "📊 Explorar mercado", "🏆 Feature importance", "📚 Metodología"])

# ══════════════════════════════════════════════════════════════
# TAB 1: PREDICTOR ESTRELLA
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Predicción de alcance para una canción nueva")
    st.caption("Ingresa las características del artista y la canción. El modelo estima su nivel de alcance antes del lanzamiento.")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### 🎤 Perfil del artista")
        c1, c2 = st.columns(2)
        with c1:
            subs_millones   = st.number_input("Suscriptores YT (millones)", 0.01, 20.0, 1.0, 0.1)
            channel_age     = st.number_input("Antigüedad del canal (años)", 0.5, 20.0, 5.0, 0.5)
            is_vevo         = st.toggle("Canal VEVO", value=False)
            is_licensed     = st.toggle("Contenido licenciado", value=True)
        with c2:
            lastfm_listeners = st.number_input("Last.fm listeners (miles)", 1, 5000, 200) * 1000
            lastfm_playcount = st.number_input("Last.fm playcount (miles)", 1, 50000, 2000) * 1000
            top_tag_score    = st.slider("Pureza de género (Last.fm)", 10, 100, 65)
            similar_match    = st.slider("Cohesión de clúster (similares)", 0.20, 0.95, 0.55)

        st.markdown("#### 🌍 Alcance territorial")
        c3, c4 = st.columns(2)
        paises = ["MX", "US", "CO", "ES", "AR", "CL", "PE", "VE", "EC", "DO", "PR", "GT", "UY", "PA", "CR", "BO"]
        with c3:
            n_territorios   = st.slider("Nº territorios con tracción", 1, 7, 2)
            territorio_1    = st.selectbox("Territorio principal", paises, index=0)
        with c4:
            if n_territorios >= 2:
                paises_2 = [p for p in paises if p != territorio_1]
                territorio_2 = st.selectbox("Segundo territorio", paises_2, index=1)
            else:
                territorio_2 = "SIN_SEGUNDO"
                st.info("Solo un territorio activo")

        top10_listeners = int(lastfm_listeners * 0.35)

        st.markdown("#### 🏷️ Contexto editorial")
        c5, c6 = st.columns(2)
        with c5:
            career_stage  = st.selectbox("Etapa de carrera", ["Emergente", "Establecido", "Consagrado"])
            label_type    = st.selectbox("Tipo de sello", ["Indie", "Major", "Regional"])
            release_month = st.selectbox("Mes de lanzamiento", list(range(1, 13)),
                                          format_func=lambda m: ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"][m-1])
        with c6:
            has_featuring = st.toggle("¿Tiene featuring?", value=False)
            n_collabs     = st.number_input("Nº de colaboradores", 0, 3, 0) if has_featuring else 0

        st.markdown("#### 🎵 Características de audio")
        c7, c8 = st.columns(2)
        with c7:
            duracion  = st.slider("Duración (segundos)", 120, 400, 210)
            tempo     = st.slider("Tempo (BPM)", 60, 140, 85)
            energia   = st.slider("Energía RMS media", 0.04, 0.31, 0.16)
            energia_s = st.slider("Energía RMS variabilidad", 0.02, 0.12, 0.05)
            brillo    = st.slider("Brillo espectral (Hz)", 1050, 3100, 2000)
        with c8:
            brillo_s  = st.slider("Brillo variabilidad", 350, 1300, 700)
            ancho     = st.slider("Ancho de banda (Hz)", 1500, 3500, 2300)
            rolloff   = st.slider("Spectral rolloff (Hz)", 2500, 8200, 4500)
            zcr       = st.slider("Zero crossing rate", 0.04, 0.21, 0.10)
            bailab    = st.slider("Bailabilidad", 0.70, 1.60, 1.10)
            complejid = st.slider("Complejidad dinámica (dB)", -28.0, -10.0, -18.0)

        st.markdown("#### 🔬 MFCCs")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mfcc1 = mc1.number_input("MFCC 1", -220.0, -110.0, -155.0)
        mfcc2 = mc2.number_input("MFCC 2", 60.0, 105.0, 80.0)
        mfcc3 = mc3.number_input("MFCC 3", -15.0, 35.0, 10.0)
        mfcc4 = mc4.number_input("MFCC 4", 5.0, 50.0, 22.0)
        mfcc5 = mc5.number_input("MFCC 5", -5.0, 30.0, 8.0)

        predecir = st.button("🔮 ANALIZAR CANCIÓN", use_container_width=True, type="primary")

    with col_result:
        st.markdown("#### 📈 Resultado de la predicción")

        if predecir:
            input_data = pd.DataFrame([{
                "yt_channel_subscribers_log":    float(np.log10(subs_millones * 1e6)),
                "channel_age_years":             channel_age,
                "is_vevo":                       int(is_vevo),
                "is_licensed_content":           int(is_licensed),
                "lastfm_artist_listeners":       lastfm_listeners,
                "lastfm_artist_playcount":       lastfm_playcount,
                "lastfm_plays_per_listener":     round(lastfm_playcount / lastfm_listeners, 2),
                "lastfm_top_tag_score":          top_tag_score,
                "lastfm_top10_listeners_mean":   top10_listeners,
                "lastfm_similar_match_mean":     similar_match,
                "territorio_top_1":              territorio_1,
                "territorio_top_2":              territorio_2,
                "n_territorios_top":             n_territorios,
                "career_stage":                  career_stage,
                "label_type":                    label_type,
                "has_featuring":                 int(has_featuring),
                "n_collaborators":               int(n_collabs),
                "release_month":                 release_month,
                "duracion_total_s":              duracion,
                "tempo_bpm":                     float(tempo),
                "energia_rms_mean":              energia,
                "energia_rms_std":               energia_s,
                "brillo_centroide_mean":         float(brillo),
                "brillo_centroide_std":          float(brillo_s),
                "ancho_banda_hz":                float(ancho),
                "caida_espectral_rolloff":       float(rolloff),
                "tasa_cruces_cero":              zcr,
                "bailabilidad_essentia":         bailab,
                "complejidad_dinamica_essentia": complejid,
                "mfcc_1": mfcc1, "mfcc_2": mfcc2, "mfcc_3": mfcc3,
                "mfcc_4": mfcc4, "mfcc_5": mfcc5,
            }])

            nivel_pred = modelo.predict(input_data).flatten()[0]
            proba      = modelo.predict_proba(input_data)[0]
            clases     = list(modelo.classes_)

            prob_dict = {c: p for c, p in zip(clases, proba)}

            color  = NIVEL_COLORS[nivel_pred]
            emoji  = NIVEL_EMOJIS[nivel_pred]

            st.markdown(f"""
            <div style="background:#1e293b; border:2px solid {color};
                        border-radius:16px; padding:2rem; text-align:center; margin-bottom:1rem;">
                <div style="font-size:3rem">{emoji}</div>
                <div style="font-size:2.5rem; font-weight:800; color:{color}">{nivel_pred.upper()}</div>
                <div style="color:#94a3b8; margin-top:0.5rem">Nivel de alcance proyectado</div>
                <div style="color:#64748b; font-size:0.85rem; margin-top:0.3rem">
                    {"< 5M vistas" if nivel_pred == "Bajo" else "5M – 100M vistas" if nivel_pred == "Medio" else "> 100M vistas"}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Barras de probabilidad
            st.markdown("##### Distribución de probabilidades")
            for nivel_c in ["Alto", "Medio", "Bajo"]:
                prob_val = prob_dict.get(nivel_c, 0)
                col_n, col_bar = st.columns([2, 5])
                col_n.markdown(f"{NIVEL_EMOJIS[nivel_c]} **{nivel_c}**")
                col_bar.progress(float(prob_val), text=f"{prob_val*100:.1f}%")

            # Canciones similares del dataset
            st.markdown("##### 🎵 Canciones similares en el dataset")
            mask = df["nivel"] == nivel_pred
            similares = df[mask][["cancion", "artista", "nivel", "n_territorios_top", "career_stage"]].sample(
                min(4, mask.sum()), random_state=42)
            st.dataframe(similares, hide_index=True, use_container_width=True)

        else:
            st.info("👈 Completa el formulario y haz click en **ANALIZAR CANCIÓN**")
            st.markdown("""
            #### ¿Qué predice HitBeat?
            El modelo estima si una canción nueva alcanzará nivel:
            - 🔴 **Alto** — más de 100M vistas
            - 🟡 **Medio** — entre 5M y 100M vistas
            - 🟢 **Bajo** — menos de 5M vistas

            Usando **34 características** del artista y la canción
            disponibles **antes del lanzamiento**.
            """)


# ══════════════════════════════════════════════════════════════
# TAB 2: EXPLORAR MERCADO
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Exploración del dataset de Balada")

    # Filtros en sidebar
    with st.sidebar:
        st.markdown("### 🔍 Filtros")
        nivel_filtro  = st.multiselect("Nivel", ["Alto", "Medio", "Bajo"],
                                        default=["Alto", "Medio", "Bajo"])
        origen_filtro = st.multiselect("Origen", ["real", "sintetico"],
                                        default=["real", "sintetico"])
        career_filtro = st.multiselect("Etapa de carrera",
                                        df["career_stage"].unique().tolist(),
                                        default=df["career_stage"].unique().tolist())

    df_f = df[
        df["nivel"].isin(nivel_filtro) &
        df["origen"].isin(origen_filtro) &
        df["career_stage"].isin(career_filtro)
    ]

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Canciones", len(df_f))
    k2.metric("Artistas únicos", df_f["artista"].nunique())
    k3.metric("Países", df_f["pais"].nunique())
    k4.metric("Con featuring", f"{df_f['has_featuring'].mean()*100:.0f}%")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        # Distribución de niveles
        nivel_counts = df_f["nivel"].value_counts().reset_index()
        fig_nivel = px.pie(
            nivel_counts, values="count", names="nivel",
            title="Distribución de niveles",
            color="nivel",
            color_discrete_map=NIVEL_COLORS,
            hole=0.4,
        )
        fig_nivel.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        st.plotly_chart(fig_nivel, use_container_width=True)

    with c2:
        # n_territorios por nivel
        fig_ter = px.box(
            df_f, x="nivel", y="n_territorios_top",
            color="nivel", color_discrete_map=NIVEL_COLORS,
            title="Territorios de impacto por nivel",
            category_orders={"nivel": ["Bajo", "Medio", "Alto"]},
        )
        fig_ter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", showlegend=False,
        )
        st.plotly_chart(fig_ter, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Tempo por nivel
        fig_tempo = px.violin(
            df_f, x="nivel", y="tempo_bpm",
            color="nivel", color_discrete_map=NIVEL_COLORS,
            title="Distribución de Tempo (BPM) por nivel",
            box=True, category_orders={"nivel": ["Bajo", "Medio", "Alto"]},
        )
        fig_tempo.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", showlegend=False,
        )
        st.plotly_chart(fig_tempo, use_container_width=True)

    with c4:
        # Suscriptores por nivel
        df_f["subs_log"] = df_f["yt_channel_subscribers_log"]
        fig_subs = px.box(
            df_f, x="nivel", y="subs_log",
            color="nivel", color_discrete_map=NIVEL_COLORS,
            title="Suscriptores del canal (log10) por nivel",
            category_orders={"nivel": ["Bajo", "Medio", "Alto"]},
        )
        fig_subs.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", showlegend=False,
        )
        st.plotly_chart(fig_subs, use_container_width=True)

    # Top países por nivel
    st.markdown("#### 🌍 Territorios de mayor impacto por nivel")
    cols_paises = st.columns(3)
    for i, nivel in enumerate(["Alto", "Medio", "Bajo"]):
        sub = df_f[df_f["nivel"] == nivel]
        pais_counts = sub["territorio_top_1"].value_counts().head(6).reset_index()
        fig_p = px.bar(
            pais_counts, x="territorio_top_1", y="count",
            title=f"{NIVEL_EMOJIS[nivel]} Nivel {nivel}",
            color_discrete_sequence=[NIVEL_COLORS[nivel]],
        )
        fig_p.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", showlegend=False,
            xaxis_title="", yaxis_title="canciones",
        )
        cols_paises[i].plotly_chart(fig_p, use_container_width=True)

    # Scatter: subs vs territorios coloreado por nivel
    st.markdown("#### 📊 Relación entre base de fans y alcance territorial")
    fig_scatter = px.scatter(
        df_f, x="yt_channel_subscribers_log", y="n_territorios_top",
        color="nivel", color_discrete_map=NIVEL_COLORS,
        hover_data=["artista", "cancion", "career_stage"],
        title="Suscriptores del canal (log10) vs Territorios de impacto",
        opacity=0.7, size_max=8,
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Importancia de variables — Modelo Balada")
    st.caption("Calculado con CatBoost Feature Importance sobre el modelo entrenado con 333 canciones.")

    importances = pd.DataFrame({
        "feature":    FEATURES,
        "importance": modelo.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    importances["bloque"] = importances["feature"].apply(lambda f:
        "Territorial" if f in ["territorio_top_1", "territorio_top_2", "n_territorios_top"]
        else "Last.fm"    if f.startswith("lastfm")
        else "YouTube"    if f in ["yt_channel_subscribers_log", "channel_age_years", "is_vevo", "is_licensed_content"]
        else "Editorial"  if f in ["career_stage", "label_type", "has_featuring", "n_collaborators", "release_month"]
        else "Audio"
    )

    bloque_colors = {
        "Territorial": "#f97316",
        "Last.fm":     "#f5c842",
        "YouTube":     "#4ade80",
        "Editorial":   "#a855f7",
        "Audio":       "#f87171",
    }

    fig_imp = px.bar(
        importances, x="importance", y="feature",
        color="bloque", color_discrete_map=bloque_colors,
        orientation="h",
        title="Feature Importance (CatBoost) — 34 variables",
        text="importance",
    )
    fig_imp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_imp.update_layout(
        height=900,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Importancia (%)",
        yaxis_title="",
        legend_title="Bloque",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Resumen por bloque
    st.markdown("#### Importancia acumulada por bloque")
    bloque_sum = importances.groupby("bloque")["importance"].sum().reset_index()
    bloque_sum = bloque_sum.sort_values("importance", ascending=False)

    fig_bloque = px.pie(
        bloque_sum, values="importance", names="bloque",
        color="bloque", color_discrete_map=bloque_colors,
        hole=0.45,
    )
    fig_bloque.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="white",
    )
    col_pie, col_table = st.columns([1, 1])
    col_pie.plotly_chart(fig_bloque, use_container_width=True)

    bloque_sum.columns = ["Bloque", "Importancia total (%)"]
    bloque_sum["Importancia total (%)"] = bloque_sum["Importancia total (%)"].round(2)
    col_table.markdown(" ")
    col_table.markdown(" ")
    col_table.dataframe(bloque_sum, hide_index=True, use_container_width=True)

    st.info("""
    **Hallazgo clave:** `n_territorios_top` (33.6%) domina el modelo con un amplio margen.
    El modelo aprendió que **cuántos territorios hispanohablantes tienen tracción con el artista
    predice mejor el alcance de una canción nueva que cualquier feature de audio individual.**
    Las variables de audio contribuyen en conjunto (~30% acumulado) pero ninguna domina individualmente,
    consistente con la literatura (Herremans 2014, Interiano 2018).
    """)


# ══════════════════════════════════════════════════════════════
# TAB 4: METODOLOGÍA
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Metodología del proyecto HitBeat")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
        #### ¿Qué predice HitBeat?
        HitBeat clasifica canciones nuevas en tres niveles de alcance
        en YouTube antes de su lanzamiento:

        | Nivel | Rango de vistas | Representación |
        |---|---|---|
        | 🔴 Alto | > 100M | ~5% de videos musicales |
        | 🟡 Medio | 5M – 100M | ~20% |
        | 🟢 Bajo | < 5M | ~75% |

        #### Dataset
        - **333 canciones** de Balada en español
        - **170 reales** + 163 sintéticas calibradas
        - Balance exacto: 111 por clase
        - Rango temporal: 2010–2024

        #### Pipeline en Databricks
        ```
        Discovery → Bronze → Silver → Gold → Dashboard
        ```
        - **Bronze**: dataset unificado en Delta Lake
        - **Silver**: selección de 34 features PRE-lanzamiento
        - **Gold**: modelo entrenado + predicciones + MLflow Registry
        """)

    with col_m2:
        st.markdown("""
        #### Modelo: CatBoost Multiclase
        - **Algoritmo**: Gradient Boosting con manejo nativo de categóricas
        - **Validación**: 5-Fold Stratified Cross Validation
        - **Accuracy CV**: **67.86% ± 2.86%** (baseline azar = 33%)
        - **Registro**: MLflow Model Registry en Unity Catalog

        #### Variables de inferencia (34 features)
        Todas disponibles **antes del lanzamiento**:

        | Bloque | Variables | Importancia |
        |---|---|---|
        | 🟠 Territorial | 3 (Last.fm geo) | ~42% |
        | 🟡 Last.fm | 6 | ~14% |
        | 🟢 YouTube | 4 | ~10% |
        | 🟣 Editorial | 5 | ~8% |
        | 🔴 Audio | 16 (librosa+essentia) | ~26% |

        #### Limitaciones
        - Dataset de Balada únicamente (Reguetón y Corridos en proceso)
        - Overfitting moderado con 333 muestras (mejora con más datos)
        - Clase Medio estructuralmente ambigua por amplitud del rango

        #### Referencias
        - Herremans et al. (2014) — modelos por género
        - Interiano et al. (2018) — predictores de hits
        - Pachet & Roy (2008) — límites del audio-only
        """)

    st.divider()
    st.markdown("""
    <div style='text-align:center; color:#64748b; font-size:0.85rem'>
    HitBeat · Trabajo Terminal · IPN ESCOM · Mayo 2026<br>
    Pipeline: Databricks (Delta Lake + MLflow) → Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)
