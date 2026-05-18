import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os

st.set_page_config(
    page_title="HitBeat — Predictor de Alcance Musical",
    page_icon="🎵", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(90deg, #1DB954, #F5C842, #A855F7, #38BDF8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { color: #94a3b8; font-size: 1rem; }
    .stTabs [data-baseweb="tab"] {
        background: #1e293b; border-radius: 8px;
        padding: 8px 20px; color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def cargar_datos():
    ruta = os.path.join(os.path.dirname(__file__), "data", "hitbeat_dashboard_data.csv")
    return pd.read_csv(ruta)

@st.cache_resource
def cargar_modelo():
    ruta = os.path.join(os.path.dirname(__file__), "data", "modelo_balada.pkl")
    with open(ruta, "rb") as f:
        return pickle.load(f)

df     = cargar_datos()
modelo = cargar_modelo()

FEATURES = [
    "yt_channel_subscribers_log", "channel_age_years",
    "is_vevo", "is_licensed_content",
    "lastfm_artist_listeners", "lastfm_artist_playcount",
    "lastfm_plays_per_listener", "lastfm_top_tag_score",
    "lastfm_top10_listeners_mean", "lastfm_similar_match_mean",
    "territorio_top_1", "territorio_top_2", "n_territorios_top",
    "career_stage", "label_type", "has_featuring", "n_collaborators",
    "release_month", "duracion_total_s", "tempo_bpm",
    "energia_rms_mean", "energia_rms_std",
    "brillo_centroide_mean", "brillo_centroide_std",
    "ancho_banda_hz", "caida_espectral_rolloff", "tasa_cruces_cero",
    "bailabilidad_essentia", "complejidad_dinamica_essentia",
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
]

NIVEL_COLORS = {"Alto": "#ef4444", "Medio": "#f59e0b", "Bajo": "#22c55e"}
NIVEL_EMOJIS = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}
PAISES = ["MX","US","CO","ES","AR","CL","PE","VE","EC","DO","PR","GT","UY","PA","CR","BO"]

def extraer_features_audio(wav_bytes):
    try:
        import librosa, io
        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=22050, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
        if tempo > 140: tempo /= 2
        rms      = librosa.feature.rms(y=y)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        zcr      = librosa.feature.zero_crossing_rate(y=y)[0]
        mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        duracion = float(librosa.get_duration(y=y, sr=sr))
        y_harm, y_perc = librosa.effects.hpss(y)
        perc_ratio = float(np.mean(np.abs(y_perc)) / (np.mean(np.abs(y_harm)) + 1e-6))
        beat_frames = librosa.beat.beat_track(y=y, sr=sr)[1]
        beat_stab   = (1 / (1 + np.std(np.diff(beat_frames)) / (np.mean(np.diff(beat_frames)) + 1e-6))
                       if len(beat_frames) > 2 else 0.5)
        bailabilidad = float(np.clip(perc_ratio * beat_stab * 2.5, 0.70, 1.60))
        rms_db   = librosa.amplitude_to_db(rms)
        comp_din = float(np.clip(np.percentile(rms_db, 5) - np.percentile(rms_db, 95), -28, -10))
        return {
            "duracion_total_s": round(duracion, 2), "tempo_bpm": round(tempo, 1),
            "energia_rms_mean": round(float(np.mean(rms)), 4),
            "energia_rms_std":  round(float(np.std(rms)), 4),
            "brillo_centroide_mean": round(float(np.mean(centroid)), 2),
            "brillo_centroide_std":  round(float(np.std(centroid)), 2),
            "ancho_banda_hz":        round(float(np.mean(bw)), 2),
            "caida_espectral_rolloff": round(float(np.mean(rolloff)), 2),
            "tasa_cruces_cero":      round(float(np.mean(zcr)), 4),
            "bailabilidad_essentia":         round(bailabilidad, 4),
            "complejidad_dinamica_essentia": round(comp_din, 4),
            "mfcc_1": round(float(mfccs[0]), 4), "mfcc_2": round(float(mfccs[1]), 4),
            "mfcc_3": round(float(mfccs[2]), 4), "mfcc_4": round(float(mfccs[3]), 4),
            "mfcc_5": round(float(mfccs[4]), 4),
        }, None
    except Exception as e:
        return None, str(e)

# ── Header ────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎵 HitBeat</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predicción de alcance musical en YouTube · Música latina en español · IPN ESCOM</p>', unsafe_allow_html=True)
st.divider()

tabs = st.tabs(["🎯 Predecir canción nueva", "📊 Explorar mercado", "🏆 Feature importance", "📚 Metodología"])

# ══════════ TAB 1: PREDICTOR ══════════
with tabs[0]:
    st.markdown("### Predicción de alcance para una canción nueva")
    st.caption("Sube el archivo de audio y completa el perfil del artista. El modelo estima el alcance antes del lanzamiento.")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### 🎵 Archivo de audio")
        wav_file = st.file_uploader(
            "Sube el archivo de la canción (.wav o .mp3)",
            type=["wav", "mp3"],
            help="Se extraen automáticamente las 16 características acústicas con Librosa."
        )

        audio_features, audio_ok = {}, False

        if wav_file is not None:
            with st.spinner("Analizando audio..."):
                feats, err = extraer_features_audio(wav_file.read())
            if err:
                st.error(f"Error al procesar el audio: {err}")
            else:
                audio_features, audio_ok = feats, True
                st.success("✅ Audio analizado correctamente")
                with st.expander("Ver características acústicas extraídas"):
                    c_a1, c_a2 = st.columns(2)
                    labels = [
                        ("Duración", f"{feats['duracion_total_s']:.1f} s"),
                        ("Tempo", f"{feats['tempo_bpm']:.1f} BPM"),
                        ("Energía RMS", f"{feats['energia_rms_mean']:.4f}"),
                        ("Brillo espectral", f"{feats['brillo_centroide_mean']:.0f} Hz"),
                        ("Bailabilidad", f"{feats['bailabilidad_essentia']:.3f}"),
                        ("Complejidad dinámica", f"{feats['complejidad_dinamica_essentia']:.2f} dB"),
                        ("Ancho de banda", f"{feats['ancho_banda_hz']:.0f} Hz"),
                        ("Spectral rolloff", f"{feats['caida_espectral_rolloff']:.0f} Hz"),
                        ("Zero crossing rate", f"{feats['tasa_cruces_cero']:.4f}"),
                        ("MFCC 1-5", f"{feats['mfcc_1']:.1f} / {feats['mfcc_2']:.1f} / {feats['mfcc_3']:.1f} / {feats['mfcc_4']:.1f} / {feats['mfcc_5']:.1f}"),
                    ]
                    for i, (label, val) in enumerate(labels):
                        (c_a1 if i < 5 else c_a2).metric(label, val)
        else:
            st.info("👆 Sube un archivo .wav o .mp3 para extraer automáticamente las características de audio.")

        st.divider()
        st.markdown("#### 📡 Contexto del artista")
        c1, c2 = st.columns(2)
        with c1:
            subs_millones    = st.number_input("Suscriptores YT (millones)", 0.01, 20.0, 1.0, 0.1)
            channel_age      = st.number_input("Antigüedad del canal (años)", 0.5, 20.0, 5.0, 0.5)
            is_vevo          = st.toggle("Canal VEVO", value=False)
            is_licensed      = st.toggle("Contenido licenciado", value=True)
        with c2:
            lastfm_listeners = st.number_input("Last.fm listeners (miles)", 1, 5000, 200) * 1000
            lastfm_playcount = st.number_input("Last.fm playcount (miles)", 1, 50000, 2000) * 1000
            top_tag_score    = st.slider("Pureza de género (Last.fm)", 10, 100, 65)
            similar_match    = st.slider("Cohesión de clúster de similares", 0.20, 0.95, 0.55)

        st.markdown("#### 🌍 Alcance territorial del artista")
        c3, c4 = st.columns(2)
        with c3:
            n_territorios = st.slider("Nº territorios con tracción", 1, 7, 2)
            territorio_1  = st.selectbox("Territorio principal", PAISES, index=0)
        with c4:
            if n_territorios >= 2:
                territorio_2 = st.selectbox("Segundo territorio", [p for p in PAISES if p != territorio_1], index=1)
            else:
                territorio_2 = "SIN_SEGUNDO"
                st.info("Con 1 territorio, top_2 = SIN_SEGUNDO")

        st.markdown("#### 🏷️ Contexto editorial")
        c5, c6 = st.columns(2)
        with c5:
            career_stage  = st.selectbox("Etapa de carrera", ["Emergente", "Establecido", "Consagrado"])
            label_type    = st.selectbox("Tipo de sello", ["Indie", "Major", "Regional"])
            release_month = st.selectbox("Mes de lanzamiento", list(range(1,13)),
                                          format_func=lambda m: ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"][m-1])
        with c6:
            has_featuring = st.toggle("¿Tiene featuring?", value=False)
            n_collabs     = st.number_input("Nº de colaboradores", 0, 3, 0) if has_featuring else 0

        st.divider()
        if not audio_ok:
            st.warning("⚠️ Sube un archivo de audio para habilitar el análisis.")
        predecir = st.button("🔮 ANALIZAR CANCIÓN", use_container_width=True,
                              type="primary", disabled=not audio_ok)

    with col_result:
        st.markdown("#### 📈 Resultado de la predicción")
        if predecir and audio_ok:
            input_data = pd.DataFrame([{
                "yt_channel_subscribers_log":  float(np.log10(subs_millones * 1e6)),
                "channel_age_years":           channel_age,
                "is_vevo":                     int(is_vevo),
                "is_licensed_content":         int(is_licensed),
                "lastfm_artist_listeners":     lastfm_listeners,
                "lastfm_artist_playcount":     lastfm_playcount,
                "lastfm_plays_per_listener":   round(lastfm_playcount / lastfm_listeners, 2),
                "lastfm_top_tag_score":        top_tag_score,
                "lastfm_top10_listeners_mean": int(lastfm_listeners * 0.35),
                "lastfm_similar_match_mean":   similar_match,
                "territorio_top_1":            territorio_1,
                "territorio_top_2":            territorio_2,
                "n_territorios_top":           n_territorios,
                "career_stage":                career_stage,
                "label_type":                  label_type,
                "has_featuring":               int(has_featuring),
                "n_collaborators":             int(n_collabs),
                "release_month":               release_month,
                **audio_features,
            }])

            nivel_pred = modelo.predict(input_data).flatten()[0]
            proba      = modelo.predict_proba(input_data)[0]
            clases     = list(modelo.classes_)
            prob_dict  = {c: p for c, p in zip(clases, proba)}
            color      = NIVEL_COLORS[nivel_pred]
            emoji      = NIVEL_EMOJIS[nivel_pred]

            st.markdown(f"""
            <div style="background:#1e293b; border:2px solid {color};
                        border-radius:16px; padding:2rem; text-align:center; margin-bottom:1.5rem;">
                <div style="font-size:3.5rem">{emoji}</div>
                <div style="font-size:2.8rem; font-weight:800; color:{color}; margin:0.3rem 0">{nivel_pred.upper()}</div>
                <div style="color:#94a3b8; font-size:1rem">Nivel de alcance proyectado</div>
                <div style="color:#64748b; font-size:0.85rem; margin-top:0.4rem">
                    {"< 5 millones de vistas" if nivel_pred=="Bajo" else "Entre 5M y 100M vistas" if nivel_pred=="Medio" else "Más de 100 millones de vistas"}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### Distribución de probabilidades")
            for nv in ["Alto", "Medio", "Bajo"]:
                p = prob_dict.get(nv, 0)
                col_n, col_b = st.columns([2, 5])
                col_n.markdown(f"{NIVEL_EMOJIS[nv]} **{nv}**")
                col_b.progress(float(p), text=f"{p*100:.1f}%")

            st.divider()
            st.markdown("##### 🎵 Canciones similares en el dataset de referencia")
            mask = df["nivel"] == nivel_pred
            similares = df[mask][["cancion","artista","nivel","n_territorios_top","career_stage","origen"]].sample(min(5, mask.sum()), random_state=42)
            st.dataframe(similares, hide_index=True, use_container_width=True)

        elif not audio_ok:
            st.info("👈 Sube un archivo de audio y completa el perfil del artista.")
            st.markdown("""
            #### ¿Cómo funciona HitBeat?
            1. **Sube el audio** (.wav o .mp3) — la app extrae automáticamente las 16 características acústicas con Librosa
            2. **Completa el perfil del artista** — datos contextuales disponibles antes del lanzamiento
            3. **Obtén la predicción** — Bajo / Medio / Alto con probabilidades

            Modelo entrenado con **333 canciones de Balada** · **Accuracy CV: 67.86%** (azar = 33%)
            """)

# ══════════ TAB 2: EXPLORAR MERCADO ══════════
with tabs[1]:
    st.markdown("### Exploración del dataset de Balada")
    with st.sidebar:
        st.markdown("### 🔍 Filtros")
        nivel_filtro  = st.multiselect("Nivel", ["Alto","Medio","Bajo"], default=["Alto","Medio","Bajo"])
        origen_filtro = st.multiselect("Origen", ["real","sintetico"], default=["real","sintetico"])
        career_filtro = st.multiselect("Etapa de carrera", df["career_stage"].unique().tolist(), default=df["career_stage"].unique().tolist())

    df_f = df[df["nivel"].isin(nivel_filtro) & df["origen"].isin(origen_filtro) & df["career_stage"].isin(career_filtro)]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Canciones", len(df_f))
    k2.metric("Artistas únicos", df_f["artista"].nunique())
    k3.metric("Países", df_f["pais"].nunique())
    k4.metric("Con featuring", f"{df_f['has_featuring'].mean()*100:.0f}%")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df_f["nivel"].value_counts().reset_index(), values="count", names="nivel",
                     title="Distribución de niveles", color="nivel", color_discrete_map=NIVEL_COLORS, hole=0.4)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df_f, x="nivel", y="n_territorios_top", color="nivel",
                     color_discrete_map=NIVEL_COLORS, title="Territorios de impacto por nivel",
                     category_orders={"nivel":["Bajo","Medio","Alto"]})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.violin(df_f, x="nivel", y="tempo_bpm", color="nivel",
                        color_discrete_map=NIVEL_COLORS, title="Tempo (BPM) por nivel",
                        box=True, category_orders={"nivel":["Bajo","Medio","Alto"]})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.box(df_f, x="nivel", y="yt_channel_subscribers_log", color="nivel",
                     color_discrete_map=NIVEL_COLORS, title="Suscriptores canal (log10) por nivel",
                     category_orders={"nivel":["Bajo","Medio","Alto"]})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🌍 Territorios de mayor impacto por nivel")
    cols_p = st.columns(3)
    for i, nivel in enumerate(["Alto","Medio","Bajo"]):
        sub = df_f[df_f["nivel"]==nivel]
        pc  = sub["territorio_top_1"].value_counts().head(6).reset_index()
        fig = px.bar(pc, x="territorio_top_1", y="count", title=f"{NIVEL_EMOJIS[nivel]} Nivel {nivel}",
                     color_discrete_sequence=[NIVEL_COLORS[nivel]])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                          showlegend=False, xaxis_title="", yaxis_title="canciones")
        cols_p[i].plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📊 Suscriptores vs Territorios de impacto")
    fig = px.scatter(df_f, x="yt_channel_subscribers_log", y="n_territorios_top",
                     color="nivel", color_discrete_map=NIVEL_COLORS,
                     hover_data=["artista","cancion","career_stage"],
                     title="Canal (log10 subs) vs Territorios de impacto", opacity=0.7)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

# ══════════ TAB 3: FEATURE IMPORTANCE ══════════
with tabs[2]:
    st.markdown("### Importancia de variables — Modelo Balada")
    importances = pd.DataFrame({
        "feature": FEATURES, "importance": modelo.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importances["bloque"] = importances["feature"].apply(lambda f:
        "Territorial" if f in ["territorio_top_1","territorio_top_2","n_territorios_top"]
        else "Last.fm"   if f.startswith("lastfm")
        else "YouTube"   if f in ["yt_channel_subscribers_log","channel_age_years","is_vevo","is_licensed_content"]
        else "Editorial" if f in ["career_stage","label_type","has_featuring","n_collaborators","release_month"]
        else "Audio"
    )
    bloque_colors = {"Territorial":"#f97316","Last.fm":"#f5c842","YouTube":"#4ade80","Editorial":"#a855f7","Audio":"#f87171"}
    fig = px.bar(importances, x="importance", y="feature", color="bloque",
                 color_discrete_map=bloque_colors, orientation="h",
                 title="Feature Importance — 34 variables", text="importance")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=900, paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                      yaxis=dict(categoryorder="total ascending"), xaxis_title="Importancia (%)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    bloque_sum = importances.groupby("bloque")["importance"].sum().reset_index().sort_values("importance", ascending=False)
    fig2 = px.pie(bloque_sum, values="importance", names="bloque", color="bloque",
                  color_discrete_map=bloque_colors, hole=0.45)
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    col_pie, col_tab = st.columns([1,1])
    col_pie.plotly_chart(fig2, use_container_width=True)
    bloque_sum.columns = ["Bloque","Importancia (%)"]
    bloque_sum["Importancia (%)"] = bloque_sum["Importancia (%)"].round(2)
    col_tab.dataframe(bloque_sum, hide_index=True, use_container_width=True)
    st.info("**Hallazgo clave:** `n_territorios_top` (33.6%) domina el modelo. La amplitud geográfica del artista predice mejor el alcance que cualquier feature de audio individual.")

# ══════════ TAB 4: METODOLOGÍA ══════════
with tabs[3]:
    st.markdown("### Metodología del proyecto HitBeat")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### ¿Qué predice HitBeat?
        | Nivel | Rango de vistas | Representación |
        |---|---|---|
        | 🔴 Alto | > 100M | ~5% |
        | 🟡 Medio | 5M – 100M | ~20% |
        | 🟢 Bajo | < 5M | ~75% |

        #### Dataset
        - 333 canciones de Balada en español
        - 170 reales + 163 sintéticas calibradas
        - Balance: 111 por clase · 2010–2024

        #### Extracción de audio
        16 features acústicas extraídas automáticamente del archivo .wav con **Librosa** (Python). No se requiere acceso posterior a la publicación.

        #### Pipeline en Databricks
        `Bronze → Silver → Gold`
        Orquestado como Job con trigger por File Arrival.
        """)
    with c2:
        st.markdown("""
        #### Modelo: CatBoost Multiclase
        - Validación: 5-Fold Stratified Cross Validation
        - **Accuracy CV: 67.86% ± 2.86%** (azar = 33%)
        - Registro: MLflow Unity Catalog

        #### Variables de inferencia (34 features)
        | Bloque | N | Importancia |
        |---|---|---|
        | 🟠 Territorial | 3 | ~42% |
        | 🟡 Last.fm | 6 | ~14% |
        | 🟢 YouTube | 4 | ~10% |
        | 🟣 Editorial | 5 | ~8% |
        | 🔴 Audio | 16 | ~26% |

        #### Limitaciones
        - Dataset de Balada únicamente (Reguetón y Corridos en proceso)
        - Overfitting moderado con 333 muestras
        - Clase Medio estructuralmente ambigua

        #### Referencias
        - Herremans et al. (2014) · Interiano et al. (2018) · Pachet & Roy (2008)
        """)
    st.divider()
    st.markdown("<div style='text-align:center;color:#64748b;font-size:0.85rem'>HitBeat · TT · IPN ESCOM · Mayo 2026 · Databricks + Streamlit Cloud</div>", unsafe_allow_html=True)
