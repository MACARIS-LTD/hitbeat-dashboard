import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HitBeat — Predicción de Alcance Musical",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Estilo: editorial, profesional, sin saturación ──────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    h1, h2, h3 { font-weight: 500 !important; letter-spacing: -0.01em; }
    .brand {
        font-family: 'Georgia', serif;
        font-size: 2.4rem; font-weight: 400;
        letter-spacing: -0.02em; color: #e2e8f0;
        margin-bottom: 0.2rem;
    }
    .brand-sub {
        color: #94a3b8; font-size: 0.95rem;
        letter-spacing: 0.02em; font-weight: 300;
    }
    .section-label {
        font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.12em; text-transform: uppercase;
        color: #64748b; margin: 1.4rem 0 0.6rem;
    }
    .helper {
        color: #64748b; font-size: 0.8rem;
        font-style: italic; margin-top: 4px;
    }
    .result-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1.5rem; margin-bottom: 1rem;
    }
    .level-pill {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 8px 22px; border-radius: 99px;
        font-size: 1.3rem; font-weight: 500;
        letter-spacing: 0.02em;
    }
    .warning-box {
        background: rgba(245, 158, 11, 0.08);
        border-left: 3px solid #f59e0b;
        padding: 10px 14px; border-radius: 4px;
        font-size: 0.85rem; color: #fbbf24; margin: 8px 0;
    }
    .error-box {
        background: rgba(239, 68, 68, 0.08);
        border-left: 3px solid #ef4444;
        padding: 10px 14px; border-radius: 4px;
        font-size: 0.85rem; color: #f87171; margin: 8px 0;
    }
    .region-row {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 12px;
        background: #1e293b;
        border-radius: 8px; margin-bottom: 6px;
    }
    .month-cell {
        text-align: center; padding: 8px 4px;
        border-radius: 6px; font-size: 0.78rem;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; border-bottom: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 0 !important;
        color: #64748b !important;
        padding: 12px 18px;
        font-size: 0.9rem; font-weight: 400;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #e2e8f0 !important;
        border-bottom: 2px solid #38bdf8 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stExpander"] {
        background: #0f172a; border: 1px solid #1e293b;
        border-radius: 8px;
    }
    /* Tooltips de criterios de trayectoria */
    .trayectoria-tooltips {
        display: flex; gap: 1.2rem; flex-wrap: wrap;
        margin: 0.3rem 0 0.8rem 0;
    }
    .trayectoria-chip {
        display: inline-flex; align-items: center; gap: 6px;
        color: #94a3b8; font-size: 0.8rem;
        cursor: help;
    }
    .trayectoria-chip .help-icon {
        display: inline-flex; align-items: center; justify-content: center;
        width: 16px; height: 16px;
        border: 1px solid #475569; border-radius: 50%;
        font-size: 0.7rem; color: #94a3b8;
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CARGA DE DATOS Y MODELO
# ══════════════════════════════════════════════════════════════
@st.cache_data
def cargar_datos():
    ruta = os.path.join(os.path.dirname(__file__), "data", "hitbeat_dashboard_data.csv")
    return pd.read_csv(ruta)

@st.cache_resource
def cargar_modelo():
    ruta = os.path.join(os.path.dirname(__file__), "data", "modelo_unificado.pkl")
    with open(ruta, "rb") as f:
        return pickle.load(f)

df     = cargar_datos()
modelo = cargar_modelo()


# ══════════════════════════════════════════════════════════════
# CONSTANTES Y REGLAS DE CONSISTENCIA
# ══════════════════════════════════════════════════════════════
# 35 features: 'genero' es la primera, el resto en el mismo orden con que se entrenó el modelo
FEATURES = [
    "genero",
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

# Géneros soportados por el modelo unificado
GENEROS = ["Balada", "Reguetón", "Regional Mexicano"]

# Criterios para cada nivel de trayectoria (usados en tooltips)
TRAYECTORIA_CRITERIOS = {
    "Emergente": (
        "Artista en construcción de audiencia. "
        "Suscriptores típicos por debajo de 500K en YouTube y oyentes en Last.fm en órdenes similares. "
        "Carrera en ascenso o reciente (menos de 5 años activos), presencia territorial limitada a 1–3 países."
    ),
    "Establecido": (
        "Artista con audiencia regional consolidada. "
        "Suscriptores típicos entre 500K y 5M, oyentes en Last.fm en el orden de cientos de miles a varios millones. "
        "Mínimo 3 años de trayectoria activa, presencia en 2–5 territorios."
    ),
    "Consagrado": (
        "Artista con audiencia masiva e internacional. "
        "Suscriptores típicos por encima de 5M (la mediana ronda los 10M), "
        "oyentes robustos en Last.fm y trayectoria mayor a 5 años. "
        "Presencia consolidada en múltiples mercados hispanohablantes y, frecuentemente, cruce a mercados anglosajones."
    ),
}

# Rangos en valores numéricos representativos del rango
SUBS_RANGOS = {
    "Menos de 100 mil":      50_000,
    "100 mil – 500 mil":     250_000,
    "500 mil – 2 millones":  1_000_000,
    "2 – 10 millones":       5_000_000,
    "Más de 10 millones":    15_000_000,
}
LISTENERS_RANGOS = {
    "Menos de 50 mil":           25_000,
    "50 mil – 500 mil":          200_000,
    "500 mil – 2 millones":      1_000_000,
    "Más de 2 millones":         3_500_000,
}
PLAYCOUNT_RANGOS = {
    "Menos de 1 millón":         500_000,
    "1 – 10 millones":           5_000_000,
    "10 – 100 millones":         50_000_000,
    "Más de 100 millones":       300_000_000,
}
ANTIGUEDAD_RANGOS = {
    "Menos de 2 años":  1.5,
    "2 – 5 años":       3.5,
    "5 – 10 años":      7.5,
    "Más de 10 años":   14.0,
}

# Reglas de compatibilidad por trayectoria (calibradas con percentiles reales)
COMPAT = {
    "Emergente": {
        "subs":       ["Menos de 100 mil", "100 mil – 500 mil", "500 mil – 2 millones"],
        "listeners":  ["Menos de 50 mil", "50 mil – 500 mil"],
        "antiguedad": ["Menos de 2 años", "2 – 5 años", "5 – 10 años", "Más de 10 años"],
        "sello":      ["Independiente", "Sello regional", "Major (Universal / Sony / Warner)"],
        "vevo":       True,
        "max_territorios": 4,
    },
    "Establecido": {
        "subs":       ["500 mil – 2 millones", "2 – 10 millones"],
        "listeners":  ["50 mil – 500 mil", "500 mil – 2 millones", "Más de 2 millones"],
        "antiguedad": ["2 – 5 años", "5 – 10 años", "Más de 10 años"],
        "sello":      ["Independiente", "Major (Universal / Sony / Warner)"],
        "vevo":       True,
        "max_territorios": 6,
    },
    "Consagrado": {
        "subs":       ["2 – 10 millones", "Más de 10 millones"],
        "listeners":  ["500 mil – 2 millones", "Más de 2 millones"],
        "antiguedad": ["5 – 10 años", "Más de 10 años"],
        "sello":      ["Independiente", "Major (Universal / Sony / Warner)"],
        "vevo":       True,
        "max_territorios": 7,
    },
}

# Pool de países hispanos
PAISES_DISPLAY = {
    "MX": "México",          "US": "Estados Unidos",  "CO": "Colombia",
    "ES": "España",          "AR": "Argentina",       "CL": "Chile",
    "PE": "Perú",            "VE": "Venezuela",       "EC": "Ecuador",
    "DO": "República Dominicana", "PR": "Puerto Rico",
    "GT": "Guatemala",       "UY": "Uruguay",         "PA": "Panamá",
    "CR": "Costa Rica",      "BO": "Bolivia",
}
PAISES_FLAGS = {
    "MX": "🇲🇽","US": "🇺🇸","CO": "🇨🇴","ES": "🇪🇸","AR": "🇦🇷","CL": "🇨🇱",
    "PE": "🇵🇪","VE": "🇻🇪","EC": "🇪🇨","DO": "🇩🇴","PR": "🇵🇷","GT": "🇬🇹",
    "UY": "🇺🇾","PA": "🇵🇦","CR": "🇨🇷","BO": "🇧🇴",
}

NIVEL_COLORS = {"Alto": "#ef4444", "Medio": "#f59e0b", "Bajo": "#22c55e"}
NIVEL_BG     = {"Alto": "rgba(239,68,68,0.12)",
                "Medio": "rgba(245,158,11,0.12)",
                "Bajo":  "rgba(34,197,94,0.12)"}
NIVEL_DESC   = {
    "Alto":  "Más de 100 millones de vistas en YouTube",
    "Medio": "Entre 5 y 100 millones de vistas",
    "Bajo":  "Menos de 5 millones de vistas",
}


# ══════════════════════════════════════════════════════════════
# EXTRACCIÓN DE FEATURES DE AUDIO
# ══════════════════════════════════════════════════════════════
def extraer_features_audio(wav_bytes):
    """Extrae 16 features acústicas con Librosa. Aproxima las 2 de Essentia."""
    try:
        import librosa
        import io

        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=22050, mono=True)

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
        if tempo > 140:
            tempo = tempo / 2

        rms      = librosa.feature.rms(y=y)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        zcr      = librosa.feature.zero_crossing_rate(y=y)[0]
        mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        duracion = float(librosa.get_duration(y=y, sr=sr))

        # Aproximaciones de Essentia
        y_harm, y_perc = librosa.effects.hpss(y)
        perc_ratio = float(np.mean(np.abs(y_perc)) / (np.mean(np.abs(y_harm)) + 1e-6))
        if len(beat_frames) > 2:
            diffs = np.diff(beat_frames)
            beat_stab = 1 / (1 + np.std(diffs) / (np.mean(diffs) + 1e-6))
        else:
            beat_stab = 0.5
        bailabilidad = float(np.clip(perc_ratio * beat_stab * 2.5, 0.70, 1.60))

        rms_db   = librosa.amplitude_to_db(rms)
        comp_din = float(np.clip(np.percentile(rms_db, 5) - np.percentile(rms_db, 95), -28, -10))

        return {
            "duracion_total_s":              round(duracion, 2),
            "tempo_bpm":                     round(tempo, 1),
            "energia_rms_mean":              round(float(np.mean(rms)), 4),
            "energia_rms_std":               round(float(np.std(rms)), 4),
            "brillo_centroide_mean":         round(float(np.mean(centroid)), 2),
            "brillo_centroide_std":          round(float(np.std(centroid)), 2),
            "ancho_banda_hz":                round(float(np.mean(bw)), 2),
            "caida_espectral_rolloff":       round(float(np.mean(rolloff)), 2),
            "tasa_cruces_cero":              round(float(np.mean(zcr)), 4),
            "bailabilidad_essentia":         round(bailabilidad, 4),
            "complejidad_dinamica_essentia": round(comp_din, 4),
            "mfcc_1": round(float(mfccs[0]), 4), "mfcc_2": round(float(mfccs[1]), 4),
            "mfcc_3": round(float(mfccs[2]), 4), "mfcc_4": round(float(mfccs[3]), 4),
            "mfcc_5": round(float(mfccs[4]), 4),
        }, None
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════
# FUNCIONES DE ANÁLISIS DE OUTPUT (filtran por género y nivel)
# ══════════════════════════════════════════════════════════════
def territorios_recomendados(genero, nivel, paises_artista):
    """Territorios con mejor recibimiento esperado, filtrados por género y nivel."""
    sub = df[(df["genero"] == genero) & (df["nivel"] == nivel)]
    if len(sub) == 0:
        # Si no hay canciones de ese género/nivel, usar solo nivel
        sub = df[df["nivel"] == nivel]

    freq = pd.concat([
        sub["territorio_top_1"].value_counts(),
        sub[sub["territorio_top_2"] != "SIN_SEGUNDO"]["territorio_top_2"].value_counts(),
    ], axis=1).fillna(0).sum(axis=1)
    freq = freq.sort_values(ascending=False).head(4)
    total = freq.sum()

    resultado = []
    for pais, count in freq.items():
        if pais == "SIN_SEGUNDO": continue
        afinidad = "Alta" if count / total > 0.25 else "Media"
        es_actual = pais in paises_artista
        resultado.append({
            "pais": pais,
            "nombre": PAISES_DISPLAY.get(pais, pais),
            "flag":   PAISES_FLAGS.get(pais, ""),
            "afinidad": afinidad,
            "actual": es_actual,
        })
    return resultado[:4]


def temporada_optima(genero, nivel):
    """Mes/temporada óptima basada en patrones del dataset, filtrada por género y nivel."""
    sub = df[(df["genero"] == genero) & (df["nivel"] == nivel)]
    if len(sub) == 0:
        sub = df[df["nivel"] == nivel]

    counts = sub["release_month"].value_counts().sort_index()
    top = counts.nlargest(4).index.tolist()
    pct = round(counts.loc[top].sum() / counts.sum() * 100)

    meses = ["Ene","Feb","Mar","Abr","May","Jun",
             "Jul","Ago","Sep","Oct","Nov","Dic"]
    estaciones = {
        (12,1,2): "el invierno",
        (3,4,5):  "la primavera",
        (6,7,8):  "el verano",
        (9,10,11):"el otoño",
    }
    estacion_dominante = None
    for rango, nombre in estaciones.items():
        if sum(m in rango for m in top) >= 3:
            estacion_dominante = nombre
            break

    return {
        "top_meses": top,
        "top_nombres": [meses[m-1] for m in top],
        "pct": pct,
        "estacion": estacion_dominante,
    }


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom: 2rem;">
    <div class="brand">HitBeat</div>
    <div class="brand-sub">Predicción de alcance musical para música latina en español</div>
</div>
""", unsafe_allow_html=True)


tabs = st.tabs(["Análisis de canción", "Explorar mercado", "Importancia de variables", "Metodología"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("##### Análisis de alcance esperado para una nueva canción")
    st.caption("Sube el audio y completa el perfil del artista. El modelo proyecta el nivel de alcance, los territorios con mejor recibimiento esperado, y el momento óptimo de lanzamiento.")

    col_form, col_result = st.columns([1.05, 1], gap="large")

    # ── FORMULARIO ────────────────────────────────────────────
    with col_form:

        # 1. GÉNERO (CAMPO ANCLA PRINCIPAL)
        st.markdown('<p class="section-label">Género musical</p>', unsafe_allow_html=True)
        st.caption("El modelo está entrenado con tres géneros de música latina en español.")
        genero = st.radio(
            "Género",
            GENEROS,
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

        # 2. AUDIO
        st.markdown('<p class="section-label">Audio de la canción</p>', unsafe_allow_html=True)
        wav_file = st.file_uploader(
            "Archivo de audio (.wav o .mp3)",
            type=["wav", "mp3"],
            help="Las 16 características acústicas se extraen automáticamente con Librosa.",
            label_visibility="collapsed",
        )

        audio_features, audio_ok = {}, False
        if wav_file is not None:
            with st.spinner("Analizando audio..."):
                feats, err = extraer_features_audio(wav_file.read())
            if err:
                st.error(f"Error al procesar el audio: {err}")
            else:
                audio_features, audio_ok = feats, True
                with st.expander("Características acústicas extraídas", expanded=False):
                    c1, c2 = st.columns(2)
                    c1.metric("Duración",     f"{feats['duracion_total_s']:.0f} s")
                    c1.metric("Tempo",        f"{feats['tempo_bpm']:.0f} BPM")
                    c1.metric("Energía RMS",  f"{feats['energia_rms_mean']:.3f}")
                    c1.metric("Bailabilidad", f"{feats['bailabilidad_essentia']:.2f}")
                    c2.metric("Brillo espectral",     f"{feats['brillo_centroide_mean']:.0f} Hz")
                    c2.metric("Ancho de banda",       f"{feats['ancho_banda_hz']:.0f} Hz")
                    c2.metric("Spectral rolloff",     f"{feats['caida_espectral_rolloff']:.0f} Hz")
                    c2.metric("Complejidad dinámica", f"{feats['complejidad_dinamica_essentia']:.1f} dB")

        # 3. TRAYECTORIA (CAMPO ANCLA SECUNDARIO)
        st.markdown('<p class="section-label">Trayectoria del artista</p>', unsafe_allow_html=True)
        st.caption("Define el nivel de fama del artista. Condiciona los rangos válidos del resto del formulario.")

        # Tooltips con criterios por categoría (pasa el mouse sobre el icono ?)
        st.markdown(f"""
        <div class="trayectoria-tooltips">
            <span class="trayectoria-chip" title="{TRAYECTORIA_CRITERIOS['Emergente']}">
                Emergente <span class="help-icon">?</span>
            </span>
            <span class="trayectoria-chip" title="{TRAYECTORIA_CRITERIOS['Establecido']}">
                Establecido <span class="help-icon">?</span>
            </span>
            <span class="trayectoria-chip" title="{TRAYECTORIA_CRITERIOS['Consagrado']}">
                Consagrado <span class="help-icon">?</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

        trayectoria = st.radio(
            "Trayectoria",
            list(COMPAT.keys()),
            index=1,
            horizontal=True,
            label_visibility="collapsed",
        )
        compat = COMPAT[trayectoria]

        # 4. CANAL DE YOUTUBE
        st.markdown('<p class="section-label">Canal del artista en YouTube</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            subs_label = st.selectbox(
                "Suscriptores del canal",
                compat["subs"],
                help="Consultar directamente en el canal de YouTube del artista.",
            )
        with c2:
            antig_label = st.selectbox(
                "Antigüedad del canal",
                compat["antiguedad"],
            )

        # Distribución comercial (selector unificado)
        distribucion = st.radio(
            "¿Cómo se distribuye comercialmente la música del artista?",
            [
                "Distribución independiente (sin contrato con sello)",
                "Sello con distribución registrada",
                "Sello con canal VEVO oficial",
            ],
            index=1,
            help=(
                "Distribución independiente: el artista publica directamente sin sello formal.  •  "
                "Sello con distribución registrada: hay un sello (indie, regional o major) "
                "que reclama derechos del video en YouTube.  •  "
                "Canal VEVO oficial: el canal del artista termina en 'VEVO', solo aplica a artistas "
                "firmados con un major label (Universal, Sony, Warner)."
            ),
        )
        if distribucion == "Distribución independiente (sin contrato con sello)":
            is_vevo, is_licensed = False, False
        elif distribucion == "Sello con distribución registrada":
            is_vevo, is_licensed = False, True
        else:
            is_vevo, is_licensed = True, True

        # 5. POPULARIDAD LAST.FM
        st.markdown('<p class="section-label">Popularidad histórica en Last.fm</p>', unsafe_allow_html=True)
        st.caption("Consultar en last.fm/music/{nombre del artista}")
        c1, c2 = st.columns(2)
        with c1:
            listeners_label = st.selectbox(
                "Oyentes únicos",
                compat["listeners"],
                help="Personas distintas que escuchan al artista en Last.fm.",
            )
        with c2:
            listeners_val = LISTENERS_RANGOS[listeners_label]
            playcount_validos = [
                k for k, v in PLAYCOUNT_RANGOS.items()
                if v >= listeners_val * 4
            ]
            if not playcount_validos:
                playcount_validos = list(PLAYCOUNT_RANGOS.keys())[-2:]
            playcount_label = st.selectbox(
                "Reproducciones totales",
                playcount_validos,
                help="Historial acumulado del artista. Debe ser mayor que los oyentes únicos.",
            )

        # 6. PERFIL EDITORIAL
        st.markdown('<p class="section-label">Perfil editorial</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if is_vevo:
                sello = "Major (Universal / Sony / Warner)"
                st.selectbox(
                    "Sello discográfico", [sello], disabled=True,
                    help="VEVO requiere sello Major.",
                )
            elif not is_licensed:
                sello = "Independiente"
                st.selectbox(
                    "Sello discográfico", [sello], disabled=True,
                    help="La distribución independiente implica que no hay un sello con derechos registrados.",
                )
            else:
                opciones_sello = [s for s in compat["sello"]
                                  if s != "Independiente" or "Independiente" in compat["sello"]]
                sello = st.selectbox("Sello discográfico", opciones_sello)
        with c2:
            featuring_opt = st.radio(
                "Colaboraciones",
                ["Sin featuring", "1 colaborador", "2 o más"],
                horizontal=True,
            )
        has_featuring = featuring_opt != "Sin featuring"
        n_collabs = {"Sin featuring": 0, "1 colaborador": 1, "2 o más": 2}[featuring_opt]

        # 7. MES DE LANZAMIENTO
        st.markdown('<p class="section-label">Mes de lanzamiento planeado</p>', unsafe_allow_html=True)
        meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        release_idx = st.select_slider(
            "Mes",
            options=list(range(12)),
            value=4,
            format_func=lambda i: meses[i],
            label_visibility="collapsed",
        )
        release_month = release_idx + 1

        # 8. PRESENCIA TERRITORIAL
        st.markdown('<p class="section-label">Presencia territorial actual del artista</p>', unsafe_allow_html=True)
        st.caption(f"¿En qué países el artista ya tiene base de fans? Máximo {compat['max_territorios']} para un artista {trayectoria.lower()}.")

        paises_principales = ["MX", "US", "CO", "ES", "AR", "CL", "PE", "Otro"]
        cols_paises = st.columns(4)
        territorios_seleccionados = []
        for i, pais in enumerate(paises_principales):
            col = cols_paises[i % 4]
            label = PAISES_DISPLAY.get(pais, "Otro país")
            seleccionado = col.checkbox(label, value=(pais in ["MX", "US"]), key=f"pais_{pais}")
            if seleccionado:
                territorios_seleccionados.append(pais)

        if len(territorios_seleccionados) > compat["max_territorios"]:
            st.markdown(f"""
            <div class="warning-box">
                Para un artista {trayectoria.lower()} es poco frecuente tener presencia en más de
                {compat["max_territorios"]} territorios. El modelo lo procesará de todas formas,
                pero considera revisar la selección.
            </div>
            """, unsafe_allow_html=True)

        # 9. VALIDACIONES Y BOTÓN
        st.markdown("---")

        errores = []
        if not audio_ok:
            errores.append("Sube un archivo de audio para habilitar el análisis.")
        if len(territorios_seleccionados) == 0:
            errores.append("Selecciona al menos un territorio donde el artista tenga presencia.")

        for err in errores:
            st.markdown(f'<div class="error-box">{err}</div>', unsafe_allow_html=True)

        predecir = st.button(
            "Analizar canción",
            use_container_width=True,
            type="primary",
            disabled=len(errores) > 0,
        )

    # ── RESULTADO ─────────────────────────────────────────────
    with col_result:
        if predecir and audio_ok and len(territorios_seleccionados) > 0:

            subs_val      = SUBS_RANGOS[subs_label]
            listeners_val = LISTENERS_RANGOS[listeners_label]
            playcount_val = PLAYCOUNT_RANGOS[playcount_label]
            antig_val     = ANTIGUEDAD_RANGOS[antig_label]

            sello_map = {
                "Independiente": "Indie",
                "Sello regional": "Regional",
                "Major (Universal / Sony / Warner)": "Major",
            }

            territorio_1 = territorios_seleccionados[0] if territorios_seleccionados else "MX"
            territorio_2 = (territorios_seleccionados[1]
                            if len(territorios_seleccionados) >= 2 else "SIN_SEGUNDO")

            # Construir input con TODAS las features en el orden que espera el modelo
            input_data = pd.DataFrame([{
                "genero":                      genero,
                "yt_channel_subscribers_log":  float(np.log10(subs_val)),
                "channel_age_years":           antig_val,
                "is_vevo":                     int(is_vevo),
                "is_licensed_content":         int(is_licensed),
                "lastfm_artist_listeners":     listeners_val,
                "lastfm_artist_playcount":     playcount_val,
                "lastfm_plays_per_listener":   round(playcount_val / listeners_val, 2),
                "lastfm_top_tag_score":        65,
                "lastfm_top10_listeners_mean": int(listeners_val * 0.35),
                "lastfm_similar_match_mean":   0.55,
                "territorio_top_1":            territorio_1,
                "territorio_top_2":            territorio_2,
                "n_territorios_top":           len(territorios_seleccionados),
                "career_stage":                trayectoria,
                "label_type":                  sello_map[sello],
                "has_featuring":               int(has_featuring),
                "n_collaborators":             n_collabs,
                "release_month":               release_month,
                **audio_features,
            }])
            # Asegurar el orden exacto que espera el modelo
            input_data = input_data[FEATURES]

            nivel_pred = modelo.predict(input_data).flatten()[0]
            proba      = modelo.predict_proba(input_data)[0]
            clases     = list(modelo.classes_)
            prob_dict  = {c: p for c, p in zip(clases, proba)}

            color  = NIVEL_COLORS[nivel_pred]
            bg     = NIVEL_BG[nivel_pred]

            # PREDICCIÓN PRINCIPAL
            st.markdown(f"""
            <div class="result-card" style="text-align: center;">
                <p style="font-size: 0.72rem; color: #64748b; letter-spacing: 0.12em;
                          text-transform: uppercase; margin-bottom: 12px;">
                    Nivel de alcance proyectado
                </p>
                <div class="level-pill" style="background: {bg}; color: {color};
                                                border: 1px solid {color}33;">
                    {nivel_pred}
                </div>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 14px;">
                    {NIVEL_DESC[nivel_pred]}
                </p>
                <p style="color: #64748b; font-size: 0.75rem; margin-top: 8px;">
                    Género analizado: {genero}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # PROBABILIDADES
            with st.container(border=False):
                st.markdown('<p class="section-label" style="margin-top:0">Distribución de probabilidad</p>',
                             unsafe_allow_html=True)
                for nv in ["Alto", "Medio", "Bajo"]:
                    p = prob_dict.get(nv, 0)
                    es_top = (nv == nivel_pred)
                    col_n, col_b = st.columns([1.5, 5])
                    col_n.markdown(
                        f"<span style='color: {NIVEL_COLORS[nv] if es_top else '#94a3b8'};"
                        f"font-weight: {'500' if es_top else '400'};'>{nv}</span>",
                        unsafe_allow_html=True
                    )
                    col_b.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="flex: 1; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden;">
                            <div style="width: {p*100}%; height: 100%; background: {NIVEL_COLORS[nv]}; border-radius: 3px;"></div>
                        </div>
                        <span style="font-size: 0.85rem; color: {NIVEL_COLORS[nv] if es_top else '#94a3b8'};
                                     min-width: 38px; text-align: right;">
                            {p*100:.0f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            # TERRITORIOS RECOMENDADOS (filtrados por género)
            st.markdown('<p class="section-label">Territorios con mejor recibimiento esperado</p>',
                         unsafe_allow_html=True)
            st.caption(f"Basado en patrones de canciones {genero.lower()} con nivel {nivel_pred} del dataset.")

            recomendados = territorios_recomendados(genero, nivel_pred, territorios_seleccionados)
            for r in recomendados:
                badge_color = "#38bdf8" if r["afinidad"] == "Alta" else "#94a3b8"
                badge_bg    = "rgba(56,189,248,0.12)" if r["afinidad"] == "Alta" else "rgba(148,163,184,0.1)"
                actual_tag  = (' <span style="color:#22c55e; font-size:0.7rem; '
                               'background:rgba(34,197,94,0.12); padding:1px 7px; border-radius:99px; '
                               'margin-left:6px">presencia actual</span>'
                               if r["actual"] else "")
                st.markdown(f"""
                <div class="region-row">
                    <span style="font-size: 1.3rem;">{r["flag"]}</span>
                    <div style="flex: 1;">
                        <div style="color: #e2e8f0; font-size: 0.92rem;">{r["nombre"]}{actual_tag}</div>
                    </div>
                    <span style="background: {badge_bg}; color: {badge_color};
                                 font-size: 0.75rem; padding: 3px 10px; border-radius: 99px;">
                        Afinidad {r["afinidad"].lower()}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # TEMPORADA ÓPTIMA (filtrada por género)
            st.markdown('<p class="section-label">Momento de lanzamiento sugerido</p>',
                         unsafe_allow_html=True)
            temporada = temporada_optima(genero, nivel_pred)

            cols_meses = st.columns(12)
            for i, mes_nombre in enumerate(meses):
                mes_num = i + 1
                es_top = mes_num in temporada["top_meses"]
                es_seleccionado = mes_num == release_month
                bg_cell = ("#f59e0b" if es_seleccionado else
                            "rgba(245,158,11,0.18)" if es_top else "#1e293b")
                color_cell = ("#0f172a" if es_seleccionado else
                               "#f59e0b" if es_top else "#64748b")
                cols_meses[i].markdown(f"""
                <div class="month-cell" style="background: {bg_cell}; color: {color_cell};">
                    {mes_nombre}
                </div>
                """, unsafe_allow_html=True)

            estacion_txt = (f"durante {temporada['estacion']}"
                             if temporada["estacion"]
                             else f"en los meses {', '.join(temporada['top_nombres'][:3])}")
            mes_actual = meses[release_month - 1]
            alineado = release_month in temporada["top_meses"]

            mensaje = (f"Las canciones de {genero.lower()} nivel {nivel_pred} suelen concentrar su mejor "
                       f"recibimiento {estacion_txt}, donde se ubica el {temporada['pct']}% de los "
                       f"casos con mejor performance. ")
            if alineado:
                mensaje += f"Tu lanzamiento en {mes_actual} está alineado con esa ventana."
            else:
                mensaje += (f"Tu lanzamiento planeado en {mes_actual} queda fuera de la temporada "
                            f"de mayor tracción para esta clase.")

            st.markdown(f"""
            <div style="background: rgba(245,158,11,0.06); border-left: 3px solid #f59e0b;
                        padding: 12px 14px; border-radius: 4px; margin-top: 12px;
                        font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
                {mensaje}
            </div>
            """, unsafe_allow_html=True)

            # CANCIONES SIMILARES (filtradas por género, EXCLUYE sintéticas en el ranking)
            with st.expander("Canciones de referencia con nivel similar", expanded=False):
                # Solo reales del mismo género y nivel
                mask = (df["nivel"] == nivel_pred) & \
                       (df["genero"] == genero) & \
                       (df["origen"] == "real")

                # Fallback 1: reales de cualquier género con el mismo nivel
                if mask.sum() < 3:
                    mask = (df["nivel"] == nivel_pred) & (df["origen"] == "real")

                # Fallback 2: si por alguna razón no hay reales, usar cualquier registro real
                if mask.sum() < 3:
                    mask = df["origen"] == "real"

                sim = df[mask][["cancion", "artista", "genero", "n_territorios_top", "career_stage"]].sample(
                    min(5, mask.sum()), random_state=42
                ).rename(columns={
                    "cancion": "Canción", "artista": "Artista", "genero": "Género",
                    "n_territorios_top": "Territorios", "career_stage": "Trayectoria",
                })
                st.dataframe(sim, hide_index=True, use_container_width=True)

        else:
            # Estado inicial
            st.markdown("""
            <div class="result-card">
                <p class="section-label" style="margin-top:0">Cómo funciona</p>
                <ol style="color: #cbd5e1; line-height: 1.9; font-size: 0.9rem; padding-left: 1.2rem;">
                    <li>Selecciona el género de la canción.</li>
                    <li>Completa el perfil del artista.</li>
                    <li>Sube el archivo de audio (.wav o .mp3). Las características acústicas
                        se extraen automáticamente.</li>
                    <li>Obtén el análisis: nivel proyectado, territorios con mejor recibimiento,
                        y temporada óptima de lanzamiento.</li>
                </ol>
                <p style="color: #64748b; font-size: 0.8rem; font-style: italic;
                          margin-top: 1rem; border-top: 1px solid #1e293b; padding-top: 12px;">
                    Modelo CatBoost unificado entrenado con 999 canciones de tres géneros latinos.
                    Accuracy validada por 5-Fold CV: 76.78% (baseline azar = 33%).
                </p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — EXPLORAR MERCADO
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("##### Cómo se comporta el mercado de la música latina")
    st.caption("Cuatro patrones de mercado descubiertos en el dataset de 999 canciones que ilustran qué decisiones realmente mueven el alcance de una canción.")

    st.divider()

    # ── KPIs NARRATIVOS DE MERCADO ────────────────────────────
    # Calcular los 4 KPIs sobre el dataset completo
    emerg_pct  = (df[df['career_stage'] == 'Emergente']['nivel']  == 'Alto').mean() * 100
    consag_pct = (df[df['career_stage'] == 'Consagrado']['nivel'] == 'Alto').mean() * 100
    multiplicador = consag_pct / emerg_pct if emerg_pct > 0 else 0

    regional_hits = (df[df['label_type'] == 'Regional']['nivel'] == 'Alto').sum()
    regional_total = (df['label_type'] == 'Regional').sum()

    reg_sin = (df[(df['genero'] == 'Reguetón') & (~df['has_featuring'])]['nivel'] == 'Alto').mean() * 100
    reg_con = (df[(df['genero'] == 'Reguetón') & (df['has_featuring'])]['nivel'] == 'Alto').mean() * 100
    boost_feat_reg = reg_con - reg_sin

    # Umbral mágico: primer n_territorios donde supera 50%
    umbral_terr = None
    for n_t in range(1, 8):
        sub = df[df['n_territorios_top'] == n_t]
        if len(sub) >= 10 and (sub['nivel'] == 'Alto').mean() > 0.5:
            umbral_terr = n_t
            break

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div style="background: rgba(56,189,248,0.08); border-left: 3px solid #38bdf8;
                    padding: 14px 16px; border-radius: 6px; height: 110px;">
            <div style="font-size: 0.7rem; color: #64748b; letter-spacing: 0.1em;
                        text-transform: uppercase; margin-bottom: 6px;">
                Salto por trayectoria
            </div>
            <div style="font-size: 1.6rem; font-weight: 600; color: #38bdf8; line-height: 1.1;">
                {multiplicador:.1f}×
            </div>
            <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 4px;">
                más probable ser hit como Consagrado vs Emergente
            </div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div style="background: rgba(168,85,247,0.08); border-left: 3px solid #a855f7;
                    padding: 14px 16px; border-radius: 6px; height: 110px;">
            <div style="font-size: 0.7rem; color: #64748b; letter-spacing: 0.1em;
                        text-transform: uppercase; margin-bottom: 6px;">
                Umbral internacional
            </div>
            <div style="font-size: 1.6rem; font-weight: 600; color: #a855f7; line-height: 1.1;">
                {umbral_terr}+ países
            </div>
            <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 4px;">
                donde la probabilidad de hit supera el 50%
            </div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div style="background: rgba(239,68,68,0.08); border-left: 3px solid #ef4444;
                    padding: 14px 16px; border-radius: 6px; height: 110px;">
            <div style="font-size: 0.7rem; color: #64748b; letter-spacing: 0.1em;
                        text-transform: uppercase; margin-bottom: 6px;">
                Techo de sellos Regional
            </div>
            <div style="font-size: 1.6rem; font-weight: 600; color: #ef4444; line-height: 1.1;">
                0 de {regional_total}
            </div>
            <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 4px;">
                canciones Regional llegan a nivel Alto
            </div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div style="background: rgba(249,115,22,0.08); border-left: 3px solid #f97316;
                    padding: 14px 16px; border-radius: 6px; height: 110px;">
            <div style="font-size: 0.7rem; color: #64748b; letter-spacing: 0.1em;
                        text-transform: uppercase; margin-bottom: 6px;">
                Featuring en Reguetón
            </div>
            <div style="font-size: 1.6rem; font-weight: 600; color: #f97316; line-height: 1.1;">
                +{boost_feat_reg:.0f} pp
            </div>
            <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 4px;">
                de probabilidad extra de hit con colaboración
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── HISTORIA 1: TRAYECTORIA × GÉNERO ──────────────────────
    st.markdown('<p class="section-label">1. La trayectoria del artista es el predictor más fuerte</p>',
                 unsafe_allow_html=True)

    TRAYECTORIA_ORDER = ["Emergente", "Establecido", "Consagrado"]
    data_1 = []
    for g in GENEROS:
        for t in TRAYECTORIA_ORDER:
            sub_t = df[(df['genero'] == g) & (df['career_stage'] == t)]
            pct = (sub_t['nivel'] == 'Alto').mean() * 100 if len(sub_t) > 0 else 0
            data_1.append({'Trayectoria': t, 'Género': g, 'pct_alto': round(pct)})
    df_1 = pd.DataFrame(data_1)

    fig1 = px.bar(df_1, x='Trayectoria', y='pct_alto', color='Género',
                   barmode='group', text='pct_alto',
                   category_orders={'Trayectoria': TRAYECTORIA_ORDER, 'Género': GENEROS},
                   color_discrete_map={"Balada": "#38bdf8", "Reguetón": "#a855f7",
                                        "Regional Mexicano": "#f97316"})
    fig1.update_traces(texttemplate='%{text}%', textposition='outside', textfont_size=11)
    fig1.add_hline(y=33.3, line_dash="dash", line_color="gray", line_width=1,
                    annotation_text="Baseline azar (33%)", annotation_position="top right",
                    annotation_font_size=10, annotation_font_color="#94a3b8")
    fig1.update_layout(
        yaxis_title="% canciones nivel Alto", xaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                     title_text=""),
        yaxis=dict(range=[0, 80], gridcolor="rgba(148,163,184,0.15)"),
        xaxis=dict(showgrid=False),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""
    <div style="background: rgba(56,189,248,0.06); border-left: 3px solid #38bdf8;
                padding: 10px 14px; border-radius: 4px; font-size: 0.85rem;
                color: #cbd5e1; margin-bottom: 2rem;">
        Un artista Consagrado tiene entre 49% y 65% de probabilidad de hit, mientras que
        un Emergente apenas roza el 5–10%. <strong>El nombre del artista define el techo
        de la canción antes de que esta se publique.</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── HISTORIA 2: TERRITORIOS ───────────────────────────────
    st.markdown('<p class="section-label">2. El alcance internacional separa hits de no-hits</p>',
                 unsafe_allow_html=True)

    data_2 = []
    for n_t in range(1, 8):
        sub = df[df['n_territorios_top'] == n_t]
        if len(sub) >= 10:
            pct = (sub['nivel'] == 'Alto').mean() * 100
            data_2.append({'n_terr': n_t, 'pct_alto': round(pct), 'n_obs': len(sub)})
    df_2 = pd.DataFrame(data_2)
    df_2['color'] = df_2['pct_alto'].apply(
        lambda v: '#ef4444' if v > 60 else '#f59e0b' if v > 30 else '#94a3b8'
    )

    fig2 = go.Figure(go.Bar(
        x=df_2['n_terr'], y=df_2['pct_alto'],
        marker_color=df_2['color'],
        text=[f"{v}%" for v in df_2['pct_alto']],
        textposition='outside', textfont_size=11,
        customdata=df_2['n_obs'],
        hovertemplate="<b>%{x} territorios</b><br>%{y}% nivel Alto<br>n=%{customdata}<extra></extra>",
    ))
    fig2.add_hline(y=33.3, line_dash="dash", line_color="gray", line_width=1,
                    annotation_text="Baseline azar (33%)", annotation_position="top right",
                    annotation_font_size=10, annotation_font_color="#94a3b8")
    fig2.update_layout(
        xaxis_title="Número de territorios donde el artista tiene tracción",
        yaxis_title="% canciones nivel Alto",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=380,
        yaxis=dict(range=[0, 115], gridcolor="rgba(148,163,184,0.15)"),
        xaxis=dict(showgrid=False, tickmode='linear'),
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    <div style="background: rgba(239,68,68,0.06); border-left: 3px solid #ef4444;
                padding: 10px 14px; border-radius: 4px; font-size: 0.85rem;
                color: #cbd5e1; margin-bottom: 2rem;">
        Pasar de 3 a 4 territorios duplica la probabilidad de hit (25% → 51%). Con 5
        territorios la probabilidad es del 73%, y con 6 o más es prácticamente garantizada.
        <strong>Expandir la audiencia internacional es la palanca más rentable que un
        sello puede activar.</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── HISTORIA 3: SELLO ─────────────────────────────────────
    st.markdown('<p class="section-label">3. Los sellos Regional no producen hits</p>',
                 unsafe_allow_html=True)

    sello_order = ['Indie', 'Regional', 'Major']
    data_3 = []
    for sello_iter in sello_order:
        sub = df[df['label_type'] == sello_iter]
        for nivel in ['Bajo', 'Medio', 'Alto']:
            count = (sub['nivel'] == nivel).sum()
            data_3.append({'Sello': sello_iter, 'Nivel': nivel, 'Canciones': count})
    df_3 = pd.DataFrame(data_3)

    fig3 = px.bar(df_3, x='Sello', y='Canciones', color='Nivel',
                   category_orders={'Sello': sello_order, 'Nivel': ['Bajo', 'Medio', 'Alto']},
                   color_discrete_map=NIVEL_COLORS, text='Canciones')
    fig3.update_traces(textposition='inside', textfont_size=11, textfont_color='white')
    fig3.update_layout(
        xaxis_title="", yaxis_title="Canciones",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                     title_text=""),
        yaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        xaxis=dict(showgrid=False),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
    <div style="background: rgba(34,197,94,0.06); border-left: 3px solid #22c55e;
                padding: 10px 14px; border-radius: 4px; font-size: 0.85rem;
                color: #cbd5e1; margin-bottom: 2rem;">
        Los sellos Major concentran más de la mitad de los hits del dataset. Los Indies
        pueden competir en géneros de audiencia leal, pero los sellos Regional sirven para
        distribución local: <strong>no logran empujar canciones al nivel Alto en ninguno
        de los tres géneros analizados.</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── HISTORIA 4: FEATURING ─────────────────────────────────
    st.markdown('<p class="section-label">4. El featuring no es estrategia universal</p>',
                 unsafe_allow_html=True)

    data_4 = []
    for g in GENEROS:
        sub = df[df['genero'] == g]
        sin_f = (sub[~sub['has_featuring']]['nivel'] == 'Alto').mean() * 100
        con_f = (sub[sub['has_featuring']]['nivel'] == 'Alto').mean() * 100
        data_4.append({'Género': g, 'Tipo': 'Sin featuring', 'pct_alto': round(sin_f)})
        data_4.append({'Género': g, 'Tipo': 'Con featuring', 'pct_alto': round(con_f)})
    df_4 = pd.DataFrame(data_4)

    fig4 = px.bar(df_4, x='Género', y='pct_alto', color='Tipo', barmode='group',
                   text='pct_alto',
                   category_orders={'Género': GENEROS,
                                     'Tipo': ['Sin featuring', 'Con featuring']},
                   color_discrete_map={'Sin featuring': '#94a3b8', 'Con featuring': '#a855f7'})
    fig4.update_traces(texttemplate='%{text}%', textposition='outside', textfont_size=11)
    fig4.add_hline(y=33.3, line_dash="dash", line_color="gray", line_width=1,
                    annotation_text="Baseline azar (33%)", annotation_position="top right",
                    annotation_font_size=10, annotation_font_color="#94a3b8")
    fig4.update_layout(
        yaxis_title="% canciones nivel Alto", xaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                     title_text=""),
        yaxis=dict(range=[0, 55], gridcolor="rgba(148,163,184,0.15)"),
        xaxis=dict(showgrid=False),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("""
    <div style="background: rgba(168,85,247,0.06); border-left: 3px solid #a855f7;
                padding: 10px 14px; border-radius: 4px; font-size: 0.85rem;
                color: #cbd5e1; margin-bottom: 1rem;">
        En Reguetón, colaborar sube la probabilidad de hit del 26% al 40% (+14 puntos).
        En Balada el efecto se invierte: el featuring está ligeramente correlacionado
        con peor performance porque el género vive de la conexión emocional con un solista.
        <strong>La decisión de hacer featuring debe pensarse según el género, no como
        receta universal.</strong>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — IMPORTANCIA DE VARIABLES
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("##### Qué variables pesan más en la predicción del modelo")
    st.caption("Análisis de la contribución relativa de cada variable a la capacidad predictiva del modelo CatBoost.")

    # ── EXPLICACIÓN DEL MÉTODO ────────────────────────────────
    st.markdown("""
    <div style="background: rgba(56,189,248,0.06); border-left: 3px solid #38bdf8;
                padding: 12px 16px; border-radius: 4px; font-size: 0.85rem;
                color: #cbd5e1; line-height: 1.55; margin: 1rem 0 1.5rem 0;">
        <strong style="color: #38bdf8;">Cómo leer estos porcentajes</strong><br>
        Los valores mostrados provienen del método <em>PredictionValuesChange</em> de CatBoost,
        que mide cuánto cambia en promedio la predicción del modelo cuando se altera el valor
        de cada variable. La suma de todas las contribuciones es <strong>100%</strong>: cada
        porcentaje indica qué proporción del poder predictivo total del modelo se atribuye a esa
        variable. Por ejemplo, si <code>n_territorios_top</code> aparece con 18%, quiere decir
        que el 18% de las decisiones de clasificación del modelo dependen, directa o
        indirectamente, de esa variable.
    </div>
    """, unsafe_allow_html=True)

    # ── IMPORTANCIA AGREGADA POR BLOQUE TEMÁTICO ──────────────
    st.markdown('<p class="section-label">Visión por bloque temático</p>',
                 unsafe_allow_html=True)
    st.caption("Suma de la importancia de las variables que componen cada bloque del dataset.")

    importances_raw = pd.DataFrame({
        "feature":    FEATURES,
        "importance": modelo.get_feature_importance(),
    })
    importances_raw["bloque"] = importances_raw["feature"].apply(lambda f:
        "Género"        if f == "genero"
        else "Territorial" if f in ["territorio_top_1","territorio_top_2","n_territorios_top"]
        else "Last.fm"     if f.startswith("lastfm")
        else "YouTube"     if f in ["yt_channel_subscribers_log","channel_age_years","is_vevo","is_licensed_content"]
        else "Editorial"   if f in ["career_stage","label_type","has_featuring","n_collaborators","release_month"]
        else "Audio"
    )

    por_bloque = (importances_raw.groupby("bloque")["importance"]
                   .sum()
                   .sort_values(ascending=True)
                   .reset_index())

    bloque_colors = {"Género":"#38bdf8","Territorial":"#f97316","Last.fm":"#fbbf24",
                      "YouTube":"#4ade80","Editorial":"#a855f7","Audio":"#f87171"}

    fig_bloque = px.bar(por_bloque, x="importance", y="bloque",
                         color="bloque", color_discrete_map=bloque_colors,
                         orientation="h", text="importance")
    fig_bloque.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              textfont_color="#e2e8f0", textfont_size=12)
    fig_bloque.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis_title="Contribución agregada del bloque a la capacidad predictiva (%)",
        yaxis_title="",
        showlegend=False,
        margin=dict(t=20, b=40),
        xaxis=dict(range=[0, max(por_bloque["importance"]) * 1.15],
                    gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_bloque, use_container_width=True)

    # Calcular el top bloque para la narrativa
    top_bloque = por_bloque.iloc[-1]["bloque"]
    top_bloque_pct = por_bloque.iloc[-1]["importance"]
    audio_pct = por_bloque[por_bloque["bloque"] == "Audio"]["importance"].values[0]

    st.markdown(f"""
    <div style="background: rgba(249,115,22,0.06); border-left: 3px solid #f97316;
                padding: 12px 14px; border-radius: 4px; font-size: 0.85rem;
                color: #cbd5e1; line-height: 1.55; margin-bottom: 2.5rem;">
        <strong>El bloque {top_bloque} domina con {top_bloque_pct:.0f}%</strong> de la
        capacidad predictiva total del modelo. La configuración geográfica del artista en
        países hispanohablantes pesa más que las cualidades intrínsecas de la canción,
        un hallazgo que matiza la asunción frecuente de la literatura clásica de
        <em>Hit Song Science</em> de que las propiedades sonoras serían el factor más
        determinante del éxito musical.
    </div>
    """, unsafe_allow_html=True)

    # ── IMPORTANCIA POR VARIABLE INDIVIDUAL ───────────────────
    st.markdown('<p class="section-label">Visión por variable individual</p>',
                 unsafe_allow_html=True)
    st.caption("Las 35 variables ordenadas por su contribución individual.")

    importances = importances_raw.sort_values("importance", ascending=False)

    fig = px.bar(importances, x="importance", y="feature",
                 color="bloque", color_discrete_map=bloque_colors,
                 orientation="h", text="importance")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                       textfont_color="#94a3b8")
    fig.update_layout(
        height=900, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        yaxis=dict(categoryorder="total ascending", showgrid=False),
        xaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        xaxis_title="Contribución a la capacidad predictiva del modelo (%)",
        yaxis_title="",
        legend_title_text="Bloque", title_font_size=14,
    )
    st.plotly_chart(fig, use_container_width=True)

    n_terr_imp = importances[importances["feature"] == "n_territorios_top"]["importance"].values[0]
    genero_imp = importances[importances["feature"] == "genero"]["importance"].values[0]

    st.markdown(f"""
    <div style="background: rgba(56,189,248,0.06); border-left: 3px solid #38bdf8;
                padding: 14px 18px; border-radius: 4px; font-size: 0.88rem;
                color: #cbd5e1; line-height: 1.6; margin-top: 1rem;">
        <strong style="color: #38bdf8;">Lectura individual de las variables:</strong><br>
        La variable <code>n_territorios_top</code> domina individualmente con {n_terr_imp:.1f}%
        de contribución al modelo. El alcance geográfico del artista, medido como número de
        países hispanohablantes con tracción real, es el predictor más fuerte del éxito en los
        tres géneros.<br><br>
        La variable <code>genero</code> aporta apenas {genero_imp:.1f}%, lo cual indica que el
        modelo ya distingue entre géneros implícitamente a través de las features acústicas,
        territoriales y editoriales. La etiqueta explícita refina la predicción pero no es
        indispensable, lo cual valida la riqueza del feature space.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — METODOLOGÍA
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("##### Metodología del proyecto HitBeat")
    st.caption("Trabajo Terminal · 2026-A134 · ESCOM-IPN")

    # ── DEFINICIÓN DEL PROBLEMA ───────────────────────────────
    st.markdown('<p class="section-label">Definición del problema</p>',
                 unsafe_allow_html=True)
    st.markdown("""
    Clasificación multiclase del nivel de alcance esperado de una canción de música latina
    en español antes de su publicación, mediante el análisis integrado de variables acústicas
    y variables contextuales del artista obtenidas de YouTube y Last.fm. La salida del modelo
    es una categoría discreta (Bajo, Medio o Alto) acompañada de su distribución de probabilidad.

    | Nivel | Rango de vistas en YouTube |
    |-------|----------------------------|
    | Alto  | más de 100 millones        |
    | Medio | entre 5 y 100 millones     |
    | Bajo  | menos de 5 millones        |
    """)

    # ── CORPUS ────────────────────────────────────────────────
    st.markdown('<p class="section-label">Corpus HitBeat</p>',
                 unsafe_allow_html=True)
    st.markdown("""
    Conjunto de **999 canciones reales** distribuidas equilibradamente entre los tres géneros
    y tres niveles de alcance. Cada canción cumple tres condiciones: pertenece a un lanzamiento
    original (sin versiones ni reinterpretaciones), cuenta con información completa y verificable
    en las tres fuentes (audio WAV, YouTube Data API v3, API de Last.fm), y su desempeño digital
    es atribuible a las propiedades de la propia canción.

    | Género | Bajo | Medio | Alto | Total |
    |--------|------|-------|------|-------|
    | Reguetón | 111 | 111 | 111 | 333 |
    | Balada | 111 | 111 | 111 | 333 |
    | Regional Mexicano | 111 | 111 | 111 | 333 |
    | **Total** | **333** | **333** | **333** | **999** |

    **Período cubierto:** 2008–2024.
    """)

    # ── VARIABLES ─────────────────────────────────────────────
    st.markdown('<p class="section-label">Conjunto de variables</p>',
                 unsafe_allow_html=True)
    st.markdown("""
    El dataset final integra **48 variables** organizadas en ocho bloques temáticos. Para el
    entrenamiento se emplean **34 variables predictoras**; las 14 restantes corresponden a
    identificadores, variables de control y al bloque objetivo, excluidas del entrenamiento
    para garantizar la integridad metodológica.

    | Bloque | Variables predictoras | Ejemplos |
    |--------|----------------------|----------|
    | Acústicas | 15 | Tempo, energía RMS, brillo espectral, MFCCs, danceability |
    | YouTube | 6 | Suscriptores del canal, antigüedad, VEVO, contenido licenciado |
    | Last.fm | 6 | Oyentes únicos, reproducciones acumuladas, intensidad por oyente |
    | Territoriales | 3 | Número de países con tracción y top territoriales |
    | Editoriales | 4 | Tipo de sello, etapa de carrera, featuring |
    | **Total predictoras** | **34** | |
    """)

    # ── PIPELINE Y EVALUACIÓN ─────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-label">Pipeline de datos</p>',
                     unsafe_allow_html=True)
        st.markdown("""
        Arquitectura **Medallion** en Databricks con tres capas:

        - **Bronce:** ingesta del dataset crudo en formato Excel
        - **Plata:** selección de las 34 variables predictoras y limpieza
        - **Oro:** entrenamiento del modelo, predicciones y exportación de artefactos

        Las tablas están versionadas en **Delta Lake** dentro de Unity Catalog,
        y el modelo se registra en **MLflow** para trazabilidad de versiones.
        El pipeline se ejecuta como Job orquestado con disparador por llegada
        de archivo al volumen de entrada.
        """)
    with c2:
        st.markdown('<p class="section-label">Protocolo de evaluación</p>',
                     unsafe_allow_html=True)
        st.markdown("""
        **5-fold stratified cross-validation** con doble estratificación por
        género y por nivel de alcance. Este diseño garantiza que las métricas
        reflejen el desempeño real sobre datos no vistos.

        Las predicciones para la matriz de confusión y curvas ROC se generan
        con estimación **out-of-fold**: cada canción es clasificada por un
        modelo que no la vio durante su entrenamiento.

        Las métricas principales son **accuracy** y **F1 macro**, este último
        especialmente apropiado para un dataset balanceado al ponderar por
        igual las tres clases.
        """)

    st.divider()

    # ── BENCHMARK DE 6 MODELOS ────────────────────────────────
    st.markdown('<p class="section-label">Benchmark de modelos</p>',
                 unsafe_allow_html=True)
    st.markdown("""
    Se entrenaron y compararon **seis algoritmos** pertenecientes a cinco familias técnicas
    distintas. La selección de familias responde a un criterio metodológico explícito:
    evidenciar qué aproximación se ajusta mejor a las propiedades del dataset, sin asumir
    de antemano el algoritmo ganador.
    """)

    benchmark = pd.DataFrame([
        {"Modelo": "CatBoost",            "Familia": "Gradient Boosting",  "Accuracy CV": "0.7678 ± 0.0100", "F1 Macro CV": "0.7658", "Δ vs CatBoost": "—"},
        {"Modelo": "XGBoost",             "Familia": "Gradient Boosting",  "Accuracy CV": "0.7568 ± 0.0164", "F1 Macro CV": "0.7548", "Δ vs CatBoost": "+1.1 pp"},
        {"Modelo": "SVM",                 "Familia": "Kernel",             "Accuracy CV": "0.7398 ± 0.0100", "F1 Macro CV": "0.7377", "Δ vs CatBoost": "+2.8 pp"},
        {"Modelo": "Random Forest",       "Familia": "Bagging",            "Accuracy CV": "0.7190",          "F1 Macro CV": "0.7110", "Δ vs CatBoost": "+4.9 pp"},
        {"Modelo": "Regresión Logística", "Familia": "Lineal",             "Accuracy CV": "0.7107",          "F1 Macro CV": "0.7111", "Δ vs CatBoost": "+5.7 pp"},
        {"Modelo": "K Vecinos Cercanos",  "Familia": "Basado en distancia","Accuracy CV": "0.6747",          "F1 Macro CV": "0.6668", "Δ vs CatBoost": "+9.3 pp"},
    ])
    st.dataframe(benchmark, hide_index=True, use_container_width=True)

    st.markdown("""
    <div style="background: rgba(168,85,247,0.06); border-left: 3px solid #a855f7;
                padding: 12px 14px; border-radius: 4px; font-size: 0.85rem;
                color: #cbd5e1; line-height: 1.55; margin: 0.5rem 0 1.5rem 0;">
        Los seis modelos superan ampliamente el baseline aleatorio (33.3%). Incluso el
        menor —K Vecinos Cercanos— duplica el azar, lo que confirma que el conjunto de
        variables contiene información predictiva real. <strong>CatBoost encabeza el
        ranking en ambas métricas con la menor dispersión entre folds (±0.0100)</strong>,
        combinando ventaja en desempeño, estabilidad entre iteraciones y manejo nativo
        de variables categóricas.
    </div>
    """, unsafe_allow_html=True)

    # ── DESEMPEÑO POR GÉNERO ──────────────────────────────────
    st.markdown('<p class="section-label">Desempeño de CatBoost por género</p>',
                 unsafe_allow_html=True)
    st.markdown("""
    El análisis del accuracy descompuesto por género revela un patrón consistente que se
    reproduce en los seis modelos del benchmark, lo que indica que constituye una propiedad
    estructural del problema y no una limitación particular de ningún algoritmo.

    | Género | Accuracy | Posición | Fundamento estructural |
    |--------|----------|----------|------------------------|
    | Regional Mexicano | 87.70% | 1° (más fácil) | Audiencias territorialmente acotadas; las variables territoriales discriminan con alta claridad |
    | Reguetón | 77.19% | 2° (intermedio) | Circulación mixta entre regional y global; señal territorial moderada |
    | Balada | 65.47% | 3° (más difícil) | Difusión transnacional sin anclaje territorial definido |
    """)

    st.divider()

    # ── HALLAZGOS PRINCIPALES ─────────────────────────────────
    st.markdown('<p class="section-label">Hallazgos principales del proyecto</p>',
                 unsafe_allow_html=True)
    st.markdown("""
    Cuatro hallazgos transversales que se sostienen en evidencia que se repite
    consistentemente a través de los seis modelos del benchmark.
    """)

    h1, h2 = st.columns(2)
    with h1:
        st.markdown("""
        <div style="background: rgba(249,115,22,0.06); border-left: 3px solid #f97316;
                    padding: 14px 16px; border-radius: 6px; margin-bottom: 12px;
                    font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
            <strong style="color: #f97316;">I. El contexto territorial domina</strong><br>
            El bloque territorial concentra el 42% de la capacidad predictiva,
            superando individualmente a todos los demás bloques e incluso a la suma
            del bloque acústico (26%). Para música latina en español, la configuración
            geográfica pesa más que las cualidades intrínsecas de la pieza.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background: rgba(239,68,68,0.06); border-left: 3px solid #ef4444;
                    padding: 14px 16px; border-radius: 6px;
                    font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
            <strong style="color: #ef4444;">III. Clase Medio: estructuralmente difícil</strong><br>
            La clase Medio cubre un orden y medio de magnitud (5M a 100M de vistas) y reúne
            tanto canciones con éxito regional consolidado como producciones con potencial
            de escalar. En los seis modelos, concentra la mayor proporción de errores,
            distribuidos casi simétricamente hacia las clases adyacentes.
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div style="background: rgba(245,158,11,0.06); border-left: 3px solid #f59e0b;
                    padding: 14px 16px; border-radius: 6px; margin-bottom: 12px;
                    font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
            <strong style="color: #f59e0b;">II. La dificultad es estructural al género</strong><br>
            El orden de dificultad (Regional 87.7% → Reguetón 77.2% → Balada 65.5%) se
            mantiene en familias técnicas tan distintas como gradient boosting, kernel y
            distancia. Esto confirma que la diferencia es estructural al género, no una
            deficiencia algorítmica.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background: rgba(56,189,248,0.06); border-left: 3px solid #38bdf8;
                    padding: 14px 16px; border-radius: 6px;
                    font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
            <strong style="color: #38bdf8;">IV. Audio y contexto son complementarios</strong><br>
            La matriz de correlaciones confirma que las correlaciones entre variables
            acústicas y variables contextuales son cercanas a cero. Cada dimensión aporta
            información distinta, lo que valida cuantitativamente el diseño multimodal
            del dataset.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── COMPARACIÓN CON LA LITERATURA ─────────────────────────
    st.markdown('<p class="section-label">Posicionamiento respecto a la literatura</p>',
                 unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(34,197,94,0.06); border-left: 3px solid #22c55e;
                padding: 14px 16px; border-radius: 6px;
                font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
        El modelo HitBeat clasifica con un <strong>accuracy de 76.78%</strong> sobre 999
        canciones reales, superando en más de 43 puntos porcentuales al baseline aleatorio.
        Este resultado se sitúa en el <strong>extremo superior del rango reportado por la
        literatura comparable (65–79%)</strong> en el campo de <em>Hit Song Science</em>.
        El aporte diferencial del proyecto se sostiene en tres planos: metodológico
        (operar exclusivamente con información pre-lanzamiento), empírico (el hallazgo
        territorial), y de implementación (cerrar el ciclo entre investigación y
        herramienta utilizable).
    </div>
    """, unsafe_allow_html=True)

    # ── LIMITACIONES Y LÍNEAS FUTURAS ─────────────────────────
    st.markdown('<p class="section-label">Limitaciones reconocidas y líneas de trabajo futuro</p>',
                 unsafe_allow_html=True)

    l1, l2 = st.columns(2)
    with l1:
        st.markdown("""
        **Limitaciones del trabajo actual**

        - La clase Medio es estructuralmente más ambigua por la amplitud
          del rango (5M a 100M de vistas)
        - El modelo no captura efectos virales no orgánicos (sincronizaciones
          en cine/TV, virales en TikTok posteriores al lanzamiento)
        - El alcance proyectado es una clasificación cualitativa de techo
          esperado, no una predicción puntual con ventana temporal específica
        - El corpus se limita a tres géneros y al período 2008–2024
        """)
    with l2:
        st.markdown("""
        **Líneas de trabajo futuro**

        - **Ampliación del corpus** a géneros adicionales como salsa,
          bachata y pop latino
        - **Refinamiento del modelado:** optimización sistemática de
          hiperparámetros y modelos especializados por género
        - **Explicabilidad individual** mediante valores SHAP por predicción
        - **Integración de fuentes adicionales:** Spotify, redes sociales
          (TikTok, Instagram) y análisis de contenido lírico
        """)

    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 1rem 0;">
        HitBeat · Trabajo Terminal II · 2026-A134 · ESCOM-IPN · Mayo 2026<br>
        Databricks (Delta Lake · MLflow Unity Catalog) → Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)
