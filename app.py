import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    ruta = os.path.join(os.path.dirname(__file__), "data", "modelo_balada.pkl")
    with open(ruta, "rb") as f:
        return pickle.load(f)

df     = cargar_datos()
modelo = cargar_modelo()


# ══════════════════════════════════════════════════════════════
# CONSTANTES Y REGLAS DE CONSISTENCIA
# ══════════════════════════════════════════════════════════════
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

# Reglas de compatibilidad por trayectoria.
# Rangos calibrados con los percentiles reales (P10–P90) del dataset de referencia.
# Las categorías se solapan deliberadamente porque la trayectoria no es función
# exclusiva de subs sino composite con antigüedad, sello y alcance internacional.
COMPAT = {
    "Emergente": {
        # Sin masa crítica de audiencia. Mediana de subs ~195K, max real 1.8M.
        # 38% de los Emergentes del dataset tiene VEVO, así que se permite.
        "subs":       ["Menos de 100 mil", "100 mil – 500 mil", "500 mil – 2 millones"],
        "listeners":  ["Menos de 50 mil", "50 mil – 500 mil"],
        "antiguedad": ["Menos de 2 años", "2 – 5 años", "5 – 10 años", "Más de 10 años"],
        "sello":      ["Independiente", "Sello regional", "Major (Universal / Sony / Warner)"],
        "vevo":       True,
        "max_territorios": 4,
    },
    "Establecido": {
        # Audiencia regional consolidada. Mediana de subs ~780K, max real 10.9M.
        "subs":       ["100 mil – 500 mil", "500 mil – 2 millones", "2 – 10 millones"],
        "listeners":  ["50 mil – 500 mil", "500 mil – 2 millones", "Más de 2 millones"],
        "antiguedad": ["2 – 5 años", "5 – 10 años", "Más de 10 años"],
        "sello":      ["Independiente", "Major (Universal / Sony / Warner)"],
        "vevo":       True,
        "max_territorios": 6,
    },
    "Consagrado": {
        # Audiencia masiva internacional. Mediana de subs ~4.1M, max real 18.7M.
        # Algunos consagrados tienen poca presencia en Last.fm por cobertura
        # desigual, por eso se permite el rango 50K-500K de listeners.
        "subs":       ["500 mil – 2 millones", "2 – 10 millones", "Más de 10 millones"],
        "listeners":  ["50 mil – 500 mil", "500 mil – 2 millones", "Más de 2 millones"],
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
# FUNCIONES DE ANÁLISIS DE OUTPUT
# ══════════════════════════════════════════════════════════════
def territorios_recomendados(nivel, paises_artista):
    """
    Estima los 3-4 territorios donde una canción de este nivel suele tener
    mejor recibimiento. Basado en frecuencias del dataset filtrando por nivel.
    """
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
        afinidad = "Alta" if count / total > 0.20 else "Media"
        es_actual = pais in paises_artista
        resultado.append({
            "pais": pais,
            "nombre": PAISES_DISPLAY.get(pais, pais),
            "flag":   PAISES_FLAGS.get(pais, ""),
            "afinidad": afinidad,
            "actual": es_actual,
        })
    return resultado[:4]


def temporada_optima(nivel):
    """Calcula el mes/temporada óptima basado en patrones del dataset."""
    sub = df[df["nivel"] == nivel]
    counts = sub["release_month"].value_counts().sort_index()
    # Top 4 meses
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

        # 1. AUDIO
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

        # 2. TRAYECTORIA (CAMPO ANCLA)
        st.markdown('<p class="section-label">Trayectoria del artista</p>', unsafe_allow_html=True)
        st.caption("Define el nivel de fama del artista. Condiciona los rangos válidos del resto del formulario.")
        trayectoria = st.radio(
            "Trayectoria",
            list(COMPAT.keys()),
            index=1,
            horizontal=True,
            label_visibility="collapsed",
            help=(
                "Emergente: sin masa crítica de audiencia, suscriptores típicos por debajo de 500K.  •  "
                "Establecido: audiencia regional consolidada, suscriptores típicos de 300K a 3M.  •  "
                "Consagrado: audiencia masiva internacional, suscriptores típicos por encima de 1M."
            ),
        )
        compat = COMPAT[trayectoria]

        # 3. CANAL DE YOUTUBE
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

        # VEVO + Licencia (con lógica encadenada)
        col_vevo, col_lic = st.columns(2)
        vevo_disponible = compat["vevo"]
        with col_vevo:
            is_vevo = st.checkbox(
                "Canal VEVO oficial",
                value=False,
                disabled=not vevo_disponible,
                help="VEVO requiere contrato activo con un sello." if not vevo_disponible
                     else "Activarlo forzará el sello a 'Major'.",
            )
        with col_lic:
            is_licensed = st.checkbox(
                "Contenido licenciado",
                value=True if vevo_disponible else False,
            )

        # 4. POPULARIDAD LAST.FM
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
            # Filtrar playcount válido según listeners
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

        # 5. PERFIL EDITORIAL
        st.markdown('<p class="section-label">Perfil editorial</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            # Si VEVO está activo, forzar sello a Major
            if is_vevo:
                sello = "Major (Universal / Sony / Warner)"
                st.selectbox("Sello discográfico", [sello], disabled=True,
                             help="VEVO requiere sello Major.")
            else:
                sello = st.selectbox("Sello discográfico", compat["sello"])
        with c2:
            featuring_opt = st.radio(
                "Colaboraciones",
                ["Sin featuring", "1 colaborador", "2 o más"],
                horizontal=True,
            )
        has_featuring = featuring_opt != "Sin featuring"
        n_collabs = {"Sin featuring": 0, "1 colaborador": 1, "2 o más": 2}[featuring_opt]

        # 6. MES DE LANZAMIENTO
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

        # 7. PRESENCIA TERRITORIAL
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

        # Validación de cantidad de territorios
        if len(territorios_seleccionados) > compat["max_territorios"]:
            st.markdown(f"""
            <div class="warning-box">
                Para un artista {trayectoria.lower()} es poco frecuente tener presencia en más de 
                {compat["max_territorios"]} territorios. El modelo lo procesará de todas formas, 
                pero considera revisar la selección.
            </div>
            """, unsafe_allow_html=True)

        # 8. VALIDACIONES Y BOTÓN
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

            # Mapeo de categorías visuales a valores del modelo
            subs_val      = SUBS_RANGOS[subs_label]
            listeners_val = LISTENERS_RANGOS[listeners_label]
            playcount_val = PLAYCOUNT_RANGOS[playcount_label]
            antig_val     = ANTIGUEDAD_RANGOS[antig_label]

            # Las etiquetas de trayectoria ya coinciden con las del modelo
            # (Emergente / Establecido / Consagrado), no se requiere mapeo.
            sello_map = {
                "Independiente": "Indie",
                "Sello regional": "Regional",
                "Major (Universal / Sony / Warner)": "Major",
            }

            territorio_1 = territorios_seleccionados[0] if territorios_seleccionados else "MX"
            territorio_2 = (territorios_seleccionados[1]
                            if len(territorios_seleccionados) >= 2 else "SIN_SEGUNDO")

            input_data = pd.DataFrame([{
                "yt_channel_subscribers_log":  float(np.log10(subs_val)),
                "channel_age_years":           antig_val,
                "is_vevo":                     int(is_vevo),
                "is_licensed_content":         int(is_licensed),
                "lastfm_artist_listeners":     listeners_val,
                "lastfm_artist_playcount":     playcount_val,
                "lastfm_plays_per_listener":   round(playcount_val / listeners_val, 2),
                "lastfm_top_tag_score":        65,  # valor medio típico
                "lastfm_top10_listeners_mean": int(listeners_val * 0.35),
                "lastfm_similar_match_mean":   0.55,  # valor medio típico
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

            # TERRITORIOS RECOMENDADOS
            st.markdown('<p class="section-label">Territorios con mejor recibimiento esperado</p>',
                         unsafe_allow_html=True)
            st.caption(f"Basado en patrones de canciones nivel {nivel_pred} del dataset de referencia.")

            recomendados = territorios_recomendados(nivel_pred, territorios_seleccionados)
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

            # TEMPORADA ÓPTIMA
            st.markdown('<p class="section-label">Momento de lanzamiento sugerido</p>',
                         unsafe_allow_html=True)
            temporada = temporada_optima(nivel_pred)
            
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
            
            mensaje = (f"Las canciones nivel {nivel_pred} del dataset suelen concentrar su mejor "
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

            # CANCIONES SIMILARES
            with st.expander("Canciones de referencia con nivel similar", expanded=False):
                mask = df["nivel"] == nivel_pred
                sim = df[mask][["cancion", "artista", "n_territorios_top", "career_stage"]].sample(
                    min(5, mask.sum()), random_state=42
                ).rename(columns={
                    "cancion": "Canción", "artista": "Artista",
                    "n_territorios_top": "Territorios", "career_stage": "Trayectoria",
                })
                st.dataframe(sim, hide_index=True, use_container_width=True)

        else:
            # Estado inicial
            st.markdown("""
            <div class="result-card">
                <p class="section-label" style="margin-top:0">Cómo funciona</p>
                <ol style="color: #cbd5e1; line-height: 1.9; font-size: 0.9rem; padding-left: 1.2rem;">
                    <li>Sube el archivo de audio (.wav o .mp3) — las características acústicas
                        se extraen automáticamente con Librosa.</li>
                    <li>Completa el perfil del artista. Los campos están encadenados por 
                        trayectoria para evitar combinaciones inconsistentes.</li>
                    <li>Obtén el análisis: nivel proyectado, territorios con mejor recibimiento, 
                        y temporada óptima de lanzamiento.</li>
                </ol>
                <p style="color: #64748b; font-size: 0.8rem; font-style: italic; 
                          margin-top: 1rem; border-top: 1px solid #1e293b; padding-top: 12px;">
                    Modelo CatBoost entrenado con 333 canciones de Balada. 
                    Accuracy validada por 5-Fold CV: 67.86% (baseline azar = 33%).
                </p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — EXPLORAR MERCADO
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("##### Exploración del dataset de Balada")
    st.caption("333 canciones · Balance 111 / 111 / 111 por nivel · Datos reales y sintéticos calibrados")

    with st.sidebar:
        st.markdown("### Filtros")
        nivel_filtro  = st.multiselect("Nivel", ["Alto","Medio","Bajo"], default=["Alto","Medio","Bajo"])
        origen_filtro = st.multiselect("Origen", ["real","sintetico"], default=["real","sintetico"])
        career_filtro = st.multiselect("Trayectoria",
                                        df["career_stage"].unique().tolist(),
                                        default=df["career_stage"].unique().tolist())

    df_f = df[df["nivel"].isin(nivel_filtro) &
              df["origen"].isin(origen_filtro) &
              df["career_stage"].isin(career_filtro)]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Canciones", len(df_f))
    k2.metric("Artistas únicos", df_f["artista"].nunique())
    k3.metric("Países", df_f["pais"].nunique())
    k4.metric("Con featuring", f"{df_f['has_featuring'].mean()*100:.0f}%")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df_f["nivel"].value_counts().reset_index(),
                     values="count", names="nivel",
                     title="Distribución de niveles", color="nivel",
                     color_discrete_map=NIVEL_COLORS, hole=0.55)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                           font_family="Arial", title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df_f, x="nivel", y="n_territorios_top",
                     color="nivel", color_discrete_map=NIVEL_COLORS,
                     title="Territorios de impacto por nivel",
                     category_orders={"nivel":["Bajo","Medio","Alto"]})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                           showlegend=False, title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.violin(df_f, x="nivel", y="tempo_bpm",
                         color="nivel", color_discrete_map=NIVEL_COLORS,
                         title="Tempo (BPM) por nivel", box=True,
                         category_orders={"nivel":["Bajo","Medio","Alto"]})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                           showlegend=False, title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.box(df_f, x="nivel", y="yt_channel_subscribers_log",
                     color="nivel", color_discrete_map=NIVEL_COLORS,
                     title="Suscriptores del canal (log10) por nivel",
                     category_orders={"nivel":["Bajo","Medio","Alto"]})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                           showlegend=False, title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("##### Importancia de variables en el modelo")
    st.caption("Calculado con CatBoost Feature Importance sobre las 34 variables de inferencia.")

    importances = pd.DataFrame({
        "feature":    FEATURES,
        "importance": modelo.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    importances["bloque"] = importances["feature"].apply(lambda f:
        "Territorial" if f in ["territorio_top_1","territorio_top_2","n_territorios_top"]
        else "Last.fm"   if f.startswith("lastfm")
        else "YouTube"   if f in ["yt_channel_subscribers_log","channel_age_years","is_vevo","is_licensed_content"]
        else "Editorial" if f in ["career_stage","label_type","has_featuring","n_collaborators","release_month"]
        else "Audio"
    )
    bloque_colors = {"Territorial":"#f97316","Last.fm":"#fbbf24",
                      "YouTube":"#4ade80","Editorial":"#a855f7","Audio":"#f87171"}

    fig = px.bar(importances, x="importance", y="feature",
                 color="bloque", color_discrete_map=bloque_colors,
                 orientation="h", text="importance")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                       textfont_color="#94a3b8")
    fig.update_layout(
        height=850, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Importancia (%)", yaxis_title="",
        legend_title_text="Bloque", title_font_size=14,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style="background: rgba(56,189,248,0.06); border-left: 3px solid #38bdf8;
                padding: 14px 18px; border-radius: 4px; font-size: 0.88rem;
                color: #cbd5e1; line-height: 1.6; margin-top: 1rem;">
        <strong style="color: #38bdf8;">Hallazgo principal:</strong> 
        la variable <code>n_territorios_top</code> domina el modelo con 33.6% de importancia. 
        El número de territorios hispanohablantes con tracción del artista es un mejor predictor 
        del alcance de una canción nueva que cualquier característica acústica individual.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — METODOLOGÍA
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("##### Metodología del proyecto HitBeat")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ##### Definición del problema
        Clasificación multiclase del nivel de alcance esperado de una canción 
        en YouTube antes de su publicación.
        
        | Nivel | Rango de vistas | Mercado |
        |-------|-----------------|---------|
        | Alto  | más de 100M     | ~5% de los videos |
        | Medio | 5M – 100M       | ~20% |
        | Bajo  | menos de 5M     | ~75% |

        ##### Dataset
        - 333 canciones de Balada en español, 2010–2024  
        - 170 reales con extracción directa de fuentes  
        - 163 sintéticas calibradas para balance de clases  
        - 88 artistas distintos, 13 países hispanohablantes
        
        ##### Pipeline de datos
        Arquitectura Medallion en Databricks (Bronze → Silver → Gold), 
        orquestada como Job con disparador por llegada de archivo. 
        Cada nueva canción ingresada al volumen reentrenará el modelo 
        automáticamente.
        """)
    with c2:
        st.markdown("""
        ##### Modelo
        CatBoost Multiclase con tratamiento nativo de variables categóricas. 
        Validación por 5-Fold Stratified Cross Validation.
        
        | Métrica | Valor |
        |---------|-------|
        | Accuracy CV | 67.86% ± 2.86% |
        | F1 macro CV | 67.11% |
        | Baseline azar | 33.3% |
        
        ##### Variables de inferencia
        34 features distribuidas en cinco bloques, todas disponibles 
        antes del lanzamiento:
        
        | Bloque | N | Importancia |
        |--------|---|-------------|
        | Territorial | 3 | ~42% |
        | Audio | 16 | ~26% |
        | Last.fm | 6 | ~14% |
        | YouTube | 4 | ~10% |
        | Editorial | 5 | ~8% |
        
        ##### Limitaciones
        - Modelo entrenado únicamente con Balada. Reguetón y Corridos 
          en proceso para arquitectura de tres modelos especializados.  
        - Overfitting moderado con 333 muestras (esperado se reduzca 
          con dataset extendido).  
        - La clase Medio es estructuralmente más ambigua por la 
          amplitud del rango (5M – 100M).
        """)
    
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 1rem 0;">
        HitBeat · Trabajo Terminal · IPN ESCOM · Mayo 2026<br>
        Databricks (Delta Lake · MLflow Unity Catalog) → Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)
