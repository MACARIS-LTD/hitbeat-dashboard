# HitBeat 🎵

**Predictor de alcance musical en YouTube para música latina en español.**

Proyecto de Trabajo Terminal · IPN ESCOM · 2026

## Stack
- **Pipeline de datos**: Databricks (Delta Lake · Medallion Architecture · MLflow)
- **Modelo**: CatBoost Multiclase · Accuracy CV 67.86%
- **Dashboard**: Streamlit Cloud

## Estructura del repo
```
app.py              ← Dashboard principal
requirements.txt    ← Dependencias
data/               ← Dataset y modelo (no versionados en git)
  hitbeat_dashboard_data.csv
  modelo_balada.pkl
```

## Cómo correr localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
