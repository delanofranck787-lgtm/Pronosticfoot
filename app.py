import streamlit as st
import pandas as pd
from datetime import datetime
from data import COMPETITIONS, get_upcoming_matches, get_finished_matches
from model import DixonColesModel

st.set_page_config(page_title="Pronostics Foot", page_icon="⚽", layout="wide")

st.title("⚽ Pronostics Football")
st.caption("Modèle Dixon-Coles · Probabilités calculées sur l'historique des matchs")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    competition_name = st.selectbox("Compétition", list(COMPETITIONS.keys()))
    competition_code = COMPETITIONS[competition_name]
    st.divider()
    st.caption("⚠️ Les pronostics sont des probabilités, pas des certitudes. Le foot reste imprévisible.")

# --- Chargement des données ---
with st.spinner("Chargement de l'historique..."):
    history, hist_err = get_finished_matches(competition_code)

if hist_err:
    st.error(f"Erreur de chargement de l'historique : {hist_err}")
    st.stop()

if history is None or len(history) < 20:
    st.warning("Pas assez de matchs historiques pour entraîner le modèle. Réessaie plus tard.")
    st.stop()

# --- Entraînement du modèle ---
model = DixonColesModel()
trained = model.fit(history)

if not trained:
    st.error("Le modèle n'a pas pu s'entraîner.")
    st.stop()

st.success(f"✅ Modèle entraîné sur {len(history)} matchs historiques de {competition_name}")

# --- Matchs à venir ---
with st.spinner("Chargement des matchs à venir..."):
    upcoming, up_err = get_upcoming_matches(competition_code)

if up_err:
    st.error(f"Erreur de chargement des matchs à venir : {up_err}")
    st.stop()

if not upcoming:
    st.info("Aucun match à venir pour cette compétition.")
    st.stop()

st.subheader(f"🔮 {len(upcoming)} matchs à venir")

for m in upcoming:
    home = m["homeTeam"]["name"]
    away = m["awayTeam"]["name"]
    date_str = datetime.strptime(m["utcDate"], "%Y-%m-%dT%H:%M:%SZ").strftime("%d/%m/%Y %H:%M")

    pred = model.predict(home, away)

    with st.container(border=True):
        st.markdown(f"### {home} 🆚 {away}")
        st.caption(f"📅 {date_str}")

        col1, col2, col3 = st.columns(3)
        col1.metric(f"🏠 {home}", f"{pred['prob_home']*100:.1f}%")
        col2.metric("🤝 Nul", f"{pred['prob_draw']*100:.1f}%")
        col3.metric(f"✈️ {away}", f"{pred['prob_away']*100:.1f}%")

        # Barre de probabilité
        st.progress(pred["prob_home"], text=f"Domicile {pred['prob_home']*100:.0f}%")
        st.progress(pred["prob_draw"], text=f"Nul {pred['prob_draw']*100:.0f}%")
        st.progress(pred["prob_away"], text=f"Extérieur {pred['prob_away']*100:.0f}%")

        with st.expander("📊 Détails du pronostic"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Score probable", pred["most_likely_score"], f"{pred['score_probability']*100:.1f}%")
            c2.metric("Over 2.5", f"{pred['over_25']*100:.1f}%")
            c3.metric("BTTS", f"{pred['btts']*100:.1f}%")
            st.caption(f"Buts attendus : {home} {pred['lambda_home']:.2f} — {away} {pred['lambda_away']:.2f}")

st.divider()
st.caption("⚠️ App à but informatif. Aucun pronostic n'est garanti. Jouer comporte des risques.")
