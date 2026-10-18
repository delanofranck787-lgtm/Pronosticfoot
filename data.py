import requests
import pandas as pd
import streamlit as st
from datetime import datetime

API_KEY = st.secrets.get("FOOTBALL_API_KEY", "")
BASE_URL = "https://api.football-data.org/v4"

# Compétitions gratuites disponibles sur le plan gratuit
COMPETITIONS = {
    "Ligue 1": "FL1",
    "Premier League": "PL",
    "La Liga": "PD",
    "Serie A": "SA",
    "Bundesliga": "BL1",
    "Champions League": "CL",
}

HEADERS = {"X-Auth-Token": API_KEY}


@st.cache_data(ttl=3600)
def get_upcoming_matches(competition_code: str):
    """Récupère les matchs à venir d'une compétition."""
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    params = {"status": "SCHEDULED"}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if r.status_code != 200:
            return None, f"Erreur API ({r.status_code})"
        data = r.json()
        matches = data.get("matches", [])[:20]
        return matches, None
    except Exception as e:
        return None, f"Erreur réseau : {e}"


@st.cache_data(ttl=3600)
def get_finished_matches(competition_code: str, limit: int = 200):
    """Récupère les matchs terminés pour entraîner le modèle."""
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    params = {"status": "FINISHED"}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if r.status_code != 200:
            return None, f"Erreur API ({r.status_code})"
        data = r.json()
        matches = data.get("matches", [])[-limit:]
        rows = []
        for m in matches:
            if m["score"]["fullTime"]["home"] is None:
                continue
            rows.append({
                "date": m["utcDate"],
                "home": m["homeTeam"]["name"],
                "away": m["awayTeam"]["name"],
                "home_goals": m["score"]["fullTime"]["home"],
                "away_goals": m["score"]["fullTime"]["away"],
            })
        return pd.DataFrame(rows), None
    except Exception as e:
        return None, f"Erreur réseau : {e}"
