import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# ================== CONSTANTS ==================
POLYMARKET_US_BASE_URL = "https://gateway.polymarket.us/v1"
ESPN_BASE = "https://site.web.api.espn.com/apis/v2"
FIFA_MENS_RANKINGS_URL = "https://api.fifa.com/api/v3/rankings"
THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"

FIFA_COUNTRY_ALIASES = {
    "CPV": ["Cabo Verde", "Cape Verde"],
    "CZE": ["Czechia", "Czech Republic"],
    "IRN": ["IR Iran", "Iran"],
    "KOR": ["Korea Republic", "South Korea"],
    "KSA": ["Saudi Arabia"],
    "TUR": ["Turkiye", "Turkey"],
    "UAE": ["United Arab Emirates"],
    "USA": ["USA", "United States", "US"],
}

st.set_page_config(page_title="Polymarket + Sharp Odds Scanner", layout="wide")
st.title("🏆 Polymarket Sports Scanner + The Odds API")
st.info("**Production v2.7** — Improved Open Market Detection")

# ================== SECRETS ==================
def get_odds_api_key():
    try:
        return st.secrets["THE_ODDS_API_KEY"]
    except:
        return None

# ================== SIDEBAR ==================
st.sidebar.header("🔑 Configuration")

BANKROLL = st.sidebar.number_input("Bankroll ($)", value=10000, min_value=100, step=500)

ODDS_API_KEY = get_odds_api_key()
if ODDS_API_KEY:
    st.sidebar.success("✅ Odds API Key loaded from secrets")
else:
    ODDS_API_KEY = st.sidebar.text_input("The Odds API Key", type="password")

RELAXED_MODE = st.sidebar.checkbox("🟢 Relaxed Mode (Show more bets)", value=True)

if RELAXED_MODE:
    MIN_EDGE_PCT = st.sidebar.number_input("Minimum Edge (%)", value=1.5, step=0.5)
    MIN_KELLY_PCT = st.sidebar.number_input("Minimum Kelly (%)", value=0.2, step=0.1)
    MIN_VOLUME = st.sidebar.number_input("Minimum Volume ($)", value=5000, step=5000)
    MIN_CONFIDENCE = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.35, 0.05)
else:
    MIN_EDGE_PCT = st.sidebar.number_input("Minimum Edge (%)", value=4.0, step=1.0)
    MIN_KELLY_PCT = st.sidebar.number_input("Minimum Kelly (%)", value=0.4, step=0.1)
    MIN_VOLUME = st.sidebar.number_input("Minimum Volume ($)", value=30000, step=10000)
    MIN_CONFIDENCE = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)

KELLY_FRACTION = st.sidebar.slider("Kelly Fraction", 0.05, 1.0, 0.25, 0.05)

USE_AUTO_MODEL = st.sidebar.checkbox("Use Sport Model", value=True)
REQUIRE_MODEL_ONLY = st.sidebar.checkbox("Require Strong Model Only", value=False)

st.sidebar.markdown("---")
CAT_ALL = st.sidebar.checkbox("All Categories", value=True)
CAT_SPORTS = st.sidebar.checkbox("Sports", value=True)
SIDE_FILTER = st.sidebar.radio("Bet Side", ["Both", "YES only", "NO only"], index=0)

AUTO_REFRESH = st.sidebar.checkbox("Auto Refresh (5 min)", value=False)
DEBUG_MODE = st.sidebar.checkbox("Show Debug Info", value=True)

# ================== HELPERS ==================
def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

def optional_float(value):
    try:
        return float(value) if value not in (None, "", "null", "None") else None
    except:
        return None

def clear_market_name(market: dict) -> str:
    parts = [market.get(k) for k in ["question", "title", "subtitle"] if market.get(k)]
    return " - ".join(filter(None, parts))

def is_open_market(market: dict) -> bool:
    """Improved logic based on actual API data"""
    if market.get("hidden", False):
        return False
    if market.get("archived", False):
        return False
    # Allow markets that are active or have prices even if marked closed (many expired games still show)
    if market.get("active") is True:
        return True
    if market.get("ep3Status") in ["OPEN", "ACTIVE"]:
        return True
    # Allow markets that still have meaningful prices
    if any(side.get("price") not in (None, 0, 1) for side in market.get("marketSides", [])):
        return True
    return False

def event_group_key(outcome):
    name = outcome.event_name.lower()
    name = re.sub(r"\s*-\s*\d+-\d+.*$", "", name)
    name = re.sub(r"\s*-\s*(yes|no)$", "", name, flags=re.I)
    return normalize_name(name)

# ================== FETCHERS (with fixed Odds API) ==================
@st.cache_data(ttl=600)
def fetch_polymarket():
    markets = []
    offset = 0
    for _ in range(50):
        try:
            resp = requests.get(
                f"{POLYMARKET_US_BASE_URL}/markets",
                params={"active": "true", "limit": 100, "offset": offset},
                timeout=25
            )
            resp.raise_for_status()
            batch = resp.json().get("markets", [])
            if not batch: break
            markets.extend(batch)
            offset += 100
        except Exception as e:
            st.warning(f"Polymarket: {e}")
            break
    return markets

# ... [Keep fetch_sports_model(), fetch_sharp_odds(), all parser functions, estimate_probabilities, etc. from v2.6]

# (For brevity, I'm assuming you have the rest from previous version. 
# If you need the full thing again, let me know.)

# ================== MAIN LOGIC ==================
markets = fetch_polymarket()
sports_data = fetch_sports_model() if USE_AUTO_MODEL else SportsModelData()
sharp_odds = fetch_sharp_odds()

outcomes = []
stats = {"total": len(markets), "open": 0, "category_pass": 0, "volume_pass": 0, "final": 0}

for market in markets:
    if not is_open_market(market):
        continue
    stats["open"] += 1

    category = (market.get("category") or "unknown").lower()
    if not (CAT_ALL or (category == "sports" and CAT_SPORTS)):
        continue
    stats["category_pass"] += 1

    volume = optional_float(market.get("volumeNum") or market.get("volume"))
    if volume and volume < MIN_VOLUME:
        continue
    stats["volume_pass"] += 1

    event_name = clear_market_name(market)
    for side in market.get("marketSides", []):
        price = optional_float(side.get("price"))
        if not price or not (0.01 <= price <= 0.99):
            continue

        outcome_text = side.get("description") or ("Yes" if side.get("long", True) else "No")
        if SIDE_FILTER == "YES only" and outcome_text.upper() != "YES": continue
        if SIDE_FILTER == "NO only" and outcome_text.upper() != "NO": continue

        team = side.get("team", {})
        outcomes.append(MarketOutcome(
            key=f"{market.get('slug') or market.get('id')}::{outcome_text.lower()}",
            market_id=market.get("id", ""),
            slug=market.get("slug", ""),
            category=category,
            event_name=event_name,
            market_name=event_name,
            outcome=outcome_text,
            participant=team.get("name", ""),
            league=team.get("league", "").lower(),
            record=team.get("record", ""),
            market_prob=price,
            volume=volume,
            liquidity=optional_float(market.get("liquidityNum"))
        ))

stats["final"] = len(outcomes)

# ... (rest of the code: estimates, value_rows, UI - same as v2.6)

# UI part (same as before)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Markets", stats["total"])
col2.metric("Open", stats["open"])
col3.metric("After Filters", stats["volume_pass"])
col4.metric("Value Bets", len(df))

st.subheader(f"🔍 Value Bets Found: {len(df)}")
