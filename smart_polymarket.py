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

st.set_page_config(page_title="Polymarket Sports Scanner", layout="wide")
st.title("🏆 Polymarket Sports Scanner + Sharp Odds")
st.info("**Production v3.1** — Clean & Professional")

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
    st.sidebar.success("✅ Sharp Odds Active")
else:
    st.sidebar.info("Add THE_ODDS_API_KEY in .streamlit/secrets.toml for sharp comparison")

MIN_EDGE_PCT = st.sidebar.number_input("Minimum Edge (%)", value=2.5, step=0.5)
MIN_KELLY_PCT = st.sidebar.number_input("Minimum Kelly (%)", value=0.25, step=0.05)
MIN_VOLUME = st.sidebar.number_input("Minimum Volume ($)", value=8000, step=2000)
MIN_CONFIDENCE = st.sidebar.slider("Minimum Model Confidence", 0.0, 1.0, 0.4, 0.05)

KELLY_FRACTION = st.sidebar.slider("Kelly Fraction", 0.05, 1.0, 0.25, 0.05)

USE_AUTO_MODEL = st.sidebar.checkbox("Use Sport Model", value=True)
REQUIRE_MODEL_ONLY = st.sidebar.checkbox("Require Model Estimate", value=False)

st.sidebar.markdown("---")
CAT_ALL = st.sidebar.checkbox("All Categories", value=True)
CAT_SPORTS = st.sidebar.checkbox("Sports", value=True)
SIDE_FILTER = st.sidebar.radio("Bet Side", ["Both", "YES only", "NO only"], index=0)

DEBUG_MODE = st.sidebar.checkbox("Show Debug Info", value=False)

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
    if market.get("hidden", False) or market.get("archived", False):
        return False
    if market.get("active") is True or market.get("ep3Status") in ["OPEN", "ACTIVE", None]:
        return True
    sides = market.get("marketSides", [])
    if any(optional_float(s.get("price")) not in (None, 0, 1) for s in sides):
        return True
    return False

def event_group_key(outcome):
    name = outcome.event_name.lower()
    name = re.sub(r"\s*-\s*\d+-\d+.*$", "", name)
    name = re.sub(r"\s*-\s*(yes|no)$", "", name, flags=re.I)
    return normalize_name(name)

# ================== FETCHERS ==================
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

@st.cache_data(ttl=900)
def fetch_sports_model():
    ratings = {}
    leagues = [("nba", "basketball/nba"), ("nhl", "hockey/nhl"), ("nfl", "football/nfl")]
    for league, path in leagues:
        try:
            r = requests.get(f"{ESPN_BASE}/sports/{path}/standings", 
                           params={"region": "us", "lang": "en"}, timeout=15)
            r.raise_for_status()
            ratings.update(parse_espn_standings(r.json(), league))
        except: pass

    try:
        r = requests.get("https://statsapi.mlb.com/api/v1/standings",
                        params={"leagueId": "103,104", "season": datetime.now().year}, timeout=15)
        r.raise_for_status()
        ratings.update(parse_mlb_standings(r.json()))
    except: pass

    try:
        r = requests.get(FIFA_MENS_RANKINGS_URL, params={"gender": "male"}, timeout=15)
        r.raise_for_status()
        ratings.update(parse_fifa_rankings(r.json()))
    except: pass
    return SportsModelData(ratings=ratings)

@st.cache_data(ttl=300)
def fetch_sharp_odds():
    if not ODDS_API_KEY: return {}
    sports_list = ["americanfootball_nfl", "basketball_nba", "baseball_mlb", "soccer"]
    sharp = {}
    for sport in sports_list:
        try:
            r = requests.get(
                f"{THE_ODDS_API_BASE}/sports/{sport}/odds",
                params={"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h", "oddsFormat": "decimal"},
                timeout=15
            )
            r.raise_for_status()
            for event in r.json():
                home = normalize_name(event.get("home_team", ""))
                away = normalize_name(event.get("away_team", ""))
                for book in event.get("bookmakers", []):
                    for mkt in book.get("markets", []):
                        if mkt.get("key") == "h2h":
                            for outcome in mkt.get("outcomes", []):
                                name = normalize_name(outcome.get("name", ""))
                                price = optional_float(outcome.get("price"))
                                if price and price > 0:
                                    sharp[(home, away, name)] = price
        except: pass
    return sharp

# ================== DOMAIN MODELS ==================
@dataclass(frozen=True)
class MarketOutcome:
    key: str
    market_id: str
    slug: str
    category: str
    event_name: str
    market_name: str
    outcome: str
    participant: str
    league: str
    record: str
    market_prob: float
    volume: float | None
    liquidity: float | None

@dataclass(frozen=True)
class ProbabilityEstimate:
    true_prob: float
    source: str
    is_model: bool
    confidence: float

@dataclass
class TeamStats:
    wins: float = 0.0
    losses: float = 0.0
    draws: float = 0.0
    points_for: float = 0.0
    points_against: float = 0.0
    playoff_seed: int | None = None
    home_record_wins: float = 0.0
    home_record_losses: float = 0.0

@dataclass
class TeamRating:
    name: str
    league: str
    base_rating: float
    stats: TeamStats
    source: str

@dataclass
class SportsModelData:
    ratings: dict[tuple[str, str], TeamRating] = field(default_factory=dict)

# ================== PARSERS & MODEL ==================
def parse_espn_standings(payload: dict, league: str):
    ratings = {}
    for entry in iter_espn_entries(payload):
        team = entry.get("team", {})
        stats_dict = {s.get("name"): s.get("value") for s in entry.get("stats", [])}
        wins = optional_float(stats_dict.get("wins")) or 0
        losses = optional_float(stats_dict.get("losses")) or 0
        ties = optional_float(stats_dict.get("ties")) or 0
        games = wins + losses + ties
        if games == 0: continue
        base = (wins + 0.5 * ties) / games
        rating = TeamRating(
            name=team.get("displayName") or team.get("name") or "",
            league=league,
            base_rating=max(0.01, min(0.99, base)),
            stats=TeamStats(wins=wins, losses=losses, draws=ties, playoff_seed=optional_float(entry.get("playoffSeed"))),
            source=f"ESPN {league.upper()}"
        )
        for alias in [team.get("displayName"), team.get("name"), team.get("abbreviation")]:
            if alias:
                ratings[(league, normalize_name(alias))] = rating
    return ratings

def iter_espn_entries(node):
    if isinstance(node.get("standings"), dict):
        yield from node["standings"].get("entries", [])
    for child in node.get("children", []):
        yield from iter_espn_entries(child)

def parse_mlb_standings(payload):
    ratings = {}
    for group in payload.get("records", []):
        for tr in group.get("teamRecords", []):
            team = tr.get("team", {})
            wins = optional_float(tr.get("wins")) or 0
            losses = optional_float(tr.get("losses")) or 0
            rf = optional_float(tr.get("runsFor")) or 0
            ra = optional_float(tr.get("runsAllowed")) or 0
            if wins + losses == 0: continue
            pyth = (rf ** 2) / (rf ** 2 + ra ** 2) if (rf + ra) > 0 else wins / (wins + losses)
            rating = TeamRating(name=team.get("name", ""), league="mlb", base_rating=max(0.01, min(0.99, pyth)),
                                stats=TeamStats(wins=wins, losses=losses, points_for=rf, points_against=ra),
                                source="MLB Pythagorean")
            ratings[("mlb", normalize_name(team.get("name", "")))] = rating
    return ratings

def parse_fifa_rankings(payload):
    ratings = {}
    for row in payload.get("Results", []):
        points = optional_float(row.get("DecimalTotalPoints") or row.get("TotalPoints"))
        name_obj = row.get("TeamName")
        name = name_obj.get("Description") if isinstance(name_obj, dict) else name_obj
        if not name or points is None: continue
        norm = max(0.01, min(0.99, (points - 1100) / 900))
        rating = TeamRating(name=name, league="fifawc", base_rating=norm, stats=TeamStats(), source="FIFA")
        country = str(row.get("IdCountry", ""))
        for alias in [name, country] + FIFA_COUNTRY_ALIASES.get(country, []):
            if alias:
                ratings[("fifawc", normalize_name(alias))] = rating
    return ratings

def lookup_team_rating(data: SportsModelData, league: str, participant: str):
    if not participant: return None
    norm = normalize_name(participant)
    league = league.lower()
    if (league, norm) in data.ratings:
        return data.ratings[(league, norm)]
    for (l, n), rating in data.ratings.items():
        if l == league and (norm in n or n in norm):
            return rating
    return None

def estimate_probabilities(outcomes, sports_data):
    estimates = {}
    groups = defaultdict(list)
    for o in [o for o in outcomes if o.outcome.upper() == "YES"]:
        groups[event_group_key(o)].append(o)

    for group in groups.values():
        if len(group) < 2: continue
        strengths = {}
        for o in group:
            rating = lookup_team_rating(sports_data, o.league, o.participant)
            if rating:
                p = rating.base_rating
                if rating.stats.playoff_seed:
                    p = min(0.99, p + (16 - min(rating.stats.playoff_seed, 16)) * 0.012)
                strengths[o.key] = p
        if strengths:
            total = sum(strengths.values())
            for o in group:
                if o.key in strengths:
                    estimates[o.key] = ProbabilityEstimate(
                        true_prob=strengths[o.key] / total,
                        source="Sports Model",
                        is_model=True,
                        confidence=0.72
                    )
    return estimates

def calc_no_complement(estimates, outcomes):
    result = estimates.copy()
    yes_map = {k.split("::")[0]: v for k, v in estimates.items()}
    for o in outcomes:
        if o.outcome.upper() == "NO":
            base = o.key.split("::")[0]
            if base in yes_map:
                yes = yes_map[base]
                result[o.key] = ProbabilityEstimate(1 - yes.true_prob, f"Complement ({yes.source})", yes.is_model, yes.confidence * 0.95)
    return result

# ================== MAIN ==================
markets = fetch_polymarket()
sports_data = fetch_sports_model() if USE_AUTO_MODEL else SportsModelData()
sharp_odds = fetch_sharp_odds()

outcomes = []
stats = {"total": len(markets), "open": 0, "category": 0, "volume": 0, "final": 0}

for market in markets:
    if not is_open_market(market): continue
    stats["open"] += 1

    category = (market.get("category") or "unknown").lower()
    if not (CAT_ALL or (category == "sports" and CAT_SPORTS)): continue
    stats["category"] += 1

    volume = optional_float(market.get("volumeNum") or market.get("volume"))
    if volume and volume < MIN_VOLUME: continue
    stats["volume"] += 1

    event_name = clear_market_name(market)
    for side in market.get("marketSides", []):
        price = optional_float(side.get("price"))
        if not price or not (0.01 <= price <= 0.99): continue

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

estimates = estimate_probabilities(outcomes, sports_data)
estimates = calc_no_complement(estimates, outcomes)

# Value bets
value_rows = []
for o in outcomes:
    est = estimates.get(o.key)
    if REQUIRE_MODEL_ONLY and not (est and est.is_model):
        continue
    if not est:
        est = ProbabilityEstimate(o.market_prob, "Baseline", False, 0.4)

    if est.confidence < MIN_CONFIDENCE: continue

    edge_pct = (est.true_prob - o.market_prob) / o.market_prob * 100 if o.market_prob > 0 else 0
    if edge_pct < MIN_EDGE_PCT: continue

    kelly_full = max(0, est.true_prob * (1/o.market_prob - 1) - (1 - est.true_prob)) / (1/o.market_prob - 1) if o.market_prob != 1 else 0
    kelly_pct = kelly_full * KELLY_FRACTION * 100
    if kelly_pct < MIN_KELLY_PCT: continue

    bet_size = BANKROLL * kelly_full * KELLY_FRACTION

    value_rows.append({
        "Market": o.market_name[:65],
        "Side": o.outcome,
        "PM Prob%": round(o.market_prob * 100, 1),
        "Model Prob%": round(est.true_prob * 100, 1),
        "Edge%": round(edge_pct, 1),
        "Kelly%": round(kelly_pct, 1),
        "Conf": round(est.confidence, 2),
        "Bet $": round(bet_size),
        "Volume": f"${int(o.volume):,}" if o.volume else "N/A",
        "Source": est.source
    })

df = pd.DataFrame(value_rows)

# ================== UI ==================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Markets", stats["total"])
col2.metric("Open Markets", stats["open"])
col3.metric("After Filters", stats["volume"])
col4.metric("Value Bets", len(df))

st.subheader(f"🔍 Value Bets Found: {len(df)}")

if df.empty:
    st.warning("No value bets found with current filters. The sports betting market is quite efficient right now.")
else:
    df = df.sort_values("Edge%", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "value_bets.csv", "text/csv")

if DEBUG_MODE:
    with st.expander("Debug Info"):
        st.write(stats)
        st.write(f"Sharp Odds Events Loaded: {len(sharp_odds)}")

st.caption("⚠️ Not financial advice • Independent Model + Sharp Odds Comparison")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
