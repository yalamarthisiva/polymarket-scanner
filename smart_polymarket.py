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
POLYMARKET_US_BASE_URL = "https://gamma-api.polymarket.com"
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

SHARP_SPORTS = [
    "americanfootball_nfl",
    "basketball_nba",
    "baseball_mlb",
    "icehockey_nhl",
    "soccer_epl",
    "soccer_usa_mls",
    "soccer_uefa_champs_league",
]

st.set_page_config(page_title="Polymarket Sports Scanner", layout="wide")
st.title("🏆 Polymarket Sports Scanner + Sharp Odds")
st.info("**Production v4.0** — Sharp Odds Integration + Improved Matching")

# ================== SECRETS ==================
def get_odds_api_key():
    try:
        return st.secrets["THE_ODDS_API_KEY"]
    except Exception:
        return None

# ================== SIDEBAR ==================
st.sidebar.header("🔑 Configuration")

BANKROLL = st.sidebar.number_input("Bankroll ($)", value=10000, min_value=100, step=500)

ODDS_API_KEY = get_odds_api_key()
if ODDS_API_KEY:
    st.sidebar.success("✅ Sharp Odds Active")
else:
    st.sidebar.warning("Add THE_ODDS_API_KEY in secrets.toml for sharp edge")

MIN_EDGE_PCT = st.sidebar.number_input("Minimum Edge (%)", value=2.0, step=0.5)
MIN_KELLY_PCT = st.sidebar.number_input("Minimum Kelly (%)", value=0.1, step=0.05)
MIN_VOLUME = st.sidebar.number_input("Minimum Volume ($)", value=1000, step=1000)
MIN_CONFIDENCE = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.35, 0.05)
KELLY_FRACTION = st.sidebar.slider("Kelly Fraction", 0.05, 1.0, 0.25, 0.05)

USE_AUTO_MODEL = st.sidebar.checkbox("Use Sport Model", value=True)
REQUIRE_MODEL_ONLY = st.sidebar.checkbox("Require Model Estimate", value=False)

st.sidebar.markdown("---")
CAT_ALL = st.sidebar.checkbox("All Categories", value=True)
CAT_SPORTS = st.sidebar.checkbox("Sports", value=True)
SIDE_FILTER = st.sidebar.radio("Bet Side", ["Both", "YES only", "NO only"], index=0)
DEBUG_MODE = st.sidebar.checkbox("Debug Info", value=False)

# ================== HELPERS ==================
def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

def optional_float(value):
    try:
        return float(value) if value not in (None, "", "null", "None") else None
    except Exception:
        return None

def clear_market_name(market: dict) -> str:
    parts = [market.get(k) for k in ["question", "title", "subtitle"] if market.get(k)]
    return " - ".join(filter(None, parts))

def is_open_market(market: dict) -> bool:
    if market.get("hidden", False) or market.get("archived", False):
        return False
    if market.get("closed", False):
        return False
    active = market.get("active")
    if active is False:
        return False
    return True

def event_group_key(outcome) -> str:
    name = outcome.event_name.lower()
    name = re.sub(r"\s*-\s*\d+-\d+.*$", "", name)
    name = re.sub(r"\s*-\s*(yes|no)$", "", name, flags=re.I)
    return normalize_name(name)

def extract_teams_from_title(title: str) -> list[str]:
    """Extract potential team/participant names from a market title."""
    # Pattern: "Will X beat Y", "X vs Y", "X to win", "X -0.5"
    patterns = [
        r"will\s+(.+?)\s+(?:beat|win|defeat|cover)",
        r"(.+?)\s+(?:vs\.?|v\.?|@)\s+(.+?)(?:\s+[-–]|\?|$)",
        r"(.+?)\s+to\s+win",
        r"(.+?)\s+(?:win|wins|advance|beat|beats)",
    ]
    teams = []
    for pat in patterns:
        m = re.search(pat, title, re.I)
        if m:
            teams.extend([g.strip() for g in m.groups() if g])
    # Fallback: split on " vs " or " - "
    if not teams:
        for sep in [" vs ", " vs. ", " v ", " - ", " @ "]:
            if sep.lower() in title.lower():
                parts = re.split(sep, title, flags=re.I, maxsplit=1)
                teams = [p.strip() for p in parts if p.strip()]
                break
    return [t for t in teams if len(t) > 2][:2]


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

@dataclass
class TeamRating:
    name: str
    league: str
    base_rating: float
    stats: TeamStats
    source: str

@dataclass
class SportsModelData:
    ratings: dict = field(default_factory=dict)


# ================== FETCHERS ==================
@st.cache_data(ttl=600)
def fetch_polymarket():
    """Fetch markets using the Gamma (public) API."""
    markets = []
    offset = 0
    for _ in range(60):
        try:
            resp = requests.get(
                f"{POLYMARKET_US_BASE_URL}/markets",
                params={"active": "true", "closed": "false", "limit": 100, "offset": offset},
                timeout=25,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            # Gamma API returns a list directly
            if isinstance(batch, list):
                markets.extend(batch)
                if len(batch) < 100:
                    break
            else:
                # Fallback for dict-wrapped response
                items = batch.get("markets", batch.get("data", []))
                if not items:
                    break
                markets.extend(items)
            offset += 100
        except Exception as e:
            st.warning(f"Polymarket fetch error at offset {offset}: {e}")
            break
    return markets


@st.cache_data(ttl=900)
def fetch_sports_model():
    ratings = {}
    leagues = [
        ("nba", "basketball/nba"),
        ("nhl", "hockey/nhl"),
        ("nfl", "football/nfl"),
    ]
    for league, path in leagues:
        try:
            r = requests.get(
                f"{ESPN_BASE}/sports/{path}/standings",
                params={"region": "us", "lang": "en"},
                timeout=15,
            )
            r.raise_for_status()
            ratings.update(parse_espn_standings(r.json(), league))
        except Exception:
            pass

    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/standings",
            params={"leagueId": "103,104", "season": datetime.now().year},
            timeout=15,
        )
        r.raise_for_status()
        ratings.update(parse_mlb_standings(r.json()))
    except Exception:
        pass

    try:
        r = requests.get(FIFA_MENS_RANKINGS_URL, params={"gender": "male"}, timeout=15)
        r.raise_for_status()
        ratings.update(parse_fifa_rankings(r.json()))
    except Exception:
        pass

    return SportsModelData(ratings=ratings)


@st.cache_data(ttl=300)
def fetch_sharp_odds(api_key: str | None):
    """
    Returns two dicts:
      sharp_by_team:  (norm_team_name) -> {decimal_price, home, away, sport}
      sharp_by_event: (norm_home, norm_away) -> {home_price, away_price, draw_price}
    """
    if not api_key:
        return {}, {}

    by_team = {}
    by_event = {}

    for sport in SHARP_SPORTS:
        try:
            r = requests.get(
                f"{THE_ODDS_API_BASE}/sports/{sport}/odds",
                params={
                    "apiKey": api_key,
                    "regions": "us,uk,eu",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                    "bookmakers": "pinnacle,betfair_ex_eu,draftkings,fanduel",
                },
                timeout=15,
            )
            r.raise_for_status()
            for event in r.json():
                home = normalize_name(event.get("home_team", ""))
                away = normalize_name(event.get("away_team", ""))
                if not home or not away:
                    continue

                # Aggregate prices across sharp books — use consensus (median)
                home_prices, away_prices, draw_prices = [], [], []
                for book in event.get("bookmakers", []):
                    for mkt in book.get("markets", []):
                        if mkt.get("key") != "h2h":
                            continue
                        for oc in mkt.get("outcomes", []):
                            p = optional_float(oc.get("price"))
                            if not p or p <= 1.0:
                                continue
                            n = normalize_name(oc.get("name", ""))
                            if n == home:
                                home_prices.append(p)
                            elif n == away:
                                away_prices.append(p)
                            else:
                                draw_prices.append(p)

                def median(lst):
                    if not lst:
                        return None
                    s = sorted(lst)
                    m = len(s) // 2
                    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

                hp = median(home_prices)
                ap = median(away_prices)
                dp = median(draw_prices)

                if hp:
                    by_team[home] = {"decimal": hp, "home": home, "away": away, "sport": sport}
                if ap:
                    by_team[away] = {"decimal": ap, "home": home, "away": away, "sport": sport}

                by_event[(home, away)] = {
                    "home_price": hp,
                    "away_price": ap,
                    "draw_price": dp,
                    "sport": sport,
                }
        except Exception:
            pass

    return by_team, by_event


# ================== PARSERS ==================
def parse_espn_standings(payload: dict, league: str):
    ratings = {}
    for entry in iter_espn_entries(payload):
        team = entry.get("team", {})
        stats_dict = {s.get("name"): s.get("value") for s in entry.get("stats", [])}
        wins = optional_float(stats_dict.get("wins")) or 0
        losses = optional_float(stats_dict.get("losses")) or 0
        ties = optional_float(stats_dict.get("ties")) or 0
        games = wins + losses + ties
        if games == 0:
            continue
        base = (wins + 0.5 * ties) / games
        rating = TeamRating(
            name=team.get("displayName") or team.get("name") or "",
            league=league,
            base_rating=max(0.01, min(0.99, base)),
            stats=TeamStats(wins=wins, losses=losses, draws=ties,
                            playoff_seed=optional_float(entry.get("playoffSeed"))),
            source=f"ESPN {league.upper()}",
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
            if wins + losses == 0:
                continue
            pyth = (rf**2) / (rf**2 + ra**2) if (rf + ra) > 0 else wins / (wins + losses)
            rating = TeamRating(
                name=team.get("name", ""),
                league="mlb",
                base_rating=max(0.01, min(0.99, pyth)),
                stats=TeamStats(wins=wins, losses=losses, points_for=rf, points_against=ra),
                source="MLB Pythagorean",
            )
            ratings[("mlb", normalize_name(team.get("name", "")))] = rating
    return ratings


def parse_fifa_rankings(payload):
    ratings = {}
    for row in payload.get("Results", []):
        points = optional_float(row.get("DecimalTotalPoints") or row.get("TotalPoints"))
        name_obj = row.get("TeamName")
        name = name_obj.get("Description") if isinstance(name_obj, dict) else name_obj
        if not name or points is None:
            continue
        norm = max(0.01, min(0.99, (points - 1100) / 900))
        rating = TeamRating(name=name, league="fifawc", base_rating=norm, stats=TeamStats(), source="FIFA")
        country = str(row.get("IdCountry", ""))
        for alias in [name, country] + FIFA_COUNTRY_ALIASES.get(country, []):
            if alias:
                ratings[("fifawc", normalize_name(alias))] = rating
    return ratings


def lookup_team_rating(data: SportsModelData, league: str, participant: str):
    if not participant:
        return None
    norm = normalize_name(participant)
    league = league.lower()
    if (league, norm) in data.ratings:
        return data.ratings[(league, norm)]
    for (l, n), rating in data.ratings.items():
        if l == league and (norm in n or n in norm):
            return rating
    return None


# ================== PROBABILITY ESTIMATION ==================

def vig_free_prob(decimal_odds_list: list[float]) -> list[float]:
    """Remove bookmaker overround and return fair probabilities."""
    if not decimal_odds_list or any(o <= 0 for o in decimal_odds_list):
        return []
    raw = [1.0 / o for o in decimal_odds_list]
    total = sum(raw)
    return [p / total for p in raw]


def match_sharp_odds_for_outcome(
    outcome: MarketOutcome,
    sharp_by_team: dict,
    sharp_by_event: dict,
) -> ProbabilityEstimate | None:
    """
    Try to find a sharp-book price for this Polymarket outcome.
    Handles YES/NO binary markets that correspond to a team winning.
    """
    market_title = outcome.event_name.lower()
    participant = normalize_name(outcome.participant) if outcome.participant else ""
    is_yes = outcome.outcome.upper() == "YES"

    # --- Strategy 1: direct team name match ---
    if participant and participant in sharp_by_team:
        entry = sharp_by_team[participant]
        dec = entry["decimal"]
        # Get the opposing price for vig removal
        home, away = entry["home"], entry["away"]
        event_key = (home, away)
        event = sharp_by_event.get(event_key, {})
        prices = [p for p in [event.get("home_price"), event.get("away_price")] if p]
        if len(prices) == 2:
            fair_probs = vig_free_prob(prices)
            # Determine which side is ours
            if normalize_name(event.get("home", "")) == participant:
                true_prob = fair_probs[0]
            else:
                true_prob = fair_probs[1]
        else:
            true_prob = min(0.97, 1.0 / dec)

        return ProbabilityEstimate(
            true_prob=true_prob if is_yes else 1 - true_prob,
            source="Sharp Books (direct)",
            is_model=True,
            confidence=0.82,
        )

    # --- Strategy 2: fuzzy team name from market title ---
    teams = extract_teams_from_title(outcome.event_name)
    for team_raw in teams:
        team_norm = normalize_name(team_raw)
        # Substring match in sharp_by_team keys
        for key, entry in sharp_by_team.items():
            if (team_norm in key or key in team_norm) and len(min(team_norm, key)) > 3:
                dec = entry["decimal"]
                home, away = entry["home"], entry["away"]
                event_key = (home, away)
                event = sharp_by_event.get(event_key, {})
                prices = [p for p in [event.get("home_price"), event.get("away_price")] if p]

                if len(prices) == 2:
                    fair_probs = vig_free_prob(prices)
                    # Assign to correct side by matching team position
                    matched_home = normalize_name(entry.get("home", ""))
                    true_prob = fair_probs[0] if (team_norm in matched_home or matched_home in team_norm) else fair_probs[1]
                else:
                    true_prob = min(0.97, 1.0 / dec)

                # For "Will X win?" market, YES maps to that team winning
                if is_yes:
                    return ProbabilityEstimate(
                        true_prob=true_prob,
                        source="Sharp Books (fuzzy)",
                        is_model=True,
                        confidence=0.70,
                    )
                else:
                    return ProbabilityEstimate(
                        true_prob=1 - true_prob,
                        source="Sharp Books (fuzzy)",
                        is_model=True,
                        confidence=0.65,
                    )

    # --- Strategy 3: event-level keyword overlap ---
    title_tokens = set(normalize_name(outcome.event_name).split())
    best_overlap, best_entry, best_event = 0, None, None
    for (home, away), event in sharp_by_event.items():
        event_tokens = set((home + " " + away).split())
        overlap = len(title_tokens & event_tokens)
        if overlap >= 2 and overlap > best_overlap:
            best_overlap = overlap
            best_entry = (home, away)
            best_event = event

    if best_event and best_entry:
        home, away = best_entry
        prices = [p for p in [best_event.get("home_price"), best_event.get("away_price")] if p]
        if len(prices) == 2:
            fair_probs = vig_free_prob(prices)
            # Heuristic: title mentions home team first → YES = home win
            home_in_title = home in " ".join(list(title_tokens)[:6])
            true_prob = fair_probs[0] if home_in_title else fair_probs[1]
            return ProbabilityEstimate(
                true_prob=true_prob if is_yes else 1 - true_prob,
                source="Sharp Books (event match)",
                is_model=True,
                confidence=0.55,
            )

    return None


def estimate_probabilities_model(outcomes, sports_data: SportsModelData) -> dict:
    estimates = {}
    groups = defaultdict(list)
    for o in outcomes:
        if o.outcome.upper() == "YES":
            groups[event_group_key(o)].append(o)

    for group in groups.values():
        if len(group) < 2:
            continue
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
                        source="Sports Model (ESPN/FIFA)",
                        is_model=True,
                        confidence=0.68,
                    )
    return estimates


def calc_no_complement(estimates: dict, outcomes) -> dict:
    result = estimates.copy()
    yes_map = {k.split("::")[0]: v for k, v in estimates.items()}
    for o in outcomes:
        if o.outcome.upper() == "NO":
            base = o.key.split("::")[0]
            if base in yes_map:
                yes = yes_map[base]
                result[o.key] = ProbabilityEstimate(
                    true_prob=max(0.01, min(0.99, 1 - yes.true_prob)),
                    source=f"Complement ({yes.source})",
                    is_model=yes.is_model,
                    confidence=yes.confidence * 0.95,
                )
    return result


# ================== MAIN ==================
markets = fetch_polymarket()
sports_data = fetch_sports_model() if USE_AUTO_MODEL else SportsModelData()
sharp_by_team, sharp_by_event = fetch_sharp_odds(ODDS_API_KEY)

outcomes: list[MarketOutcome] = []
stats = {
    "total": len(markets),
    "open": 0,
    "category_pass": 0,
    "volume_pass": 0,
    "final": 0,
}

for market in markets:
    if not is_open_market(market):
        continue
    stats["open"] += 1

    category = (market.get("category") or "unknown").lower()
    if not (CAT_ALL or (category == "sports" and CAT_SPORTS)):
        continue
    stats["category_pass"] += 1

    volume = optional_float(market.get("volumeNum") or market.get("volume"))
    if volume is not None and volume < MIN_VOLUME:
        continue
    stats["volume_pass"] += 1

    event_name = clear_market_name(market)

    # Gamma API: outcomes are in "tokens" list or "outcomePrices" + "outcomes" parallel lists
    tokens = market.get("tokens", [])
    outcome_names = market.get("outcomes", [])
    outcome_prices_raw = market.get("outcomePrices", [])

    # Build (outcome_text, price) pairs
    pairs: list[tuple[str, float]] = []

    if tokens:
        for tok in tokens:
            name = tok.get("outcome", tok.get("title", ""))
            price = optional_float(tok.get("price"))
            if name and price is not None:
                pairs.append((name, price))
    elif outcome_names and outcome_prices_raw:
        # outcome_prices may be a JSON string in the API
        if isinstance(outcome_prices_raw, str):
            try:
                outcome_prices_raw = json.loads(outcome_prices_raw)
            except Exception:
                outcome_prices_raw = []
        if isinstance(outcome_names, str):
            try:
                outcome_names = json.loads(outcome_names)
            except Exception:
                outcome_names = []
        for name, price_str in zip(outcome_names, outcome_prices_raw):
            price = optional_float(price_str)
            if name and price is not None:
                pairs.append((name, price))
    else:
        # Legacy marketSides fallback
        for side in market.get("marketSides", []):
            price = optional_float(side.get("price"))
            name = side.get("description") or ("Yes" if side.get("long", True) else "No")
            if price is not None:
                pairs.append((name, price))

    for outcome_text, price in pairs:
        if not (0.01 <= price <= 0.99):
            continue
        if SIDE_FILTER == "YES only" and outcome_text.upper() != "YES":
            continue
        if SIDE_FILTER == "NO only" and outcome_text.upper() != "NO":
            continue

        # Try to extract participant from market title
        extracted = extract_teams_from_title(event_name)
        # Assign participant: for YES outcomes take first team, NO take second
        if extracted:
            if outcome_text.upper() == "YES":
                participant = extracted[0]
            elif len(extracted) > 1:
                participant = extracted[1]
            else:
                participant = extracted[0]
        else:
            participant = outcome_text  # fallback: use outcome text itself

        league = market.get("league", market.get("sport", "")).lower()

        slug = market.get("slug") or market.get("id") or market.get("conditionId", "")
        outcomes.append(
            MarketOutcome(
                key=f"{slug}::{outcome_text.lower()}",
                market_id=str(market.get("id") or market.get("conditionId", "")),
                slug=str(slug),
                category=category,
                event_name=event_name,
                market_name=event_name,
                outcome=outcome_text,
                participant=participant,
                league=league,
                record="",
                market_prob=price,
                volume=volume,
                liquidity=optional_float(market.get("liquidityNum") or market.get("liquidity")),
            )
        )

stats["final"] = len(outcomes)

# ---- Estimate probabilities ----
# 1. Sports model (ESPN/FIFA)
model_estimates = estimate_probabilities_model(outcomes, sports_data) if USE_AUTO_MODEL else {}

# 2. Sharp books — run for EVERY outcome not already covered by model
sharp_estimates = {}
for o in outcomes:
    if o.key in model_estimates:
        continue
    est = match_sharp_odds_for_outcome(o, sharp_by_team, sharp_by_event)
    if est:
        sharp_estimates[o.key] = est

# Merge: model takes precedence if both available (higher confidence)
combined_estimates = {**sharp_estimates, **model_estimates}

# 3. Complement: fill NO sides from YES estimates
combined_estimates = calc_no_complement(combined_estimates, outcomes)

# ---- Value bet calculation ----
value_rows = []
for o in outcomes:
    est = combined_estimates.get(o.key)

    if REQUIRE_MODEL_ONLY and not (est and est.is_model):
        continue

    if est is None:
        # No estimate — skip (was causing 0-edge false positives before)
        continue

    if est.confidence < MIN_CONFIDENCE:
        continue

    edge_pct = (
        (est.true_prob - o.market_prob) / o.market_prob * 100
        if o.market_prob > 0 else 0
    )
    if edge_pct < MIN_EDGE_PCT:
        continue

    b = (1.0 / o.market_prob) - 1  # decimal odds - 1
    kelly_full = max(0.0, (est.true_prob * (b + 1) - 1) / b) if b > 0 else 0
    kelly_pct = kelly_full * KELLY_FRACTION * 100
    if kelly_pct < MIN_KELLY_PCT:
        continue

    bet_size = BANKROLL * kelly_full * KELLY_FRACTION

    link = f"https://polymarket.com/event/{o.slug}" if o.slug else ""

    value_rows.append(
        {
            "Market": o.market_name[:70],
            "Side": o.outcome,
            "PM Prob%": round(o.market_prob * 100, 1),
            "Model Prob%": round(est.true_prob * 100, 1),
            "Edge%": round(edge_pct, 1),
            "Kelly%": round(kelly_pct, 2),
            "Conf": round(est.confidence, 2),
            "Bet $": round(bet_size),
            "Volume": f"${int(o.volume):,}" if o.volume else "N/A",
            "Source": est.source,
            "Link": link,
        }
    )

df = pd.DataFrame(value_rows)

# ================== UI ==================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Markets", stats["total"])
col2.metric("Open Markets", stats["open"])
col3.metric("After Volume Filter", stats["volume_pass"])
col4.metric("Outcomes Parsed", stats["final"])
col5.metric("Value Bets", len(df))

# Data source status
src_cols = st.columns(3)
src_cols[0].info(f"📊 ESPN/FIFA Ratings: {len(sports_data.ratings)} teams")
src_cols[1].info(f"🎯 Sharp Book Events: {len(sharp_by_event)}")
src_cols[2].info(f"🔍 Estimates Generated: {len(combined_estimates)}")

st.subheader(f"🔍 Value Bets Found: {len(df)}")

if df.empty:
    st.warning(
        "No value bets found with current filters. Try: lowering Minimum Edge %, "
        "lowering Minimum Volume, reducing Minimum Confidence, or unchecking "
        "'Require Model Estimate'."
    )
    if not ODDS_API_KEY:
        st.error("🔑 No THE_ODDS_API_KEY found — sharp odds comparison unavailable. "
                 "Add it to secrets.toml for best results.")
else:
    df_display = df.sort_values("Edge%", ascending=False).reset_index(drop=True)

    # Colour-code by confidence
    def highlight_conf(row):
        c = row["Conf"]
        if c >= 0.75:
            return ["background-color: #1a4a1a"] * len(row)
        elif c >= 0.55:
            return ["background-color: #3a3a10"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_display.style.apply(highlight_conf, axis=1),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn("Link", display_text="View"),
        },
    )
    st.download_button(
        "📥 Download CSV",
        df_display.to_csv(index=False).encode("utf-8"),
        "value_bets.csv",
        "text/csv",
    )

if DEBUG_MODE:
    with st.expander("🐛 Debug Info"):
        st.write("**Pipeline stats:**", stats)
        st.write(f"**Sharp teams indexed:** {len(sharp_by_team)}")
        st.write(f"**Sharp events indexed:** {len(sharp_by_event)}")
        st.write(f"**Model estimates:** {len(model_estimates)}")
        st.write(f"**Sharp estimates:** {len(sharp_estimates)}")
        st.write(f"**Combined estimates:** {len(combined_estimates)}")
        if sharp_by_team:
            st.write("**Sample sharp teams:**", list(sharp_by_team.keys())[:20])
        if outcomes:
            st.write("**Sample outcomes (first 5):**")
            for o in outcomes[:5]:
                st.write(f"  {o.event_name[:60]} | {o.outcome} | prob={o.market_prob:.2f} | participant={o.participant}")

st.caption("⚠️ Not financial advice • Independent Model + Sharp Odds Comparison • v4.0")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
