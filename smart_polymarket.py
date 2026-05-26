import json
import re
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

FUTURES_KEYWORDS = [
    "world series", "super bowl", "nba champion", "nfl champion",
    "stanley cup", "championship", "league champion", "win the league",
    "win the division", "make the playoffs", "reach the playoffs",
    "make playoffs", "advance to", "win the cup", "mvp", "cy young",
    "heisman", "golden boot", "ballon d'or", "season wins",
    "regular season", "win the season", "finish first", "win their division",
    "qualify for", "relegated", "promoted", "finish top", "finish bottom",
    "presidential", "gubernatorial", "election", "senate", "congress",
    "primary", "governor", "mayor", "president",
]

MAX_PROB_RATIO = 8.0   
MAX_EDGE_PCT   = 400.0 

st.set_page_config(page_title="Polymarket Sports Scanner", layout="wide")
st.title("🏆 Polymarket Sports Scanner + Sharp Odds")

# ================== SECRETS ==================
def get_odds_api_key():
    try:
        if "THE_ODDS_API_KEY" in st.secrets:
            return st.secrets["THE_ODDS_API_KEY"]
        return None
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

MIN_EDGE_PCT    = st.sidebar.number_input("Minimum Edge (%)",   value=2.0,  step=0.5)
MIN_KELLY_PCT   = st.sidebar.number_input("Minimum Kelly (%)",  value=0.1,  step=0.05)
MIN_VOLUME      = st.sidebar.number_input("Minimum Volume ($)", value=1000, step=1000)
MIN_CONFIDENCE  = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.50, 0.05)
KELLY_FRACTION  = st.sidebar.slider("Kelly Fraction", 0.05, 1.0, 0.25, 0.05)

USE_AUTO_MODEL    = st.sidebar.checkbox("Use Sport Model",      value=True)
REQUIRE_MODEL_ONLY = st.sidebar.checkbox("Require Model Estimate", value=False)

st.sidebar.markdown("---")
CAT_ALL    = st.sidebar.checkbox("All Categories", value=False)
CAT_SPORTS = st.sidebar.checkbox("Sports Only",    value=True)
SIDE_FILTER = st.sidebar.radio("Bet Side", ["Both", "YES only", "NO only"], index=0)
DEBUG_MODE  = st.sidebar.checkbox("Debug Info", value=False)

# ================== HELPERS ==================
def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

def optional_float(value):
    try:
        if value in (None, "", "null", "None"):
            return None
        return float(value)
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
    if market.get("active") is False:
        return False
    return True

def event_group_key(outcome) -> str:
    name = outcome.event_name.lower()
    name = re.sub(r"\s*-\s*\d+-\d+.*$", "", name)
    name = re.sub(r"\s*-\s*(yes|no)$", "", name, flags=re.I)
    return normalize_name(name)

def is_futures_market(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in FUTURES_KEYWORDS)

def extract_teams_from_title(title: str) -> list[str]:
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
            if isinstance(batch, list):
                markets.extend(batch)
                if len(batch) < 100:
                    break
            else:
                items = batch.get("markets", batch.get("data", []))
                if not items:
                    break
                markets.extend(items)
            offset += 100
        except Exception as e:
            st.sidebar.error(f"Polymarket API offset {offset} failure: {e}")
            break
    return markets


@st.cache_data(ttl=900)
def fetch_sports_model():
    ratings = {}
    leagues = [("nba", "basketball/nba"), ("nhl", "hockey/nhl"), ("nfl", "football/nfl")]
    for league, path in leagues:
        try:
            r = requests.get(
                f"{ESPN_BASE}/sports/{path}/standings",
                params={"region": "us", "lang": "en"}, timeout=15,
            )
            r.raise_for_status()
            ratings.update(parse_espn_standings(r.json(), league))
        except Exception:
            pass
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/standings",
            params={"leagueId": "103,104", "season": datetime.now().year}, timeout=15,
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
    if not api_key:
        return {}, {}
    by_team, by_event = {}, {}
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
                    if not lst: return None
                    s = sorted(lst); m = len(s) // 2
                    return s[m] if len(s) % 2 else (s[m-1]+s[m])/2

                hp, ap, dp = median(home_prices), median(away_prices), median(draw_prices)
                if hp: by_team[home] = {"decimal": hp, "home": home, "away": away, "sport": sport}
                if ap: by_team[away] = {"decimal": ap, "home": home, "away": away, "sport": sport}
                by_event[(home, away)] = {"home_price": hp, "away_price": ap,
                                          "draw_price": dp, "sport": sport}
        except Exception as e:
            st.sidebar.error(f"Odds API Error: {e}")
            pass
    return by_team, by_event


# ================== PARSERS ==================
def parse_espn_standings(payload: dict, league: str):
    ratings = {}
    for entry in iter_espn_entries(payload):
        team = entry.get("team", {})
        stats_dict = {s.get("name"): s.get("value") for s in entry.get("stats", [])}
        wins   = optional_float(stats_dict.get("wins"))   or 0
        losses = optional_float(stats_dict.get("losses")) or 0
        ties   = optional_float(stats_dict.get("ties"))   or 0
        games  = wins + losses + ties
        if games == 0: continue
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
            wins   = optional_float(tr.get("wins")) or 0
            losses = optional_float(tr.get("losses")) or 0
            rf = optional_float(tr.get("runsFor"))      or 0
            ra = optional_float(tr.get("runsAllowed"))  or 0
            if wins + losses == 0: continue
            pyth = (rf**2)/(rf**2+ra**2) if (rf+ra)>0 else wins/(wins+losses)
            rating = TeamRating(
                name=team.get("name",""), league="mlb",
                base_rating=max(0.01, min(0.99, pyth)),
                stats=TeamStats(wins=wins, losses=losses, points_for=rf, points_against=ra),
                source="MLB Pythagorean",
            )
            ratings[("mlb", normalize_name(team.get("name","")))] = rating
    return ratings

def parse_fifa_rankings(payload):
    ratings = {}
    for row in payload.get("Results", []):
        points = optional_float(row.get("DecimalTotalPoints") or row.get("TotalPoints"))
        name_obj = row.get("TeamName")
        name = name_obj.get("Description") if isinstance(name_obj, dict) else name_obj
        if not name or points is None: continue
        norm = max(0.01, min(0.99, (points - 1100) / 900))
        rating = TeamRating(name=name, league="fifawc", base_rating=norm,
                            stats=TeamStats(), source="FIFA")
        country = str(row.get("IdCountry",""))
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


# ================== PROBABILITY ESTIMATION ==================
def vig_free_prob(decimal_odds_list: list[float]) -> list[float]:
    if not decimal_odds_list or any(o <= 0 for o in decimal_odds_list):
        return []
    raw = [1.0/o for o in decimal_odds_list]
    total = sum(raw)
    return [p/total for p in raw]


def is_plausible_match(market_prob: float, model_prob: float) -> bool:
    if model_prob <= 0 or market_prob <= 0:
        return False
    ratio = model_prob / market_prob
    inv_ratio = market_prob / model_prob
    if max(ratio, inv_ratio) > MAX_PROB_RATIO:
        return False
    if market_prob < 0.04:
        return False
    return True


def match_sharp_odds_for_outcome(
    outcome: MarketOutcome,
    sharp_by_team: dict,
    sharp_by_event: dict,
) -> ProbabilityEstimate | None:
    if is_futures_market(outcome.event_name):
        return None

    is_yes = outcome.outcome.upper() == "YES"
    participant = normalize_name(outcome.participant) if outcome.participant else ""

    def make_estimate(true_prob: float, source: str, conf: float) -> ProbabilityEstimate | None:
        tp = true_prob if is_yes else 1 - true_prob
        tp = max(0.01, min(0.99, tp))
        if not is_plausible_match(outcome.market_prob, tp):
            return None
        return ProbabilityEstimate(true_prob=tp, source=source, is_model=False, confidence=conf)

    def get_fair_prob(home: str, away: str, side: str) -> float | None:
        event = sharp_by_event.get((home, away), {})
        prices = [p for p in [event.get("home_price"), event.get("away_price")] if p]
        if len(prices) < 2:
            return None
        fair = vig_free_prob(prices)
        return fair[0] if side == "home" else fair[1]

    if participant and participant in sharp_by_team:
        entry = sharp_by_team[participant]
        home, away = entry["home"], entry["away"]
        side = "home" if normalize_name(home) == participant else "away"
        fp = get_fair_prob(home, away, side)
        if fp:
            return make_estimate(fp, "Sharp Books (direct)", conf=0.82)

    teams = extract_teams_from_title(outcome.event_name)
    for team_raw in teams:
        team_norm = normalize_name(team_raw)
        for key, entry in sharp_by_team.items():
            if len(min(team_norm, key, key=len)) <= 3:
                continue
            if team_norm in key or key in team_norm:
                home, away = entry["home"], entry["away"]
                matched_home = normalize_name(home)
                side = "home" if (team_norm in matched_home or matched_home in team_norm) else "away"
                fp = get_fair_prob(home, away, side)
                if fp:
                    return make_estimate(fp, "Sharp Books (fuzzy)", conf=0.68)

    title_tokens = set(normalize_name(outcome.event_name).split())
    stop = {"the","a","an","to","of","in","and","or","for","is","will","who","vs","at"}
    sig_tokens = title_tokens - stop

    best_overlap, best_event_key = 0, None
    for (home, away) in sharp_by_event:
        event_tokens = set((home + " " + away).split()) - stop
        overlap = len(sig_tokens & event_tokens)
        if overlap >= 3 and overlap > best_overlap:
            best_overlap = overlap
            best_event_key = (home, away)

    if best_event_key:
        home, away = best_event_key
        event = sharp_by_event[best_event_key]
        prices = [p for p in [event.get("home_price"), event.get("away_price")] if p]
        if len(prices) == 2:
            fair = vig_free_prob(prices)
            home_in_title = home in " ".join(list(sig_tokens)[:6])
            fp = fair[0] if home_in_title else fair[1]
            return make_estimate(fp, "Sharp Books (event match)", conf=0.55)

    return None


def estimate_probabilities_model(outcomes, sports_data: SportsModelData) -> dict:
    estimates = {}
    groups = defaultdict(list)
    for o in outcomes:
        if o.outcome.upper() == "YES":
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
                if o.key in strengths and total > 0:
                    tp = strengths[o.key] / total
                    if is_plausible_match(o.market_prob, tp):
                        estimates[o.key] = ProbabilityEstimate(
                            true_prob=tp,
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


# ================== MAIN EXECUTION PIPELINE ==================
markets     = fetch_polymarket()
sports_data = fetch_sports_model() if USE_AUTO_MODEL else SportsModelData()
sharp_by_team, sharp_by_event = fetch_sharp_odds(ODDS_API_KEY)

outcomes: list[MarketOutcome] = []
stats = {
    "total": len(markets), "open": 0,
    "category_pass": 0, "volume_pass": 0, "final": 0,
    "futures_skipped": 0,
}

for market in markets:
    if not is_open_market(market): continue
    stats["open"] += 1

    # ========================================================
    # V5.2 RESILIENT CATEGORY & SPORT DETECTION
    # ========================================================
    category = str(market.get("category") or "").lower()
    sport_field = str(market.get("sport") or "").lower()
    league_field = str(market.get("league") or "").lower()
    
    # Harvest any embedded tag tags or labels
    tags = [str(t).lower() for t in market.get("tags", [])]
    
    is_sports_market = (
        category == "sports" or 
        sport_field != "" or 
        league_field != "" or 
        "sports" in tags or
        any(s in league_field for s in ["nba", "nfl", "mlb", "nhl", "soccer", "ufc", "tennis", "fifa"])
    )

    if not CAT_ALL:
        if CAT_SPORTS and not is_sports_market: 
            continue
            
    stats["category_pass"] += 1

    # ========================================================
    # V5.1 AGGRESSIVE VOLUME EXTRACTION FIX
    # ========================================================
    raw_vols = [
        market.get("volumeNum"), 
        market.get("volume"), 
        market.get("volume24hr"),
        market.get("liquidityNum"), 
        market.get("liquidity")
    ]
    
    valid_vols = []
    for v in raw_vols:
        val = optional_float(v)
        if val is not None:
            valid_vols.append(val)
            
    current_volume = max(valid_vols) if valid_vols else 0.0
    
    if current_volume < MIN_VOLUME: continue
    stats["volume_pass"] += 1

    event_name = clear_market_name(market)

    tokens             = market.get("tokens", [])
    outcome_names      = market.get("outcomes", [])
    outcome_prices_raw = market.get("outcomePrices", [])

    pairs: list[tuple[str, float]] = []
    if tokens:
        for tok in tokens:
            name  = tok.get("outcome", tok.get("title", ""))
            price = optional_float(tok.get("price"))
            if name and price is not None:
                pairs.append((name, price))
                
    if not pairs and outcome_names and outcome_prices_raw:
        if isinstance(outcome_prices_raw, str):
            try: outcome_prices_raw = json.loads(outcome_prices_raw)
            except Exception: outcome_prices_raw = []
        if isinstance(outcome_names, str):
            try: outcome_names = json.loads(outcome_names)
            except Exception: outcome_names = []
        for name, price_str in zip(outcome_names, outcome_prices_raw):
            price = optional_float(price_str)
            if name and price is not None:
                pairs.append((name, price))
                
    if not pairs:
        for side in market.get("marketSides", []):
            price = optional_float(side.get("price"))
            name  = side.get("description") or ("Yes" if side.get("long", True) else "No")
            if price is not None:
                pairs.append((name, price))

    for outcome_text, price in pairs:
        if not (0.01 <= price <= 0.99): continue
        if SIDE_FILTER == "YES only" and outcome_text.upper() != "YES": continue
        if SIDE_FILTER == "NO only"  and outcome_text.upper() != "NO":  continue

        extracted = extract_teams_from_title(event_name)
        if extracted:
            participant = extracted[0] if outcome_text.upper() == "YES" else (
                extracted[1] if len(extracted) > 1 else extracted[0])
        else:
            participant = outcome_text

        league_val = league_field if league_field else sport_field
        slug   = market.get("slug") or market.get("id") or market.get("conditionId","")

        outcomes.append(MarketOutcome(
            key=f"{slug}::{outcome_text.lower()}",
            market_id=str(market.get("id") or market.get("conditionId","")),
            slug=str(slug),
            category=category if category else "sports",
            event_name=event_name,
            market_name=event_name,
            outcome=outcome_text,
            participant=participant,
            league=league_val,
            record="",
            market_prob=price,
            volume=current_volume,
            liquidity=optional_float(market.get("liquidityNum") or market.get("liquidity")),
        ))

stats["final"] = len(outcomes)
stats["futures_skipped"] = sum(1 for o in outcomes if is_futures_market(o.event_name))

model_estimates = estimate_probabilities_model(outcomes, sports_data) if USE_AUTO_MODEL else {}

sharp_estimates = {}
for o in outcomes:
    if o.key in model_estimates: continue
    est = match_sharp_odds_for_outcome(o, sharp_by_team, sharp_by_event)
    if est:
        sharp_estimates[o.key] = est

combined = {**sharp_estimates, **model_estimates}
combined = calc_no_complement(combined, outcomes)

value_rows = []
for o in outcomes:
    est = combined.get(o.key)
    if REQUIRE_MODEL_ONLY and not (est and est.is_model): continue
    if est is None: continue
    if est.confidence < MIN_CONFIDENCE: continue

    edge_pct = (est.true_prob - o.market_prob) / o.market_prob * 100 if o.market_prob > 0 else 0
    if edge_pct < MIN_EDGE_PCT: continue
    if edge_pct > MAX_EDGE_PCT: continue

    b = (1.0 / o.market_prob) - 1
    kelly_full = max(0.0, (est.true_prob * (b + 1) - 1) / b) if b > 0 else 0
    kelly_pct  = kelly_full * KELLY_FRACTION * 100
    if kelly_pct < MIN_KELLY_PCT: continue

    bet_size = BANKROLL * kelly_full * KELLY_FRACTION
    link     = f"https://polymarket.com/event/{o.slug}" if o.slug else ""

    value_rows.append({
        "Market":       o.market_name[:70],
        "Side":         o.outcome,
        "PM Prob%":     round(o.market_prob * 100, 1),
        "Model Prob%":  round(est.true_prob  * 100, 1),
        "Edge%":        round(edge_pct, 1),
        "Kelly%":       round(kelly_pct, 2),
        "Conf":         round(est.confidence, 2),
        "Bet $":        round(bet_size),
        "Volume":       f"${int(o.volume):,}" if o.volume else "N/A",
        "Source":       est.source,
        "Link":         link,
    })

df = pd.DataFrame(value_rows)

# ================== UI DISPLAY LAYOUT ==================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Markets",     stats["total"])
col2.metric("Open Markets",      stats["open"])
col3.metric("After Vol. Filter", stats["volume_pass"])
col4.metric("Outcomes Parsed",   stats["final"])
col5.metric("✅ Value Bets",     len(df))

src_cols = st.columns(4)
src_cols[0].info(f"📊 ESPN/FIFA Teams: {len(sports_data.ratings)}")
src_cols[1].info(f"🎯 Sharp Events: {len(sharp_by_event)}")
src_cols[2].info(f"🔍 Estimates: {len(combined)}")
src_cols[3].warning(f"🚫 Futures filtered: {stats['futures_skipped']}")

st.subheader(f"🔍 Value Bets Found: {len(df)}")

if df.empty:
    st.warning(
        "No value bets found. Try: lower Minimum Edge %, lower Minimum Confidence "
        "(currently guards against false positives), or reduce Minimum Volume."
    )
    if not ODDS_API_KEY:
        st.error("🔑 No THE_ODDS_API_KEY — ensure it is correctly named and formatted in secrets.toml.")
else:
    df_display = df.sort_values("Edge%", ascending=False).reset_index(drop=True)

    def highlight_conf(row):
        c = row["Conf"]
        if c >= 0.75: return ["background-color: #1a4a1a"] * len(row)
        if c >= 0.60: return ["background-color: #2a3a10"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_display.style.apply(highlight_conf, axis=1),
        use_container_width=True,
        hide_index=True,
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="View")},
    )

    sc = st.columns(4)
    sc[0].metric("Avg Edge%",    f"{df_display['Edge%'].mean():.1f}%")
    sc[1].metric("Avg Conf",     f"{df_display['Conf'].mean():.2f}")
    sc[2].metric("Total Bet $",  f"${df_display['Bet $'].sum():,.0f}")
    sc[3].metric("Max Kelly%",   f"{df_display['Kelly%'].max():.2f}%")

    st.download_button(
        "📥 Download CSV",
        df_display.to_csv(index=False).encode("utf-8"),
        "value_bets.csv", "text/csv",
    )

if DEBUG_MODE:
    with st.expander("🐛 Debug Info"):
        st.write("**Pipeline stats:**", stats)
        st.write(f"**Sharp teams indexed:** {len(sharp_by_team)}")
        st.write(f"**Sharp events indexed:** {len(sharp_by_event)}")
        st.write(f"**Model estimates:** {len(model_estimates)}")
        st.write(f"**Sharp estimates:** {len(sharp_estimates)}")
        st.write(f"**Combined:** {len(combined)}")
        if sharp_by_team:
            st.write("**Sample sharp teams:**", list(sharp_by_team.keys())[:20])
        if outcomes:
            st.write("**Sample outcomes (first 8):**")
            for o in outcomes[:8]:
                est = combined.get(o.key)
                is_fut = is_futures_market(o.event_name)
                st.write(f"  {'🚫FUTURE' if is_fut else '✅GAME  '} "
                         f"{o.event_name[:55]} | {o.outcome} "
                         f"pm={o.market_prob:.2f} "
                         f"model={est.true_prob if est else 0.0:.2f} "
                         f"src={est.source if est else '—'}")

st.caption("⚠️ Not financial advice • Sharp Odds + Plausibility Guards • v5.2")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
