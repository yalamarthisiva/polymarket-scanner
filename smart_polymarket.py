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
MIN_CONFIDENCE  = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.40, 0.05)
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
    if market.get("closed", False) or str(market.get("closed")).lower() == "true":
        return False
    if market.get("active") is False or str(market.get("active")).lower() == "false":
        return False
    return True

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
                home = event.get("home_team", "")
                away = event.get("away_team", "")
                home_norm = normalize_name(home)
                away_norm = normalize_name(away)
                if not home_norm or not away_norm:
                    continue
                
                home_prices, away_prices = [], []
                for book in event.get("bookmakers", []):
                    for mkt in book.get("markets", []):
                        if mkt.get("key") != "h2h":
                            continue
                        for oc in mkt.get("outcomes", []):
                            p = optional_float(oc.get("price"))
                            if not p or p <= 1.0:
                                continue
                            n = normalize_name(oc.get("name", ""))
                            if n == home_norm:
                                home_prices.append(p)
                            elif n == away_norm:
                                away_prices.append(p)

                def median(lst):
                    if not lst: return None
                    s = sorted(lst); m = len(s) // 2
                    return s[m] if len(s) % 2 else (s[m-1]+s[m])/2

                hp, ap = median(home_prices), median(away_prices)
                if hp and ap:
                    event_data = {
                        "home": home, "away": away, 
                        "home_price": hp, "away_price": ap, "sport": sport
                    }
                    by_team[home_norm] = event_data
                    by_team[away_norm] = event_data
                    by_event[(home_norm, away_norm)] = event_data
        except Exception as e:
            st.sidebar.error(f"Odds API Error on {sport}: {e}")
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
    if not decimal_odds_list or any(o <= 1.0 for o in decimal_odds_list):
        return [0.5, 0.5]
    raw = [1.0/o for o in decimal_odds_list]
    total = sum(raw)
    return [p/total for p in raw] if total > 0 else [0.5, 0.5]

def is_plausible_match(market_prob: float, model_prob: float) -> bool:
    if model_prob <= 0 or market_prob <= 0:
        return False
    ratio = model_prob / market_prob
    inv_ratio = market_prob / model_prob
    if max(ratio, inv_ratio) > MAX_PROB_RATIO:
        return False
    if market_prob < 0.03 or market_prob > 0.97:
        return False
    return True


def match_sharp_odds_for_outcome(
    outcome: MarketOutcome,
    sharp_by_team: dict,
    sharp_by_event: dict,
) -> ProbabilityEstimate | None:
    if is_futures_market(outcome.event_name):
        return None

    title_norm = " " + normalize_name(outcome.event_name) + " "
    is_yes = outcome.outcome.upper() == "YES"
    part_norm = normalize_name(outcome.participant) if outcome.participant else ""

    # Dynamic Matcher Loop
    for (home_key, away_key), event in sharp_by_event.items():
        home_words = home_key.split()
        away_words = away_key.split()
        home_last = home_words[-1] if home_words else ""
        away_last = away_words[-1] if away_words else ""
        
        # Match cross-references
        has_home = home_key in title_norm or (home_last and f" {home_last} " in title_norm)
        has_away = away_key in title_norm or (away_last and f" {away_last} " in title_norm)
        
        if has_home and has_away:
            prices = [event.get("home_price"), event.get("away_price")]
            if None in prices: continue
            fair_probs = vig_free_prob(prices) # [home_prob, away_prob]
            
            # Identify targeted asset
            if part_norm and (part_norm in home_key or home_last in part_norm):
                true_prob = fair_probs[0] if is_yes else fair_probs[1]
            elif part_norm and (part_norm in away_key or away_last in part_norm):
                true_prob = fair_probs[1] if is_yes else fair_probs[0]
            else:
                # Fallback to positional appearance order
                h_idx = title_norm.find(home_last) if home_last else -1
                a_idx = title_norm.find(away_last) if away_last else -1
                is_home_first = h_idx < a_idx if (h_idx != -1 and a_idx != -1) else True
                
                if is_home_first:
                    true_prob = fair_probs[0] if is_yes else fair_probs[1]
                else:
                    true_prob = fair_probs[1] if is_yes else fair_probs[0]
                    
            true_prob = max(0.01, min(0.99, true_prob))
            if is_plausible_match(outcome.market_prob, true_prob):
                return ProbabilityEstimate(
                    true_prob=true_prob, 
                    source=f"Sharp ({event['home']} vs {event['away']})", 
                    is_model=False, confidence=0.85
                )
    return None


def estimate_probabilities_model(outcomes, sports_data: SportsModelData) -> dict:
    estimates = {}
    for o in outcomes:
        if is_futures_market(o.event_name): continue
        rating = lookup_team_rating(sports_data, o.league, o.participant)
        if not rating: continue
        
        opponent_rating = None
        teams = extract_teams_from_title(o.event_name)
        for t in teams:
            if normalize_name(t) != normalize_name(o.participant):
                opp_r = lookup_team_rating(sports_data, o.league, t)
                if opp_r:
                    opponent_rating = opp_r
                    break
                    
        is_yes = o.outcome.upper() == "YES"
        if opponent_rating:
            total = rating.base_rating + opponent_rating.base_rating
            tp = rating.base_rating / total if total > 0 else 0.5
        else:
            tp = rating.base_rating
            
        tp = tp if is_yes else (1.0 - tp)
        tp = max(0.01, min(0.99, tp))
        
        if is_plausible_match(o.market_prob, tp):
            estimates[o.key] = ProbabilityEstimate(
                true_prob=tp, source=f"Model ({rating.source})", is_model=True, confidence=0.60
            )
    return estimates


# ================== MAIN EXECUTION PIPELINE ==================
markets     = fetch_polymarket()
sports_data = fetch_sports_model() if USE_AUTO_MODEL else SportsModelData()
sharp_by_team, sharp_by_event = fetch_sharp_odds(ODDS_API_KEY)

outcomes: list[MarketOutcome] = []
stats = {
    "total": len(markets), "open": 0, "category_pass": 0, 
    "volume_pass": 0, "final": 0, "futures_skipped": 0,
}

for market in markets:
    if not is_open_market(market): continue
    stats["open"] += 1

    # Safe Tag Parsing Matrix
    tag_list = market.get("tags", [])
    tag_words = []
    if isinstance(tag_list, list):
        for t in tag_list:
            if isinstance(t, dict):
                tag_words.append(str(t.get("name", "")).lower())
                tag_words.append(str(t.get("slug", "")).lower())
            else:
                tag_words.append(str(t).lower())

    category = str(market.get("category") or "").lower()
    sport_field = str(market.get("sport") or "").lower()
    league_field = str(market.get("league") or "").lower()
    title_or_q = (str(market.get("question", "")) + " " + str(market.get("title", ""))).lower()
    
    is_sports_market = (
        category == "sports" or
        sport_field != "" or
        league_field != "" or
        "sports" in tag_words or
        any(s in league_field for s in ["nba", "nfl", "mlb", "nhl", "soccer", "ufc", "tennis", "fifa"]) or
        any(s in title_or_q for s in ["nba", "nfl", "mlb", "nhl", "ufc", "champions league", "premier league", "world cup", "soccer", "basketball", "baseball"])
    )

    if not CAT_ALL and CAT_SPORTS and not is_sports_market: 
        continue
            
    stats["category_pass"] += 1

    # Dynamic Variable Harvesting (Bypasses API field shifts)
    valid_vols = []
    for k, v in market.items():
        if "volume" in k.lower() or "liquidity" in k.lower():
            val = optional_float(v)
            if val is not None: valid_vols.append(val)
            
    current_volume = max(valid_vols) if valid_vols else 0.0
    if current_volume < MIN_VOLUME: continue
    stats["volume_pass"] += 1

    event_name = clear_market_name(market)
    pairs = []
    
    # Format parsing matrices
    outcomes_list = market.get("outcomes")
    prices_list = market.get("outcomePrices")
    if isinstance(prices_list, str):
        try: prices_list = json.loads(prices_list)
        except Exception: pass
    if isinstance(outcomes_list, str):
        try: outcomes_list = json.loads(outcomes_list)
        except Exception: pass
        
    if isinstance(outcomes_list, list) and isinstance(prices_list, list):
        for name, pr in zip(outcomes_list, prices_list):
            p_val = optional_float(pr)
            if name and p_val is not None: pairs.append((str(name), p_val))
                
    if not pairs and isinstance(market.get("tokens"), list):
        for tok in market["tokens"]:
            if isinstance(tok, dict):
                n = tok.get("outcome") or tok.get("title")
                p = tok.get("price")
                if n and p is not None: pairs.append((str(n), optional_float(p)))

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

        league_val = league_field if league_field else (sport_field if sport_field else "sports")
        slug = market.get("slug") or market.get("id") or market.get("conditionId","")

        outcomes.append(MarketOutcome(
            key=f"{slug}::{outcome_text.lower()}",
            market_id=str(market.get("id") or market.get("conditionId","")),
            slug=str(slug), category=category, event_name=event_name,
            market_name=event_name, outcome=outcome_text, participant=participant,
            league=league_val, record="", market_prob=price, volume=current_volume,
            liquidity=optional_float(market.get("liquidityNum") or market.get("liquidity")),
        ))

stats["final"] = len(outcomes)
stats["futures_skipped"] = sum(1 for o in outcomes if is_futures_market(o.event_name))

model_estimates = estimate_probabilities_model(outcomes, sports_data) if USE_AUTO_MODEL else {}
sharp_estimates = {}
for o in outcomes:
    if o.key in model_estimates: continue
    est = match_sharp_odds_for_outcome(o, sharp_by_team, sharp_by_event)
    if est: sharp_estimates[o.key] = est

combined = {**sharp_estimates, **model_estimates}

value_rows = []
for o in outcomes:
    est = combined.get(o.key)
    if REQUIRE_MODEL_ONLY and not (est and est.is_model): continue
    if est is None or est.confidence < MIN_CONFIDENCE: continue

    edge_pct = (est.true_prob - o.market_prob) / o.market_prob * 100 if o.market_prob > 0 else 0
    if edge_pct < MIN_EDGE_PCT or edge_pct > MAX_EDGE_PCT: continue

    b = (1.0 / o.market_prob) - 1
    kelly_full = max(0.0, (est.true_prob * (b + 1) - 1) / b) if b > 0 else 0
    kelly_pct  = kelly_full * KELLY_FRACTION * 100
    if kelly_pct < MIN_KELLY_PCT: continue

    bet_size = BANKROLL * kelly_full * KELLY_FRACTION
    link     = f"https://polymarket.com/event/{o.slug}" if o.slug else ""

    value_rows.append({
        "Market":       o.market_name[:75],
        "Side":         o.outcome,
        "PM Prob%":     round(o.market_prob * 100, 1),
        "True Prob%":   round(est.true_prob  * 100, 1),
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
col3.metric("After Filters",     stats["volume_pass"])
col4.metric("Outcomes Parsed",   stats["final"])
col5.metric("✅ Value Bets",     len(df))

src_cols = st.columns(4)
src_cols[0].info(f"📊 Model Teams Indexed: {len(sports_data.ratings)}")
src_cols[1].info(f"🎯 Sharp Line Matchers: {len(sharp_by_event)}")
src_cols[2].info(f"🔍 Pipeline Estimates: {len(combined)}")
src_cols[3].warning(f"🚫 Futures Hidden: {stats['futures_skipped']}")

st.subheader(f"🔍 Value Bets Found: {len(df)}")

if df.empty:
    st.warning("No valuation edges found matching current configuration thresholds.")
else:
    df_display = df.sort_values("Edge%", ascending=False).reset_index(drop=True)
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open Market")},
    )

    sc = st.columns(4)
    sc[0].metric("Avg Edge%",    f"{df_display['Edge%'].mean():.1f}%")
    sc[1].metric("Avg Conf",     f"{df_display['Conf'].mean():.2f}")
    sc[2].metric("Total Allocation",  f"${df_display['Bet $'].sum():,.0f}")
    sc[3].metric("Max Kelly%",   f"{df_display['Kelly%'].max():.2f}%")

if DEBUG_MODE:
    with st.expander("🐛 Raw Debug Matrix"):
        st.write("Execution Steps Data:", stats)
        if outcomes:
            st.write("First 5 Outcomes Sample Log:")
            for o in outcomes[:5]:
                est = combined.get(o.key)
                st.write(f"Title: {o.event_name} | Price: {o.market_prob} | Target: {o.participant} | Match: {est.source if est else 'None'}")

st.caption("⚠️ Automated scanning array • Sharp Odds Execution Layout • v5.3")
