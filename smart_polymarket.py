"""
Smart Polymarket Value Tool - fixed / current API version

Run:
    pip install streamlit pandas requests
    streamlit run smart_polymarket_value_tool_fixed.py

What this version fixes:
- Uses Polymarket Gamma API for discovery instead of the stale/fragile gateway URL.
- Handles Gamma /events and /markets response shapes, including paginated list payloads.
- Parses outcomes, outcomePrices, clobTokenIds, event metadata, tags, volume, and liquidity defensively.
- Optionally refreshes buy prices from the public CLOB orderbook using token_id.
- Avoids treating unsupported/no-model markets as value bets unless you explicitly allow them.
- Improves participant/league inference so sports rating models have a better chance to match.

Important: this app is an analysis tool, not financial advice and not an automated trading bot.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import pandas as pd
import requests
import streamlit as st


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
FIFA_MENS_RANKINGS_URL = "https://api.fifa.com/api/v3/rankings"

REQUEST_HEADERS = {
    "User-Agent": "smart-polymarket-value-tool/1.1",
    "Accept": "application/json",
}

FIFA_RATING_BASELINE = 1500.0
FIFA_ELO_SCALE = 400.0

FIFA_COUNTRY_ALIASES = {
    "ARG": ["Argentina"],
    "BEL": ["Belgium"],
    "BRA": ["Brazil"],
    "CAN": ["Canada"],
    "CPV": ["Cabo Verde", "Cape Verde"],
    "CRO": ["Croatia"],
    "CZE": ["Czechia", "Czech Republic"],
    "ENG": ["England"],
    "ESP": ["Spain"],
    "FRA": ["France"],
    "GER": ["Germany"],
    "IRN": ["IR Iran", "Iran"],
    "ITA": ["Italy"],
    "JPN": ["Japan"],
    "KOR": ["Korea Republic", "South Korea", "Korea"],
    "KSA": ["Saudi Arabia"],
    "MAR": ["Morocco"],
    "MEX": ["Mexico"],
    "NED": ["Netherlands", "Holland"],
    "POR": ["Portugal"],
    "SUI": ["Switzerland"],
    "TUR": ["Turkiye", "Turkey"],
    "UAE": ["United Arab Emirates", "UAE"],
    "USA": ["USA", "United States", "United States of America", "US"],
    "URU": ["Uruguay"],
}


# ================== DOMAIN MODEL ==================
@dataclass(frozen=True)
class MarketOutcome:
    key: str
    market_id: str
    condition_id: str
    slug: str
    category: str
    event_name: str
    market_name: str
    outcome: str
    participant: str
    league: str
    record: str
    market_prob: float
    price_source: str
    token_id: str
    live_bid: float | None
    live_ask: float | None
    live_mid: float | None
    live_spread_pct: float | None
    volume: float | None
    liquidity: float | None
    end_date: str


@dataclass(frozen=True)
class ProbabilityEstimate:
    true_prob: float
    source: str
    model_probability: float | None = None
    market_prior: float | None = None
    confidence: float = 0.0
    effective_model_weight: float = 0.0
    max_shift_pp: float = 0.0


class ProbabilityProvider(Protocol):
    def get(self, outcome: MarketOutcome) -> ProbabilityEstimate | None:
        ...


@dataclass
class AnalysisConfig:
    bankroll: float
    kelly_fraction: float
    min_edge_pct: float
    min_edge_pp: float
    min_kelly_pct: float
    min_market_prob: float
    max_market_prob: float
    min_volume: float
    min_liquidity: float
    max_bet_pct: float
    max_live_spread_pct: float
    side_filter: str
    sort_by: str
    sort_ascending: bool
    include_all_categories: bool
    include_sports: bool
    include_politics: bool
    include_crypto: bool
    include_culture: bool
    include_finance: bool
    use_auto_model: bool
    model_blend: float
    min_model_confidence: float
    max_model_shift_pp: float
    min_event_market_coverage: float
    use_market_consensus_fallback: bool
    require_actionable_model: bool
    allow_record_only_value: bool
    skip_politics_auto_model: bool
    use_live_clob_prices: bool
    live_clob_max_tokens: int
    fetch_pages: int
    page_size: int


@dataclass(frozen=True)
class ParsedOutcome:
    outcome: str
    price: float
    participant: str
    league: str
    record: str
    token_id: str
    live_bid: float | None = None
    live_ask: float | None = None
    live_mid: float | None = None
    live_spread_pct: float | None = None
    price_source: str = "gamma"


@dataclass(frozen=True)
class TeamRating:
    name: str
    league: str
    rating: float
    source: str
    games: float = 0.0
    wins: float | None = None
    losses: float | None = None
    draws_or_ot: float = 0.0
    confidence: float = 0.35


@dataclass(frozen=True)
class SportsModelData:
    ratings: dict[tuple[str, str], TeamRating]


@dataclass(frozen=True)
class RecordSignal:
    rating: float
    games: float
    wins: float
    losses: float
    draws_or_ot: float
    confidence: float


@dataclass(frozen=True)
class ModelSignal:
    strength: float
    source: str
    confidence: float


@dataclass
class AutoModelProbabilityProvider:
    by_key: dict[str, ProbabilityEstimate]

    def get(self, outcome: MarketOutcome) -> ProbabilityEstimate | None:
        return self.by_key.get(outcome.key)


# ================== UI ==================
st.set_page_config(page_title="Smart Polymarket Value Tool", layout="wide")
st.title("Smart Polymarket Value Tool")
st.info(
    "Uses Polymarket Gamma API for market discovery and optional public CLOB orderbook "
    "prices. Probability estimates are conservative heuristics, not guaranteed edge."
)

st.sidebar.header("Bankroll & Filters")
BANKROLL = st.sidebar.number_input("Bankroll ($)", value=10_000, min_value=100, step=500)
KELLY_FRACTION = st.sidebar.slider(
    "Kelly Fraction",
    min_value=0.05,
    max_value=1.0,
    value=0.20,
    step=0.05,
    help="0.20 = one-fifth Kelly. Use smaller fractions for noisy probability estimates.",
)
MIN_EDGE_PCT = st.sidebar.number_input(
    "Minimum Edge (%)",
    value=8.0,
    min_value=-100.0,
    max_value=500.0,
    step=1.0,
    help="Relative edge: (model true probability - buy price) / buy price.",
)
MIN_EDGE_PP = st.sidebar.number_input(
    "Minimum Edge (percentage points)",
    value=2.0,
    min_value=-100.0,
    max_value=100.0,
    step=0.5,
)
MIN_KELLY = st.sidebar.number_input(
    "Minimum Kelly % of bankroll", value=0.1, min_value=0.0, max_value=100.0, step=0.1
)
PROB_RANGE = st.sidebar.slider(
    "Buy probability range",
    min_value=0.01,
    max_value=0.99,
    value=(0.02, 0.95),
    step=0.01,
)
MIN_VOLUME = st.sidebar.number_input("Minimum Volume ($)", value=0.0, min_value=0.0, step=10_000.0)
MIN_LIQUIDITY = st.sidebar.number_input("Minimum Liquidity ($)", value=0.0, min_value=0.0, step=10_000.0)
MAX_BET_PCT = st.sidebar.slider(
    "Max suggested bet % of bankroll",
    min_value=0.1,
    max_value=10.0,
    value=1.5,
    step=0.1,
)
MAX_LIVE_SPREAD_PCT = st.sidebar.slider(
    "Max live spread %",
    min_value=1.0,
    max_value=100.0,
    value=25.0,
    step=1.0,
    help="Only applied when live CLOB bid/ask is available.",
)

SORT_BY = st.sidebar.selectbox(
    "Sort by",
    options=[
        "Edge %",
        "Kelly %",
        "EV %",
        "Suggested Bet ($)",
        "Buy Probability",
        "Live Spread %",
        "Market",
    ],
)
SORT_ASCENDING = st.sidebar.checkbox("Sort ascending", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Live Market Data")
USE_LIVE_CLOB_PRICES = st.sidebar.checkbox(
    "Use live CLOB orderbook buy prices",
    value=True,
    help="Uses public orderbook asks when token IDs are available. Slower, but closer to executable pricing.",
)
LIVE_CLOB_MAX_TOKENS = st.sidebar.slider(
    "Max live CLOB tokens per refresh", min_value=10, max_value=250, value=80, step=10
)
FETCH_PAGES = st.sidebar.slider("Gamma pages to fetch", min_value=1, max_value=10, value=5, step=1)
PAGE_SIZE = st.sidebar.slider("Gamma page size", min_value=25, max_value=100, value=100, step=25)

st.sidebar.markdown("---")
st.sidebar.subheader("Automation")
USE_AUTO_MODEL = st.sidebar.checkbox("Use automated true probabilities", value=True)
MODEL_BLEND = st.sidebar.slider(
    "Model weight",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Maximum model weight before confidence shrinkage. Actual weight is lower when data quality is weak.",
)
MIN_MODEL_CONFIDENCE = st.sidebar.slider(
    "Minimum model confidence",
    min_value=0.00,
    max_value=1.00,
    value=0.18,
    step=0.01,
    help="Filters weak estimates. Low-confidence records/partial matchups will remain baseline instead of creating fake edge.",
)
MAX_MODEL_SHIFT_PP = st.sidebar.slider(
    "Max model move from market (pp)",
    min_value=1.0,
    max_value=25.0,
    value=8.0,
    step=0.5,
    help="Safety cap: after blending, the estimated true probability cannot move more than this many points from the de-vigged market prior.",
)
MIN_EVENT_MARKET_COVERAGE = st.sidebar.slider(
    "Minimum event market coverage",
    min_value=0.25,
    max_value=1.20,
    value=0.70,
    step=0.05,
)
USE_MARKET_CONSENSUS_FALLBACK = st.sidebar.checkbox(
    "Fill unsupported markets with no-edge market baseline", value=True
)
REQUIRE_ACTIONABLE_MODEL = st.sidebar.checkbox(
    "Exclude no-edge baseline from value bets", value=True
)
ALLOW_RECORD_ONLY_VALUE = st.sidebar.checkbox(
    "Allow record-only value bets",
    value=False,
    help="Record-only estimates are weak. Leave off for stricter recommendations.",
)
SKIP_POLITICS_AUTO_MODEL = st.sidebar.checkbox(
    "Skip politics automation until polling/news model exists", value=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("Categories")
CAT_ALL = st.sidebar.checkbox("Show All", value=True)
CAT_SPORTS = st.sidebar.checkbox("Sports", value=True)
CAT_POLITICS = st.sidebar.checkbox("Politics", value=True)
CAT_CRYPTO = st.sidebar.checkbox("Crypto", value=True)
CAT_CULTURE = st.sidebar.checkbox("Culture / Entertainment", value=False)
CAT_FINANCE = st.sidebar.checkbox("Finance / Economy", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Bet Side")
SIDE_FILTER = st.sidebar.radio(
    "Show bets",
    options=["YES only", "NO only", "Both"],
    index=2,
    help="Use Both for sports/multi-outcome markets where outcomes are team/player names instead of YES/NO.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data")
DEBUG_MODE = st.sidebar.checkbox("Debug: show raw rows", value=False)
AUTO_REFRESH = st.sidebar.checkbox("Auto Refresh", value=False)
REFRESH_MIN = st.sidebar.slider("Refresh every (minutes)", 3, 15, 5)

CONFIG = AnalysisConfig(
    bankroll=BANKROLL,
    kelly_fraction=KELLY_FRACTION,
    min_edge_pct=MIN_EDGE_PCT,
    min_edge_pp=MIN_EDGE_PP,
    min_kelly_pct=MIN_KELLY,
    min_market_prob=PROB_RANGE[0],
    max_market_prob=PROB_RANGE[1],
    min_volume=MIN_VOLUME,
    min_liquidity=MIN_LIQUIDITY,
    max_bet_pct=MAX_BET_PCT,
    max_live_spread_pct=MAX_LIVE_SPREAD_PCT,
    side_filter=SIDE_FILTER,
    sort_by=SORT_BY,
    sort_ascending=SORT_ASCENDING,
    include_all_categories=CAT_ALL,
    include_sports=CAT_SPORTS,
    include_politics=CAT_POLITICS,
    include_crypto=CAT_CRYPTO,
    include_culture=CAT_CULTURE,
    include_finance=CAT_FINANCE,
    use_auto_model=USE_AUTO_MODEL,
    model_blend=MODEL_BLEND,
    min_model_confidence=MIN_MODEL_CONFIDENCE,
    max_model_shift_pp=MAX_MODEL_SHIFT_PP,
    min_event_market_coverage=MIN_EVENT_MARKET_COVERAGE,
    use_market_consensus_fallback=USE_MARKET_CONSENSUS_FALLBACK,
    require_actionable_model=REQUIRE_ACTIONABLE_MODEL,
    allow_record_only_value=ALLOW_RECORD_ONLY_VALUE,
    skip_politics_auto_model=SKIP_POLITICS_AUTO_MODEL,
    use_live_clob_prices=USE_LIVE_CLOB_PRICES,
    live_clob_max_tokens=LIVE_CLOB_MAX_TOKENS,
    fetch_pages=FETCH_PAGES,
    page_size=PAGE_SIZE,
)


# ================== GENERIC HELPERS ==================
def optional_float(value: Any) -> float | None:
    try:
        if value in (None, "", "null", "None"):
            return None
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def clamp_probability(value: float | None, low: float = 0.001, high: float = 0.999) -> float | None:
    if value is None:
        return None
    return max(low, min(high, value))


def normalize_probability(value: Any) -> float | None:
    prob = optional_float(value)
    if prob is None:
        return None
    if prob > 1:
        prob /= 100
    if 0 <= prob <= 1:
        return prob
    return None


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def parse_jsonish_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        # Some upstream fields are comma-separated strings.
        if "," in text and not text.startswith("["):
            return [part.strip() for part in text.split(",") if part.strip()]
    return []


def first_present(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def response_to_list(payload: Any, preferred_keys: tuple[str, ...] = ("events", "markets", "data", "results")) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def tag_text(obj: dict) -> str:
    tags = obj.get("tags") or []
    pieces: list[str] = []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except json.JSONDecodeError:
            tags = [tags]
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict):
                pieces.append(first_present(tag.get("label"), tag.get("name"), tag.get("slug")))
            else:
                pieces.append(str(tag))
    return " ".join(piece for piece in pieces if piece)


# ================== POLYMARKET DATA LAYER ==================
@st.cache_data(ttl=45, show_spinner="Fetching Polymarket events from Gamma API...")
def fetch_gamma_events(page_size: int, pages: int) -> tuple[list[dict], dict[str, int | str]]:
    events: list[dict] = []
    stats: dict[str, int | str] = {"endpoint": "/events", "pages_requested": pages, "events_raw": 0}

    for page in range(pages):
        params = {
            "active": "true",
            "closed": "false",
            "limit": page_size,
            "offset": page * page_size,
            "order": "volume24hr",
            "ascending": "false",
        }
        response = requests.get(
            f"{GAMMA_BASE_URL}/events", params=params, headers=REQUEST_HEADERS, timeout=25
        )
        response.raise_for_status()
        batch = response_to_list(response.json(), preferred_keys=("events", "data", "results"))
        if not batch:
            break
        events.extend(batch)
        if len(batch) < page_size:
            break

    stats["events_raw"] = len(events)
    return events, stats


@st.cache_data(ttl=45, show_spinner="Fetching Polymarket markets from Gamma API...")
def fetch_gamma_markets(page_size: int, pages: int) -> tuple[list[dict], dict[str, int | str]]:
    markets: list[dict] = []
    stats: dict[str, int | str] = {"endpoint": "/markets", "pages_requested": pages, "markets_raw": 0}

    for page in range(pages):
        params = {
            "active": "true",
            "closed": "false",
            "limit": page_size,
            "offset": page * page_size,
            "order": "volumeNum",
            "ascending": "false",
        }
        response = requests.get(
            f"{GAMMA_BASE_URL}/markets", params=params, headers=REQUEST_HEADERS, timeout=25
        )
        response.raise_for_status()
        batch = response_to_list(response.json(), preferred_keys=("markets", "data", "results"))
        if not batch:
            break
        markets.extend(batch)
        if len(batch) < page_size:
            break

    stats["markets_raw"] = len(markets)
    return markets, stats


def flatten_event_markets(events: list[dict]) -> list[dict]:
    flattened: list[dict] = []
    for event in events:
        event_title = first_present(event.get("title"), event.get("question"), event.get("slug"))
        event_category = first_present(event.get("category"), event.get("type"), event.get("collectionType"))
        event_tags = tag_text(event)
        event_slug = first_present(event.get("slug"), event.get("id"))
        event_end = first_present(event.get("endDate"), event.get("endDateIso"))

        for market in event.get("markets") or []:
            if not isinstance(market, dict):
                continue
            merged = dict(market)
            merged["_event_title"] = event_title
            merged["_event_category"] = event_category
            merged["_event_tags"] = event_tags
            merged["_event_slug"] = event_slug
            merged["_event_end_date"] = event_end
            flattened.append(merged)
    return flattened


@st.cache_data(ttl=15, show_spinner=False)
def fetch_order_book(token_id: str) -> dict[str, Any] | None:
    if not token_id:
        return None
    try:
        response = requests.get(
            f"{CLOB_BASE_URL}/book",
            params={"token_id": token_id},
            headers=REQUEST_HEADERS,
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except requests.RequestException:
        return None


def best_price(levels: Any, side: str) -> float | None:
    if not isinstance(levels, list):
        return None
    prices = [optional_float(level.get("price")) for level in levels if isinstance(level, dict)]
    prices = [price for price in prices if price is not None]
    if not prices:
        return None
    return max(prices) if side == "bid" else min(prices)


def order_book_snapshot(token_id: str) -> tuple[float | None, float | None, float | None, float | None]:
    book = fetch_order_book(token_id)
    if not book:
        return None, None, None, None
    bid = best_price(book.get("bids"), "bid")
    ask = best_price(book.get("asks"), "ask")
    mid = None
    spread_pct = None
    if bid is not None and ask is not None and ask >= bid:
        mid = (bid + ask) / 2
        if mid > 0:
            spread_pct = ((ask - bid) / mid) * 100
    elif bid is not None:
        mid = bid
    elif ask is not None:
        mid = ask
    return bid, ask, mid, spread_pct


def fetch_market_data(config: AnalysisConfig) -> tuple[list[dict], dict[str, int | str]]:
    """Fetch via /events first because Polymarket docs recommend events for discovery."""
    try:
        events, stats = fetch_gamma_events(config.page_size, config.fetch_pages)
        markets = flatten_event_markets(events)
        if markets:
            stats["markets_flattened"] = len(markets)
            return markets, stats
    except Exception as exc:  # show fallback reason but keep app alive
        st.warning(f"Gamma /events fetch failed; falling back to /markets. Reason: {exc}")

    try:
        markets, stats = fetch_gamma_markets(config.page_size, config.fetch_pages)
        return markets, stats
    except Exception as exc:
        st.error(f"Gamma market fetch failed: {exc}")
        return [], {"endpoint": "failed", "error": str(exc)}


# ================== SPORTS DATA LAYER ==================
@st.cache_data(ttl=900, show_spinner="Fetching public sports standings...")
def fetch_sports_model_data() -> SportsModelData:
    ratings: dict[tuple[str, str], TeamRating] = {}

    for league, url, source in [
        (
            "nba",
            "https://site.web.api.espn.com/apis/v2/sports/basketball/nba/standings"
            "?region=us&lang=en&contentorigin=espn&type=0&level=2"
            "&sort=playoffseed%3Aasc",
            "ESPN NBA standings",
        ),
        (
            "nhl",
            "https://site.web.api.espn.com/apis/v2/sports/hockey/nhl/standings"
            "?region=us&lang=en&contentorigin=espn&type=0&level=2"
            "&sort=playoffseed%3Aasc",
            "ESPN NHL standings",
        ),
    ]:
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
            response.raise_for_status()
            ratings.update(parse_espn_team_ratings(response.json(), league, source))
        except requests.RequestException:
            continue

    try:
        response = requests.get(
            "https://statsapi.mlb.com/api/v1/standings",
            params={
                "leagueId": "103,104",
                "season": datetime.now(timezone.utc).year,
                "standingsTypes": "regularSeason",
            },
            headers=REQUEST_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        ratings.update(parse_mlb_team_ratings(response.json()))
    except requests.RequestException:
        pass

    try:
        response = requests.get(
            FIFA_MENS_RANKINGS_URL,
            params={"gender": "male"},
            headers=REQUEST_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        ratings.update(parse_fifa_team_ratings(response.json()))
    except requests.RequestException:
        pass

    return SportsModelData(ratings=ratings)


def rating_keys(league: str, *names: str) -> list[tuple[str, str]]:
    return [(league.lower(), normalize_name(name)) for name in names if str(name).strip()]


def add_team_rating(
    ratings: dict[tuple[str, str], TeamRating],
    league: str,
    rating: TeamRating,
    *aliases: str,
) -> None:
    for key in rating_keys(league, *aliases):
        ratings[key] = rating


def iter_espn_standing_entries(node: dict):
    standings = node.get("standings")
    if isinstance(standings, dict):
        for entry in standings.get("entries", []) or []:
            yield entry
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            yield from iter_espn_standing_entries(child)


def stat_float(stats: dict, *names: str) -> float | None:
    for name in names:
        value = optional_float(stats.get(name))
        if value is not None:
            return value
    return None


def bayesian_rate(
    wins: float,
    losses: float,
    draws_or_ot: float = 0.0,
    prior_rate: float = 0.50,
    prior_games: float = 12.0,
) -> float:
    games = wins + losses + draws_or_ot
    if games <= 0:
        return prior_rate
    effective_wins = wins + 0.5 * draws_or_ot
    return max(0.01, min(0.99, (effective_wins + prior_rate * prior_games) / (games + prior_games)))


def games_confidence(games: float, full_season_games: float) -> float:
    if games <= 0:
        return 0.0
    # Smoothly rises with sample size. Even a full season is not perfect predictive signal.
    return max(0.05, min(0.85, games / max(full_season_games, 1)))


def league_full_season_games(league: str) -> float:
    return {"nba": 82.0, "nhl": 82.0, "mlb": 162.0, "fifawc": 12.0}.get(league.lower(), 50.0)


def team_rating_from_record(
    name: str,
    league: str,
    wins: float,
    losses: float,
    draws_or_ot: float,
    source: str,
    point_diff_per_game: float | None = None,
) -> TeamRating | None:
    games = wins + losses + draws_or_ot
    if games <= 0:
        return None

    record_rate = bayesian_rate(wins, losses, draws_or_ot, prior_rate=0.50, prior_games=12.0)

    # Point differential is often more predictive than raw W/L. Use it only as a small adjustment.
    if point_diff_per_game is not None:
        # Convert margin into a probability-like strength shift; keep it bounded to avoid overfitting.
        pd_shift = max(-0.10, min(0.10, point_diff_per_game / 120.0))
        rating_value = max(0.01, min(0.99, record_rate + pd_shift))
        source = f"{source}; Bayesian record + point-diff adjustment"
    else:
        rating_value = record_rate
        source = f"{source}; Bayesian record"

    return TeamRating(
        name=name,
        league=league,
        rating=rating_value,
        source=source,
        games=games,
        wins=wins,
        losses=losses,
        draws_or_ot=draws_or_ot,
        confidence=games_confidence(games, league_full_season_games(league)),
    )


def parse_espn_team_ratings(payload: dict, league: str, source: str) -> dict[tuple[str, str], TeamRating]:
    ratings: dict[tuple[str, str], TeamRating] = {}
    for entry in iter_espn_standing_entries(payload):
        team = entry.get("team", {})
        stats = {
            stat.get("name"): stat.get("value")
            for stat in entry.get("stats", [])
            if isinstance(stat, dict)
        }
        wins = stat_float(stats, "wins", "overallWins") or 0.0
        losses = stat_float(stats, "losses", "overallLosses") or 0.0
        draws_or_ot = stat_float(stats, "otLosses", "ties", "draws") or 0.0
        games = wins + losses + draws_or_ot
        if games <= 0:
            continue

        point_diff = stat_float(
            stats,
            "pointDifferential",
            "pointsDifferential",
            "differential",
            "avgPointDifferential",
        )
        point_diff_per_game = None
        if point_diff is not None:
            # ESPN sometimes exposes total differential and sometimes average differential.
            point_diff_per_game = point_diff / games if abs(point_diff) > 40 else point_diff

        team_name = first_present(team.get("displayName"), team.get("name"))
        rating = team_rating_from_record(
            name=team_name,
            league=league,
            wins=wins,
            losses=losses,
            draws_or_ot=draws_or_ot,
            source=source,
            point_diff_per_game=point_diff_per_game,
        )
        if rating is None:
            continue
        add_team_rating(
            ratings,
            league,
            rating,
            team.get("displayName", ""),
            team.get("shortDisplayName", ""),
            team.get("name", ""),
            team.get("abbreviation", ""),
            team.get("location", ""),
        )
    return ratings

def parse_mlb_team_ratings(payload: dict) -> dict[tuple[str, str], TeamRating]:
    ratings: dict[tuple[str, str], TeamRating] = {}
    for record_group in payload.get("records", []) or []:
        for team_record in record_group.get("teamRecords", []) or []:
            team = team_record.get("team", {})
            wins = optional_float(team_record.get("wins")) or 0.0
            losses = optional_float(team_record.get("losses")) or 0.0
            games = wins + losses
            if games <= 0:
                continue
            rating = team_rating_from_record(
                name=team.get("name", ""),
                league="mlb",
                wins=wins,
                losses=losses,
                draws_or_ot=0.0,
                source="MLB Stats API standings",
            )
            if rating is None:
                continue
            add_team_rating(
                ratings,
                "mlb",
                rating,
                team.get("name", ""),
                team.get("abbreviation", ""),
                team.get("teamName", ""),
                team.get("clubName", ""),
            )
    return ratings

def localized_description(value: Any, locale: str = "en-GB") -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    fallback = ""
    for item in value:
        if not isinstance(item, dict):
            continue
        description = first_present(item.get("Description"), item.get("description"))
        if not fallback:
            fallback = description
        if item.get("Locale") == locale and description:
            return description
    return fallback


def parse_fifa_team_ratings(payload: dict) -> dict[tuple[str, str], TeamRating]:
    ratings: dict[tuple[str, str], TeamRating] = {}
    rows = payload.get("Results") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return ratings
    for row in rows:
        if not isinstance(row, dict):
            continue
        country_code = str(row.get("IdCountry") or "")
        name = localized_description(row.get("TeamName"))
        points = optional_float(row.get("DecimalTotalPoints") or row.get("TotalPoints"))
        if not name or points is None:
            continue
        pub_date = str(row.get("PubDate") or "")[:10]
        source = "FIFA men's ranking" + (f" {pub_date}" if pub_date else "")
        # Convert FIFA point differences into a relative strength. Normalization happens inside each event group.
        relative_strength = max(0.01, 10 ** ((points - FIFA_RATING_BASELINE) / FIFA_ELO_SCALE))
        rating = TeamRating(
            name=name,
            league="fifawc",
            rating=relative_strength,
            source=f"{source}; FIFA rating points",
            games=12.0,
            confidence=0.55,
        )
        add_team_rating(ratings, "fifawc", rating, name, country_code, *FIFA_COUNTRY_ALIASES.get(country_code, []))
    return ratings


# ================== MARKET PARSING ==================
def is_open_market(market: dict) -> bool:
    if market.get("closed") is True or market.get("archived") is True or market.get("hidden") is True:
        return False
    if market.get("active") is False:
        return False
    # Gamma markets usually do not include ep3Status. If it appears and is not OPEN, skip.
    ep3 = market.get("ep3Status")
    if ep3 and str(ep3).upper() != "OPEN":
        return False
    return True


def clear_market_name(market: dict) -> str:
    pieces = []
    for piece in (
        market.get("question"),
        market.get("title"),
        market.get("groupItemTitle"),
        market.get("subtitle"),
    ):
        text = first_present(piece)
        if text and text not in pieces:
            pieces.append(text)
    return " - ".join(pieces)


def event_name_for_market(market: dict) -> str:
    return first_present(market.get("_event_title"), market.get("eventTitle"), clear_market_name(market))


def infer_category(market: dict) -> str:
    text = " ".join(
        [
            first_present(market.get("category")),
            first_present(market.get("_event_category")),
            tag_text(market),
            first_present(market.get("_event_tags")),
            clear_market_name(market),
            event_name_for_market(market),
        ]
    ).lower()

    sports_words = ["sports", "nba", "nfl", "nhl", "mlb", "soccer", "football", "tennis", "ufc", "fifa"]
    politics_words = ["politic", "election", "senate", "congress", "president", "trump", "biden"]
    crypto_words = ["crypto", "bitcoin", "ethereum", "solana", "btc", "eth", "token"]
    culture_words = ["culture", "entertainment", "movie", "music", "oscars", "grammy", "celebrity"]
    finance_words = ["finance", "economy", "fed", "inflation", "rates", "stock", "nasdaq", "s&p", "dollar"]

    if any(word in text for word in sports_words):
        return "sports"
    if any(word in text for word in politics_words):
        return "politics"
    if any(word in text for word in crypto_words):
        return "crypto"
    if any(word in text for word in culture_words):
        return "culture"
    if any(word in text for word in finance_words):
        return "finance"
    return first_present(market.get("category"), market.get("_event_category"), "unknown").lower()


def infer_league(market: dict, participant: str = "") -> str:
    text = " ".join(
        [
            first_present(market.get("category")),
            first_present(market.get("_event_category")),
            tag_text(market),
            first_present(market.get("_event_tags")),
            clear_market_name(market),
            event_name_for_market(market),
            participant,
        ]
    ).lower()
    if "nba" in text or "basketball" in text:
        return "nba"
    if "nhl" in text or "hockey" in text:
        return "nhl"
    if "mlb" in text or "baseball" in text:
        return "mlb"
    if "fifa" in text or "world cup" in text or "uefa" in text or "soccer" in text:
        return "fifawc"
    return ""


def infer_participant(market: dict, outcome_name: str) -> str:
    direct = first_present(
        market.get("groupItemTitle"),
        market.get("shortTitle"),
        market.get("teamName"),
        market.get("teamAName"),
        market.get("teamBName"),
    )
    if direct:
        return direct

    if outcome_name.strip().lower() not in {"yes", "no"}:
        return outcome_name.strip()

    question = first_present(market.get("question"), market.get("title"))
    patterns = [
        r"^will\s+(.+?)\s+(?:win|beat|make|reach|qualify|score|be)",
        r"^will\s+(.+?)\?*$",
        r"^(.+?)\s+to\s+win\b",
        r"^(.+?)\s+winner\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" ?:-")
            if 2 <= len(candidate) <= 80:
                return candidate
    return ""


def parse_record_from_text(*values: str) -> str:
    text = " ".join(values)
    match = re.search(r"\b(\d{1,3}-\d{1,3}(?:-\d{1,3})?)\b", text)
    return match.group(1) if match else ""


def market_side_price(side: dict) -> float | None:
    for key in ("price", "lastPrice", "bestAsk", "bestBid"):
        prob = normalize_probability(side.get(key))
        if prob is not None:
            return prob
    quote = side.get("quote") or side.get("bestAskQuote") or side.get("bestBidQuote")
    if isinstance(quote, dict):
        return normalize_probability(quote.get("value"))
    return None


def market_outcomes(market: dict, config: AnalysisConfig) -> list[ParsedOutcome]:
    parsed: list[ParsedOutcome] = []

    # Some newer/alternate payloads expose explicit sides. Keep this branch for compatibility.
    sides = market.get("marketSides")
    if isinstance(sides, list) and sides:
        for side in sides:
            if not isinstance(side, dict):
                continue
            name = first_present(side.get("description"), "Yes" if side.get("long") else "No")
            gamma_price = market_side_price(side)
            if gamma_price is None:
                continue
            team = side.get("team") or {}
            participant = first_present(
                team.get("safeName"), team.get("name"), team.get("alias"), team.get("displayAbbreviation"), infer_participant(market, name)
            )
            league = first_present(team.get("league"), infer_league(market, participant)).lower()
            token_id = first_present(side.get("clobTokenId"), side.get("token_id"), side.get("asset_id"))
            price, bid, ask, mid, spread, source = enrich_price_with_clob(gamma_price, token_id, config)
            parsed.append(
                ParsedOutcome(
                    outcome=name,
                    price=price,
                    participant=participant,
                    league=league,
                    record=first_present(team.get("record"), parse_record_from_text(participant, clear_market_name(market))),
                    token_id=token_id,
                    live_bid=bid,
                    live_ask=ask,
                    live_mid=mid,
                    live_spread_pct=spread,
                    price_source=source,
                )
            )
        return parsed

    names = [str(x) for x in parse_jsonish_list(market.get("outcomes"))]
    prices = parse_jsonish_list(market.get("outcomePrices"))
    token_ids = [str(x) for x in parse_jsonish_list(market.get("clobTokenIds"))]

    for index, name in enumerate(names):
        gamma_price = normalize_probability(prices[index]) if index < len(prices) else None
        if gamma_price is None:
            continue
        token_id = token_ids[index] if index < len(token_ids) else ""
        participant = infer_participant(market, name)
        league = infer_league(market, participant)
        price, bid, ask, mid, spread, source = enrich_price_with_clob(gamma_price, token_id, config)
        parsed.append(
            ParsedOutcome(
                outcome=name,
                price=price,
                participant=participant,
                league=league,
                record=parse_record_from_text(participant, clear_market_name(market), event_name_for_market(market)),
                token_id=token_id,
                live_bid=bid,
                live_ask=ask,
                live_mid=mid,
                live_spread_pct=spread,
                price_source=source,
            )
        )
    return parsed


def enrich_price_with_clob(
    gamma_price: float,
    token_id: str,
    config: AnalysisConfig,
) -> tuple[float, float | None, float | None, float | None, float | None, str]:
    if not config.use_live_clob_prices or not token_id:
        return gamma_price, None, None, None, None, "gamma"
    bid, ask, mid, spread_pct = order_book_snapshot(token_id)
    # For a buy/value scanner, the executable cost is best ask. Fall back to Gamma implied price.
    if ask is not None and 0 < ask < 1:
        return ask, bid, ask, mid, spread_pct, "clob_ask"
    if mid is not None and 0 < mid < 1:
        return mid, bid, ask, mid, spread_pct, "clob_mid"
    return gamma_price, bid, ask, mid, spread_pct, "gamma_fallback"


def side_allowed(outcome: str, config: AnalysisConfig) -> bool:
    side = outcome.strip().upper()
    if config.side_filter == "YES only":
        return side == "YES"
    if config.side_filter == "NO only":
        return side == "NO"
    return True


def category_allowed(category: str, config: AnalysisConfig) -> bool:
    if config.include_all_categories:
        return True
    if category == "sports":
        return config.include_sports
    if category == "politics":
        return config.include_politics
    if category == "crypto":
        return config.include_crypto
    if category in {"culture", "entertainment"}:
        return config.include_culture
    if category in {"finance", "economy"}:
        return config.include_finance
    return False


def get_market_volume(market: dict) -> float | None:
    for key in ("volumeNum", "volume", "volume24hr", "volume24hrClob", "volumeClob"):
        value = optional_float(market.get(key))
        if value is not None:
            return value
    return None


def get_market_liquidity(market: dict) -> float | None:
    for key in ("liquidityNum", "liquidity", "liquidityClob", "liquidityAmm"):
        value = optional_float(market.get(key))
        if value is not None:
            return value
    return None


def end_date_for_market(market: dict) -> str:
    return first_present(
        market.get("endDateIso"), market.get("endDate"), market.get("_event_end_date"), market.get("gameStartTime")
    )


def clear_outcome_market_name(market: dict, parsed_outcome: ParsedOutcome) -> str:
    base = clear_market_name(market)
    pieces = [base]
    if parsed_outcome.participant and parsed_outcome.participant.lower() not in base.lower():
        pieces.append(parsed_outcome.participant)
    if parsed_outcome.record and parsed_outcome.record not in " - ".join(pieces):
        pieces.append(parsed_outcome.record)
    return " - ".join(piece for piece in pieces if piece)


def make_market_key(market: dict, outcome: str, token_id: str = "") -> str:
    slug = first_present(market.get("slug"), market.get("conditionId"), market.get("id"), clear_market_name(market))
    token_suffix = f"::{token_id}" if token_id else ""
    return f"{slug}::{outcome.strip().lower()}{token_suffix}"


def build_market_outcomes(markets: list[dict], config: AnalysisConfig) -> tuple[list[MarketOutcome], dict[str, int]]:
    rows: list[MarketOutcome] = []
    stats = {
        "Markets scanned": 0,
        "Closed/inactive skipped": 0,
        "Category skipped": 0,
        "No price data skipped": 0,
        "Side skipped": 0,
        "Probability skipped": 0,
        "Volume skipped": 0,
        "Volume unavailable": 0,
        "Liquidity skipped": 0,
        "Liquidity unavailable": 0,
        "Spread skipped": 0,
        "Candidate outcomes": 0,
    }

    live_tokens_used = 0

    for market in markets:
        stats["Markets scanned"] += 1
        if not is_open_market(market):
            stats["Closed/inactive skipped"] += 1
            continue

        category = infer_category(market)
        if not category_allowed(category, config):
            stats["Category skipped"] += 1
            continue

        volume = get_market_volume(market)
        liquidity = get_market_liquidity(market)

        if volume is None:
            stats["Volume unavailable"] += 1
        elif volume < config.min_volume:
            stats["Volume skipped"] += 1
            continue

        if liquidity is None:
            stats["Liquidity unavailable"] += 1
        elif liquidity < config.min_liquidity:
            stats["Liquidity skipped"] += 1
            continue

        # Do not spend unlimited time hitting orderbook endpoints.
        local_config = config
        if config.use_live_clob_prices:
            token_count = len(parse_jsonish_list(market.get("clobTokenIds")))
            if live_tokens_used + token_count > config.live_clob_max_tokens:
                local_config = AnalysisConfig(**{**config.__dict__, "use_live_clob_prices": False})
            else:
                live_tokens_used += token_count

        outcomes = market_outcomes(market, local_config)
        if not outcomes:
            stats["No price data skipped"] += 1
            continue

        for parsed_outcome in outcomes:
            if not side_allowed(parsed_outcome.outcome, config):
                stats["Side skipped"] += 1
                continue
            if not (config.min_market_prob <= parsed_outcome.price <= config.max_market_prob):
                stats["Probability skipped"] += 1
                continue
            if parsed_outcome.live_spread_pct is not None and parsed_outcome.live_spread_pct > config.max_live_spread_pct:
                stats["Spread skipped"] += 1
                continue

            rows.append(
                MarketOutcome(
                    key=make_market_key(market, parsed_outcome.outcome, parsed_outcome.token_id),
                    market_id=str(market.get("id", "")),
                    condition_id=str(market.get("conditionId", "")),
                    slug=str(market.get("slug", "")),
                    category=category,
                    event_name=event_name_for_market(market),
                    market_name=clear_outcome_market_name(market, parsed_outcome),
                    outcome=parsed_outcome.outcome,
                    participant=parsed_outcome.participant,
                    league=parsed_outcome.league,
                    record=parsed_outcome.record,
                    market_prob=parsed_outcome.price,
                    price_source=parsed_outcome.price_source,
                    token_id=parsed_outcome.token_id,
                    live_bid=parsed_outcome.live_bid,
                    live_ask=parsed_outcome.live_ask,
                    live_mid=parsed_outcome.live_mid,
                    live_spread_pct=parsed_outcome.live_spread_pct,
                    volume=volume,
                    liquidity=liquidity,
                    end_date=end_date_for_market(market),
                )
            )

    stats["Candidate outcomes"] = len(rows)
    return rows, stats


# ================== AUTOMATED MODELS ==================
def parse_record_signal(record: str) -> RecordSignal | None:
    if not record:
        return None
    match = re.fullmatch(r"\s*(\d+)-(\d+)(?:-(\d+))?\s*", record)
    if not match:
        return None
    wins = float(match.group(1))
    losses = float(match.group(2))
    draws_or_ot = float(match.group(3) or 0)
    games = wins + losses + draws_or_ot
    if games <= 0:
        return None
    rating = bayesian_rate(wins, losses, draws_or_ot, prior_rate=0.50, prior_games=12.0)
    # A record extracted from a market title is weaker than a trusted standings feed.
    confidence = min(0.35, games_confidence(games, 82.0) * 0.50)
    return RecordSignal(
        rating=rating,
        games=games,
        wins=wins,
        losses=losses,
        draws_or_ot=draws_or_ot,
        confidence=confidence,
    )


def parse_record_rating(record: str) -> float | None:
    signal = parse_record_signal(record)
    return signal.rating if signal else None


def safe_logit(probability: float) -> float:
    p = max(0.01, min(0.99, probability))
    return math.log(p / (1 - p))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def event_group_key(outcome: MarketOutcome) -> str:
    name = outcome.event_name or outcome.market_name
    # Group all candidate outcomes under the same event, but avoid grouping spreads/totals with moneyline-style bets.
    market_text = f"{outcome.market_name} {outcome.outcome}".lower()
    if re.search(r"\b(spread|total|over|under|handicap)\b|[+-]\d+(?:\.\d+)?", market_text):
        name = outcome.market_name
    return normalize_name(name)


def estimate_quality(estimate: ProbabilityEstimate | None) -> str:
    if estimate is None:
        return "Missing"
    if estimate.source.startswith("auto:market-baseline"):
        return "Baseline"
    if estimate.source.startswith("auto:rating:"):
        return "Model"
    if estimate.source.startswith("auto:record-in-market"):
        return "Record"
    return "Other"


def is_actionable_estimate(estimate: ProbabilityEstimate | None) -> bool:
    return estimate is not None and estimate_quality(estimate) != "Baseline"


def is_supported_model_outcome(outcome: MarketOutcome) -> bool:
    side = outcome.outcome.strip().upper()
    if side == "NO":
        return False
    if outcome.category == "politics":
        return False
    if outcome.category != "sports" and outcome.league not in {"nba", "nhl", "mlb", "fifawc"}:
        return False
    if not outcome.participant:
        return False

    text = f"{outcome.market_name} {outcome.outcome} {outcome.participant}".lower()
    # Do not treat spread/total/over-under lines as simple win probabilities.
    if re.search(r"\b(over|under|total|spread|handicap)\b", text):
        return False
    if re.search(r"(?:^|\s)[ou]\s*\d+(?:\.\d+)?\b", text):
        return False
    if re.search(r"[+-]\d+(?:\.\d+)?", outcome.outcome):
        return False
    return side == "YES" or bool(outcome.participant)


def lookup_team_rating(data: SportsModelData, league: str, participant: str) -> TeamRating | None:
    if not league or not participant:
        return None
    normalized = normalize_name(participant)
    exact = data.ratings.get((league.lower(), normalized))
    if exact:
        return exact

    # Token-overlap match is safer than raw substring matching for names like NY, LA, O/U, etc.
    participant_tokens = {token for token in normalized.split() if len(token) >= 3}
    if not participant_tokens:
        return None
    best: tuple[int, TeamRating] | None = None
    for (rating_league, rating_name), rating in data.ratings.items():
        if rating_league != league.lower():
            continue
        rating_tokens = {token for token in rating_name.split() if len(token) >= 3}
        overlap = len(participant_tokens & rating_tokens)
        if overlap <= 0:
            continue
        if best is None or overlap > best[0]:
            best = (overlap, rating)
    return best[1] if best else None


def outcome_signal(outcome: MarketOutcome, sports_data: SportsModelData) -> ModelSignal | None:
    rating = lookup_team_rating(sports_data, outcome.league, outcome.participant)
    if rating is not None:
        return ModelSignal(
            strength=rating.rating,
            source=f"auto:rating:{rating.source}; games={rating.games:.0f}; confidence={rating.confidence:.2f}",
            confidence=rating.confidence,
        )

    record_signal = parse_record_signal(outcome.record)
    if record_signal is not None:
        return ModelSignal(
            strength=record_signal.rating,
            source=f"auto:record-in-market; games={record_signal.games:.0f}; confidence={record_signal.confidence:.2f}",
            confidence=record_signal.confidence,
        )
    return None


def event_exponent(group: list[MarketOutcome]) -> float:
    if any(outcome.league == "fifawc" for outcome in group):
        return 1.0
    event = group[0].event_name.lower() if group else ""
    if any(word in event for word in ["champion", "winner", "mvp", "award", "golden boot"]):
        return 1.75
    return 1.25


def signal_log_score(signal: ModelSignal) -> float:
    # Ratings in 0..1 are treated as win-rate-like. FIFA relative strengths can be >1, so log them.
    if 0 < signal.strength < 1:
        return safe_logit(signal.strength)
    return math.log(max(0.001, signal.strength))


def de_vigged_market_probs(group: list[MarketOutcome]) -> dict[str, float]:
    total_prob = sum(outcome.market_prob for outcome in group if outcome.market_prob > 0)
    if total_prob <= 0:
        return {}
    return {outcome.key: outcome.market_prob / total_prob for outcome in group}


def no_edge_market_baseline(outcome: MarketOutcome) -> ProbabilityEstimate:
    return ProbabilityEstimate(
        true_prob=max(0.001, min(0.999, outcome.market_prob)),
        source="auto:market-baseline (no edge)",
        model_probability=None,
        market_prior=outcome.market_prob,
        confidence=0.0,
        effective_model_weight=0.0,
        max_shift_pp=0.0,
    )


def blend_model_with_market_prior(
    outcome: MarketOutcome,
    model_prob: float,
    market_prior: float,
    signal_source: str,
    confidence: float,
    config: AnalysisConfig,
) -> ProbabilityEstimate | None:
    confidence = max(0.0, min(1.0, confidence))
    if confidence < config.min_model_confidence:
        return None

    effective_weight = max(0.0, min(config.model_blend, config.model_blend * confidence))
    blended = market_prior + effective_weight * (model_prob - market_prior)

    # Hard cap model movement from the market prior. This prevents weak standings signals from creating absurd edges.
    max_shift = (config.max_model_shift_pp / 100.0) * max(0.25, confidence)
    shift = max(-max_shift, min(max_shift, blended - market_prior))
    true_prob = max(0.001, min(0.999, market_prior + shift))

    return ProbabilityEstimate(
        true_prob=true_prob,
        source=(
            f"{signal_source}; market_prior={market_prior:.1%}; "
            f"raw_model={model_prob:.1%}; confidence={confidence:.2f}; "
            f"effective_weight={effective_weight:.2f}; shift_cap={max_shift * 100:.1f}pp"
        ),
        model_probability=model_prob,
        market_prior=market_prior,
        confidence=confidence,
        effective_model_weight=effective_weight,
        max_shift_pp=max_shift * 100,
    )


def build_event_model_estimates(
    group: list[MarketOutcome],
    sports_data: SportsModelData,
    config: AnalysisConfig,
) -> dict[str, ProbabilityEstimate]:
    if not group:
        return {}
    if sum(outcome.market_prob for outcome in group) < config.min_event_market_coverage:
        return {}

    signals: dict[str, ModelSignal] = {}
    for outcome in group:
        if not is_supported_model_outcome(outcome):
            continue
        signal = outcome_signal(outcome, sports_data)
        if signal is not None:
            signals[outcome.key] = signal

    # Need at least two modeled competitors in the same event. A single YES market without the opponent is not enough.
    if len(signals) < 2:
        return {}

    exponent = event_exponent(group)
    raw_scores: dict[str, float] = {}
    for key, signal in signals.items():
        score = signal_log_score(signal) * exponent
        # Avoid overflow while preserving ordering.
        raw_scores[key] = math.exp(max(-8.0, min(8.0, score)))

    score_total = sum(raw_scores.values())
    if score_total <= 0:
        return {}

    market_probs = de_vigged_market_probs([outcome for outcome in group if outcome.key in signals])
    coverage = sum(outcome.market_prob for outcome in group if outcome.key in signals)
    coverage_confidence = max(0.0, min(1.0, coverage))
    avg_signal_confidence = sum(signal.confidence for signal in signals.values()) / len(signals)
    group_confidence = min(avg_signal_confidence, coverage_confidence)

    source_kinds = {"record" if signals[key].source.startswith("auto:record-in-market") else "rating" for key in signals}
    source_prefix = "auto:record-in-market" if source_kinds == {"record"} else "auto:rating:conservative sports model"

    estimates: dict[str, ProbabilityEstimate] = {}
    for outcome in group:
        if outcome.key not in raw_scores:
            continue
        model_prob = raw_scores[outcome.key] / score_total
        market_prior = market_probs.get(outcome.key, outcome.market_prob)
        source = f"{source_prefix}; {signals[outcome.key].source}"
        estimate = blend_model_with_market_prior(
            outcome=outcome,
            model_prob=model_prob,
            market_prior=market_prior,
            signal_source=source,
            confidence=group_confidence,
            config=config,
        )
        if estimate is not None:
            estimates[outcome.key] = estimate
    return estimates


def build_auto_model_provider(
    outcomes: list[MarketOutcome], sports_data: SportsModelData, config: AnalysisConfig
) -> AutoModelProbabilityProvider:
    if not config.use_auto_model:
        return AutoModelProbabilityProvider(by_key={})

    by_key: dict[str, ProbabilityEstimate] = {}
    groups: dict[str, list[MarketOutcome]] = defaultdict(list)

    for outcome in outcomes:
        if config.skip_politics_auto_model and outcome.category == "politics":
            continue
        if not is_supported_model_outcome(outcome):
            continue
        groups[event_group_key(outcome)].append(outcome)

    for group in groups.values():
        by_key.update(build_event_model_estimates(group, sports_data, config))

    if config.use_market_consensus_fallback:
        for group in groups.values():
            for outcome in group:
                by_key.setdefault(outcome.key, no_edge_market_baseline(outcome))

    return AutoModelProbabilityProvider(by_key=by_key)


# ================== VALUE MATH ==================
def relative_edge(true_prob: float, market_prob: float) -> float:
    if market_prob <= 0:
        return 0.0
    return (true_prob - market_prob) / market_prob


def expected_value_per_dollar(true_prob: float, market_prob: float) -> float:
    # Buying one binary share at price p returns $1 if correct; ROI on cost = true_prob / p - 1.
    return relative_edge(true_prob, market_prob)


def full_kelly_fraction(true_prob: float, market_prob: float) -> float:
    if not 0 < market_prob < 1:
        return 0.0
    odds_profit = (1.0 / market_prob) - 1.0
    if odds_profit <= 0:
        return 0.0
    losing_prob = 1.0 - true_prob
    fraction = ((odds_profit * true_prob) - losing_prob) / odds_profit
    return max(0.0, fraction)


def build_estimate_display_df(outcomes: list[MarketOutcome], provider: ProbabilityProvider) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        estimate = provider.get(outcome)
        rows.append(
            {
                "Market": outcome.market_name,
                "Outcome": outcome.outcome,
                "Participant": outcome.participant,
                "Buy Probability": outcome.market_prob * 100,
                "Auto True Probability %": estimate.true_prob * 100 if estimate else None,
                "Raw Model Probability %": estimate.model_probability * 100 if estimate and estimate.model_probability is not None else None,
                "Market Prior %": estimate.market_prior * 100 if estimate and estimate.market_prior is not None else None,
                "Model Confidence": estimate.confidence if estimate else None,
                "Effective Model Weight": estimate.effective_model_weight if estimate else None,
                "Shift Cap pp": estimate.max_shift_pp if estimate else None,
                "Model Source": estimate.source if estimate else "no model",
                "Estimate Quality": estimate_quality(estimate),
                "Category": outcome.category,
                "League": outcome.league,
                "Record": outcome.record,
                "Price Source": outcome.price_source,
                "Live Bid": outcome.live_bid,
                "Live Ask": outcome.live_ask,
                "Live Spread %": outcome.live_spread_pct,
                "Event Group": event_group_key(outcome),
                "Token ID": outcome.token_id,
            }
        )
    return pd.DataFrame(rows)


def count_auto_estimates(estimate_df: pd.DataFrame) -> int:
    if estimate_df.empty or "Auto True Probability %" not in estimate_df:
        return 0
    return int(estimate_df["Auto True Probability %"].notna().sum())


def count_quality(estimate_df: pd.DataFrame, quality: str) -> int:
    if estimate_df.empty or "Estimate Quality" not in estimate_df:
        return 0
    return int((estimate_df["Estimate Quality"] == quality).sum())


def analyze_value(
    outcomes: list[MarketOutcome], provider: ProbabilityProvider, config: AnalysisConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    stats = {
        "Candidate outcomes": len(outcomes),
        "Missing true probability": 0,
        "Baseline skipped": 0,
        "Record-only skipped": 0,
        "Filtered by edge": 0,
        "Filtered by edge pp": 0,
        "Filtered by Kelly": 0,
        "Value rows": 0,
    }

    for outcome in outcomes:
        estimate = provider.get(outcome)
        if estimate is None:
            stats["Missing true probability"] += 1
            continue
        quality = estimate_quality(estimate)
        if config.require_actionable_model and quality == "Baseline":
            stats["Baseline skipped"] += 1
            continue
        if config.require_actionable_model and quality == "Record" and not config.allow_record_only_value:
            stats["Record-only skipped"] += 1
            continue

        edge = relative_edge(estimate.true_prob, outcome.market_prob)
        edge_pct = edge * 100
        edge_pp = (estimate.true_prob - outcome.market_prob) * 100
        if edge_pct < config.min_edge_pct:
            stats["Filtered by edge"] += 1
            continue
        if edge_pp < config.min_edge_pp:
            stats["Filtered by edge pp"] += 1
            continue

        ev = expected_value_per_dollar(estimate.true_prob, outcome.market_prob)
        adjusted_kelly = full_kelly_fraction(estimate.true_prob, outcome.market_prob) * config.kelly_fraction
        capped_kelly = min(adjusted_kelly, config.max_bet_pct / 100)
        kelly_pct = adjusted_kelly * 100
        if kelly_pct < config.min_kelly_pct:
            stats["Filtered by Kelly"] += 1
            continue

        rows.append(
            {
                "Market": outcome.market_name,
                "Outcome": outcome.outcome,
                "Participant": outcome.participant,
                "Buy Probability": outcome.market_prob,
                "My True Probability": estimate.true_prob,
                "Edge %": edge_pct,
                "Edge pp": edge_pp,
                "EV %": ev * 100,
                "EV / $100": ev * 100,
                "Kelly %": kelly_pct,
                "Bet %": capped_kelly * 100,
                "Suggested Bet ($)": config.bankroll * capped_kelly,
                "Capped": capped_kelly < adjusted_kelly,
                "Volume": outcome.volume,
                "Liquidity": outcome.liquidity,
                "Live Bid": outcome.live_bid,
                "Live Ask": outcome.live_ask,
                "Live Spread %": outcome.live_spread_pct,
                "Price Source": outcome.price_source,
                "Source Quality": quality,
                "Raw Model Probability": estimate.model_probability,
                "Market Prior": estimate.market_prior,
                "Model Confidence": estimate.confidence,
                "Effective Model Weight": estimate.effective_model_weight,
                "Shift Cap pp": estimate.max_shift_pp,
                "Source": estimate.source,
                "End Date": outcome.end_date,
                "Market Key": outcome.key,
                "Token ID": outcome.token_id,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        sort_col = {
            "Buy Probability": "Buy Probability",
            "Suggested Bet ($)": "Suggested Bet ($)",
            "Kelly %": "Kelly %",
            "Edge %": "Edge %",
            "EV %": "EV %",
            "Live Spread %": "Live Spread %",
            "Market": "Market",
        }[config.sort_by]
        df = df.sort_values(sort_col, ascending=config.sort_ascending, na_position="last")

    stats["Value rows"] = len(df)
    return df, stats


def build_edge_review_df(
    outcomes: list[MarketOutcome], provider: ProbabilityProvider, config: AnalysisConfig, limit: int = 40
) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        estimate = provider.get(outcome)
        if not is_actionable_estimate(estimate):
            continue
        edge = relative_edge(estimate.true_prob, outcome.market_prob)
        adjusted_kelly = full_kelly_fraction(estimate.true_prob, outcome.market_prob) * config.kelly_fraction
        edge_pct = edge * 100
        edge_pp = (estimate.true_prob - outcome.market_prob) * 100
        blockers = []
        if estimate_quality(estimate) == "Record" and not config.allow_record_only_value:
            blockers.append("record-only disabled")
        if edge_pct < config.min_edge_pct:
            blockers.append("edge %")
        if edge_pp < config.min_edge_pp:
            blockers.append("edge pp")
        if adjusted_kelly * 100 < config.min_kelly_pct:
            blockers.append("Kelly")
        rows.append(
            {
                "Market": outcome.market_name,
                "Outcome": outcome.outcome,
                "Participant": outcome.participant,
                "Buy Probability": outcome.market_prob,
                "My True Probability": estimate.true_prob,
                "Edge %": edge_pct,
                "Edge pp": edge_pp,
                "EV %": expected_value_per_dollar(estimate.true_prob, outcome.market_prob) * 100,
                "Kelly %": adjusted_kelly * 100,
                "Live Spread %": outcome.live_spread_pct,
                "Price Source": outcome.price_source,
                "Source Quality": estimate_quality(estimate),
                "Raw Model Probability": estimate.model_probability,
                "Market Prior": estimate.market_prior,
                "Model Confidence": estimate.confidence,
                "Effective Model Weight": estimate.effective_model_weight,
                "Shift Cap pp": estimate.max_shift_pp,
                "Source": estimate.source,
                "Status": "Passes filters" if not blockers else "Below: " + ", ".join(blockers),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Edge %", ascending=False).head(limit)


# ================== DISPLAY HELPERS ==================
def format_optional_money(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${float(value):,.0f}"


def format_optional_pct(value: Any, decimals: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}%"


def display_value_table(df: pd.DataFrame) -> None:
    display_df = df.copy()
    display_df["Buy Probability"] = display_df["Buy Probability"].map("{:.1%}".format)
    display_df["My True Probability"] = display_df["My True Probability"].map("{:.1%}".format)
    display_df["Edge %"] = display_df["Edge %"].map("{:+.1f}%".format)
    display_df["Edge pp"] = display_df["Edge pp"].map("{:+.1f}pp".format)
    display_df["EV %"] = display_df["EV %"].map("{:+.1f}%".format)
    display_df["EV / $100"] = display_df["EV / $100"].map("${:+.2f}".format)
    display_df["Kelly %"] = display_df["Kelly %"].map("{:.2f}%".format)
    display_df["Bet %"] = display_df["Bet %"].map("{:.2f}%".format)
    display_df["Suggested Bet ($)"] = display_df["Suggested Bet ($)"].map("${:,.0f}".format)
    display_df["Capped"] = display_df["Capped"].map(lambda value: "Yes" if value else "No")
    display_df["Volume"] = display_df["Volume"].map(format_optional_money)
    display_df["Liquidity"] = display_df["Liquidity"].map(format_optional_money)
    display_df["Live Bid"] = display_df["Live Bid"].map(lambda v: "N/A" if pd.isna(v) else f"{v:.3f}")
    display_df["Live Ask"] = display_df["Live Ask"].map(lambda v: "N/A" if pd.isna(v) else f"{v:.3f}")
    display_df["Live Spread %"] = display_df["Live Spread %"].map(lambda v: format_optional_pct(v, 1))
    for col in ["Raw Model Probability", "Market Prior"]:
        if col in display_df:
            display_df[col] = display_df[col].map(lambda v: "N/A" if pd.isna(v) else f"{v:.1%}")
    for col in ["Model Confidence", "Effective Model Weight"]:
        if col in display_df:
            display_df[col] = display_df[col].map(lambda v: "N/A" if pd.isna(v) else f"{v:.2f}")
    if "Shift Cap pp" in display_df:
        display_df["Shift Cap pp"] = display_df["Shift Cap pp"].map(lambda v: "N/A" if pd.isna(v) else f"{v:.1f}pp")

    st.dataframe(
        display_df[
            [
                "Market",
                "Outcome",
                "Participant",
                "Buy Probability",
                "My True Probability",
                "Edge %",
                "Edge pp",
                "EV %",
                "EV / $100",
                "Kelly %",
                "Bet %",
                "Suggested Bet ($)",
                "Capped",
                "Volume",
                "Liquidity",
                "Live Bid",
                "Live Ask",
                "Live Spread %",
                "Price Source",
                "Source Quality",
                "Raw Model Probability",
                "Market Prior",
                "Model Confidence",
                "Effective Model Weight",
                "Shift Cap pp",
                "Source",
                "End Date",
            ]
        ],
        width="stretch",
        height=620,
        hide_index=True,
    )


def display_edge_review_table(df: pd.DataFrame) -> None:
    display_df = df.copy()
    display_df["Buy Probability"] = display_df["Buy Probability"].map("{:.1%}".format)
    display_df["My True Probability"] = display_df["My True Probability"].map("{:.1%}".format)
    display_df["Edge %"] = display_df["Edge %"].map("{:+.1f}%".format)
    display_df["Edge pp"] = display_df["Edge pp"].map("{:+.1f}pp".format)
    display_df["EV %"] = display_df["EV %"].map("{:+.1f}%".format)
    display_df["Kelly %"] = display_df["Kelly %"].map("{:.2f}%".format)
    display_df["Live Spread %"] = display_df["Live Spread %"].map(lambda v: format_optional_pct(v, 1))
    for col in ["Raw Model Probability", "Market Prior"]:
        if col in display_df:
            display_df[col] = display_df[col].map(lambda v: "N/A" if pd.isna(v) else f"{v:.1%}")
    for col in ["Model Confidence", "Effective Model Weight"]:
        if col in display_df:
            display_df[col] = display_df[col].map(lambda v: "N/A" if pd.isna(v) else f"{v:.2f}")
    if "Shift Cap pp" in display_df:
        display_df["Shift Cap pp"] = display_df["Shift Cap pp"].map(lambda v: "N/A" if pd.isna(v) else f"{v:.1f}pp")
    st.dataframe(display_df, width="stretch", height=420, hide_index=True)


# ================== APP FLOW ==================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

col_btn, col_time = st.columns([1, 4])
with col_btn:
    manual_refresh = st.button("Refresh Now", type="primary")

if manual_refresh:
    st.cache_data.clear()
    st.session_state.last_refresh = time.time()

markets, fetch_stats = fetch_market_data(CONFIG)
outcomes, scan_stats = build_market_outcomes(markets, CONFIG)

with col_time:
    st.caption(
        f"Last updated: {datetime.now().strftime('%H:%M:%S')} | "
        f"{len(markets)} markets | {len(outcomes)} candidate outcomes | "
        f"source: {fetch_stats.get('endpoint', 'unknown')}"
    )

if DEBUG_MODE:
    st.subheader("Raw candidate outcomes")
    debug_df = pd.DataFrame([outcome.__dict__ for outcome in outcomes])
    st.dataframe(debug_df, width="stretch", hide_index=True)
    with st.expander("Fetch and scan stats", expanded=True):
        st.write(fetch_stats)
        st.dataframe(pd.DataFrame([{"Step": key, "Count": value} for key, value in scan_stats.items()]), hide_index=True)
    st.stop()

if not outcomes:
    st.warning("No markets match the current filters. Lower filters or fetch more Gamma pages.")
    with st.expander("Fetch and scan stats", expanded=True):
        st.write(fetch_stats)
        st.dataframe(pd.DataFrame([{"Step": key, "Count": value} for key, value in scan_stats.items()]), hide_index=True)
    st.stop()

st.subheader("1. Automated True Probability Estimates")
st.caption(
    "Model-backed estimates currently cover simple NBA/NHL/MLB/FIFA-style winner markets "
    "when participants can be matched to public standings/rankings. Unsupported markets are marked baseline."
)

sports_model_data = fetch_sports_model_data() if CONFIG.use_auto_model else SportsModelData(ratings={})
auto_provider = build_auto_model_provider(outcomes, sports_model_data, CONFIG)
estimate_df = build_estimate_display_df(outcomes, auto_provider)
auto_estimate_count = count_auto_estimates(estimate_df)
actionable_estimate_count = (
    count_quality(estimate_df, "Model") + count_quality(estimate_df, "Record") + count_quality(estimate_df, "Other")
)
baseline_estimate_count = count_quality(estimate_df, "Baseline")

metric_cols = st.columns(6)
metric_cols[0].metric("Candidate outcomes", len(outcomes))
metric_cols[1].metric("Non-baseline estimates", actionable_estimate_count)
metric_cols[2].metric("Baseline only", baseline_estimate_count)
metric_cols[3].metric("Still missing", max(0, len(outcomes) - auto_estimate_count))
metric_cols[4].metric("Sports rating keys", len(sports_model_data.ratings))
metric_cols[5].metric("Live CLOB rows", int((estimate_df.get("Price Source", pd.Series(dtype=str)).astype(str).str.contains("clob")).sum()))

if actionable_estimate_count == 0:
    st.warning(
        "No non-baseline estimates were generated for the current filters. This is safer than inventing fake edge. "
        "For true sports/politics/news edge, add a real domain model before betting."
    )

with st.expander("Model coverage and scan stats", expanded=False):
    coverage_df = estimate_df.copy()
    if not coverage_df.empty:
        coverage_df["Buy Probability"] = coverage_df["Buy Probability"].map("{:.1f}%".format)
        coverage_df["Auto True Probability %"] = coverage_df["Auto True Probability %"].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.1f}%"
        )
        coverage_df["Raw Model Probability %"] = coverage_df["Raw Model Probability %"].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.1f}%"
        )
        coverage_df["Market Prior %"] = coverage_df["Market Prior %"].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.1f}%"
        )
        coverage_df["Model Confidence"] = coverage_df["Model Confidence"].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.2f}"
        )
        coverage_df["Effective Model Weight"] = coverage_df["Effective Model Weight"].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.2f}"
        )
        coverage_df["Shift Cap pp"] = coverage_df["Shift Cap pp"].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.1f}pp"
        )
        coverage_df["Live Spread %"] = coverage_df["Live Spread %"].map(lambda v: format_optional_pct(v, 1))
        st.dataframe(
            coverage_df[
                [
                    "Market",
                    "Outcome",
                    "Participant",
                    "Buy Probability",
                    "Auto True Probability %",
                    "Raw Model Probability %",
                    "Market Prior %",
                    "Model Confidence",
                    "Effective Model Weight",
                    "Shift Cap pp",
                    "Estimate Quality",
                    "Model Source",
                    "Category",
                    "League",
                    "Record",
                    "Price Source",
                    "Live Spread %",
                    "Event Group",
                ]
            ],
            width="stretch",
            hide_index=True,
            height=420,
        )
    st.write("Fetch stats", fetch_stats)
    st.dataframe(pd.DataFrame([{"Step": key, "Count": value} for key, value in scan_stats.items()]), hide_index=True)

st.subheader("2. Model Edge Review")
edge_review_df = build_edge_review_df(outcomes, auto_provider, CONFIG)
if edge_review_df.empty:
    st.info("No model-backed edges to review for the current filters.")
else:
    display_edge_review_table(edge_review_df)

st.subheader("3. Value Bets")
value_df, value_stats = analyze_value(outcomes, auto_provider, CONFIG)
if value_df.empty:
    if actionable_estimate_count == 0:
        st.warning("No value bets because no non-baseline estimates were generated.")
    else:
        st.warning("No value bets passed your filters. Review the model edge table and the stats below.")
    with st.expander("Why no value bets?", expanded=False):
        st.dataframe(pd.DataFrame([{"Step": key, "Count": value} for key, value in value_stats.items()]), hide_index=True)
else:
    value_cols = st.columns(4)
    value_cols[0].metric("Value rows", len(value_df))
    value_cols[1].metric("Best edge", f"{value_df['Edge %'].max():.1f}%")
    value_cols[2].metric("Best Kelly", f"{value_df['Kelly %'].max():.2f}%")
    value_cols[3].metric("Total suggested", f"${value_df['Suggested Bet ($)'].sum():,.0f}")
    display_value_table(value_df)
    csv = value_df.to_csv(index=False).encode()
    st.download_button("Download value bets CSV", csv, "polymarket_value_bets.csv", "text/csv")

st.caption(
    "Not financial advice. A positive edge in this app only means the selected heuristic model is above the market price. "
    "Before risking money, validate the probability model, market rules, liquidity, spread, fees, and legal/compliance constraints."
)

if AUTO_REFRESH:
    elapsed = time.time() - st.session_state.last_refresh
    remaining = max(0, REFRESH_MIN * 60 - elapsed)
    st.caption(f"Next auto-refresh check in {int(remaining)}s")
    time.sleep(1)
    if remaining <= 1:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
    st.rerun()
