import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd
import requests
import streamlit as st


POLYMARKET_US_BASE_URL = "https://gateway.polymarket.us/v1"
FIFA_MENS_RANKINGS_URL = "https://api.fifa.com/api/v3/rankings"
FIFA_RATING_BASELINE = 1500.0
FIFA_ELO_SCALE = 400.0

FIFA_COUNTRY_ALIASES = {
    "CPV": ["Cabo Verde", "Cape Verde"],
    "CZE": ["Czechia", "Czech Republic"],
    "IRN": ["IR Iran", "Iran"],
    "KOR": ["Korea Republic", "South Korea", "Korea"],
    "KSA": ["Saudi Arabia"],
    "TUR": ["Turkiye", "Turkey"],
    "UAE": ["United Arab Emirates", "UAE"],
    "USA": ["USA", "United States", "United States of America", "US"],
}


st.set_page_config(page_title="Smart Polymarket Value Tool", layout="wide")
st.title("Smart Polymarket Value Tool")
st.info(
    "Using the Polymarket US public API only. Automated estimates are simple "
    "heuristics from ESPN/MLB standings, FIFA rankings, in-market records, "
    "and conservative market-consensus fallback."
)


# ================== DOMAIN MODEL ==================
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


class ProbabilityProvider(Protocol):
    """Small extension point for automated probability models."""

    def get(self, outcome: MarketOutcome) -> ProbabilityEstimate | None:
        ...


@dataclass
class AnalysisConfig:
    bankroll: float
    kelly_fraction: float
    min_edge_pct: float
    min_kelly_pct: float
    min_market_prob: float
    max_market_prob: float
    min_volume: float
    min_liquidity: float
    side_filter: str
    sort_by: str
    sort_ascending: bool
    include_all_categories: bool
    include_sports: bool
    include_politics: bool
    include_crypto: bool
    include_culture: bool
    use_auto_model: bool
    model_blend: float
    use_market_consensus_fallback: bool
    skip_politics_auto_model: bool


@dataclass(frozen=True)
class ParsedOutcome:
    outcome: str
    price: float
    participant: str
    league: str
    record: str


@dataclass(frozen=True)
class TeamRating:
    name: str
    league: str
    rating: float
    source: str


@dataclass(frozen=True)
class SportsModelData:
    ratings: dict[tuple[str, str], TeamRating]


@dataclass
class AutoModelProbabilityProvider:
    by_key: dict[str, ProbabilityEstimate]

    def get(self, outcome: MarketOutcome) -> ProbabilityEstimate | None:
        return self.by_key.get(outcome.key)


# ================== SIDEBAR CONTROLS ==================
st.sidebar.header("Bankroll & Filters")
BANKROLL = st.sidebar.number_input("Bankroll ($)", value=10_000, min_value=100, step=500)
KELLY_FRACTION = st.sidebar.slider(
    "Kelly Fraction",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="0.25 = quarter-Kelly. Use smaller fractions for noisy probability estimates.",
)
MIN_EDGE_PCT = st.sidebar.number_input(
    "Minimum Edge (%)",
    value=8.0,
    min_value=-100.0,
    max_value=500.0,
    step=1.0,
    help="Relative edge: (your true probability - market probability) / market probability.",
)
MIN_KELLY = st.sidebar.number_input(
    "Minimum Kelly % of bankroll", value=0.1, min_value=0.0, max_value=100.0, step=0.1
)
PROB_RANGE = st.sidebar.slider(
    "Polymarket probability range",
    min_value=0.01,
    max_value=0.99,
    value=(0.01, 0.99),
    step=0.01,
)
MIN_VOLUME = st.sidebar.number_input(
    "Minimum Volume ($, if available)", value=0.0, min_value=0.0, step=10_000.0
)
MIN_LIQUIDITY = st.sidebar.number_input(
    "Minimum Liquidity ($, if available)", value=0.0, min_value=0.0, step=10_000.0
)

SORT_BY = st.sidebar.selectbox(
    "Sort by",
    options=[
        "Edge %",
        "Kelly %",
        "EV %",
        "Suggested Bet ($)",
        "Polymarket Probability",
        "Market",
    ],
)
SORT_ASCENDING = st.sidebar.checkbox("Sort ascending", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Automation")
USE_AUTO_MODEL = st.sidebar.checkbox("Use automated true probabilities", value=True)
MODEL_BLEND = st.sidebar.slider(
    "Model weight",
    min_value=0.05,
    max_value=1.0,
    value=0.35,
    step=0.05,
    help=(
        "How much to trust the simple statistical model versus de-vigged market "
        "consensus. Lower is more conservative."
    ),
)
USE_MARKET_CONSENSUS_FALLBACK = st.sidebar.checkbox(
    "Fill unsupported markets with market consensus", value=True
)
SKIP_POLITICS_AUTO_MODEL = st.sidebar.checkbox(
    "Skip politics automation until polling model exists", value=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("Categories")
CAT_ALL = st.sidebar.checkbox("Show All", value=True)
CAT_SPORTS = st.sidebar.checkbox("Sports", value=True)
CAT_POLITICS = st.sidebar.checkbox("Politics", value=True)
CAT_CRYPTO = st.sidebar.checkbox("Crypto", value=True)
CAT_CULTURE = st.sidebar.checkbox("Culture / Entertainment", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Bet Side")
SIDE_FILTER = st.sidebar.radio(
    "Show bets",
    options=["YES only", "NO only", "Both"],
    index=0,
    help="For multi-outcome markets, YES means that listed outcome wins.",
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
    min_kelly_pct=MIN_KELLY,
    min_market_prob=PROB_RANGE[0],
    max_market_prob=PROB_RANGE[1],
    min_volume=MIN_VOLUME,
    min_liquidity=MIN_LIQUIDITY,
    side_filter=SIDE_FILTER,
    sort_by=SORT_BY,
    sort_ascending=SORT_ASCENDING,
    include_all_categories=CAT_ALL,
    include_sports=CAT_SPORTS,
    include_politics=CAT_POLITICS,
    include_crypto=CAT_CRYPTO,
    include_culture=CAT_CULTURE,
    use_auto_model=USE_AUTO_MODEL,
    model_blend=MODEL_BLEND,
    use_market_consensus_fallback=USE_MARKET_CONSENSUS_FALLBACK,
    skip_politics_auto_model=SKIP_POLITICS_AUTO_MODEL,
)


# ================== DATA LAYER ==================
@st.cache_data(show_spinner="Fetching Polymarket US markets...")
def fetch_market_data() -> list[dict]:
    try:
        response = requests.get(
            f"{POLYMARKET_US_BASE_URL}/markets",
            params={
                "active": "true",
                "closed": "false",
                "archived": "false",
                "includeHidden": "false",
                "limit": 500,
            },
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("markets", []) if isinstance(payload, dict) else []
    except Exception as exc:
        st.warning(f"Fetch failed: {exc}")
        return []


@st.cache_data(ttl=900, show_spinner="Fetching public sports standings...")
def fetch_sports_model_data() -> SportsModelData:
    ratings: dict[tuple[str, str], TeamRating] = {}

    for league, url, source in [
        (
            "nba",
            "https://site.web.api.espn.com/apis/v2/sports/basketball/nba/standings"
            "?region=us&lang=en&contentorigin=espn&type=0&level=2"
            "&sort=playoffseed%3Aasc",
            "ESPN standings",
        ),
        (
            "nhl",
            "https://site.web.api.espn.com/apis/v2/sports/hockey/nhl/standings"
            "?region=us&lang=en&contentorigin=espn&type=0&level=2"
            "&sort=playoffseed%3Aasc",
            "ESPN standings",
        ),
    ]:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            ratings.update(parse_espn_team_ratings(response.json(), league, source))
        except Exception:
            continue

    try:
        response = requests.get(
            "https://statsapi.mlb.com/api/v1/standings",
            params={
                "leagueId": "103,104",
                "season": datetime.now().year,
                "standingsTypes": "regularSeason",
            },
            timeout=15,
        )
        response.raise_for_status()
        ratings.update(parse_mlb_team_ratings(response.json()))
    except Exception:
        pass

    try:
        response = requests.get(
            FIFA_MENS_RANKINGS_URL,
            params={"gender": "male"},
            timeout=15,
        )
        response.raise_for_status()
        ratings.update(parse_fifa_team_ratings(response.json()))
    except Exception:
        pass

    return SportsModelData(ratings=ratings)


# ================== PARSING HELPERS ==================
def optional_float(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_probability(value) -> float | None:
    prob = optional_float(value)
    if prob is None:
        return None
    if prob > 1:
        prob = prob / 100
    if 0 <= prob <= 1:
        return prob
    return None


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def rating_keys(league: str, *names: str) -> list[tuple[str, str]]:
    keys = []
    for name in names:
        if name:
            keys.append((league.lower(), normalize_name(name)))
    return keys


def add_team_rating(
    ratings: dict[tuple[str, str], TeamRating],
    league: str,
    rating: TeamRating,
    *aliases: str,
):
    for key in rating_keys(league, *aliases):
        ratings[key] = rating


def iter_espn_standing_entries(node: dict):
    standings = node.get("standings")
    if isinstance(standings, dict):
        for entry in standings.get("entries", []) or []:
            yield entry

    for child in node.get("children", []) or []:
        yield from iter_espn_standing_entries(child)


def parse_espn_team_ratings(payload: dict, league: str, source: str) -> dict[tuple[str, str], TeamRating]:
    ratings: dict[tuple[str, str], TeamRating] = {}

    for entry in iter_espn_standing_entries(payload):
        team = entry.get("team", {})
        stats = {stat.get("name"): stat.get("value") for stat in entry.get("stats", [])}
        wins = optional_float(stats.get("wins")) or 0.0
        losses = optional_float(stats.get("losses")) or 0.0
        ot_losses = optional_float(stats.get("otLosses")) or 0.0
        points = optional_float(stats.get("points"))
        games = wins + losses + ot_losses

        if games <= 0:
            continue

        if league == "nhl" and points is not None:
            rating_value = points / (2 * games)
        else:
            rating_value = wins / games

        rating = TeamRating(
            name=team.get("displayName") or team.get("name") or "",
            league=league,
            rating=max(0.01, min(0.99, rating_value)),
            source=source,
        )
        add_team_rating(
            ratings,
            league,
            rating,
            team.get("displayName", ""),
            team.get("shortDisplayName", ""),
            team.get("name", ""),
            team.get("abbreviation", ""),
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

            rating = TeamRating(
                name=team.get("name", ""),
                league="mlb",
                rating=max(0.01, min(0.99, wins / games)),
                source="MLB Stats API standings",
            )
            add_team_rating(ratings, "mlb", rating, team.get("name", ""))

    return ratings


def localized_description(value, locale: str = "en-GB") -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""

    fallback = ""
    for item in value:
        if not isinstance(item, dict):
            continue
        description = item.get("Description") or item.get("description") or ""
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
        source = "FIFA men's ranking"
        if pub_date:
            source = f"{source} {pub_date}"

        # FIFA ranking points are Elo-like. Convert point differences into a
        # relative strength, then normalize within each Polymarket event.
        rating = TeamRating(
            name=name,
            league="fifawc",
            rating=max(0.01, 10 ** ((points - FIFA_RATING_BASELINE) / FIFA_ELO_SCALE)),
            source=source,
        )
        add_team_rating(
            ratings,
            "fifawc",
            rating,
            name,
            country_code,
            *FIFA_COUNTRY_ALIASES.get(country_code, []),
        )

    return ratings


def parse_record_rating(record: str) -> float | None:
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

    return max(0.01, min(0.99, (wins + 0.5 * draws_or_ot) / games))


def get_quote_value(value) -> float | None:
    if isinstance(value, dict):
        return optional_float(value.get("value"))
    return optional_float(value)


def get_market_volume(market: dict) -> float | None:
    return optional_float(market.get("volumeNum") or market.get("volume"))


def get_market_liquidity(market: dict) -> float | None:
    return optional_float(market.get("liquidityNum") or market.get("liquidity"))


def is_open_market(market: dict) -> bool:
    if market.get("closed") or market.get("archived") or market.get("hidden"):
        return False
    if not market.get("active", True):
        return False
    return market.get("ep3Status", "OPEN") == "OPEN"


def clear_market_name(market: dict) -> str:
    question = market.get("question") or ""
    title = market.get("title") or ""
    subtitle = market.get("subtitle") or ""
    pieces = []
    for piece in (question, title, subtitle):
        if piece and piece not in pieces:
            pieces.append(piece)
    return " - ".join(pieces)


def clear_outcome_market_name(market: dict, parsed_outcome: ParsedOutcome) -> str:
    pieces = [clear_market_name(market)]
    if parsed_outcome.participant and parsed_outcome.participant not in pieces[0]:
        pieces.append(parsed_outcome.participant)
    if (
        parsed_outcome.record
        and not re.search(r"\b\d+-\d+(?:-\d+)?\b", " - ".join(pieces))
    ):
        pieces.append(parsed_outcome.record)
    return " - ".join(piece for piece in pieces if piece)


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
    return False


def side_allowed(outcome: str, config: AnalysisConfig) -> bool:
    side = outcome.strip().upper()
    if config.side_filter == "YES only":
        return side == "YES"
    if config.side_filter == "NO only":
        return side == "NO"
    return True


def market_side_price(side: dict) -> float | None:
    return (
        optional_float(side.get("price"))
        or get_quote_value(side.get("quote"))
        or get_quote_value(side.get("bestAskQuote"))
        or get_quote_value(side.get("bestBidQuote"))
    )


def market_outcomes(market: dict) -> list[ParsedOutcome]:
    sides = market.get("marketSides")
    if isinstance(sides, list) and sides:
        outcomes = []
        for side in sides:
            name = side.get("description") or ("Yes" if side.get("long") else "No")
            price = market_side_price(side)
            team = side.get("team") or {}
            if price is not None:
                outcomes.append(
                    ParsedOutcome(
                        outcome=name,
                        price=price,
                        participant=team.get("safeName")
                        or team.get("name")
                        or team.get("alias")
                        or team.get("displayAbbreviation")
                        or "",
                        league=(team.get("league") or "").lower(),
                        record=team.get("record") or "",
                    )
                )
        return outcomes

    try:
        names = json.loads(market.get("outcomes", "[]"))
        prices = json.loads(market.get("outcomePrices", "[]"))
    except (json.JSONDecodeError, TypeError):
        return []

    outcomes = []
    for name, price_value in zip(names, prices):
        price = optional_float(price_value)
        if price is not None:
            outcomes.append(
                ParsedOutcome(
                    outcome=name,
                    price=price,
                    participant="",
                    league="",
                    record="",
                )
            )
    return outcomes


def make_market_key(market: dict, outcome: str) -> str:
    slug = market.get("slug") or market.get("id") or clear_market_name(market)
    return f"{slug}::{outcome.strip().lower()}"


def build_market_outcomes(markets: list[dict], config: AnalysisConfig) -> tuple[list[MarketOutcome], dict[str, int]]:
    rows = []
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
        "Candidate outcomes": 0,
    }

    for market in markets:
        stats["Markets scanned"] += 1
        if not is_open_market(market):
            stats["Closed/inactive skipped"] += 1
            continue

        category = (market.get("category") or "unknown").lower()
        if not category_allowed(category, config):
            stats["Category skipped"] += 1
            continue

        event_name = clear_market_name(market)
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

        outcomes = market_outcomes(market)
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

            rows.append(
                MarketOutcome(
                    key=make_market_key(market, parsed_outcome.outcome),
                    market_id=str(market.get("id", "")),
                    slug=str(market.get("slug", "")),
                    category=category,
                    event_name=event_name,
                    market_name=clear_outcome_market_name(market, parsed_outcome),
                    outcome=parsed_outcome.outcome,
                    participant=parsed_outcome.participant,
                    league=parsed_outcome.league,
                    record=parsed_outcome.record,
                    market_prob=parsed_outcome.price,
                    volume=volume,
                    liquidity=liquidity,
                )
            )

    stats["Candidate outcomes"] = len(rows)
    return rows, stats


# ================== AUTOMATED MODELS ==================
def lookup_team_rating(data: SportsModelData, league: str, participant: str) -> TeamRating | None:
    if not league or not participant:
        return None

    normalized = normalize_name(participant)
    exact = data.ratings.get((league.lower(), normalized))
    if exact:
        return exact

    for (rating_league, rating_name), rating in data.ratings.items():
        if rating_league != league.lower():
            continue
        if normalized and (normalized in rating_name or rating_name in normalized):
            return rating

    return None


def outcome_strength(
    outcome: MarketOutcome, sports_data: SportsModelData
) -> tuple[float, str] | None:
    rating = lookup_team_rating(sports_data, outcome.league, outcome.participant)
    if rating is not None:
        return rating.rating, f"auto:rating:{rating.source}"

    record_rating = parse_record_rating(outcome.record)
    if record_rating is not None:
        return record_rating, "auto:record-in-market"

    return None


def event_exponent(group: list[MarketOutcome]) -> float:
    if any(outcome.league == "fifawc" for outcome in group):
        return 1.0

    event = group[0].event_name.lower()
    if "champion" in event or "winner" in event:
        return 4.0
    return 2.0


def de_vigged_market_probs(group: list[MarketOutcome]) -> dict[str, float]:
    total_prob = sum(outcome.market_prob for outcome in group if outcome.market_prob > 0)
    if total_prob <= 0:
        return {}
    return {outcome.key: outcome.market_prob / total_prob for outcome in group}


def build_event_model_estimates(
    group: list[MarketOutcome],
    sports_data: SportsModelData,
    config: AnalysisConfig,
) -> dict[str, ProbabilityEstimate]:
    strengths = {}
    sources = {}

    for outcome in group:
        strength = outcome_strength(outcome, sports_data)
        if strength is None:
            continue
        strengths[outcome.key] = strength[0]
        sources[outcome.key] = strength[1]

    if len(strengths) < 2:
        return {}

    exponent = event_exponent(group)
    raw_scores = {
        key: max(0.001, strength) ** exponent
        for key, strength in strengths.items()
    }
    score_total = sum(raw_scores.values())
    if score_total <= 0:
        return {}

    market_probs = de_vigged_market_probs(group)
    estimates = {}
    for outcome in group:
        if outcome.key not in raw_scores:
            continue

        model_prob = raw_scores[outcome.key] / score_total
        market_prob = market_probs.get(outcome.key, outcome.market_prob)
        true_prob = (
            config.model_blend * model_prob
            + (1 - config.model_blend) * market_prob
        )
        estimates[outcome.key] = ProbabilityEstimate(
            true_prob=max(0.001, min(0.999, true_prob)),
            source=f"{sources[outcome.key]} ({config.model_blend:.0%} model blend)",
        )

    return estimates


def build_auto_model_provider(
    outcomes: list[MarketOutcome],
    sports_data: SportsModelData,
    config: AnalysisConfig,
) -> AutoModelProbabilityProvider:
    if not config.use_auto_model:
        return AutoModelProbabilityProvider(by_key={})

    by_key: dict[str, ProbabilityEstimate] = {}
    groups: dict[str, list[MarketOutcome]] = defaultdict(list)

    for outcome in outcomes:
        if config.skip_politics_auto_model and outcome.category == "politics":
            continue
        if outcome.outcome.strip().upper() != "YES":
            continue
        groups[outcome.event_name].append(outcome)

    for group in groups.values():
        by_key.update(build_event_model_estimates(group, sports_data, config))

    if config.use_market_consensus_fallback:
        for group in groups.values():
            market_probs = de_vigged_market_probs(group)
            for outcome in group:
                if outcome.key in by_key:
                    continue
                true_prob = market_probs.get(outcome.key, outcome.market_prob)
                by_key[outcome.key] = ProbabilityEstimate(
                    true_prob=max(0.001, min(0.999, true_prob)),
                    source="auto:market-consensus",
                )

    return AutoModelProbabilityProvider(by_key=by_key)


def build_estimate_display_df(
    outcomes: list[MarketOutcome], provider: ProbabilityProvider
) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        estimate = provider.get(outcome)
        rows.append(
            {
                "Market": outcome.market_name,
                "Outcome": outcome.outcome,
                "Polymarket Probability": outcome.market_prob * 100,
                "Auto True Probability %": (
                    estimate.true_prob * 100 if estimate is not None else None
                ),
                "Model Source": estimate.source if estimate is not None else "no model",
                "Category": outcome.category,
                "League": outcome.league,
                "Record": outcome.record,
                "Market Key": outcome.key,
            }
        )
    return pd.DataFrame(rows)


def count_auto_estimates(estimate_df: pd.DataFrame) -> int:
    if estimate_df.empty or "Auto True Probability %" not in estimate_df:
        return 0
    return sum(
        normalize_probability(value) is not None
        for value in estimate_df["Auto True Probability %"]
    )


# ================== VALUE MATH ==================
def relative_edge(true_prob: float, market_prob: float) -> float:
    if market_prob <= 0:
        return 0.0
    return (true_prob - market_prob) / market_prob


def expected_value_per_dollar(true_prob: float, market_prob: float) -> float:
    # Buying one share at price p returns $1 if correct; ROI on cost is true_prob / p - 1.
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


def analyze_value(
    outcomes: list[MarketOutcome],
    provider: ProbabilityProvider,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    stats = {
        "Candidate outcomes": len(outcomes),
        "Missing true probability": 0,
        "Filtered by edge": 0,
        "Filtered by Kelly": 0,
        "Value rows": 0,
    }

    for outcome in outcomes:
        estimate = provider.get(outcome)
        if estimate is None:
            stats["Missing true probability"] += 1
            continue

        edge = relative_edge(estimate.true_prob, outcome.market_prob)
        edge_pct = edge * 100
        if edge_pct < config.min_edge_pct:
            stats["Filtered by edge"] += 1
            continue

        ev = expected_value_per_dollar(estimate.true_prob, outcome.market_prob)
        full_kelly = full_kelly_fraction(estimate.true_prob, outcome.market_prob)
        adjusted_kelly = full_kelly * config.kelly_fraction
        kelly_pct = adjusted_kelly * 100

        if kelly_pct < config.min_kelly_pct:
            stats["Filtered by Kelly"] += 1
            continue

        rows.append(
            {
                "Market": outcome.market_name,
                "Outcome": outcome.outcome,
                "Polymarket Probability": outcome.market_prob,
                "My True Probability": estimate.true_prob,
                "Edge %": edge_pct,
                "Edge pp": (estimate.true_prob - outcome.market_prob) * 100,
                "EV %": ev * 100,
                "EV / $100": ev * 100,
                "Kelly %": kelly_pct,
                "Suggested Bet ($)": config.bankroll * adjusted_kelly,
                "Volume": outcome.volume,
                "Liquidity": outcome.liquidity,
                "Source": estimate.source,
                "Market Key": outcome.key,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        sort_col = {
            "Polymarket Probability": "Polymarket Probability",
            "Suggested Bet ($)": "Suggested Bet ($)",
            "Kelly %": "Kelly %",
            "Edge %": "Edge %",
            "EV %": "EV %",
            "Market": "Market",
        }[config.sort_by]
        df = df.sort_values(sort_col, ascending=config.sort_ascending)

    stats["Value rows"] = len(df)
    return df, stats


# ================== DISPLAY HELPERS ==================
def format_optional_money(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def display_value_table(df: pd.DataFrame):
    display_df = df.copy()
    display_df["Polymarket Probability"] = display_df["Polymarket Probability"].map(
        "{:.1%}".format
    )
    display_df["My True Probability"] = display_df["My True Probability"].map("{:.1%}".format)
    display_df["Edge %"] = display_df["Edge %"].map("{:+.1f}%".format)
    display_df["Edge pp"] = display_df["Edge pp"].map("{:+.1f}pp".format)
    display_df["EV %"] = display_df["EV %"].map("{:+.1f}%".format)
    display_df["EV / $100"] = display_df["EV / $100"].map("${:+.2f}".format)
    display_df["Kelly %"] = display_df["Kelly %"].map("{:.2f}%".format)
    display_df["Suggested Bet ($)"] = display_df["Suggested Bet ($)"].map("${:,.0f}".format)
    display_df["Volume"] = display_df["Volume"].map(format_optional_money)
    display_df["Liquidity"] = display_df["Liquidity"].map(format_optional_money)

    st.dataframe(
        display_df[
            [
                "Market",
                "Outcome",
                "Polymarket Probability",
                "My True Probability",
                "Edge %",
                "Edge pp",
                "EV %",
                "EV / $100",
                "Kelly %",
                "Suggested Bet ($)",
                "Volume",
                "Liquidity",
                "Source",
            ]
        ],
        width="stretch",
        height=620,
        hide_index=True,
    )


# ================== APP FLOW ==================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

col_btn, col_time = st.columns([1, 4])
with col_btn:
    manual_refresh = st.button("Refresh Now", type="primary")

markets = fetch_market_data()
if manual_refresh:
    st.cache_data.clear()
    st.session_state.last_refresh = time.time()
    markets = fetch_market_data()

outcomes, scan_stats = build_market_outcomes(markets, CONFIG)

with col_time:
    st.caption(
        f"Last updated: {datetime.now().strftime('%H:%M:%S')} | "
        f"{len(markets)} markets | {len(outcomes)} candidate outcomes"
    )

if DEBUG_MODE:
    st.subheader("Raw candidate outcomes")
    debug_df = pd.DataFrame([outcome.__dict__ for outcome in outcomes])
    st.dataframe(debug_df, width="stretch", hide_index=True)
    st.stop()

st.subheader("1. Automated True Probability Estimates")
st.caption(
    "The app estimates probabilities automatically from public NBA/NHL/MLB "
    "standings, FIFA men's rankings, and records embedded in markets. "
    "Unsupported markets use a neutral market-consensus fallback or stay blank, "
    "depending on your sidebar settings."
)

if not outcomes:
    st.warning("No markets match the current filters.")
    st.stop()

sports_model_data = fetch_sports_model_data() if CONFIG.use_auto_model else SportsModelData(ratings={})
auto_provider = build_auto_model_provider(outcomes, sports_model_data, CONFIG)
estimate_df = build_estimate_display_df(outcomes, auto_provider)
auto_estimate_count = count_auto_estimates(estimate_df)

estimate_cols = st.columns(4)
estimate_cols[0].metric("Candidate outcomes", len(outcomes))
estimate_cols[1].metric("Automated estimates", auto_estimate_count)
estimate_cols[2].metric("Still missing", max(0, len(outcomes) - auto_estimate_count))
estimate_cols[3].metric("Sports rating keys", len(sports_model_data.ratings))

if auto_estimate_count == 0:
    st.warning(
        "No automated estimates were generated. Turn on automated probabilities, "
        "enable market-consensus fallback, or widen the market filters."
    )

with st.expander("Model coverage and scan stats", expanded=False):
    coverage_df = estimate_df.copy()
    if coverage_df.empty:
        st.info("No model coverage to display for the current filters.")
    else:
        coverage_df["Polymarket Probability"] = coverage_df["Polymarket Probability"].map(
            "{:.1f}%".format
        )
        coverage_df["Auto True Probability %"] = coverage_df["Auto True Probability %"].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.1f}%"
        )
        st.dataframe(
            coverage_df[
                [
                    "Market",
                    "Outcome",
                    "Polymarket Probability",
                    "Auto True Probability %",
                    "Model Source",
                    "Category",
                    "League",
                    "Record",
                ]
            ],
            width="stretch",
            hide_index=True,
            height=360,
        )
    st.dataframe(
        pd.DataFrame(
            [{"Step": key, "Count": value} for key, value in scan_stats.items()]
        ),
        width="stretch",
        hide_index=True,
    )

probability_provider = auto_provider
value_df, value_stats = analyze_value(outcomes, probability_provider, CONFIG)

st.subheader("2. Value Bets")
if value_df.empty:
    if auto_estimate_count == 0:
        st.warning("No value bets yet because no automated estimates were generated.")
    else:
        st.warning(
            "No value bets passed your filters. Lower Minimum Edge / Minimum Kelly, "
            "or review model coverage in the table above."
        )
    with st.expander("Why no value bets?"):
        st.dataframe(
            pd.DataFrame(
                [{"Step": key, "Count": value} for key, value in value_stats.items()]
            ),
            width="stretch",
            hide_index=True,
        )
else:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Value rows", len(value_df))
    metric_cols[1].metric("Best edge", f"{value_df['Edge %'].max():.1f}%")
    metric_cols[2].metric("Best Kelly", f"{value_df['Kelly %'].max():.2f}%")
    metric_cols[3].metric("Total suggested", f"${value_df['Suggested Bet ($)'].sum():,.0f}")

    display_value_table(value_df)
    csv = value_df.to_csv(index=False).encode()
    st.download_button("Download value bets CSV", csv, "polymarket_value_bets.csv", "text/csv")

st.caption(
    "This is not financial advice. Kelly sizing is only as good as your true probability estimates."
)

if AUTO_REFRESH:
    elapsed = time.time() - st.session_state.last_refresh
    remaining = max(0, REFRESH_MIN * 60 - elapsed)
    st.empty().caption(f"Next auto-refresh in {int(remaining)}s")
    time.sleep(1)
    if remaining <= 1:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
    st.rerun()
