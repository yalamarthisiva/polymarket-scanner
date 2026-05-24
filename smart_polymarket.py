import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd
import requests
import streamlit as st


POLYMARKET_US_BASE_URL = "https://gateway.polymarket.us/v1"


st.set_page_config(page_title="Smart Polymarket Value Tool", layout="wide")
st.title("Smart Polymarket Value Tool")
st.info(
    "Using the Polymarket US public API only. This tool only calculates value "
    "when you provide your own true probability."
)


# ================== DOMAIN MODEL ==================
@dataclass(frozen=True)
class MarketOutcome:
    key: str
    market_id: str
    slug: str
    category: str
    market_name: str
    outcome: str
    market_prob: float
    volume: float | None
    liquidity: float | None


@dataclass(frozen=True)
class ProbabilityEstimate:
    true_prob: float
    source: str


class ProbabilityProvider(Protocol):
    """Small extension point for manual inputs, files, or future models."""

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


@dataclass
class FileProbabilityProvider:
    by_key: dict[str, ProbabilityEstimate]
    by_market_outcome: dict[tuple[str, str], ProbabilityEstimate]
    group_rules: list[dict]

    def get(self, outcome: MarketOutcome) -> ProbabilityEstimate | None:
        if outcome.key in self.by_key:
            return self.by_key[outcome.key]

        name_key = (outcome.market_name.lower(), outcome.outcome.lower())
        if name_key in self.by_market_outcome:
            return self.by_market_outcome[name_key]

        for rule in self.group_rules:
            contains = rule.get("contains")
            category = rule.get("category")
            rule_outcome = rule.get("outcome")

            if contains and contains.lower() not in outcome.market_name.lower():
                continue
            if category and category.lower() != outcome.category.lower():
                continue
            if rule_outcome and rule_outcome.lower() != outcome.outcome.lower():
                continue

            return ProbabilityEstimate(rule["true_prob"], rule["source"])

        return None


@dataclass
class ManualProbabilityProvider:
    by_key: dict[str, ProbabilityEstimate]

    def get(self, outcome: MarketOutcome) -> ProbabilityEstimate | None:
        return self.by_key.get(outcome.key)


@dataclass
class CompositeProbabilityProvider:
    providers: list[ProbabilityProvider]

    def get(self, outcome: MarketOutcome) -> ProbabilityEstimate | None:
        for provider in self.providers:
            estimate = provider.get(outcome)
            if estimate is not None:
                return estimate
        return None


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
    value=(0.20, 0.80),
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


def market_outcomes(market: dict) -> list[tuple[str, float]]:
    sides = market.get("marketSides")
    if isinstance(sides, list) and sides:
        outcomes = []
        for side in sides:
            name = side.get("description") or ("Yes" if side.get("long") else "No")
            price = market_side_price(side)
            if price is not None:
                outcomes.append((name, price))
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
            outcomes.append((name, price))
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

        market_name = clear_market_name(market)
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

        for outcome, market_prob in outcomes:
            if not side_allowed(outcome, config):
                stats["Side skipped"] += 1
                continue

            if not (config.min_market_prob <= market_prob <= config.max_market_prob):
                stats["Probability skipped"] += 1
                continue

            rows.append(
                MarketOutcome(
                    key=make_market_key(market, outcome),
                    market_id=str(market.get("id", "")),
                    slug=str(market.get("slug", "")),
                    category=category,
                    market_name=market_name,
                    outcome=outcome,
                    market_prob=market_prob,
                    volume=volume,
                    liquidity=liquidity,
                )
            )

    stats["Candidate outcomes"] = len(rows)
    return rows, stats


# ================== PROBABILITY INPUTS ==================
def empty_file_provider() -> FileProbabilityProvider:
    return FileProbabilityProvider(by_key={}, by_market_outcome={}, group_rules=[])


def estimate_from_row(row: dict, source: str) -> tuple[str | None, ProbabilityEstimate | None]:
    true_prob = normalize_probability(
        row.get("true_prob")
        or row.get("my_true_prob")
        or row.get("My True Probability")
        or row.get("My True Probability %")
    )
    if true_prob is None:
        return None, None

    key = row.get("key") or row.get("market_key") or row.get("Market Key")
    return key, ProbabilityEstimate(true_prob=true_prob, source=source)


def load_probability_file(uploaded_file) -> FileProbabilityProvider:
    if uploaded_file is None:
        return empty_file_provider()

    by_key: dict[str, ProbabilityEstimate] = {}
    by_market_outcome: dict[tuple[str, str], ProbabilityEstimate] = {}
    group_rules: list[dict] = []
    filename = uploaded_file.name

    try:
        if filename.lower().endswith(".csv"):
            source_df = pd.read_csv(uploaded_file)
            records = source_df.to_dict("records")
        else:
            payload = json.load(uploaded_file)
            if isinstance(payload, dict) and "probabilities" in payload:
                records = payload["probabilities"]
            elif isinstance(payload, dict):
                records = [{"key": key, "true_prob": value} for key, value in payload.items()]
            elif isinstance(payload, list):
                records = payload
            else:
                records = []
    except Exception as exc:
        st.warning(f"Could not read probability file: {exc}")
        return empty_file_provider()

    for row in records:
        if not isinstance(row, dict):
            continue

        true_prob = normalize_probability(
            row.get("true_prob")
            or row.get("my_true_prob")
            or row.get("My True Probability")
            or row.get("My True Probability %")
        )
        if true_prob is None:
            continue

        estimate = ProbabilityEstimate(true_prob=true_prob, source=f"file:{filename}")
        key = row.get("key") or row.get("market_key") or row.get("Market Key")
        market = row.get("market") or row.get("Market")
        outcome = row.get("outcome") or row.get("Outcome")
        contains = row.get("contains") or row.get("market_contains")
        category = row.get("category") or row.get("Category")

        if key:
            by_key[str(key)] = estimate
        elif market and outcome:
            by_market_outcome[(str(market).lower(), str(outcome).lower())] = estimate
        elif contains or category:
            group_rules.append(
                {
                    "contains": str(contains) if contains else None,
                    "category": str(category) if category else None,
                    "outcome": str(outcome) if outcome else None,
                    "true_prob": true_prob,
                    "source": f"file:{filename}",
                }
            )

    return FileProbabilityProvider(
        by_key=by_key,
        by_market_outcome=by_market_outcome,
        group_rules=group_rules,
    )


def build_probability_editor_df(
    outcomes: list[MarketOutcome], provider: ProbabilityProvider
) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        estimate = provider.get(outcome)
        rows.append(
            {
                "Market Key": outcome.key,
                "Market": outcome.market_name,
                "Outcome": outcome.outcome,
                "Polymarket Probability": outcome.market_prob * 100,
                "My True Probability %": (
                    estimate.true_prob * 100 if estimate is not None else None
                ),
                "Source": estimate.source if estimate is not None else "",
                "Category": outcome.category,
                "Volume": outcome.volume,
                "Liquidity": outcome.liquidity,
            }
        )
    return pd.DataFrame(rows)


def manual_provider_from_editor(
    editor_df: pd.DataFrame, file_provider: ProbabilityProvider
) -> ManualProbabilityProvider:
    by_key = {}
    if editor_df.empty:
        return ManualProbabilityProvider(by_key=by_key)

    for row in editor_df.to_dict("records"):
        key = row.get("Market Key")
        true_prob = normalize_probability(row.get("My True Probability %"))
        if key and true_prob is not None:
            seed_outcome = MarketOutcome(
                key=str(key),
                market_id="",
                slug="",
                category=str(row.get("Category") or ""),
                market_name=str(row.get("Market") or ""),
                outcome=str(row.get("Outcome") or ""),
                market_prob=0.0,
                volume=None,
                liquidity=None,
            )
            file_estimate = file_provider.get(seed_outcome)
            if file_estimate and abs(file_estimate.true_prob - true_prob) < 0.000001:
                continue
            by_key[str(key)] = ProbabilityEstimate(true_prob=true_prob, source="manual")

    return ManualProbabilityProvider(by_key=by_key)


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
        use_container_width=True,
        height=620,
        hide_index=True,
    )


def download_template(outcomes: list[MarketOutcome]) -> bytes:
    template_df = pd.DataFrame(
        [
            {
                "Market Key": outcome.key,
                "Market": outcome.market_name,
                "Outcome": outcome.outcome,
                "Polymarket Probability": round(outcome.market_prob, 4),
                "My True Probability %": "",
            }
            for outcome in outcomes
        ]
    )
    return template_df.to_csv(index=False).encode()


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
    st.dataframe(debug_df, use_container_width=True, hide_index=True)
    st.stop()

st.subheader("1. Add Your True Probabilities")
st.caption(
    "Upload estimates or type them below. Values may be entered as 0.62 or 62 for 62%."
)

uploaded_probabilities = st.file_uploader(
    "Load true probabilities from CSV or JSON",
    type=["csv", "json"],
)
file_provider = load_probability_file(uploaded_probabilities)

template_col, stats_col = st.columns([1, 3])
with template_col:
    st.download_button(
        "Download CSV template",
        download_template(outcomes),
        "polymarket_true_prob_template.csv",
        "text/csv",
    )
with stats_col:
    with st.expander("File formats and scan stats"):
        st.markdown(
            """
CSV columns accepted: `Market Key`, `My True Probability %`.

Optional CSV/JSON columns for bulk rules: `contains`, `category`, `outcome`, `true_prob`.

JSON accepted: `{"market-key::yes": 0.62}` or a list of rows with `key` and `true_prob`.
"""
        )
        st.dataframe(
            pd.DataFrame(
                [{"Step": key, "Count": value} for key, value in scan_stats.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

editor_seed_provider = CompositeProbabilityProvider([file_provider])
editor_df = build_probability_editor_df(outcomes, editor_seed_provider)

if editor_df.empty:
    st.warning("No markets match the current filters.")
    st.stop()

edited_df = st.data_editor(
    editor_df,
    use_container_width=True,
    hide_index=True,
    height=360,
    disabled=[
        "Market Key",
        "Market",
        "Outcome",
        "Polymarket Probability",
        "Source",
        "Category",
        "Volume",
        "Liquidity",
    ],
    column_order=[
        "Market",
        "Outcome",
        "Polymarket Probability",
        "My True Probability %",
        "Source",
        "Category",
        "Volume",
        "Liquidity",
        "Market Key",
    ],
    key="true_probability_editor",
)

manual_provider = manual_provider_from_editor(edited_df, file_provider)
probability_provider = CompositeProbabilityProvider([manual_provider, file_provider])
value_df, value_stats = analyze_value(outcomes, probability_provider, CONFIG)

st.subheader("2. Value Bets")
if value_df.empty:
    st.warning(
        "No value bets yet. Add true probabilities above, lower Minimum Edge, "
        "or widen the Polymarket probability range."
    )
    with st.expander("Why no value bets?"):
        st.dataframe(
            pd.DataFrame(
                [{"Step": key, "Count": value} for key, value in value_stats.items()]
            ),
            use_container_width=True,
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
