import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime

st.set_page_config(page_title="Smart Polymarket Scanner", layout="wide")
st.title("🧠 Smart Polymarket Value Bets")
st.info(
    "🇺🇸 Filtered for US Polymarket app markets. "
    "Sports are fully live; Politics & Crypto expanding. "
    "Not available in AZ, IL, MA, MD, MI, MT, NJ, NV, OH."
)

# ================== SIDEBAR CONTROLS ==================
st.sidebar.header("Filters")
BANKROLL   = st.sidebar.number_input("Bankroll ($)", value=10_000, min_value=1_000)
MIN_KELLY  = st.sidebar.slider("Minimum Kelly %", 1, 40, 5)
MIN_VOLUME = st.sidebar.number_input("Minimum Volume ($)", value=50_000, step=50_000)
MIN_PROB   = st.sidebar.slider("Minimum Probability", 0.50, 0.85, 0.55)
MAX_PROB   = st.sidebar.slider("Maximum Probability", 0.85, 0.99, 0.95)
EDGE_FLOOR = st.sidebar.slider(
    "Min Edge %", min_value=1, max_value=20, value=2,
    help="How much higher your true-probability estimate must be above the market price."
)
FRAC_KELLY = st.sidebar.slider(
    "Kelly Fraction", min_value=0.10, max_value=1.0, value=0.25, step=0.05,
    help="0.25 = quarter-Kelly (recommended)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Categories")
CAT_SPORTS   = st.sidebar.checkbox("⚽ Sports", value=True)
CAT_POLITICS = st.sidebar.checkbox("🏛 Politics", value=True)
CAT_CRYPTO   = st.sidebar.checkbox("₿ Crypto", value=True)
CAT_CULTURE  = st.sidebar.checkbox("🎬 Culture / Entertainment", value=False)
CAT_ALL      = st.sidebar.checkbox("🌐 Show All (ignore category filter)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Bet Side")
SIDE_FILTER = st.sidebar.radio(
    "Show bets",
    options=["YES only", "NO only", "Both"],
    index=0,  # default: YES only
    help="YES = you think it will happen. NO = you think it won't."
)

st.sidebar.markdown("---")
DEBUG_MODE   = st.sidebar.checkbox("🔍 Debug: show raw titles", value=False)
AUTO_REFRESH = st.sidebar.checkbox("Auto Refresh", value=True)
REFRESH_MIN  = st.sidebar.slider("Refresh every (minutes)", 3, 15, 5)

# ================== KEYWORD MAPS ==================
# Politics checked FIRST so political figures don't bleed into Sports
POLITICS_KW = [
    "election", "president", "senate", "congress", "vote", "poll",
    "approval", "bill", "policy", "democrat", "republican", "governor",
    "tariff", "federal reserve", "fed rate", "supreme court", "white house",
    "executive order", "secretary of", "minister", "parliament", "referendum",
    "trump", "biden", "harris", "kamala", "macron", "modi", "zelensky", "putin",
    "aoc", "ocasio-cortez", "ocasio", "desantis", "newsom", "pelosi", "pence",
    "nato", "g7", "g20", "sanctions", "impeach", "cabinet", "inaugur",
    "midterm", "primary", "ballot", "legislation", "veto", "nominee",
    "nomination", "presidential", "republican nomination", "democratic nomination",
    "ossoff", "carlson", "tucker", "ron paul", "rfk",
]

SPORTS_KW = [
    "nba", "nfl", "mlb", "nhl", "ufc", "mma", "nascar", "pga", "tennis",
    "soccer", "mls", "championship", "playoffs", "world series", "stanley cup",
    "super bowl", "superbowl", "nba finals", "nfl draft", "mvp", "transfer",
    "tournament", "grand slam", "wimbledon", "us open", "masters", "ryder cup",
    "world cup", "fifa", "euros", "copa america", "champions league",
    "lakers", "celtics", "warriors", "bulls", "heat", "knicks", "nets",
    "yankees", "dodgers", "astros", "braves", "mets", "cubs", "red sox",
    "chiefs", "eagles", "cowboys", "patriots", "49ers", "bills", "ravens",
    "liverpool", "arsenal", "chelsea", "manchester", "barcelona", "real madrid",
    "coach fired", "traded", "roster", "signed", "free agent",
]

CRYPTO_KW = [
    "bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol",
    "defi", "nft", "token", "blockchain", "etf", "coinbase", "binance",
    "xrp", "ripple", "doge", "dogecoin", "microstrategy", "kraken",
    "altcoin", "stablecoin", "usdc", "tether", "usdt", "web3",
    "halving", "mining", "wallet", "exchange",
]

CULTURE_KW = [
    "oscar", "grammy", "emmy", "bafta", "box office", "album", "movie",
    "film", "show", "celebrity", "award", "netflix", "spotify", "billboard",
    "taylor swift", "beyonce", "kardashian", "number one", "chart",
]


def detect_category(title: str) -> str | None:
    """Politics is checked before Sports to avoid misclassifying political figures."""
    if CAT_ALL:
        return "🌐 All"
    t = title.lower()
    if CAT_POLITICS and any(kw in t for kw in POLITICS_KW):
        return "🏛 Politics"
    if CAT_SPORTS and any(kw in t for kw in SPORTS_KW):
        return "⚽ Sports"
    if CAT_CRYPTO and any(kw in t for kw in CRYPTO_KW):
        return "₿ Crypto"
    if CAT_CULTURE and any(kw in t for kw in CULTURE_KW):
        return "🎬 Culture"
    return None


# ================== DATA LAYER ==================
@st.cache_data(ttl=180, show_spinner="Fetching market data...")
def fetch_market_data() -> list[dict]:
    try:
        r = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"active": "true", "closed": "false", "limit": 500},
            timeout=25,
        )
        r.raise_for_status()
        events = r.json()
        return [m for e in events for m in e.get("markets", [])]
    except Exception as exc:
        st.warning(f"Fetch failed: {exc}")
        return []


# ================== ANALYSIS LAYER ==================
def implied_edge(market_prob: float, edge_pct: float) -> float:
    return min(market_prob * (1 + edge_pct / 100), 0.999)


def kelly_fraction(true_prob: float, market_prob: float) -> float:
    b = (1.0 / market_prob) - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (true_prob * (b + 1) - 1) / b)


def analyze(markets) -> pd.DataFrame:
    results = []
    for m in markets:
        title  = m.get("question", "")
        volume = float(m.get("volumeNum", 0))

        if volume < MIN_VOLUME:
            continue

        cat_label = detect_category(title)
        if cat_label is None:
            continue

        try:
            outcomes = json.loads(m.get("outcomes", "[]"))
            probs    = json.loads(m.get("outcomePrices", "[]"))
        except (json.JSONDecodeError, TypeError):
            continue

        for side, p_str in zip(outcomes, probs):
            try:
                market_prob = float(p_str)
            except ValueError:
                continue

            # Apply side filter
            side_upper = side.strip().upper()
            if SIDE_FILTER == "YES only" and side_upper != "YES":
                continue
            if SIDE_FILTER == "NO only" and side_upper != "NO":
                continue

            if not (MIN_PROB <= market_prob <= MAX_PROB):
                continue

            true_prob = implied_edge(market_prob, EDGE_FLOOR)
            raw_kelly = kelly_fraction(true_prob, market_prob)
            adj_kelly = raw_kelly * FRAC_KELLY
            edge      = (true_prob - market_prob) * 100

            if adj_kelly * 100 < MIN_KELLY:
                continue

            bet_size  = round(BANKROLL * adj_kelly, 2)
            exp_value = (true_prob * (1 / market_prob - 1) - (1 - true_prob)) * 100

            results.append({
                "Category":   cat_label,
                "Market":     title[:75] + "…" if len(title) > 75 else title,
                "Side":       side,
                "Mkt Prob":   market_prob,
                "True Prob":  true_prob,
                "Edge (pp)":  round(edge, 1),
                "Kelly %":    round(adj_kelly * 100, 1),
                "EV %":       round(exp_value, 1),
                "Bet ($)":    bet_size,
                "Volume ($)": volume,
            })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Kelly %", ascending=False)
    return df


# ================== AUTO-REFRESH STATE ==================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# ================== UI ==================
col_btn, col_time = st.columns([1, 4])
with col_btn:
    manual_refresh = st.button("🔄 Refresh Now", type="primary")

markets = fetch_market_data()

if manual_refresh:
    st.cache_data.clear()
    markets = fetch_market_data()

# ── DEBUG MODE ──────────────────────────────────────────────────────────────
if DEBUG_MODE:
    st.subheader("🔍 Raw market titles (first 50)")
    debug_rows = []
    for m in markets[:50]:
        title = m.get("question", "")
        cat   = detect_category(title) or "— no match —"
        debug_rows.append({
            "title":    title[:80],
            "category": cat,
            "volume":   float(m.get("volumeNum", 0)),
        })
    st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)
    st.stop()
# ───────────────────────────────────────────────────────────────────────────

df = analyze(markets)

with col_time:
    st.caption(
        f"Last updated: {datetime.now().strftime('%H:%M:%S')} • "
        f"{len(markets)} total markets • {len(df)} matches"
    )

if df.empty:
    st.warning(
        "No bets match your filters. Try: enabling '🌐 Show All', "
        "switching Side to 'Both', lowering Min Volume, or widening probability range."
    )
else:
    # Category breakdown metrics
    cat_counts = df["Category"].value_counts()
    cols = st.columns(len(cat_counts))
    for col, (cat, count) in zip(cols, cat_counts.items()):
        col.metric(cat, count)

    display_df = df.copy()
    display_df["Mkt Prob"]   = display_df["Mkt Prob"].map("{:.1%}".format)
    display_df["True Prob"]  = display_df["True Prob"].map("{:.1%}".format)
    display_df["Edge (pp)"]  = display_df["Edge (pp)"].map("{:+.1f}pp".format)
    display_df["Kelly %"]    = display_df["Kelly %"].map("{:.1f}%".format)
    display_df["EV %"]       = display_df["EV %"].map("{:+.1f}%".format)
    display_df["Bet ($)"]    = display_df["Bet ($)"].map("${:,.0f}".format)
    display_df["Volume ($)"] = display_df["Volume ($)"].map("${:,.0f}".format)

    st.dataframe(display_df, use_container_width=True, height=600)
    st.success(f"Found **{len(df)}** opportunities")

    csv = df.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv, "polymarket_bets.csv", "text/csv")

st.caption(
    "💡 Edge % is a placeholder — replace `implied_edge()` with a real model for live use. "
    "Always do your own research."
)

# ================== NON-BLOCKING AUTO-REFRESH ==================
if AUTO_REFRESH:
    elapsed   = time.time() - st.session_state.last_refresh
    remaining = max(0, REFRESH_MIN * 60 - elapsed)
    st.empty().caption(f"⏱ Next auto-refresh in {int(remaining)}s")
    time.sleep(1)
    if remaining <= 1:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
    st.rerun()