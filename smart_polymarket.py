import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime

st.set_page_config(page_title="Smart Polymarket Scanner", layout="wide")
st.title("🧠 Smart Polymarket Value Bets — US Edition")
st.info(
    "🇺🇸 Filtered for **US Polymarket app** markets. "
    "Sports are fully live; Politics & Crypto expanding. "
    "Not available in AZ, IL, MA, MD, MI, MT, NJ, NV, OH."
)

# ================== SIDEBAR CONTROLS ==================
st.sidebar.header("Filters")
BANKROLL   = st.sidebar.number_input("Bankroll ($)", value=10_000, min_value=1_000)
MIN_KELLY  = st.sidebar.slider("Minimum Kelly %", 1, 40, 10)
MIN_VOLUME = st.sidebar.number_input("Minimum Volume ($)", value=100_000, step=50_000)
MIN_PROB   = st.sidebar.slider("Minimum Probability", 0.50, 0.85, 0.60)
MAX_PROB   = st.sidebar.slider("Maximum Probability", 0.85, 0.99, 0.93)
EDGE_FLOOR = st.sidebar.slider(
    "Min Edge %",
    min_value=1, max_value=20, value=3,
    help="How much higher your true-probability estimate must be above the market price."
)
FRAC_KELLY = st.sidebar.slider(
    "Kelly Fraction", min_value=0.10, max_value=1.0, value=0.25, step=0.05,
    help="0.25 = quarter-Kelly (recommended). Lower = smaller, safer bets."
)

st.sidebar.markdown("---")
st.sidebar.subheader("US Market Categories")
CAT_SPORTS   = st.sidebar.checkbox("⚽ Sports", value=True)
CAT_POLITICS = st.sidebar.checkbox("🏛 Politics", value=True)
CAT_CRYPTO   = st.sidebar.checkbox("₿ Crypto", value=False)
CAT_CULTURE  = st.sidebar.checkbox("🎬 Culture / Current Events", value=False)

st.sidebar.markdown("---")
DEBUG_MODE   = st.sidebar.checkbox("🔍 Show raw tags (debug)", value=False)
AUTO_REFRESH = st.sidebar.checkbox("Auto Refresh", value=True)
REFRESH_MIN  = st.sidebar.slider("Refresh every (minutes)", 3, 15, 5)

ACTIVE_CATS = []
if CAT_SPORTS:   ACTIVE_CATS.append("sports")
if CAT_POLITICS: ACTIVE_CATS.append("politics")
if CAT_CRYPTO:   ACTIVE_CATS.append("crypto")
if CAT_CULTURE:  ACTIVE_CATS.extend(["culture", "entertainment", "current events"])

SPORTS_KEYWORDS = [
    "nba", "nfl", "mlb", "nhl", "ufc", "mma", "nascar", "pga", "tennis",
    "soccer", "mls", "championship", "playoffs", "world series", "stanley cup",
    "superbowl", "super bowl", "finals", "draft", "mvp", "title", "transfer",
    "win the", "score", "game winner", "match", "season", "tournament",
    "league", "cup", "series", "players championship", "grand slam",
]

POLITICS_KEYWORDS = [
    "election", "president", "senate", "congress", "vote", "poll",
    "approval", "bill", "policy", "democrat", "republican", "governor",
    "tariff", "fed rate", "supreme court", "white house", "executive order",
    "secretary", "minister", "parliament", "referendum",
]

CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol",
    "defi", "nft", "token", "blockchain", "etf", "coinbase", "binance",
    "xrp", "ripple", "doge", "dogecoin",
]

CULTURE_KEYWORDS = [
    "oscar", "grammy", "emmy", "box office", "album", "movie", "show",
    "celebrity", "award", "netflix", "spotify", "billboard",
]


def detect_category(title: str, tags_str: str) -> str | None:
    """Return category label if market matches any active category, else None."""
    combined = (title + " " + tags_str).lower()

    if "sports" in ACTIVE_CATS:
        if any(kw in combined for kw in SPORTS_KEYWORDS) or "sport" in combined:
            return "⚽ Sports"

    if "politics" in ACTIVE_CATS:
        if any(kw in combined for kw in POLITICS_KEYWORDS) or "politic" in combined:
            return "🏛 Politics"

    if "crypto" in ACTIVE_CATS:
        if any(kw in combined for kw in CRYPTO_KEYWORDS):
            return "₿ Crypto"

    if any(c in ACTIVE_CATS for c in ["culture", "entertainment", "current events"]):
        if any(kw in combined for kw in CULTURE_KEYWORDS):
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


def analyze(markets, bankroll, min_kelly, min_volume,
            min_prob, max_prob, edge_pct, frac_kelly) -> pd.DataFrame:
    results = []

    for m in markets:
        title  = m.get("question", "")
        volume = float(m.get("volumeNum", 0))

        if volume < min_volume:
            continue

        tags_str  = str(m.get("tags", "")) + " " + str(m.get("categories", ""))
        cat_label = detect_category(title, tags_str)
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

            if not (min_prob <= market_prob <= max_prob):
                continue

            true_prob = implied_edge(market_prob, edge_pct)
            raw_kelly = kelly_fraction(true_prob, market_prob)
            adj_kelly = raw_kelly * frac_kelly
            edge      = (true_prob - market_prob) * 100

            if adj_kelly * 100 < min_kelly:
                continue

            bet_size  = round(bankroll * adj_kelly, 2)
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
    st.subheader("🔍 Raw API sample (first 30 markets)")
    st.caption("Use this to see actual tag/category values from the API")
    debug_rows = []
    for m in markets[:30]:
        debug_rows.append({
            "title": m.get("question", "")[:70],
            "tags":  str(m.get("tags", ""))[:80],
            "cats":  str(m.get("categories", ""))[:80],
            "vol":   m.get("volumeNum", 0),
        })
    st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)
    st.stop()
# ───────────────────────────────────────────────────────────────────────────

df = analyze(
    markets,
    bankroll=BANKROLL,
    min_kelly=MIN_KELLY,
    min_volume=MIN_VOLUME,
    min_prob=MIN_PROB,
    max_prob=MAX_PROB,
    edge_pct=EDGE_FLOOR,
    frac_kelly=FRAC_KELLY,
)

with col_time:
    st.caption(
        f"Last updated: {datetime.now().strftime('%H:%M:%S')} • "
        f"{len(markets)} total markets • {len(df)} US matches"
    )

if not ACTIVE_CATS:
    st.warning("No categories selected — check at least one box in the sidebar.")
elif df.empty:
    st.warning(
        "No bets match your filters. Try: lowering Min Volume to $50k, "
        "lowering Min Kelly % to 1, or widening the probability range."
    )
else:
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
    st.success(f"Found **{len(df)}** tradeable opportunities")

    csv = df.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv, "polymarket_us_bets.csv", "text/csv")

st.caption(
    "💡 Edge % assumes your estimate is X% above market price — "
    "replace `implied_edge()` with a real model for live use. "
    "Sports markets are fully live on the US app. Always do your own research."
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
    st.rerun()# v3
