import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime

st.set_page_config(page_title="Smart Polymarket Scanner", layout="wide")
st.title("🧠 Smart Polymarket Value Bets")
st.caption("Free version • Adjustable filters")

# ================== SIDEBAR CONTROLS ==================
st.sidebar.header("Filters")
BANKROLL    = st.sidebar.number_input("Bankroll ($)", value=10_000, min_value=1_000)
MIN_KELLY   = st.sidebar.slider("Minimum Kelly %", 1, 40, 10)
MIN_VOLUME  = st.sidebar.number_input("Minimum Volume ($)", value=300_000, step=100_000)
MIN_PROB    = st.sidebar.slider("Minimum Probability", 0.50, 0.85, 0.68)
MAX_PROB    = st.sidebar.slider("Maximum Probability", 0.85, 0.99, 0.93)
EDGE_FLOOR  = st.sidebar.slider(
    "Min Edge % (your prob vs market)",
    min_value=1, max_value=20, value=5,
    help="How much higher your true-probability estimate must be above the market price."
)
CATEGORY    = st.sidebar.selectbox("Category", ["All", "Politics", "Sports", "World"])
FRAC_KELLY  = st.sidebar.slider(
    "Kelly Fraction", min_value=0.10, max_value=1.0, value=0.25, step=0.05,
    help="Fractional Kelly reduces variance. 0.25 = quarter-Kelly is standard."
)

st.sidebar.markdown("---")
AUTO_REFRESH  = st.sidebar.checkbox("Auto Refresh", value=True)
REFRESH_MIN   = st.sidebar.slider("Refresh every (minutes)", 3, 15, 5)

# ================== DATA LAYER (cached independently of filters) ==================
@st.cache_data(ttl=180, show_spinner="Fetching market data...")
def fetch_market_data() -> list[dict]:
    """Raw fetch — no filter logic here so the cache is filter-agnostic."""
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
    """
    Estimate a 'true' probability by assuming the market is cheap by `edge_pct`.
    Real use-case: replace this with your own model / news signal.
    """
    return min(market_prob * (1 + edge_pct / 100), 0.999)


def kelly_fraction(true_prob: float, market_prob: float) -> float:
    """
    Full Kelly for a binary market.

    b = net odds on a $1 bet  = (1 / market_prob) - 1
    f* = (p*(b+1) - 1) / b   = (p - market_prob) / (1 - market_prob)

    Note: f* > 0  iff  true_prob > market_prob  (you have positive edge).
    If market_prob == true_prob there is zero edge and Kelly is 0.
    """
    b = (1.0 / market_prob) - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (true_prob * (b + 1) - 1) / b)


def analyze(
    markets: list[dict],
    bankroll: float,
    min_kelly: float,
    min_volume: float,
    min_prob: float,
    max_prob: float,
    edge_pct: float,
    frac_kelly: float,
    category: str,
) -> pd.DataFrame:
    results = []

    for m in markets:
        title  = m.get("question", "")
        volume = float(m.get("volumeNum", 0))

        if volume < min_volume:
            continue

        if category != "All":
            tags = str(m.get("tags", "")) + str(m.get("categories", ""))
            if category.lower() not in tags.lower():
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

            true_prob  = implied_edge(market_prob, edge_pct)
            raw_kelly  = kelly_fraction(true_prob, market_prob)
            adj_kelly  = raw_kelly * frac_kelly          # fractional Kelly
            edge       = (true_prob - market_prob) * 100 # percentage points

            if adj_kelly * 100 < min_kelly:
                continue

            bet_size = round(bankroll * adj_kelly, 2)
            exp_value = (true_prob * (1 / market_prob - 1) - (1 - true_prob)) * 100

            results.append(
                {
                    "Market":       title[:78] + "…" if len(title) > 78 else title,
                    "Side":         side,
                    "Mkt Prob":     market_prob,
                    "True Prob":    true_prob,
                    "Edge (pp)":    round(edge, 1),
                    "Kelly %":      round(adj_kelly * 100, 1),
                    "EV %":         round(exp_value, 1),
                    "Bet ($)":      bet_size,
                    "Volume ($)":   volume,
                }
            )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Kelly %", ascending=False)
    return df


# ================== AUTO-REFRESH (no extra packages) ==================
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

df = analyze(
    markets,
    bankroll=BANKROLL,
    min_kelly=MIN_KELLY,
    min_volume=MIN_VOLUME,
    min_prob=MIN_PROB,
    max_prob=MAX_PROB,
    edge_pct=EDGE_FLOOR,
    frac_kelly=FRAC_KELLY,
    category=CATEGORY,
)

with col_time:
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} • {len(markets)} markets loaded")

if df.empty:
    st.warning("No bets match your current filters. Try lowering Kelly %, Volume, or Edge %.")
else:
    # Pretty-format for display only
    display_df = df.copy()
    display_df["Mkt Prob"]  = display_df["Mkt Prob"].map("{:.1%}".format)
    display_df["True Prob"] = display_df["True Prob"].map("{:.1%}".format)
    display_df["Edge (pp)"] = display_df["Edge (pp)"].map("{:+.1f}pp".format)
    display_df["Kelly %"]   = display_df["Kelly %"].map("{:.1f}%".format)
    display_df["EV %"]      = display_df["EV %"].map("{:+.1f}%".format)
    display_df["Bet ($)"]   = display_df["Bet ($)"].map("${:,.0f}".format)
    display_df["Volume ($)"] = display_df["Volume ($)"].map("${:,.0f}".format)

    st.dataframe(display_df, use_container_width=True, height=650)
    st.success(f"Found **{len(df)}** opportunities")

    csv = df.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv, "polymarket_bets.csv", "text/csv")

st.caption(
    "💡 **Edge %** is a placeholder for your own probability model — "
    "replace `implied_edge()` with real signal for live use. "
    "Always do your own research."
)

# ================== NON-BLOCKING AUTO-REFRESH LOOP ==================
# Sits at the bottom AFTER all UI is rendered. Uses a 1-second sleep + rerun
# to poll the clock. The user sees a live, interactive page the whole time.
if AUTO_REFRESH:
    elapsed = time.time() - st.session_state.last_refresh
    remaining = max(0, REFRESH_MIN * 60 - elapsed)
    countdown_slot = st.empty()
    countdown_slot.caption(f"⏱ Next auto-refresh in {int(remaining)}s")
    time.sleep(1)
    if remaining <= 1:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
    st.rerun()