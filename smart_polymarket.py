import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re
from typing import Dict, List, Tuple, Any

# ==========================================
# 1. SYSTEM CONFIGURATION & UI ROUTINES
# ==========================================
st.set_page_config(
    page_title="Quantum Trading Scout — Live Production Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-Minimalist Tech Dashboard Customization
st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #ecf0f1;
            font-family: 'Inter', -apple-system, sans-serif;
        }
        .metric-container-box {
            background-color: #161a23;
            border: 1px solid #242b3d;
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
        }
        .metric-big-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #00ffcc;
        }
        .metric-sub-label {
            font-size: 0.8rem;
            color: #8a99ad;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }
        .status-tag {
            background-color: #0b2e24;
            color: #00ffaa;
            border: 1px solid #00aa77;
            padding: 0.3rem 0.7rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SYSTEM BOUNDARY CONFIGURATIONS
# ==========================================
class MarketCategory:
    NBA = "BASKETBALL_NBA"
    MLB = "BASEBALL_MLB"
    NFL = "FOOTBALL_NFL"
    SOCCER = "SOCCER"
    CRICKET = "CRICKET"
    UNKNOWN = "UNKNOWN"

ODDS_API_SPORT_MAP = {
    "basketball_nba": MarketCategory.NBA,
    "baseball_mlb": MarketCategory.MLB,
    "americanfootball_nfl": MarketCategory.NFL,
    "soccer_epl": MarketCategory.SOCCER,
    "cricket_ipl": MarketCategory.CRICKET
}

# ==========================================
# 3. LIVE DATA CONNECTIONS & INGESTION
# ==========================================
class LiveDataPipeline:
    
    @staticmethod
    def fetch_live_polymarket() -> List[Dict[str, Any]]:
        """Queries the public Polymarket API. Maps outcomePrices to curPrice to prevent KeyErrors."""
        url = "https://gamma-api.polymarket.com/markets"
        params = {"closed": "false", "active": "true", "limit": "150", "core": "true"}
        normalized = []
        try:
            res = requests.get(url, params=params, timeout=8)
            if res.status_code == 200:
                markets = res.json()
                for m in markets:
                    prices = m.get("outcomePrices")
                    if not prices: 
                        continue
                    
                    # Handle text/string encoded arrays safely
                    if isinstance(prices, str):
                        try: 
                            prices = json.loads(prices)
                        except: 
                            continue
                    
                    if len(prices) >= 1:
                        # CRITICAL: Maps live outcome prices directly to 'curPrice' to feed the risk model
                        normalized.append({
                            "title": m.get("title", ""),
                            "curPrice": float(prices[0]),  
                            "side": "Yes",
                            "volume": int(float(m.get("volume", 0)))
                        })
                return normalized
        except Exception as e:
            st.sidebar.error(f"Polymarket API Error: {e}")
        return []

    @staticmethod
    def fetch_sharp_lines(api_key: str) -> List[Dict[str, Any]]:
        """Queries The Odds API for real consensus lines using your production key."""
        if not api_key:
            return []
            
        aggregated = []
        sports_to_scan = ["basketball_nba", "baseball_mlb", "soccer_epl"]
        
        for sport in sports_to_scan:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
            params = {"apiKey": api_key, "regions": "us", "markets": "h2h"}
            try:
                res = requests.get(url, params=params, timeout=5)
                if res.status_code == 200:
                    for item in res.json():
                        books = item.get("bookmakers", [])
                        if not books: continue
                        outcomes = books[0].get("markets", [{}])[0].get("outcomes", [])
                        for out in outcomes:
                            odds = float(out.get("price", 0))
                            if odds > 0:
                                aggregated.append({
                                    "event_name": f"{item.get('home_team')} vs {item.get('away_team')} - {out.get('name')}",
                                    "sport_key": sport,
                                    "true_prob": 1.0 / odds
                                })
            except Exception:
                continue
        return aggregated

    @staticmethod
    def clean_and_tokenize(text: str) -> set:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return {w for w in text.split() if w not in {'will', 'win', 'to', 'the', 'is', 'at', 'least', 'vs', 'for'}}

    @classmethod
    def evaluate_overlap(cls, str1: str, str2: str) -> float:
        t1, t2 = cls.clean_and_tokenize(str1), cls.clean_and_tokenize(str2)
        if not t1 or not t2: return 0.0
        return len(t1.intersection(t2)) / min(len(t1), len(t2))

# ==========================================
# 4. INTERNAL CALCULATION HELPERS
# ==========================================
def calculate_risk_bounds(price: float) -> Tuple[str, str]:
    """Sample calculation routine matching your internal dataframe apply requirements."""
    if price < 0.30:
        return "High Risk", "Asymmetric Speculative (10-30%)"
    elif price < 0.70:
        return "Moderate Risk", "Balanced Dynamic (30-70%)"
    else:
        return "Low Risk", "High Probability Core (70-99%)"

# ==========================================
# 5. RUNTIME APPLICATION ENGINE
# ==========================================
def main():
    c_left, c_right = st.columns([3, 1])
    with c_left:
        st.title("⚡ QUANTUM TRADING SCOUT")
        st.markdown("<p style='color:#8a99ad; margin-top:-15px;'>Real-Time Value Arbitrage Processing Suite</p>", unsafe_allow_html=True)
    with c_right:
        st.markdown("<div style='text-align:right; margin-top:25px;'><span class='status-tag'>LIVE ENGINE ACTIVE</span></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Left Sidebar Navigation and Parameter Configuration
    st.sidebar.header("Risk Constraints")
    bankroll = st.sidebar.number_input("Capital Pool Bankroll ($)", min_value=100.0, value=1000.0, step=100.0)
    min_edge_pct = st.sidebar.slider("Minimum Edge Threshold (%)", 0.0, 10.0, 2.0, 0.5) / 100.0
    kelly_fraction = st.sidebar.slider("Kelly Scale Multiplier", 0.05, 1.00, 0.25, 0.05)
    max_single_cap = st.sidebar.slider("Max Single Bet Cap (%)", 1, 25, 5) / 100.0
    text_match_threshold = st.sidebar.slider("Text Match Verification Threshold", 0.10, 1.00, 0.45, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Consensus Feed Authentication")
    odds_api_key = st.sidebar.text_input("The Odds API Token Key", type="password")
    
    # Pipeline Data Collection Step
    pm_raw_data = LiveDataPipeline.fetch_live_polymarket()
    sharp_feed = LiveDataPipeline.fetch_sharp_lines(odds_api_key)

    processed_execution_vector = []
    
    # Process Dataframe Logic Safely if Everything is Connected
    if pm_raw_data:
        # Build DataFrame out of raw entries safely populated with 'curPrice'
        stock_info_df = pd.DataFrame(pm_raw_data)
        
        # Apply your exact calculation block without hitting a KeyError
        if not stock_info_df.empty and 'curPrice' in stock_info_df.columns:
            stock_info_df[['risk_range', 'probability_range']] = stock_info_df['curPrice'].apply(
                lambda x: pd.Series(calculate_risk_bounds(x))
            )
            
            # Cross-reference with live sportsbook data if key is assigned
            if odds_api_key and sharp_feed:
                for idx, row in stock_info_df.iterrows():
                    poly_title = row["title"]
                    
                    best_match = None
                    max_score = 0.0
                    for sharp in sharp_feed:
                        score = LiveDataPipeline.evaluate_overlap(poly_title, sharp["event_name"])
                        if score > max_score:
                            max_score = score
                            best_match = sharp
                    
                    # Validate against your adjustable text slider boundary
                    if best_match and max_score >= text_match_threshold:
                        true_probability = best_match["true_prob"]
                        poly_price = row["curPrice"]
                        
                        if true_probability > poly_price:
                            edge = true_probability - poly_price
                            if edge >= min_edge_pct:
                                raw_kelly = edge / (1.0 - poly_price)
                                safe_kelly = min(raw_kelly * kelly_fraction, max_single_cap)
                                
                                processed_execution_vector.append({
                                    "Market Asset": poly_title,
                                    "Risk Segment": row["risk_range"],
                                    "PM Live Price": f"${poly_price:.2f}",
                                    "Sharp Consensus": f"{true_probability * 100:.1f}%",
                                    "Edge Matrix": f"{edge * 100:.1f}%",
                                    "Kelly Weight": f"{safe_kelly * 100:.2f}%",
                                    "Target Trade": f"${bankroll * safe_kelly:.2f}",
                                    "24h Volume": f"${row['volume']:,}"
                                })

    # Render Active KPI Metrics Panel
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"<div class='metric-container-box'><div class='metric-big-value'>{len(pm_raw_data)}</div><div class='metric-sub-label'>Live Contracts Pulled</div></div>", unsafe_allow_html=True)
    with kpi2:
        sharp_count = len(sharp_feed) if odds_api_key else "0 (No Key)"
        st.markdown(f"<div class='metric-container-box'><div class='metric-big-value'>{sharp_count}</div><div class='metric-sub-label'>Sharp Book Lines Parsed</div></div>", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"<div class='metric-container-box'><div class='metric-big-value'>{len(processed_execution_vector)}</div><div class='metric-sub-label'>Verified Value Signals</div></div>", unsafe_allow_html=True)
    with kpi4:
        state_tag = "RUNNING LIVE" if odds_api_key else "WAITING FOR ODDS KEY"
        st.markdown(f"<div class='metric-container-box'><div class='metric-big-value'>{state_tag}</div><div class='metric-sub-label'>Pipeline Stream State</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⚡ Live Arbitrage Processing Matrix")

    # Inefficiency Display Logic Router
    if not odds_api_key:
        st.info("🔑 API Core operational. Drop your key into 'The Odds API Token Key' field in the sidebar to run live mathematical matrix cross-checks.")
    elif processed_execution_vector:
        df_matrix = pd.DataFrame(processed_execution_vector)
        st.dataframe(df_matrix, use_container_width=True, hide_index=True)
    else:
        st.warning("Live feed parsing active. No market variations matched your current text threshold and edge requirements. Try easing your text threshold match slider down slightly.")

if __name__ == "__main__":
    main()
