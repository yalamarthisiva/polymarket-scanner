import streamlit as st
import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Tuple, Any
from difflib import SequenceMatcher
from datetime import datetime

# ==========================================
# 1. SYSTEM CONFIGURATION & UI INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="Quantum Trading Scout - Production Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Minimalist Ultra-Dark Tech Aesthetic
st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #ecf0f1;
            font-family: 'Inter', -apple-system, sans-serif;
        }
        .metric-card {
            background-color: #161a23;
            border: 1px solid #242b3d;
            padding: 1.2rem;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #00ffcc;
            margin-bottom: 0.2rem;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #8a99ad;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .status-badge {
            background-color: #0b2e24;
            color: #00ffaa;
            border: 1px solid #00aa77;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==========================================
# 2. STRICT CATEGORY FIREWALL DEFINITIONS
# ==========================================
class MarketCategory:
    NBA = "BASKETBALL_NBA"
    MLB = "BASEBALL_MLB"
    NFL = "FOOTBALL_NFL"
    SOCCER = "SOCCER"
    CRICKET = "CRICKET"
    MACRO = "ECONOMICS_MACRO"
    UNKNOWN = "UNKNOWN"

# Maps The Odds API sport_key values directly to our strict internal categories
ODDS_API_SPORT_MAP = {
    "basketball_nba": MarketCategory.NBA,
    "baseball_mlb": MarketCategory.MLB,
    "americanfootball_nfl": MarketCategory.NFL,
    "soccer_epl": MarketCategory.SOCCER,
    "soccer_uefa_champs_league": MarketCategory.SOCCER,
    "soccer_fifa_world_cup": MarketCategory.SOCCER,
    "cricket_ipl": MarketCategory.CRICKET
}

# ==========================================
# 3. LIVE DATA INGESTION ENGINE (API LAYER)
# ==========================================
class DataIngestionEngine:
    """Handles raw data extraction from production API endpoints safely."""
    
    @staticmethod
    def fetch_polymarket_active_markets() -> List[Dict[str, Any]]:
        """Queries Polymarket Gamma API for active, unresolved binary markets."""
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            "closed": "false",
            "active": "true",
            "limit": 100
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                markets = response.json()
                # Return normalized format containing essential data items
                normalized = []
                for m in markets:
                    if isinstance(m, dict) and "title" in m and "outcomePrices" in m:
                        try:
                            prices = eval(m["outcomePrices"]) if isinstance(m["outcomePrices"], str) else m["outcomePrices"]
                            yes_price = float(prices[0]) if prices else 0.0
                            normalized.append({
                                "title": m.get("title", ""),
                                "tags": m.get("tags", []),
                                "yes_price": yes_price,
                                "group_by": m.get("category", "")
                            })
                        except Exception:
                            continue
                return normalized
            else:
                st.error(f"Polymarket API Error: Status {response.status_code}")
                return []
        except Exception as e:
            st.error(f"Polymarket Connection Failure: {str(e)}")
            return []

    @staticmethod
    def fetch_the_odds_lines(api_key: str, sports: List[str]) -> List[Dict[str, Any]]:
        """Queries The Odds API for sharp multi-bookmaker implied probabilities."""
        aggregated_sharp_lines = []
        if not api_key:
            return []

        for sport in sports:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
            params = {
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "decimal"
            }
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        # Extract the sharp consensus line from bookmakers (e.g., Circa / Pinnacle if available)
                        bookmakers = item.get("bookmakers", [])
                        if not bookmakers:
                            continue
                        
                        # Use first available bookmaker as our baseline sharp model proxy
                        market_data = bookmakers[0].get("markets", [{}])[0]
                        outcomes = market_data.get("outcomes", [])
                        
                        for outcome in outcomes:
                            dec_odds = float(outcome.get("price", 0.0))
                            if dec_odds > 1:
                                # Convert decimal odds to implied true probability
                                implied_prob = 1.0 / dec_odds
                                aggregated_sharp_lines.append({
                                    "event_name": f"{item.get('home_team')} vs {item.get('away_team')} - {outcome.get('name')}",
                                    "sport_key": sport,
                                    "true_probability": implied_prob
                                })
                elif response.status_code == 401:
                    st.error("The Odds API: Invalid Authentication Key.")
                    return []
            except Exception as e:
                logging.error(f"Failed to pull sharp odds for sport {sport}: {str(e)}")
                continue
        return aggregated_sharp_lines

# ==========================================
# 4. QUANT RISK & MATCHING EXECUTION LAYER
# ==========================================
class ProductionDataPipeline:
    def __init__(self, match_threshold: float = 0.80, fractional_kelly: float = 0.25, max_bet_allocation: float = 0.05):
        self.match_threshold = match_threshold
        self.fractional_kelly = fractional_kelly
        self.max_bet_allocation = max_bet_allocation

    @staticmethod
    def classify_polymarket_event(title: str, tags: List[str]) -> str:
        """Determines the explicit context using keyword heuristics."""
        combined_text = f"{title} {' '.join(tags or [])}".lower()
        if any(kw in combined_text for kw in ["nba", "basketball", "lebron", "celtics", "lakers", "playoffs"]):
            return MarketCategory.NBA
        if any(kw in combined_text for kw in ["mlb", "baseball", "yankees", "red sox", "world series"]):
            return MarketCategory.MLB
        if any(kw in combined_text for kw in ["nfl", "football", "super bowl", "quarterback", "chiefs"]):
            return MarketCategory.NFL
        if any(kw in combined_text for kw in ["world cup", "fifa", "epl", "premier league", "soccer", "la liga", "champions league"]):
            return MarketCategory.SOCCER
        if any(kw in combined_text for kw in ["ipl", "cricket", "t20", "dhoni", "royals", "super kings"]):
            return MarketCategory.CRICKET
        if any(kw in combined_text for kw in ["inflation", "cpi", "fed rate", "gdp", "recession", "economics", "powell"]):
            return MarketCategory.MACRO
        return MarketCategory.UNKNOWN

    def compute_string_similarity(self, str1: str, str2: str) -> float:
        """Calculates token sequence ratios for validation."""
        return SequenceMatcher(None, str1.lower().strip(), str2.lower().strip()).ratio()

    def calculate_safe_kelly(self, true_prob: float, poly_price: float) -> Tuple[float, float]:
        """Calculates mathematical edge and applies fraction limitations."""
        if true_prob <= poly_price or poly_price <= 0 or poly_price >= 1:
            return 0.0, 0.0
        edge = true_prob - poly_price
        raw_kelly = edge / (1.0 - poly_price)
        allocated_capital = raw_kelly * self.fractional_kelly
        return edge, min(allocated_capital, self.max_bet_allocation)

    def evaluate_and_filter_markets(self, polymarket_raw: List[Dict], sharp_data_raw: List[Dict], bankroll: float) -> pd.DataFrame:
        """Executes targeted entity resolution over isolated categories."""
        validated_value_bets = []

        for poly_market in polymarket_raw:
            poly_title = poly_market.get("title", "")
            poly_tags = poly_market.get("tags", [])
            poly_price = float(poly_market.get("yes_price", 0.0))
            
            # Identify true category
            poly_category = self.classify_polymarket_event(poly_title, poly_tags)
            if poly_category == MarketCategory.UNKNOWN:
                continue

            best_match_sharp = None
            highest_match_score = 0.0

            # Isolated Sport Domain Loop Guardrail
            for sharp_item in sharp_data_raw:
                sport_key = sharp_item.get("sport_key", "")
                sharp_category = ODDS_API_SPORT_MAP.get(sport_key, MarketCategory.UNKNOWN)

                # FIREWALL REJECTION BLOCK FOR OUT-OF-BOUND DOMAINS
                if sharp_category != poly_category:
                    continue  

                similarity = self.compute_string_similarity(poly_title, sharp_item.get("event_name", ""))
                if similarity > highest_match_score:
                    highest_match_score = similarity
                    best_match_sharp = sharp_item

            # Strict Boundary Verification Guardrail
            if best_match_sharp and highest_match_score >= self.match_threshold:
                true_probability = float(best_match_sharp.get("true_probability"))
                edge, kelly_fraction = self.calculate_safe_kelly(true_probability, poly_price)
                
                # Enforce baseline minimum edge requirements to avoid noise entry placement
                if edge > 0.01:  
                    validated_value_bets.append({
                        "Market": poly_title,
                        "Category": poly_category,
                        "Source": f"Sharp API ({sport_key.upper()})",
                        "Polymarket Price": poly_price,
                        "True Prob%": true_probability,
                        "Edge%": edge,
                        "Allocation%": kelly_fraction,
                        "Bet Size": bankroll * kelly_fraction
                    })

        df_output = pd.DataFrame(validated_value_bets)
        if df_output.empty:
            return pd.DataFrame(columns=["Market", "Category", "Source", "Polymarket Price", "True Prob%", "Edge%", "Allocation%", "Bet Size"])
        return df_output

# ==========================================
# 5. STREAMLIT APP ENGINE RUNTIME
# ==========================================
def main():
    # Top Identity Layout Bar
    col_header, col_status = st.columns([3, 1])
    with col_header:
        st.title("⚡ QUANTUM TRADING SCOUT")
        st.markdown("<p style='color:#8a99ad; margin-top:-15px;'>Polymarket Live Value Arbitrage Execution Engine</p>", unsafe_allow_html=True)
    with col_status:
        st.markdown("<div style='text-align: right; margin-top: 25px;'><span class='status-badge'>SHARP PROTECTION ACTIVE</span></div>", unsafe_allow_html=True)
    
    st.markdown("---")

    # Interactive Risk Control Sidebars
    st.sidebar.header("System Controls")
    bankroll = st.sidebar.number_input("Capital Pool Bankroll ($)", min_value=100.0, max_value=1000000.0, value=1000.0, step=100.0)
    match_threshold = st.sidebar.slider("Text Match Verification Threshold", min_value=0.50, max_value=1.00, value=0.75, step=0.05)
    fractional_kelly = st.sidebar.slider("Kelly Criterion Multiplier Scale", min_value=0.05, max_value=1.00, value=0.25, step=0.05)
    max_allocation = st.sidebar.slider("Max Single-Bet Allocation Cap", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

    # API Authentication Setup Block
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Keys Configuration")
    
    # Resolves from secrets.toml or environment variables smoothly
    try:
        default_key = st.secrets["THE_ODDS_API_KEY"]
    except Exception:
        default_key = ""
        
    odds_api_key = st.sidebar.text_input("The Odds API Key", value=default_key, type="password")

    if not odds_api_key:
        st.warning("⚠️ Application deployment holding. Please enter your `THE_ODDS_API_KEY` inside the sidebar controller configuration to execute parsing.")
        return

    # Ingest Live Market Feed Matrices
    with st.spinner("Fetching data structures across live API endpoints..."):
        polymarket_data = DataIngestionEngine.fetch_polymarket_active_markets()
        
        target_sports = list(ODDS_API_SPORT_MAP.keys())
        sharp_data = DataIngestionEngine.fetch_the_odds_lines(odds_api_key, target_sports)

    if not polymarket_data or not sharp_data:
        st.info("Waiting for data stream updates. Please verify API credits or endpoint connection states.")
        return

    # Execute Safe Matching Pipeline Core
    pipeline = ProductionDataPipeline(
        match_threshold=match_threshold,
        fractional_kelly=fractional_kelly,
        max_bet_allocation=max_allocation
    )
    bets_df = pipeline.evaluate_and_filter_markets(polymarket_data, sharp_data, bankroll=bankroll)

    # Performance KPI Card Grid Layout
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(polymarket_data)}</div><div class='metric-label'>Polymarket Scanned</div></div>", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(bets_df)}</div><div class='metric-label'>Validated Value Bets</div></div>", unsafe_allow_html=True)
    with m_col3:
        total_alloc_pct = bets_df['Allocation%'].sum() * 100 if not bets_df.empty else 0.0
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_alloc_pct:.1f}%</div><div class='metric-label'>Total Exposure</div></div>", unsafe_allow_html=True)
    with m_col4:
        total_capital_deployed = bets_df['Bet Size'].sum() if not bets_df.empty else 0.0
        st.markdown(f"<div class='metric-card'><div class='metric-value'>${total_capital_deployed:.2f}</div><div class='metric-label'>Committed Funds</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Verified Execution Orders Vector")

    # Render Clean Tabular DataFrame Visualizations
    if not bets_df.empty:
        display_df = bets_df.copy()
        display_df['Polymarket Price'] = display_df['Polymarket Price'].map(lambda x: f"${x:.2f}")
        display_df['True Prob%'] = display_df['True Prob%'].map(lambda x: f"{x * 100:.1f}%")
        display_df['Edge%'] = display_df['Edge%'].map(lambda x: f"{x * 100:.1f}%")
        display_df['Allocation%'] = display_df['Allocation%'].map(lambda x: f"{x * 100:.1f}%")
        display_df['Bet Size'] = display_df['Bet Size'].map(lambda x: f"${x:.2f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Summary Analytics Matrix Section Base Footer
        st.markdown("---")
        avg_edge = bets_df['Edge%'].mean() * 100
        max_single_kelly = bets_df['Allocation%'].max() * 100
        
        c_summary1, c_summary2 = st.columns(2)
        with c_summary1:
            st.markdown(f"📈 **Mean Analytical Advantage (Avg Edge%):** `{avg_edge:.2f}%`")
        with c_summary2:
            st.markdown(f"🛡️ **Peak Risk Portfolio Weight (Max Kelly% Cap):** `{max_single_kelly:.2f}%`")
    else:
        st.info("System filtering active. All current live lines match sharp parameters perfectly. No anomalies detected.")

if __name__ == "__main__":
    main()
