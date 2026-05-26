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

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] { background-color: #0e1117; color: #ecf0f1; font-family: 'Inter', sans-serif; }
        .metric-container-box { background-color: #161a23; border: 1px solid #242b3d; padding: 1rem; border-radius: 6px; text-align: center; }
        .metric-big-value { font-size: 1.75rem; font-weight: 700; color: #00ffcc; }
        .metric-sub-label { font-size: 0.8rem; color: #8a99ad; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }
        .status-tag { background-color: #0b2e24; color: #00ffaa; border: 1px solid #00aa77; padding: 0.3rem 0.7rem; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA PIPELINE (LIVE INGESTION)
# ==========================================
class LiveDataPipeline:
    @staticmethod
    def fetch_live_polymarket() -> List[Dict[str, Any]]:
        url = "https://gamma-api.polymarket.com/markets"
        params = {"closed": "false", "active": "true", "limit": "150", "core": "true"}
        normalized = []
        try:
            res = requests.get(url, params=params, timeout=8)
            if res.status_code == 200:
                for m in res.json():
                    prices = m.get("outcomePrices")
                    if not prices: continue
                    # Handle JSON-encoded strings safely
                    if isinstance(prices, str):
                        try: prices = json.loads(prices)
                        except: continue
                    
                    if len(prices) >= 1:
                        normalized.append({
                            "title": m.get("title", ""),
                            "curPrice": float(prices[0]),
                            "volume": int(float(m.get("volume", 0)))
                        })
                return normalized
        except Exception as e:
            st.sidebar.error(f"Pipeline API Error: {e}")
        return []

    @staticmethod
    def fetch_sharp_lines(api_key: str) -> List[Dict[str, Any]]:
        if not api_key: return []
        aggregated = []
        # Add your sports segments here
        return aggregated

# ==========================================
# 3. ANALYTICAL ENGINE
# ==========================================
def calculate_risk_bounds(price: float) -> Tuple[str, str]:
    if price < 0.30: return "High Risk", "Asymmetric Speculative"
    elif price < 0.70: return "Moderate Risk", "Balanced Dynamic"
    else: return "Low Risk", "High Probability Core"

# ==========================================
# 4. MAIN APPLICATION
# ==========================================
def main():
    st.title("⚡ QUANTUM TRADING SCOUT")
    st.markdown("---")

    # Sidebar
    bankroll = st.sidebar.number_input("Capital Pool ($)", value=1000.0)
    
    # Ingestion
    pm_raw = LiveDataPipeline.fetch_live_polymarket()
    
    if pm_raw:
        df = pd.DataFrame(pm_raw)
        
        # Apply Logic
        df[['risk_range', 'prob_range']] = df['curPrice'].apply(
            lambda x: pd.Series(calculate_risk_bounds(x))
        )
        
        # Display Data
        st.dataframe(df, use_container_width=True)
        
    else:
        st.warning("Waiting for data stream...")

if __name__ == "__main__":
    main()
