import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re
from typing import Dict, List, Tuple, Any

# ==========================================
# 1. CORE SYSTEM INITIALIZATION
# ==========================================
st.set_page_config(page_title="Polymarket Scanner Pro", layout="wide")

# ==========================================
# 2. DATA INGESTION PIPELINES (LIVE INTEGRATION)
# ==========================================
class AdvancedMarketPipeline:
    
    @staticmethod
    def fetch_live_polymarket_dump() -> List[Dict[str, Any]]:
        """Integrated fetcher for Polymarket Gamma API."""
        url = "https://gamma-api.polymarket.com/markets"
        params = {"active": "true", "closed": "false", "limit": "250", "core": "true"}
        raw_records = []
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 200:
                for m in res.json():
                    # CRITICAL: Decode double-encoded strings
                    prices_raw = m.get("outcomePrices")
                    if not prices_raw: continue
                    
                    try:
                        # Ensure it's a list even if API returns JSON string
                        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                    except: continue
                    
                    if isinstance(prices, list) and len(prices) >= 1:
                        raw_records.append({
                            "title": m.get("title", m.get("question", "")),
                            "curPrice": float(prices[0]), # Map first outcome to Yes-price
                            "side": "Yes",
                            "volume": int(float(m.get("volume", 0))),
                            "is_sports": any(t in m.get("slug", "").lower() for t in ["nba", "nfl", "mlb", "nhl", "fifa", "ipl"])
                        })
                return raw_records
        except Exception as e:
            st.error(f"Pipeline Error: {e}")
        return []

    @staticmethod
    def fetch_sharp_consensus_lines(api_key: str) -> List[Dict[str, Any]]:
        if not api_key: return []
        aggregated = []
        # ... [Keep your existing fetch_sharp_consensus_lines logic here] ...
        return aggregated

# ==========================================
# 3. ANALYTICAL ENGINE & MAIN
# ==========================================
# ... [Insert your existing AnalyticsEngine class here] ...

def main():
    st.title("🏆 Polymarket Sports Scanner + Sharp Odds")
    
    # Run Pipelines
    raw_pm_data = AdvancedMarketPipeline.fetch_live_polymarket_dump()
    
    if raw_pm_data:
        df_scan = pd.DataFrame(raw_pm_data)
        st.write(f"Pipeline active. Records loaded: {len(df_scan)}")
        # ... [Continue with your existing filtering and display logic] ...
    else:
        st.warning("Pipeline inactive. Ensure Gamma API is reachable.")

if __name__ == "__main__":
    main()
