import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re
from typing import Dict, List, Tuple, Any

# ==========================================
# 1. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="Polymarket Scanner Pro", layout="wide")

# ==========================================
# 2. ROBUST DATA PIPELINE
# ==========================================
class AdvancedMarketPipeline:
    @staticmethod
    def fetch_live_polymarket_dump() -> pd.DataFrame:
        url = "https://gamma-api.polymarket.com/markets"
        params = {"active": "true", "closed": "false", "limit": "250", "core": "true"}
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 200:
                data = res.json()
                records = []
                for m in data:
                    prices_raw = m.get("outcomePrices")
                    # Handle double-encoded strings from Polymarket API
                    prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                    
                    if prices and isinstance(prices, list):
                        records.append({
                            "title": m.get("title", ""),
                            "curPrice": float(prices[0]),
                            "volume": float(m.get("volume", 0)),
                            "side": "Yes",
                            "slug": m.get("slug", "")
                        })
                return pd.DataFrame(records)
        except Exception as e:
            st.error(f"Pipeline Error: {e}")
        return pd.DataFrame()

# ==========================================
# 3. MAIN APPLICATION
# ==========================================
def main():
    st.title("🏆 Polymarket Data Explorer")
    
    # 1. Fetch
    df = AdvancedMarketPipeline.fetch_live_polymarket_dump()
    
    if df.empty:
        st.warning("No data returned from API. Check your internet or API status.")
        return

    st.success(f"Pipeline Active. {len(df)} records retrieved.")

    # 2. Debugging Tool: Toggle to see what the API actually returned
    if st.checkbox("Show Raw Data Preview"):
        st.write(df.head(10))

    # 3. Filtering Logic
    min_vol = st.number_input("Min Volume", value=1000)
    df_filtered = df[df["volume"] >= min_vol]

    # 4. Final Render
    if not df_filtered.empty:
        # Convert all columns to strings to ensure Streamlit can serialize them
        df_display = df_filtered.astype(str)
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No records match your filters.")

if __name__ == "__main__":
    main()
