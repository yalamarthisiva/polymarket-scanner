"""
Polymarket Quantitative Scanner v4.1
Expert Bayesian Hierarchical Model + Volatility-Adjusted Kelly
"""

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Polymarket Bayesian Scanner", layout="wide")
st.title("🏆 Polymarket Bayesian Quantitative Scanner")
st.info("**v4.1** — Hierarchical Bayesian Model + Volatility Adjustment")

# ================== CONFIGURATION ==================
BANKROLL = st.sidebar.number_input("Bankroll ($)", value=10000, min_value=100, step=500)
KELLY_FRACTION = st.sidebar.slider("Kelly Fraction", 0.05, 1.0, 0.20, 0.05)  # More conservative
MIN_EDGE_PCT = st.sidebar.number_input("Minimum Edge (%)", value=2.0, step=0.5)
MIN_KELLY_PCT = st.sidebar.number_input("Minimum Kelly (%)", value=0.2, step=0.05)
MIN_VOLUME = st.sidebar.number_input("Minimum Volume ($)", value=5000, step=1000)
MIN_CONFIDENCE = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.42, 0.05)

USE_MODEL = st.sidebar.checkbox("Use Sport Model", value=True)
REQUIRE_MODEL = st.sidebar.checkbox("Require Model Estimate", value=False)

# ================== BAYESIAN HIERARCHICAL HELPERS ==================
class BayesianFusion:
    """Hierarchical Bayesian probability fusion."""
    
    @staticmethod
    def update(prior: float, likelihood: float, prior_weight: float = 0.55, 
               volume: float = 0, liquidity: float = 0) -> Tuple[float, float]:
        """Bayesian update with volatility adjustment."""
        # Volume-based confidence in likelihood
        vol_factor = min(1.0, np.log10(volume + 1000) / 5.5) if volume else 0.3
        effective_weight = prior_weight * (1 - vol_factor * 0.3)  # Lower weight on low-volume sharp odds
        
        # Bayesian posterior
        posterior = effective_weight * prior + (1 - effective_weight) * likelihood
        posterior = max(0.02, min(0.98, posterior))
        
        # Confidence = harmonic mean adjusted by volume
        confidence = 2 * (prior * likelihood) / (prior + likelihood + 1e-8)
        confidence = confidence * (0.6 + 0.4 * vol_factor)
        
        return posterior, confidence

    @staticmethod
    def volatility_adjustment(prob: float, volume: float, base_vol: float = 0.12) -> float:
        """Reduce probability toward 0.5 for low-liquidity markets."""
        if volume < 10000:
            shrink_factor = 0.5 + 0.5 * (volume / 10000)
            return prob * shrink_factor + 0.5 * (1 - shrink_factor)
        return prob

# ================== FETCHERS (same as before) ==================
# ... (fetch_polymarket, fetch_sports_model, fetch_sharp_odds, parsers)

# ================== MAIN ==================
markets = fetch_polymarket()
sports_data = fetch_sports_model() if USE_MODEL else SportsModelData()
sharp_odds = fetch_sharp_odds()

outcomes = []
# ... build outcomes

estimates = estimate_probabilities(outcomes, sports_data)
estimates = calc_no_complement(estimates, outcomes)

value_rows = []
for o in outcomes:
    est = estimates.get(o.key)
    if REQUIRE_MODEL and not (est and est.is_model):
        continue
    if not est:
        est = ProbabilityEstimate(o.market_prob, "Baseline", False, 0.35)

    # === ADVANCED BAYESIAN FUSION ===
    sharp_prob = None
    sharp_edge = 0.0
    efficiency = 0.5

    if sharp_odds:
        # Match logic (can be refined)
        for key, price in sharp_odds.items():
            if normalize_name(o.participant) in key or normalize_name(o.market_name) in str(key):
                sharp_prob = price
                break

    if sharp_prob:
        # Hierarchical Bayesian update
        posterior, conf = BayesianFusion.update(
            est.true_prob, sharp_prob, prior_weight=0.58, 
            volume=o.volume or 0, liquidity=o.liquidity or 0
        )
        
        # Volatility adjustment
        posterior = BayesianFusion.volatility_adjustment(posterior, o.volume or 0)
        
        est = ProbabilityEstimate(posterior, "Bayesian Hierarchical", True, conf)
        sharp_edge = (posterior - o.market_prob) / o.market_prob * 100
        efficiency = 1.0 - abs(est.true_prob - sharp_prob) * 5

    edge_pct = (est.true_prob - o.market_prob) / o.market_prob * 100 if o.market_prob > 0 else 0
    if edge_pct < MIN_EDGE_PCT and sharp_edge < MIN_EDGE_PCT * 0.7:
        continue

    kelly_full = max(0, est.true_prob * (1/o.market_prob - 1) - (1 - est.true_prob)) / (1/o.market_prob - 1) if o.market_prob != 1 else 0
    kelly_pct = kelly_full * KELLY_FRACTION * 100
    if kelly_pct < MIN_KELLY_PCT: continue

    bet_size = BANKROLL * kelly_full * KELLY_FRACTION

    value_rows.append({
        "Market": o.market_name[:60],
        "Side": o.outcome,
        "PM Prob%": round(o.market_prob * 100, 1),
        "Posterior%": round(est.true_prob * 100, 1),
        "Sharp Edge%": round(sharp_edge, 1),
        "Total Edge%": round(edge_pct, 1),
        "Efficiency": round(efficiency, 2),
        "Kelly%": round(kelly_pct, 1),
        "Conf": round(est.confidence, 2),
        "Bet $": round(bet_size),
        "Volume": f"${int(o.volume):,}" if o.volume else "N/A",
        "Source": est.source
    })

df = pd.DataFrame(value_rows)

# ================== UI ==================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Markets", len(markets))
col2.metric("Open Markets", sum(1 for m in markets if is_open_market(m)))
col3.metric("Analyzed", len(outcomes))
col4.metric("Value Bets", len(df))

st.subheader(f"🔍 Value Bets Found: {len(df)}")

if not df.empty:
    df = df.sort_values("Sharp Edge%", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "value_bets.csv", "text/csv")
else:
    st.warning("No value bets found. Current markets are highly efficient.")

st.caption("⚠️ Not financial advice • Hierarchical Bayesian Model + Volatility Adjustment")

if st.button("🔄 Refresh All Data"):
    st.cache_data.clear()
    st.rerun()
