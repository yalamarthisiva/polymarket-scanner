import requests
import pandas as pd
import logging
import json
import time
from typing import List, Dict, Callable, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_market_data(limit: int = 400) -> List[Dict]:
    url = "https://gamma-api.polymarket.com/events"
    all_events = []
    offset = 0
    page = 0
    
    while True:
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
            "order": "volume_24hr",
            "ascending": "false"
        }
        
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list) or not data:
                break
                
            all_events.extend(data)
            page += 1
            if page % 3 == 0:  # Reduce log spam
                logging.info(f"Page {page}: {len(data)} events (total: {len(all_events)})")
            
            if len(data) < limit:
                break
                
            offset += limit
            if len(all_events) > 2500:
                break
        except requests.RequestException as e:
            logging.error(f"Error: {e}")
            break
    
    all_markets = [market for event in all_events for market in event.get('markets', [])]
    logging.info(f"✅ Total markets extracted: {len(all_markets)}")
    return all_markets


def analyze_markets(markets: List[Dict], criteria: Callable, 
                   category_filter: Optional[str] = None, 
                   yes_only: bool = False,
                   assumed_edge: float = 0.05,
                   min_kelly: float = 10.0,
                   bankroll: float = 10000) -> List[Dict]:
    best_bets = []
    
    for market in markets:
        market_id = market.get('id')
        title = market.get('question', 'N/A')
        volume = float(market.get('volumeNum', 0))
        liquidity = float(market.get('liquidityNum', 0))
        
        if category_filter:
            tags = str(market.get('tags', '')) + str(market.get('categories', ''))
            if category_filter.lower() not in tags.lower():
                continue
        
        try:
            outcomes = json.loads(market.get('outcomes', '[]'))
            probabilities = json.loads(market.get('outcomePrices', '[]'))
            
            for name, prob_str in zip(outcomes, probabilities):
                prob = float(prob_str)
                
                if not criteria({'probability': prob, 'volume': volume, 'liquidity': liquidity}):
                    continue
                if yes_only and name.lower() != "yes":
                    continue
                    
                # Kelly Calculation
                b = (1 / prob) - 1
                believed_p = min(prob + assumed_edge, 0.98)
                kelly = max(0, (believed_p * (b + 1) - 1) / b) if b > 0 else 0
                
                if kelly * 100 < min_kelly:
                    continue
                
                suggested_bet = round(bankroll * kelly, 2)
                
                best_bets.append({
                    'market_id': str(market_id)[:8] + "...",
                    'title': title[:62] + "..." if len(title) > 62 else title,
                    'outcome': name,
                    'probability': round(prob, 4),
                    'implied_odds': round(1 / prob, 2),
                    'volume': int(volume),
                    'kelly_%': round(kelly * 100, 1),
                    'suggested_bet': f"${suggested_bet:,.0f}",
                    'return_on_1k': round(1000 * b, 1)
                })
        except:
            continue
    
    return best_bets


def value_criteria(outcome: Dict, min_prob=0.68, max_prob=0.93, 
                  min_volume=500_000, min_liquidity=30_000) -> bool:
    prob = outcome['probability']
    if not (min_prob <= prob <= max_prob):
        return False
    if outcome.get('volume', 0) < min_volume or outcome.get('liquidity', 0) < min_liquidity:
        return False
    return True


def export_to_html(df: pd.DataFrame, filename: str = "polymarket_bets.html"):
    html = f"""
    <html>
    <head><title>Polymarket Value Bets - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
    </head>
    <body>
    <h1>Polymarket Value Bets Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    {df.to_html(index=False)}
    </body>
    </html>
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📄 HTML report saved as {filename}")


def main():
    bankroll = 100  # <-- CHANGE YOUR BANKROLL HERE
    
    print(f"🔄 Starting Polymarket Scanner | Bankroll: ${bankroll:,.0f}")
    
    while True:  # Auto-refresh loop
        try:
            markets = fetch_market_data(limit=400)
            
            if not markets:
                print("❌ No markets fetched.")
                time.sleep(60)
                continue
            
            # ================== CONFIGURATION ==================
            category = None          # "politics", "sports", "crypto", None
            yes_only = False
            min_prob = 0.68
            max_prob = 0.93
            min_volume = 500_000
            assumed_edge = 0.05
            min_kelly = 12.0         # Minimum Kelly % to show
            # ==================================================
            
            best_bets = analyze_markets(
                markets, 
                lambda o: value_criteria(o, min_prob, max_prob, min_volume),
                category_filter=category,
                yes_only=yes_only,
                assumed_edge=assumed_edge,
                min_kelly=min_kelly,
                bankroll=bankroll
            )
            
            if best_bets:
                df = pd.DataFrame(best_bets)
                df = df.sort_values('kelly_%', ascending=False)
                
                print(f"\n✅ Found {len(df)} good value bets (Kelly >= {min_kelly}%):\n")
                print(df.to_string(index=False))
                
                df.to_csv('polymarket_value_bets.csv', index=False)
                export_to_html(df)
                
            else:
                print("\nNo bets meeting criteria right now.")
            
            print(f"\n⏳ Next refresh in 5 minutes... (Ctrl+C to stop)")
            time.sleep(300)  # 5 minutes
            
        except KeyboardInterrupt:
            print("\n👋 Scanner stopped by user.")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()