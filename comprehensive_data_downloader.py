#!/usr/bin/env python3
"""
Comprehensive Historical Data Downloader
Gets 6+ months of data using multiple strategies
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import yfinance as yf
import numpy as np

class ComprehensiveDataDownloader:
    def __init__(self, api_token: str, account_id: str):
        self.api_token = api_token
        self.account_id = account_id
        self.api_url = 'https://api-fxpractice.oanda.com'
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
    
    def download_multiple_periods(self):
        """
        Download data using multiple smaller requests
        """
        print("📊 COMPREHENSIVE DATA DOWNLOAD STRATEGY")
        print("=" * 45)
        
        all_data = []
        
        # Strategy 1: Weekly chunks going back 6 months
        end_date = datetime.now()
        weeks_back = 26  # 6 months = ~26 weeks
        
        print(f"\n📅 Strategy 1: Weekly chunks ({weeks_back} weeks)")
        
        for week in range(weeks_back):
            week_end = end_date - timedelta(weeks=week)
            week_start = week_end - timedelta(weeks=1)
            
            print(f"   Week {week+1}/{weeks_back}: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
            
            try:
                url = f"{self.api_url}/v3/instruments/EUR_USD/candles"
                params = {
                    'granularity': 'M5',
                    'from': week_start.strftime('%Y-%m-%dT00:00:00.000000000Z'),
                    'to': week_end.strftime('%Y-%m-%dT23:59:59.000000000Z'),
                    'price': 'MBA'
                }
                
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    candles = data['candles']
                    
                    week_data = []
                    for candle in candles:
                        if candle['complete']:
                            week_data.append({
                                'timestamp': pd.to_datetime(candle['time']),
                                'open': float(candle['mid']['o']),
                                'high': float(candle['mid']['h']),
                                'low': float(candle['mid']['l']),
                                'close': float(candle['mid']['c']),
                                'volume': int(candle['volume'])
                            })
                    
                    all_data.extend(week_data)
                    print(f"      ✅ {len(week_data)} candles")
                    
                    time.sleep(2)  # Rate limiting
                    
                elif response.status_code == 400:
                    print(f"      ⚠️ Week too large, trying daily chunks...")
                    # Fallback to daily chunks for this week
                    daily_data = self._download_daily_chunks(week_start, week_end)
                    if daily_data:
                        all_data.extend(daily_data)
                        
                else:
                    print(f"      ❌ Failed: {response.status_code}")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue
        
        if all_data:
            # Create DataFrame
            df = pd.DataFrame(all_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
            
            return df
        else:
            return None
    
    def _download_daily_chunks(self, start_date, end_date):
        """
        Fallback: Download daily chunks for a given period
        """
        daily_data = []
        current_date = start_date
        
        while current_date < end_date:
            day_end = current_date + timedelta(days=1)
            
            try:
                url = f"{self.api_url}/v3/instruments/EUR_USD/candles"
                params = {
                    'granularity': 'M5',
                    'from': current_date.strftime('%Y-%m-%dT00:00:00.000000000Z'),
                    'to': day_end.strftime('%Y-%m-%dT00:00:00.000000000Z'),
                    'price': 'MBA'
                }
                
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    candles = data['candles']
                    
                    for candle in candles:
                        if candle['complete']:
                            daily_data.append({
                                'timestamp': pd.to_datetime(candle['time']),
                                'open': float(candle['mid']['o']),
                                'high': float(candle['mid']['h']),
                                'low': float(candle['mid']['l']),
                                'close': float(candle['mid']['c']),
                                'volume': int(candle['volume'])
                            })
                    
                    time.sleep(1)  # Rate limiting
                    
            except Exception as e:
                print(f"        Daily error: {e}")
            
            current_date = day_end
        
        return daily_data
    
    def fallback_yahoo_finance(self):
        """
        Fallback: Use Yahoo Finance for additional data
        """
        print(f"\n📈 Fallback Strategy: Yahoo Finance EURUSD")
        
        try:
            # Yahoo Finance EUR/USD symbol
            ticker = yf.Ticker("EURUSD=X")
            
            # Get 1 year of hourly data (max available)
            hist = ticker.history(period="1y", interval="1h")
            
            if not hist.empty:
                # Convert to our format
                yahoo_data = []
                for timestamp, row in hist.iterrows():
                    yahoo_data.append({
                        'timestamp': timestamp,
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume']) if not np.isnan(row['Volume']) else 1000
                    })
                
                df = pd.DataFrame(yahoo_data)
                df.set_index('timestamp', inplace=True)
                
                print(f"   ✅ Yahoo Finance: {len(df)} hourly candles")
                print(f"   📅 Period: {df.index[0]} to {df.index[-1]}")
                
                return df
            
        except Exception as e:
            print(f"   ❌ Yahoo Finance failed: {e}")
            
        return None

def download_comprehensive_data():
    """
    Download comprehensive historical data using multiple strategies
    """
    print("🎯 COMPREHENSIVE HISTORICAL DATA COLLECTION")
    print("=" * 50)
    
    # Use your OANDA credentials
    api_token = "fdf72f4b5166602e39e001d1c1cb38b0-524c6662baa3187260b34c006eeab1fd"
    account_id = "101-001-31365224-001"
    
    downloader = ComprehensiveDataDownloader(api_token, account_id)
    
    # Strategy 1: OANDA comprehensive download
    print("\n🔄 Starting OANDA comprehensive download...")
    oanda_data = downloader.download_multiple_periods()
    
    # Strategy 2: Yahoo Finance fallback
    yahoo_data = downloader.fallback_yahoo_finance()
    
    # Combine datasets
    datasets = []
    if oanda_data is not None:
        datasets.append(("OANDA", oanda_data))
    if yahoo_data is not None:
        datasets.append(("Yahoo", yahoo_data))
    
    if not datasets:
        print("❌ No data downloaded from any source")
        return False
    
    # Use the largest dataset
    best_source, best_data = max(datasets, key=lambda x: len(x[1]))
    
    print(f"\n📊 BEST DATASET: {best_source}")
    print(f"   Candles: {len(best_data):,}")
    print(f"   Period: {best_data.index[0]} to {best_data.index[-1]}")
    print(f"   Time Span: {(best_data.index[-1] - best_data.index[0]).days} days")
    print(f"   Price Range: {best_data['close'].min():.4f} - {best_data['close'].max():.4f}")
    
    # Save comprehensive dataset
    filepath = "/Users/jonspinogatti/Desktop/spin36TB/comprehensive_eurusd_data.csv"
    best_data.to_csv(filepath)
    
    print(f"\n💾 Comprehensive data saved: comprehensive_eurusd_data.csv")
    print(f"🎯 Ready for proper 6-month validation!")
    
    return True

if __name__ == "__main__":
    download_comprehensive_data()