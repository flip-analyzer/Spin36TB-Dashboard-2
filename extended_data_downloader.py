#!/usr/bin/env python3
"""
Extended OANDA Data Downloader
Downloads 6 months of data using multiple API calls
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time

class ExtendedOandaDownloader:
    def __init__(self, api_token: str, account_id: str):
        self.api_token = api_token
        self.account_id = account_id
        self.api_url = 'https://api-fxpractice.oanda.com'
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
    
    def download_extended_data(self, months_back=6):
        """
        Download 6 months of data using multiple requests
        """
        print(f"📊 DOWNLOADING {months_back} MONTHS OF EURUSD DATA")
        print("=" * 50)
        
        all_data = []
        
        # Calculate date ranges for multiple requests
        end_date = datetime.now()
        days_per_request = 30  # ~30 days per request to stay within limits
        
        for i in range(months_back):
            request_end = end_date - timedelta(days=i * days_per_request)
            request_start = request_end - timedelta(days=days_per_request)
            
            print(f"\n📅 Request {i+1}/{months_back}: {request_start.date()} to {request_end.date()}")
            
            try:
                # OANDA API call with date range
                url = f"{self.api_url}/v3/instruments/EUR_USD/candles"
                params = {
                    'granularity': 'M5',
                    'from': request_start.strftime('%Y-%m-%dT%H:%M:%S.000000000Z'),
                    'to': request_end.strftime('%Y-%m-%dT%H:%M:%S.000000000Z'),
                    'price': 'MBA'
                }
                
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    candles = data['candles']
                    
                    # Process candles
                    batch_data = []
                    for candle in candles:
                        if candle['complete']:
                            batch_data.append({
                                'timestamp': pd.to_datetime(candle['time']),
                                'open': float(candle['mid']['o']),
                                'high': float(candle['mid']['h']),
                                'low': float(candle['mid']['l']),
                                'close': float(candle['mid']['c']),
                                'volume': int(candle['volume'])
                            })
                    
                    all_data.extend(batch_data)
                    print(f"   ✅ Downloaded {len(batch_data)} candles")
                    
                    # Rate limiting
                    time.sleep(1)  # 1 second between requests
                    
                else:
                    print(f"   ❌ Request failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        if all_data:
            # Create DataFrame
            df = pd.DataFrame(all_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)  # Sort by time
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            print(f"\n✅ DOWNLOAD COMPLETE!")
            print(f"   Total Candles: {len(df):,}")
            print(f"   Date Range: {df.index[0]} to {df.index[-1]}")
            print(f"   Time Span: {(df.index[-1] - df.index[0]).days} days")
            print(f"   Price Range: {df['close'].min():.4f} - {df['close'].max():.4f}")
            
            return df
        else:
            print("❌ No data downloaded")
            return None

def download_6_months():
    """Download 6 months of OANDA data"""
    
    # Use your OANDA credentials
    api_token = "fdf72f4b5166602e39e001d1c1cb38b0-524c6662baa3187260b34c006eeab1fd"
    account_id = "101-001-31365224-001"
    
    downloader = ExtendedOandaDownloader(api_token, account_id)
    df = downloader.download_extended_data(months_back=6)
    
    if df is not None:
        # Save to CSV
        filepath = "/Users/jonspinogatti/Desktop/spin36TB/oanda_eurusd_6months.csv"
        df.to_csv(filepath)
        print(f"\n💾 Data saved to: oanda_eurusd_6months.csv")
        print(f"   Ready for enhanced system validation!")
        return True
    else:
        print("❌ Failed to download extended data")
        return False

if __name__ == "__main__":
    download_6_months()