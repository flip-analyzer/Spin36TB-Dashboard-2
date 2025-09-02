#!/usr/bin/env python3
"""
OANDA Historical Data Downloader
Downloads real EURUSD data for system validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import requests

# OANDA Configuration - YOU NEED TO FILL THESE IN
OANDA_CONFIG = {
    'api_token': 'fdf72f4b5166602e39e001d1c1cb38b0-524c6662baa3187260b34c006eeab1fd',  # Your OANDA API token
    'account_id': '101-001-31365224-001',  # Your OANDA account ID
    'environment': 'practice',  # 'practice' or 'live'
}

class OandaDataDownloader:
    """
    Download historical data from OANDA
    """
    
    def __init__(self, api_token: str, account_id: str, environment: str = 'practice'):
        self.api_token = api_token
        self.account_id = account_id
        
        if environment == 'practice':
            self.api_url = 'https://api-fxpractice.oanda.com'
        else:
            self.api_url = 'https://api-fxtrade.oanda.com'
        
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
        
        print(f"🔌 OANDA Connection Initialized")
        print(f"   Environment: {environment}")
        print(f"   Account: {account_id}")
    
    def test_connection(self):
        """
        Test OANDA API connection
        """
        try:
            url = f"{self.api_url}/v3/accounts/{self.account_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                account_data = response.json()
                print("✅ OANDA connection successful!")
                print(f"   Currency: {account_data['account']['currency']}")
                print(f"   Balance: {account_data['account']['balance']}")
                return True
            else:
                print(f"❌ Connection failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def download_historical_data(self, instrument: str = 'EUR_USD', 
                                granularity: str = 'M5',
                                count: int = 5000):
        """
        Download historical OHLC data from OANDA
        """
        print(f"📊 Downloading {instrument} data...")
        print(f"   Granularity: {granularity} (5-minute bars)")
        print(f"   Count: {count} candles")
        
        try:
            # OANDA API endpoint for historical data
            url = f"{self.api_url}/v3/instruments/{instrument}/candles"
            
            params = {
                'granularity': granularity,
                'count': count,
                'price': 'MBA'  # Mid, Bid, Ask
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"❌ Data download failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
            
            data = response.json()
            candles = data['candles']
            
            # Convert to pandas DataFrame
            ohlc_data = []
            for candle in candles:
                if candle['complete']:  # Only use complete candles
                    timestamp = pd.to_datetime(candle['time'])
                    
                    # Use mid prices (average of bid/ask)
                    ohlc_data.append({
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            df = pd.DataFrame(ohlc_data)
            df.index = pd.to_datetime([c['time'] for c in candles if c['complete']])
            
            print(f"✅ Downloaded {len(df)} complete candles")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Price range: {df['close'].min():.4f} - {df['close'].max():.4f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None
    
    def save_data(self, df: pd.DataFrame, filename: str = 'oanda_eurusd_data.csv'):
        """
        Save downloaded data to CSV file
        """
        try:
            filepath = f"/Users/jonspinogatti/Desktop/spin36TB/{filename}"
            df.to_csv(filepath)
            print(f"💾 Data saved to: {filename}")
            print(f"   Location: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Save error: {e}")
            return False

def setup_oanda_connection():
    """
    Help user set up OANDA connection
    """
    print("🔧 OANDA SETUP GUIDE")
    print("=" * 25)
    
    print("\n1️⃣ GET YOUR API CREDENTIALS:")
    print("   • Login to OANDA account")
    print("   • Go to: Manage Funds → Manage API")
    print("   • Generate Personal Access Token")
    print("   • Copy your Account ID (xxx-xxx-xxxxxxx-xxx)")
    
    print("\n2️⃣ ENTER CREDENTIALS BELOW:")
    
    try:
        api_token = input("📝 Enter your API Token: ").strip()
        account_id = input("📝 Enter your Account ID: ").strip()
        
        if not api_token or not api_token.startswith(('b')):
            print("⚠️  API Token should start with 'b' (bearer token)")
        
        if not account_id or '-' not in account_id:
            print("⚠️  Account ID should have format: xxx-xxx-xxxxxxx-xxx")
        
        return api_token, account_id
        
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled")
        return None, None

def download_and_validate():
    """
    Complete download and validation process
    """
    print("📊 OANDA HISTORICAL DATA VALIDATION")
    print("=" * 40)
    
    # Get credentials
    api_token, account_id = setup_oanda_connection()
    
    if not api_token or not account_id:
        print("❌ Cannot proceed without credentials")
        return False
    
    # Initialize downloader
    downloader = OandaDataDownloader(
        api_token=api_token,
        account_id=account_id,
        environment='practice'  # Safe to start with practice
    )
    
    # Test connection
    if not downloader.test_connection():
        print("❌ Connection failed - check your credentials")
        return False
    
    # Download 6 months of data (approximately 50,000 5-minute candles)
    print(f"\n📈 Downloading 6 months of EURUSD data...")
    df = downloader.download_historical_data(
        instrument='EUR_USD',
        granularity='M5',  # 5-minute bars
        count=50000  # ~6 months of 5-minute data (max available)
    )
    
    if df is None:
        print("❌ Data download failed")
        return False
    
    # Save data
    if downloader.save_data(df, 'oanda_eurusd_6months.csv'):
        print(f"\n✅ SUCCESS! Ready for validation testing")
        print(f"   File: oanda_eurusd_6months.csv")
        print(f"   Candles: {len(df):,}")
        print(f"   Time span: {(df.index[-1] - df.index[0]).days} days")
        
        return True
    
    return False

def quick_setup_guide():
    """
    Show quick setup guide without actually connecting
    """
    print("🚀 QUICK OANDA SETUP GUIDE")
    print("=" * 30)
    
    print("\n📋 TO GET YOUR CREDENTIALS:")
    print("1. Login to your OANDA account")
    print("2. Navigate to: Manage Funds → Manage API")  
    print("3. Click 'Generate' for Personal Access Token")
    print("4. Copy the token (starts with letters)")
    print("5. Note your Account ID (format: 123-456-7890123-001)")
    
    print("\n💻 TO RUN THE DOWNLOAD:")
    print("1. Edit this file (oanda_data_downloader.py)")
    print("2. Replace 'YOUR_API_TOKEN_HERE' with your token")
    print("3. Replace 'YOUR_ACCOUNT_ID_HERE' with your account ID")
    print("4. Run: python oanda_data_downloader.py")
    
    print("\n⚡ OR MANUAL SETUP:")
    print("1. Run this file as-is")
    print("2. It will ask for your credentials")
    print("3. Paste them when prompted")
    print("4. Data downloads automatically")
    
    print("\n🎯 RESULT:")
    print("   • File: oanda_eurusd_6months.csv")
    print("   • ~35,000 real price bars")
    print("   • Ready for system validation")

if __name__ == "__main__":
    # Check if user has configured credentials
    if (OANDA_CONFIG['api_token'] == 'YOUR_API_TOKEN_HERE' or 
        OANDA_CONFIG['account_id'] == 'YOUR_ACCOUNT_ID_HERE'):
        
        print("🔧 CREDENTIALS NOT CONFIGURED")
        print("=" * 35)
        quick_setup_guide()
        
        print(f"\n🚀 Starting interactive setup...")
        download_and_validate()
    else:
        # Use configured credentials
        downloader = OandaDataDownloader(
            api_token=OANDA_CONFIG['api_token'],
            account_id=OANDA_CONFIG['account_id'],
            environment=OANDA_CONFIG['environment']
        )
        
        if downloader.test_connection():
            df = downloader.download_historical_data()
            if df is not None:
                downloader.save_data(df)