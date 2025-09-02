import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta


class FinancialDataHandler:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data = {}
        
    def fetch_data(self, 
                   start_date: str,
                   end_date: str,
                   interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for symbols"""
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for {symbol}")
                
            self.data[symbol] = df
            
        return self.data
    
    def get_returns(self, 
                    symbol: str,
                    method: str = "log",
                    periods: int = 1) -> pd.Series:
        """Calculate returns"""
        if symbol not in self.data:
            raise ValueError(f"Data for {symbol} not found")
            
        prices = self.data[symbol]['Close']
        
        if method == "log":
            returns = np.log(prices / prices.shift(periods))
        elif method == "simple":
            returns = prices.pct_change(periods)
        else:
            raise ValueError("Method must be 'log' or 'simple'")
            
        return returns.dropna()
    
    def get_volatility(self,
                       symbol: str,
                       window: int = 20,
                       method: str = "ewm") -> pd.Series:
        """Calculate rolling volatility"""
        returns = self.get_returns(symbol)
        
        if method == "rolling":
            vol = returns.rolling(window=window).std()
        elif method == "ewm":
            vol = returns.ewm(span=window).std()
        else:
            raise ValueError("Method must be 'rolling' or 'ewm'")
            
        return vol * np.sqrt(252)  # Annualized
    
    def calculate_dollar_bars(self,
                              symbol: str,
                              dollar_threshold: float) -> pd.DataFrame:
        """Create dollar bars from tick data"""
        df = self.data[symbol].copy()
        df['dollar_volume'] = df['Close'] * df['Volume']
        df['cumsum_dollar'] = df['dollar_volume'].cumsum()
        
        # Find bar boundaries
        bar_indices = []
        last_threshold = 0
        
        for i, cum_dollar in enumerate(df['cumsum_dollar']):
            if cum_dollar >= last_threshold + dollar_threshold:
                bar_indices.append(i)
                last_threshold = cum_dollar
                
        # Create bars
        bars = []
        start_idx = 0
        
        for end_idx in bar_indices:
            bar_data = df.iloc[start_idx:end_idx+1]
            
            bar = {
                'timestamp': bar_data.index[-1],
                'open': bar_data['Open'].iloc[0],
                'high': bar_data['High'].max(),
                'low': bar_data['Low'].min(),
                'close': bar_data['Close'].iloc[-1],
                'volume': bar_data['Volume'].sum(),
                'dollar_volume': bar_data['dollar_volume'].sum()
            }
            bars.append(bar)
            start_idx = end_idx + 1
            
        return pd.DataFrame(bars).set_index('timestamp')
    
    def validate_data_quality(self, symbol: str) -> Dict[str, float]:
        """Validate data quality metrics"""
        if symbol not in self.data:
            raise ValueError(f"Data for {symbol} not found")
            
        df = self.data[symbol]
        
        metrics = {
            'missing_values_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
            'zero_volume_days': (df['Volume'] == 0).sum(),
            'negative_prices': (df[['Open', 'High', 'Low', 'Close']] <= 0).sum().sum(),
            'outliers_returns': self._detect_return_outliers(symbol),
            'data_completeness': len(df) / self._expected_trading_days(df.index[0], df.index[-1])
        }
        
        return metrics
    
    def _detect_return_outliers(self, symbol: str, threshold: float = 3.0) -> int:
        """Detect outliers in returns using z-score"""
        returns = self.get_returns(symbol)
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        return (z_scores > threshold).sum()
    
    def _expected_trading_days(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
        """Estimate expected trading days"""
        total_days = (end_date - start_date).days
        return int(total_days * 5/7)  # Approximate weekdays