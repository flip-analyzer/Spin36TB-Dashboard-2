import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from .fractional_diff import FractionalDifferentiator


class MomentumFeatureEngineer:
    """
    Momentum feature engineering following López de Prado's methodology.
    
    Creates features that capture momentum patterns while maintaining
    stationarity through fractional differentiation.
    """
    
    def __init__(self, 
                 lookback_periods: List[int] = [5, 10, 21, 63, 126],
                 volatility_windows: List[int] = [10, 21, 63],
                 use_fractional_diff: bool = True,
                 frac_diff_threshold: float = 1e-5):
        """
        Parameters:
        -----------
        lookback_periods : list
            Lookback periods for momentum calculations
        volatility_windows : list
            Windows for volatility calculations
        use_fractional_diff : bool
            Whether to apply fractional differentiation
        frac_diff_threshold : float
            Threshold for fractional differentiation
        """
        self.lookback_periods = lookback_periods
        self.volatility_windows = volatility_windows
        self.use_fractional_diff = use_fractional_diff
        self.frac_diff_threshold = frac_diff_threshold
        self.frac_diff = FractionalDifferentiator(threshold=frac_diff_threshold)
        self.scaler = StandardScaler()
        
    def create_momentum_features(self, 
                                 prices: pd.Series,
                                 volume: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Create comprehensive momentum features
        
        Parameters:
        -----------
        prices : pd.Series
            Price series (typically close prices)
        volume : pd.Series, optional
            Volume series
            
        Returns:
        --------
        pd.DataFrame
            Feature matrix
        """
        features = {}
        
        # Basic returns
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # 1. Price momentum features
        price_features = self._create_price_momentum_features(prices, returns)
        features.update(price_features)
        
        # 2. Volatility features
        vol_features = self._create_volatility_features(returns)
        features.update(vol_features)
        
        # 3. Technical momentum indicators
        tech_features = self._create_technical_features(prices, returns)
        features.update(tech_features)
        
        # 4. Volume-based features (if available)
        if volume is not None:
            volume_features = self._create_volume_features(prices, volume, returns)
            features.update(volume_features)
        
        # 5. Microstructure features
        micro_features = self._create_microstructure_features(returns)
        features.update(micro_features)
        
        # 6. Regime detection features
        regime_features = self._create_regime_features(returns)
        features.update(regime_features)
        
        # Combine all features
        feature_df = pd.DataFrame(features, index=prices.index)
        
        # Apply fractional differentiation if enabled
        if self.use_fractional_diff:
            feature_df = self._apply_fractional_diff(feature_df)
            
        return feature_df.dropna()
    
    def _create_price_momentum_features(self, 
                                        prices: pd.Series,
                                        returns: pd.Series) -> Dict:
        """Create price-based momentum features"""
        features = {}
        
        # Simple momentum
        for period in self.lookback_periods:
            features[f'momentum_{period}'] = (prices / prices.shift(period) - 1)
            features[f'log_momentum_{period}'] = np.log(prices / prices.shift(period))
            
            # Cumulative returns
            features[f'cum_returns_{period}'] = returns.rolling(period).sum()
            
            # Risk-adjusted momentum (Sharpe ratio)
            rolling_mean = returns.rolling(period).mean()
            rolling_std = returns.rolling(period).std()
            features[f'sharpe_{period}'] = rolling_mean / rolling_std * np.sqrt(252)
            
        # Cross-sectional momentum (trend strength)
        for short, long in [(5, 21), (10, 63), (21, 126)]:
            short_mom = prices / prices.shift(short) - 1
            long_mom = prices / prices.shift(long) - 1
            features[f'mom_strength_{short}_{long}'] = short_mom / (long_mom + 1e-8)
            
        # Momentum acceleration
        for period in [10, 21, 63]:
            mom = prices / prices.shift(period) - 1
            features[f'mom_accel_{period}'] = mom - mom.shift(period//2)
            
        return features
    
    def _create_volatility_features(self, returns: pd.Series) -> Dict:
        """Create volatility-based features"""
        features = {}
        
        for window in self.volatility_windows:
            # Rolling volatility
            features[f'vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            
            # EWMA volatility
            features[f'ewm_vol_{window}'] = returns.ewm(span=window).std() * np.sqrt(252)
            
            # Volatility of volatility
            vol = returns.rolling(window).std()
            features[f'vol_of_vol_{window}'] = vol.rolling(window).std()
            
            # Skewness and kurtosis
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()
            
            # Downside volatility
            downside_returns = returns[returns < 0]
            features[f'downside_vol_{window}'] = (
                downside_returns.rolling(window).std() * np.sqrt(252)
            )
            
        # Volatility ratios
        features['vol_ratio_short_long'] = (
            features['vol_10'] / (features['vol_63'] + 1e-8)
        )
        
        return features
    
    def _create_technical_features(self, 
                                   prices: pd.Series,
                                   returns: pd.Series) -> Dict:
        """Create technical momentum indicators"""
        features = {}
        
        # RSI (Relative Strength Index)
        for period in [14, 21, 63]:
            rsi = self._calculate_rsi(prices, period)
            features[f'rsi_{period}'] = rsi
            features[f'rsi_norm_{period}'] = (rsi - 50) / 50  # Normalized RSI
            
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(prices)
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = histogram
        
        # Bollinger Bands position
        for period in [20, 50]:
            bb_pos = self._bollinger_position(prices, period)
            features[f'bb_position_{period}'] = bb_pos
            
        # Williams %R
        for period in [14, 21]:
            williams_r = self._williams_r(prices, period)
            features[f'williams_r_{period}'] = williams_r
            
        # Stochastic Oscillator
        for period in [14, 21]:
            stoch_k, stoch_d = self._stochastic_oscillator(prices, period)
            features[f'stoch_k_{period}'] = stoch_k
            features[f'stoch_d_{period}'] = stoch_d
            
        return features
    
    def _create_volume_features(self, 
                                prices: pd.Series,
                                volume: pd.Series,
                                returns: pd.Series) -> Dict:
        """Create volume-based momentum features"""
        features = {}
        
        # Price-volume relationships
        for period in [10, 21, 63]:
            # Volume-weighted returns
            vol_weighted_ret = (returns * volume).rolling(period).sum() / volume.rolling(period).sum()
            features[f'vol_weighted_ret_{period}'] = vol_weighted_ret
            
            # On-balance volume momentum
            obv = self._calculate_obv(prices, volume)
            features[f'obv_momentum_{period}'] = obv / obv.shift(period) - 1
            
            # Volume rate of change
            features[f'volume_roc_{period}'] = volume / volume.shift(period) - 1
            
            # Price-volume correlation
            features[f'price_vol_corr_{period}'] = (
                prices.rolling(period).corr(volume)
            )
            
        # Volume momentum
        for period in [5, 21]:
            features[f'volume_momentum_{period}'] = (
                volume.rolling(period).mean() / volume.rolling(period*2).mean() - 1
            )
            
        # Accumulation/Distribution Line
        adl = self._accumulation_distribution_line(prices, volume)
        features['adl_momentum_21'] = adl / adl.shift(21) - 1
        
        return features
    
    def _create_microstructure_features(self, returns: pd.Series) -> Dict:
        """Create microstructure-based features"""
        features = {}
        
        # Autocorrelation at various lags
        for lag in [1, 5, 10]:
            features[f'autocorr_lag_{lag}'] = returns.rolling(63).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
            
        # Hurst exponent
        features['hurst_21'] = returns.rolling(63).apply(self._hurst_exponent)
        
        # Variance ratio test
        features['variance_ratio'] = returns.rolling(63).apply(
            lambda x: self._variance_ratio(x, 2) if len(x) >= 10 else np.nan
        )
        
        # Return clustering (ARCH effects)
        abs_returns = returns.abs()
        for period in [5, 10, 21]:
            features[f'return_clustering_{period}'] = (
                abs_returns.rolling(period).std() / abs_returns.rolling(period).mean()
            )
            
        return features
    
    def _create_regime_features(self, returns: pd.Series) -> Dict:
        """Create regime detection features"""
        features = {}
        
        # Rolling correlations with market regimes
        for window in [21, 63, 126]:
            # Trend strength
            price_ma = returns.rolling(window).mean().cumsum().diff()
            features[f'trend_strength_{window}'] = price_ma
            
            # Volatility regime
            vol = returns.rolling(window).std()
            vol_ma = vol.rolling(window).mean()
            features[f'vol_regime_{window}'] = vol / vol_ma - 1
            
            # Distribution shape regime
            skew = returns.rolling(window).skew()
            features[f'skew_regime_{window}'] = skew - skew.rolling(window*2).mean()
            
        return features
    
    def _apply_fractional_diff(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply fractional differentiation to features"""
        frac_diff_features = {}
        
        for col in features.columns:
            if features[col].notna().sum() > 100:  # Minimum observations
                try:
                    # Find optimal d for this feature
                    optimal_d = self.frac_diff.find_optimal_d(
                        features[col].dropna(),
                        d_range=(0.0, 0.8),
                        step=0.1
                    )
                    
                    # Apply fractional differentiation
                    frac_diff_series = self.frac_diff.frac_diff_ffd(
                        features[col].dropna(),
                        optimal_d,
                        threshold=self.frac_diff_threshold
                    )
                    
                    frac_diff_features[col] = frac_diff_series
                    
                except Exception as e:
                    # If fractional diff fails, use original feature
                    frac_diff_features[col] = features[col]
                    
        return pd.DataFrame(frac_diff_features)
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, 
                        prices: pd.Series,
                        fast: int = 12,
                        slow: int = 26,
                        signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper_band = ma + (2 * std)
        lower_band = ma - (2 * std)
        
        position = (prices - lower_band) / (upper_band - lower_band)
        return position
    
    def _williams_r(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high = prices.rolling(period).max()
        low = prices.rolling(period).min()
        
        williams_r = -100 * ((high - prices) / (high - low))
        return williams_r
    
    def _stochastic_oscillator(self, 
                               prices: pd.Series,
                               period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        high = prices.rolling(period).max()
        low = prices.rolling(period).min()
        
        k_percent = 100 * ((prices - low) / (high - low))
        d_percent = k_percent.rolling(3).mean()
        
        return k_percent, d_percent
    
    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        price_change = prices.diff()
        obv = np.where(price_change > 0, volume, 
                       np.where(price_change < 0, -volume, 0)).cumsum()
        return pd.Series(obv, index=prices.index)
    
    def _accumulation_distribution_line(self, 
                                        prices: pd.Series,
                                        volume: pd.Series,
                                        high: Optional[pd.Series] = None,
                                        low: Optional[pd.Series] = None) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        if high is None:
            high = prices
        if low is None:
            low = prices
            
        clv = ((prices - low) - (high - prices)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        adl = (clv * volume).cumsum()
        
        return adl
    
    def _hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent"""
        if len(returns) < 10:
            return np.nan
            
        try:
            lags = range(2, min(20, len(returns)//2))
            tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) 
                   for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return np.nan
    
    def _variance_ratio(self, returns: pd.Series, q: int) -> float:
        """Calculate variance ratio for mean reversion test"""
        if len(returns) < 2*q:
            return np.nan
            
        try:
            n = len(returns)
            variance_1 = np.var(returns, ddof=1)
            variance_q = np.var(returns.rolling(q).sum().dropna(), ddof=1) / q
            
            if variance_1 == 0:
                return np.nan
                
            return variance_q / variance_1
        except:
            return np.nan