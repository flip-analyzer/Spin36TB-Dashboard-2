#!/usr/bin/env python3
"""
López de Prado's "Advances in Financial Machine Learning" Implementation

Key concepts from the book:
1. Triple Barrier Labeling (Chapter 3)
2. Sample Weights based on Label Uniqueness (Chapter 4)
3. Fractional Differentiation for Stationarity (Chapter 5)
4. Purged Cross-Validation (Chapter 7)
5. Feature Importance via Mean Decrease Impurity (Chapter 8)
6. Bet Sizing via Kelly Criterion (Chapter 10)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, log_loss
import warnings
warnings.filterwarnings('ignore')

class LopezDePradoSystem:
    def __init__(self, 
                 pt_sl=[1, 1],      # profit_taking, stop_loss multiples
                 min_ret=0.005,     # minimum return threshold  
                 num_days=1,        # holding period
                 molecule_size=30): # pattern window size
        
        self.pt_sl = pt_sl
        self.min_ret = min_ret
        self.num_days = num_days
        self.molecule_size = molecule_size
        
        self.model = None
        self.features_df = None
        self.labels_df = None
        self.sample_weights = None
        
    def get_daily_volatility(self, close, span=100):
        """
        Chapter 2: Daily Volatility Estimates
        Uses exponentially weighted moving average
        """
        df = close.index.searchsorted(close.index - pd.Timedelta(days=1))
        df = df[df > 0]
        df = pd.Series(close.index[df - 1], index=close.index[close.shape[0] - df.shape[0]:])
        
        try:
            df = close.loc[df.index] / close.loc[df.values].values - 1
        except:
            # Fallback for intraday data
            df = close.pct_change()
        
        df = df.dropna().ewm(span=span).std()
        return df
    
    def get_events(self, close, t_events, pt_sl, target, min_ret=0.0):
        """
        Chapter 3: Triple Barrier Labeling
        Get timestamps of events (barriers being hit)
        """
        # Get target (volatility)
        target = target.loc[t_events]
        target = target[target > min_ret]  # min_ret
        
        if len(target) == 0:
            return pd.DataFrame()
        
        # Get time barriers (t1)
        num_days = pd.Timedelta(days=self.num_days)
        t1 = close.index.searchsorted(target.index + num_days)
        t1 = t1[t1 < close.shape[0]]
        t1 = pd.Series(close.index[t1], index=target.index[:t1.shape[0]])
        
        # Form events object, apply stop loss on vertical barrier
        events = pd.concat({'t1': t1, 'target': target}, axis=1)
        events = events.dropna()
        
        return events
    
    def apply_triple_barrier(self, close, events, pt_sl):
        """
        Chapter 3: Apply Triple Barrier Method
        """
        events = events.dropna(subset=['t1'])
        
        # Get barriers
        up = pt_sl[0] * events['target']
        dn = -pt_sl[1] * events['target'] 
        
        labels = []
        
        for loc, t1 in events['t1'].fillna(close.index[-1]).items():
            df0 = close[loc:t1]  # Path prices
            df0 = (df0 / close[loc] - 1) * events.at[loc, 'target']
            
            # Determine which barrier was hit first
            if df0.max() > up[loc]:
                label = 1  # Profit taking
            elif df0.min() < dn[loc]:
                label = -1  # Stop loss
            else:
                label = 0  # Time barrier
            
            labels.append(label)
        
        labels = pd.Series(labels, index=events.index)
        return labels
    
    def get_sample_weights(self, t_events, close):
        """
        Chapter 4: Sample Weights
        Weight observations by their uniqueness (number of concurrent labels)
        """
        # Calculate average uniqueness of labels
        c = pd.Series(index=close.index, dtype=float)
        
        for i, (t0, t1) in t_events.items():
            c.loc[t0:t1] = c.loc[t0:t1].add(1, fill_value=0)
        
        # Calculate weights (inverse of average uniqueness)
        weights = 1.0 / c.loc[t_events.index]
        weights = weights / weights.sum() * len(weights)  # Normalize
        
        return weights.fillna(1.0)
    
    def frac_diff_fixed_width(self, series, d, thresh=1e-5):
        """
        Chapter 5: Fractional Differentiation
        Make series stationary while preserving memory
        """
        # Get weights for fractional differentiation
        w = [1.0]
        k = 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thresh:
                break
            w.append(w_)
            k += 1
        
        w = np.array(w)
        
        # Apply fractional differentiation
        df = {}
        for name in series.columns:
            seriesF = series[name].fillna(method='ffill').dropna()
            df_ = pd.Series(index=seriesF.index, dtype=float)
            
            for iloc in range(w.shape[0], seriesF.shape[0]):
                loc = seriesF.index[iloc]
                if not np.isfinite(series.loc[loc, name]):
                    continue
                df_.loc[loc] = np.dot(w, seriesF.iloc[iloc-w.shape[0]+1:iloc+1])
            
            df[name] = df_.copy(deep=True)
        
        df = pd.DataFrame(df)
        return df.dropna()
    
    def create_features(self, data):
        """
        Create features following López de Prado's methodology
        """
        print("🔧 Creating López de Prado Features...")
        
        close = data['Close']
        
        # Basic price features
        features = pd.DataFrame(index=close.index)
        
        # Returns and volatility
        features['returns'] = close.pct_change()
        features['volatility'] = self.get_daily_volatility(close)
        
        # Technical indicators (Chapter 2 style)
        features['rsi'] = self._calculate_rsi(close)
        features['bb_upper'], features['bb_lower'] = self._calculate_bollinger_bands(close)
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Momentum features
        for window in [5, 10, 20]:
            features[f'momentum_{window}'] = close.pct_change(window)
            features[f'vol_{window}'] = features['returns'].rolling(window).std()
        
        # Microstructure features (if volume available)
        if 'Volume' in data.columns:
            features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
            features['price_volume'] = features['returns'] * np.log(data['Volume'])
        
        # Apply fractional differentiation to make features stationary
        print("   📊 Applying fractional differentiation...")
        features_stationary = self.frac_diff_fixed_width(features, d=0.4)
        
        print(f"   ✅ Created {len(features_stationary.columns)} stationary features")
        return features_stationary
    
    def _calculate_rsi(self, prices, window=14):
        """RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Bollinger Bands calculation"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    
    def create_labels(self, data):
        """
        Chapter 3: Create Triple Barrier Labels
        """
        print("🏷️  Creating Triple Barrier Labels...")
        
        close = data['Close']
        
        # Get daily volatility for barriers
        daily_vol = self.get_daily_volatility(close)
        
        # Sample events (could be based on structural breaks, here we use regular sampling)
        t_events = close.index[::20]  # Sample every 20 bars
        
        # Get events using triple barrier method
        events = self.get_events(close, t_events, self.pt_sl, daily_vol, self.min_ret)
        
        if len(events) == 0:
            print("   ❌ No events generated")
            return None, None
        
        # Apply triple barrier labeling
        labels = self.apply_triple_barrier(close, events, self.pt_sl)
        
        # Get sample weights based on label uniqueness
        sample_weights = self.get_sample_weights(events[['t1']], close)
        
        print(f"   ✅ Created {len(labels)} labels")
        print(f"      Long (1): {(labels == 1).sum()}")
        print(f"      Short (-1): {(labels == -1).sum()}")
        print(f"      Neutral (0): {(labels == 0).sum()}")
        
        return labels, sample_weights
    
    def purged_cross_val_score(self, X, y, sample_weight=None, n_splits=5, embargo_pct=0.01):
        """
        Chapter 7: Purged Cross-Validation
        Prevents data leakage in time series
        """
        print("🔀 Performing Purged Cross-Validation...")
        
        # Simple implementation - in practice would use PurgedKFold
        scores = []
        n = len(X)
        fold_size = n // n_splits
        embargo_size = int(n * embargo_pct)
        
        for i in range(n_splits):
            # Define test indices
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)
            
            # Define train indices with purging and embargo
            train_indices = list(range(0, max(0, test_start - embargo_size)))
            train_indices.extend(range(min(n, test_end + embargo_size), n))
            
            if len(train_indices) < 50:  # Minimum training samples
                continue
                
            test_indices = list(range(test_start, test_end))
            
            # Train and evaluate
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            
            # Sample weights
            if sample_weight is not None:
                sw_train = sample_weight.iloc[train_indices]
            else:
                sw_train = None
            
            # Fit model
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train, sample_weight=sw_train)
            
            # Score
            y_pred = rf.predict(X_test)
            score = (y_pred == y_test).mean()
            scores.append(score)
        
        print(f"   📊 CV Scores: {scores}")
        print(f"   📈 Mean CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        return np.array(scores)
    
    def train_lopez_model(self, data):
        """
        Train the complete López de Prado system
        """
        print("\n🎯 TRAINING LÓPEZ DE PRADO SYSTEM")
        print("=" * 45)
        
        # Create features
        self.features_df = self.create_features(data)
        
        # Create labels
        self.labels_df, self.sample_weights = self.create_labels(data)
        
        if self.labels_df is None:
            return False
        
        # Align features and labels
        common_index = self.features_df.index.intersection(self.labels_df.index)
        X = self.features_df.loc[common_index].dropna()
        y = self.labels_df.loc[X.index]
        sample_weights = self.sample_weights.loc[X.index] if self.sample_weights is not None else None
        
        print(f"📊 Training Data:")
        print(f"   Features: {X.shape}")
        print(f"   Labels: {len(y)}")
        print(f"   Sample weights: {'Yes' if sample_weights is not None else 'No'}")
        
        # Purged cross-validation
        cv_scores = self.purged_cross_val_score(X, y, sample_weights)
        
        # Train final model on all data
        print("\n🧠 Training Final Model...")
        self.model = RandomForestClassifier(
            n_estimators=500,  # More trees for better performance
            max_depth=10,      # Prevent overfitting
            min_samples_leaf=50, # López de Prado suggests larger leaf sizes
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y, sample_weight=sample_weights)
        
        # Feature importance (Chapter 8)
        feature_importance = pd.Series(
            self.model.feature_importances_, 
            index=X.columns
        ).sort_values(ascending=False)
        
        print("📈 Top Feature Importances:")
        for feat, imp in feature_importance.head(10).items():
            print(f"   {feat}: {imp:.3f}")
        
        return True
    
    def calculate_kelly_bet_size(self, probabilities, max_bet=0.25):
        """
        Chapter 10: Kelly Criterion for Bet Sizing
        """
        # probabilities = [prob_up, prob_down, prob_neutral]
        prob_up, prob_down, prob_neutral = probabilities
        
        # Assuming 1:1 risk/reward ratio (can be improved)
        if prob_up > prob_down and prob_up > 0.5:
            # Long position
            edge = prob_up - prob_down
            kelly_fraction = edge / 1.0  # Assuming 1:1 payoff
            return min(kelly_fraction * 0.25, max_bet)  # 25% of Kelly for safety
        elif prob_down > prob_up and prob_down > 0.5:
            # Short position  
            edge = prob_down - prob_up
            kelly_fraction = edge / 1.0
            return -min(kelly_fraction * 0.25, max_bet)  # Negative for short
        else:
            return 0.0  # No edge
    
    def predict_and_size(self, current_features):
        """
        Make prediction and calculate position size using López de Prado methodology
        """
        if self.model is None:
            return "HOLD", 0.0, [0.33, 0.33, 0.33]
        
        # Get prediction probabilities
        proba = self.model.predict_proba(current_features.values.reshape(1, -1))[0]
        
        # Map to [prob_down, prob_neutral, prob_up] 
        if len(proba) == 3:  # [-1, 0, 1] classes
            prob_down, prob_neutral, prob_up = proba
        else:
            # Handle binary case
            prob_neutral = 0.1
            prob_up = proba[1] if len(proba) == 2 else 0.45
            prob_down = 1 - prob_up - prob_neutral
        
        probabilities = [prob_up, prob_down, prob_neutral]
        
        # Calculate Kelly bet size
        bet_size = self.calculate_kelly_bet_size(probabilities)
        
        # Determine direction
        if bet_size > 0:
            direction = "BUY"
        elif bet_size < 0:
            direction = "SELL"
            bet_size = abs(bet_size)
        else:
            direction = "HOLD"
            bet_size = 0.0
        
        return direction, bet_size, probabilities


def main():
    """
    Run López de Prado's complete methodology
    """
    print("📚 LÓPEZ DE PRADO'S 'ADVANCES IN FINANCIAL MACHINE LEARNING'")
    print("=" * 65)
    
    # Load data with proper splits (no overfitting)
    print("📊 Loading EURUSD Data...")
    try:
        data_path = "/Users/jonspinogatti/Desktop/spin35TB/data/candles/eurusd_m5_500k.csv"
        data = pd.read_csv(data_path)
        
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        
        print(f"   ✅ Loaded {len(data):,} EURUSD 5-minute bars")
        
        # Use only first 60% for training (avoid overfitting)
        train_size = int(len(data) * 0.6)
        train_data = data.iloc[:train_size]
        
        print(f"   🧠 Training on {len(train_data):,} bars (60%)")
        print(f"   🔒 Keeping {len(data) - len(train_data):,} bars for validation/testing")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Initialize López de Prado system
    system = LopezDePradoSystem(
        pt_sl=[2, 2],        # 2x volatility for profit/stop
        min_ret=0.002,       # Minimum 0.2% move 
        num_days=0.02,       # ~30 minutes for 5-minute bars
        molecule_size=30
    )
    
    # Train the complete system
    if system.train_lopez_model(train_data):
        print(f"\n✅ LÓPEZ DE PRADO SYSTEM TRAINED!")
        print(f"   📊 Features: Fractionally differentiated")
        print(f"   🏷️  Labels: Triple barrier method")
        print(f"   ⚖️  Weights: Based on label uniqueness")
        print(f"   🔀 Validation: Purged cross-validation")
        print(f"   🤖 Model: Random Forest with proper regularization")
        print(f"   💰 Sizing: Kelly Criterion")
    else:
        print("❌ Training failed")
        return
    
    return system


if __name__ == "__main__":
    lopez_system = main()