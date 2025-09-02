#!/usr/bin/env python3
"""
López de Prado Core System - Streamlined Implementation

Focus on the most impactful concepts from "Advances in Financial Machine Learning":
1. Triple Barrier Labeling (Chapter 3) - Most important
2. Sample Weights (Chapter 4) - Critical for time series
3. Purged Cross-Validation (Chapter 7) - Prevents overfitting  
4. Kelly Criterion Sizing (Chapter 10) - Optimal bet sizing
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class LopezCoreSystem:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
    def get_daily_vol(self, close, span=100):
        """Estimate daily volatility using EWMA"""
        returns = close.pct_change().dropna()
        return returns.ewm(span=span).std()
    
    def triple_barrier_labels(self, close, vol, pt_sl=[1.5, 1.5], num_bars=6):
        """
        Chapter 3: Triple Barrier Method
        Simplified for 5-minute bars with 30-minute exits
        """
        print("🏷️  Creating Triple Barrier Labels...")
        
        labels = []
        barrier_info = []
        
        # Process each potential entry point
        for i in range(len(close) - num_bars):
            entry_price = close.iloc[i]
            entry_vol = vol.iloc[i]
            
            if pd.isna(entry_vol) or entry_vol == 0:
                continue
            
            # Set barriers
            upper_barrier = entry_price * (1 + pt_sl[0] * entry_vol)
            lower_barrier = entry_price * (1 - pt_sl[1] * entry_vol)
            
            # Look at next num_bars for barrier hits
            future_prices = close.iloc[i+1:i+1+num_bars]
            
            # Check which barrier hit first
            hit_upper = (future_prices >= upper_barrier).any()
            hit_lower = (future_prices <= lower_barrier).any()
            
            if hit_upper and hit_lower:
                # Both hit - check which came first
                upper_idx = (future_prices >= upper_barrier).idxmax() if hit_upper else len(future_prices)
                lower_idx = (future_prices <= lower_barrier).idxmax() if hit_lower else len(future_prices)
                label = 1 if upper_idx < lower_idx else -1
            elif hit_upper:
                label = 1  # Profit target hit
            elif hit_lower:
                label = -1  # Stop loss hit
            else:
                # Time barrier - check final return
                final_price = future_prices.iloc[-1]
                final_return = (final_price / entry_price) - 1
                label = 1 if final_return > 0.001 else (-1 if final_return < -0.001 else 0)
            
            labels.append(label)
            barrier_info.append({
                'timestamp': close.index[i],
                'entry_price': entry_price,
                'upper_barrier': upper_barrier,
                'lower_barrier': lower_barrier,
                'label': label
            })
        
        labels_df = pd.DataFrame(barrier_info).set_index('timestamp')
        
        print(f"   ✅ Created {len(labels_df)} labels")
        print(f"      Buy signals (1): {(labels_df['label'] == 1).sum()}")
        print(f"      Sell signals (-1): {(labels_df['label'] == -1).sum()}")
        print(f"      Neutral (0): {(labels_df['label'] == 0).sum()}")
        
        return labels_df
    
    def get_sample_weights(self, labels_df, close):
        """
        Chapter 4: Sample Weights based on Label Uniqueness
        Weight samples by how unique/concurrent they are
        """
        print("⚖️  Calculating Sample Weights...")
        
        # Count concurrent labels at each timestamp
        weights = pd.Series(1.0, index=labels_df.index)
        
        # Simplified: weight by label frequency (rarer labels get higher weight)
        label_counts = labels_df['label'].value_counts()
        total_labels = len(labels_df)
        
        for idx, row in labels_df.iterrows():
            label = row['label']
            # Inverse frequency weighting
            weights[idx] = total_labels / (label_counts[label] * len(label_counts))
        
        # Normalize weights
        weights = weights / weights.mean()
        
        print(f"   ✅ Sample weights computed (mean: {weights.mean():.2f}, std: {weights.std():.2f})")
        
        return weights
    
    def create_features(self, data, lookback=30):
        """Create features for each timestamp"""
        print("🔧 Creating Features...")
        
        close = data['Close']
        high = data['High'] 
        low = data['Low']
        volume = data['Volume'] if 'Volume' in data.columns else None
        
        features = pd.DataFrame(index=close.index)
        
        # Price momentum features
        for period in [5, 10, 20]:
            features[f'return_{period}'] = close.pct_change(period)
            features[f'volatility_{period}'] = close.pct_change().rolling(period).std()
        
        # Technical features
        features['rsi'] = self.calculate_rsi(close, 14)
        
        # Bollinger Bands
        bb_mean = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_position'] = (close - (bb_mean - 2*bb_std)) / (4 * bb_std)
        
        # High-Low features
        features['hl_ratio'] = (high - low) / close
        features['close_position'] = (close - low) / (high - low)
        
        # Volume features (if available)
        if volume is not None:
            features['volume_sma'] = volume / volume.rolling(20).mean()
            features['volume_trend'] = volume.pct_change(5)
        
        # Pattern features - simplified version of 30-candle windows
        for i in range(1, 6):  # Last 5 bars
            features[f'return_lag_{i}'] = close.pct_change().shift(i)
            features[f'volatility_lag_{i}'] = close.pct_change().rolling(5).std().shift(i)
        
        # Remove infinities and fill NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        print(f"   ✅ Created {len(features.columns)} features")
        
        return features
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def purged_kfold_validate(self, X, y, sample_weights, n_splits=3, embargo_pct=0.02):
        """
        Chapter 7: Purged Cross-Validation
        Simplified implementation to prevent data leakage
        """
        print("🔀 Purged Cross-Validation...")
        
        n = len(X)
        fold_size = n // n_splits
        embargo_size = int(n * embargo_pct)
        
        scores = []
        
        for i in range(n_splits):
            # Test set
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)
            
            # Training set with purging and embargo
            train_idx = list(range(0, max(0, test_start - embargo_size)))
            train_idx.extend(range(min(n, test_end + embargo_size), n))
            
            if len(train_idx) < 100:  # Need minimum samples
                continue
            
            test_idx = list(range(test_start, test_end))
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            sw_train = sample_weights.iloc[train_idx]
            
            # Train model
            rf = RandomForestClassifier(n_estimators=100, max_depth=8, 
                                     min_samples_leaf=20, random_state=42)
            rf.fit(X_train, y_train, sample_weight=sw_train)
            
            # Evaluate
            score = rf.score(X_test, y_test)
            scores.append(score)
            
            print(f"   Fold {i+1}: {score:.3f}")
        
        print(f"   📊 CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        return np.mean(scores)
    
    def kelly_criterion_sizing(self, win_prob, loss_prob, win_size=1.0, loss_size=1.0, max_bet=0.05):
        """
        Chapter 10: Kelly Criterion for Position Sizing
        """
        if win_prob <= loss_prob:
            return 0.0  # No edge
        
        # Kelly formula: f = (bp - q) / b
        # b = win_size/loss_size (payoff ratio)
        # p = win_prob
        # q = loss_prob = 1 - p (assuming binary outcome)
        
        b = win_size / loss_size
        p = win_prob / (win_prob + loss_prob)  # Normalize probabilities
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety factor (quarter Kelly)
        safe_kelly = kelly_fraction * 0.25
        
        return min(max(safe_kelly, 0), max_bet)
    
    def train_system(self, data):
        """Train the complete López de Prado system"""
        print("\n🎯 TRAINING LÓPEZ DE PRADO CORE SYSTEM")
        print("=" * 50)
        
        # Use only 60% for training
        train_size = int(len(data) * 0.6)
        train_data = data.iloc[:train_size]
        
        print(f"📊 Training on {len(train_data):,} bars")
        
        # Step 1: Create features
        features = self.create_features(train_data)
        
        # Step 2: Create triple barrier labels
        vol = self.get_daily_vol(train_data['Close'])
        labels_df = self.triple_barrier_labels(train_data['Close'], vol)
        
        # Step 3: Calculate sample weights
        sample_weights = self.get_sample_weights(labels_df, train_data['Close'])
        
        # Step 4: Align data
        common_idx = features.index.intersection(labels_df.index)
        X = features.loc[common_idx].dropna()
        y = labels_df.loc[X.index, 'label']
        sw = sample_weights.loc[X.index]
        
        print(f"📋 Aligned Data: {len(X)} samples, {len(X.columns)} features")
        
        # Step 5: Purged cross-validation
        cv_score = self.purged_kfold_validate(X, y, sw)
        
        # Step 6: Train final model
        print("🧠 Training Final Model...")
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10, 
            min_samples_leaf=30,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y, sample_weight=sw)
        self.feature_columns = X.columns
        
        # Feature importance
        importance = pd.Series(self.model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\n📈 Top Features:")
        for feat, imp in importance.head(10).items():
            print(f"   {feat}: {imp:.3f}")
        
        print(f"\n✅ LÓPEZ DE PRADO SYSTEM TRAINED!")
        print(f"   🎯 CV Score: {cv_score:.3f}")
        print(f"   📊 Features: {len(self.feature_columns)}")
        print(f"   🤖 Model: Random Forest with sample weighting")
        
        return True
    
    def predict_and_size(self, current_data):
        """Make prediction and calculate Kelly position size"""
        if self.model is None:
            return "HOLD", 0.0, [0.33, 0.33, 0.33]
        
        # Create features for current data
        features = self.create_features(current_data)
        current_features = features.iloc[-1][self.feature_columns]
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba([current_features])[0]
        
        # Map to classes [-1, 0, 1] = [sell, hold, buy]
        if len(probabilities) == 3:
            prob_sell, prob_hold, prob_buy = probabilities
        else:
            # Handle binary classification
            prob_buy = probabilities[1] if len(probabilities) == 2 else 0.4
            prob_sell = probabilities[0] if len(probabilities) == 2 else 0.4
            prob_hold = 1 - prob_buy - prob_sell
        
        # Calculate Kelly position sizes
        buy_size = self.kelly_criterion_sizing(prob_buy, prob_sell + prob_hold)
        sell_size = self.kelly_criterion_sizing(prob_sell, prob_buy + prob_hold)
        
        # Make decision
        if buy_size > sell_size and buy_size > 0.01:
            return "BUY", buy_size, [prob_buy, prob_sell, prob_hold]
        elif sell_size > buy_size and sell_size > 0.01:
            return "SELL", sell_size, [prob_buy, prob_sell, prob_hold]
        else:
            return "HOLD", 0.0, [prob_buy, prob_sell, prob_hold]


def main():
    """Run López de Prado core system"""
    print("📚 LÓPEZ DE PRADO CORE METHODOLOGY")
    print("='Advances in Financial Machine Learning'")
    print("=" * 50)
    
    # Load EURUSD data
    try:
        data_path = "/Users/jonspinogatti/Desktop/spin35TB/data/candles/eurusd_m5_500k.csv"
        data = pd.read_csv(data_path)
        
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        
        print(f"📊 Loaded {len(data):,} EURUSD 5-minute bars")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize and train system
    system = LopezCoreSystem()
    
    if system.train_system(data):
        # Test current market prediction
        print(f"\n🔮 CURRENT MARKET ANALYSIS:")
        direction, position_size, probabilities = system.predict_and_size(data.tail(100))
        
        prob_buy, prob_sell, prob_hold = probabilities
        
        print(f"   🎯 Signal: {direction}")
        print(f"   📏 Kelly Size: {position_size:.2%}")
        print(f"   📊 Probabilities:")
        print(f"      Buy: {prob_buy:.1%}")
        print(f"      Sell: {prob_sell:.1%}")
        print(f"      Hold: {prob_hold:.1%}")
        
        if position_size > 0:
            print(f"   ✅ TRADE: {direction} {position_size:.1%} of portfolio")
        else:
            print(f"   ⚠️  NO TRADE: Insufficient edge")
    
    return system


if __name__ == "__main__":
    lopez_system = main()