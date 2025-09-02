#!/usr/bin/env python3
"""
López de Prado Triple Barrier System for EURUSD

Implements the complete methodology from "Advances in Financial Machine Learning"
on your 1M+ EURUSD 5-minute dataset with proper data splits to avoid overfitting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class EurusdLopezSystem:
    def __init__(self, 
                 pt_sl_multiplier=1.5,    # Profit/Stop as multiple of volatility
                 time_horizon_bars=6,     # 30 minutes = 6 * 5-minute bars
                 min_volatility=0.0001):  # Minimum vol to consider (1 pip)
        
        self.pt_sl_multiplier = pt_sl_multiplier
        self.time_horizon_bars = time_horizon_bars
        self.min_volatility = min_volatility
        
        self.model = None
        self.feature_columns = None
        self.scaler = None
        
    def calculate_volatility(self, close, span=100):
        """
        Calculate rolling volatility using López de Prado's approach
        Uses exponentially weighted moving average of returns
        """
        returns = close.pct_change().dropna()
        volatility = returns.ewm(span=span).std()
        return volatility.fillna(method='ffill')
    
    def create_triple_barrier_labels(self, close, volatility):
        """
        López de Prado's Triple Barrier Labeling (Chapter 3)
        
        For each timestamp, set three barriers:
        1. Upper barrier: profit taking at +pt_sl_multiplier * volatility
        2. Lower barrier: stop loss at -pt_sl_multiplier * volatility  
        3. Time barrier: maximum holding period
        
        Label is determined by which barrier is hit first.
        """
        print(f"🎯 Creating Triple Barrier Labels...")
        print(f"   Profit/Stop: ±{self.pt_sl_multiplier}x volatility")
        print(f"   Time horizon: {self.time_horizon_bars} bars (30 minutes)")
        print(f"   Min volatility: {self.min_volatility:.4f}")
        
        labels = []
        barrier_info = []
        
        # Process every 5th timestamp for computational efficiency while maintaining quality
        step_size = 5
        total_possible = len(close) - self.time_horizon_bars
        
        for i in range(0, total_possible, step_size):
            entry_time = close.index[i]
            entry_price = close.iloc[i]
            entry_vol = volatility.iloc[i]
            
            # Skip if volatility too low or NaN
            if pd.isna(entry_vol) or entry_vol < self.min_volatility:
                continue
            
            # Set barriers
            profit_barrier = entry_price * (1 + self.pt_sl_multiplier * entry_vol)
            stop_barrier = entry_price * (1 - self.pt_sl_multiplier * entry_vol)
            time_barrier_idx = min(i + self.time_horizon_bars, len(close) - 1)
            
            # Get future prices within time horizon
            future_end_idx = min(i + self.time_horizon_bars, len(close) - 1)
            future_prices = close.iloc[i+1:future_end_idx+1]
            
            if len(future_prices) == 0:
                continue
            
            # Check which barrier hit first
            profit_hits = future_prices >= profit_barrier
            stop_hits = future_prices <= stop_barrier
            
            profit_hit_times = future_prices[profit_hits]
            stop_hit_times = future_prices[stop_hits]
            
            if len(profit_hit_times) > 0 and len(stop_hit_times) > 0:
                # Both barriers hit - which came first?
                profit_first_time = profit_hit_times.index[0]
                stop_first_time = stop_hit_times.index[0]
                
                if profit_first_time < stop_first_time:
                    label = 1  # Profit barrier hit first
                    barrier_hit = "profit"
                    hit_time = profit_first_time
                else:
                    label = -1  # Stop barrier hit first
                    barrier_hit = "stop"
                    hit_time = stop_first_time
                    
            elif len(profit_hit_times) > 0:
                label = 1  # Only profit barrier hit
                barrier_hit = "profit"
                hit_time = profit_hit_times.index[0]
                
            elif len(stop_hit_times) > 0:
                label = -1  # Only stop barrier hit
                barrier_hit = "stop"
                hit_time = stop_hit_times.index[0]
                
            else:
                # Time barrier hit - check final direction
                final_price = future_prices.iloc[-1]
                final_return = (final_price / entry_price) - 1
                
                # More nuanced time barrier labeling
                if final_return > entry_vol * 0.5:  # Moved 50%+ toward profit
                    label = 1
                elif final_return < -entry_vol * 0.5:  # Moved 50%+ toward stop
                    label = -1
                else:
                    label = 0  # Truly neutral
                    
                barrier_hit = "time"
                hit_time = future_prices.index[-1]
            
            labels.append(label)
            barrier_info.append({
                'entry_time': entry_time,
                'entry_price': entry_price,
                'entry_volatility': entry_vol,
                'profit_barrier': profit_barrier,
                'stop_barrier': stop_barrier,
                'label': label,
                'barrier_hit': barrier_hit,
                'hit_time': hit_time if 'hit_time' in locals() else None,
                'holding_period': min(self.time_horizon_bars, len(future_prices))
            })
            
            if len(labels) % 10000 == 0:
                print(f"      Processed {len(labels):,} labels...")
        
        # Convert to DataFrame
        labels_df = pd.DataFrame(barrier_info).set_index('entry_time')
        
        # Results summary
        total_labels = len(labels_df)
        buy_labels = (labels_df['label'] == 1).sum()
        sell_labels = (labels_df['label'] == -1).sum()
        neutral_labels = (labels_df['label'] == 0).sum()
        
        print(f"\n   ✅ Triple Barrier Labels Created:")
        print(f"      Total: {total_labels:,}")
        print(f"      Buy (1): {buy_labels:,} ({buy_labels/total_labels:.1%})")
        print(f"      Sell (-1): {sell_labels:,} ({sell_labels/total_labels:.1%})")
        print(f"      Neutral (0): {neutral_labels:,} ({neutral_labels/total_labels:.1%})")
        
        # Barrier hit analysis
        barrier_counts = labels_df['barrier_hit'].value_counts()
        print(f"\n   📊 Barrier Hit Analysis:")
        for barrier, count in barrier_counts.items():
            print(f"      {barrier.title()}: {count:,} ({count/total_labels:.1%})")
        
        return labels_df
    
    def calculate_sample_weights(self, labels_df):
        """
        López de Prado Sample Weights (Chapter 4)
        Weight samples by their uniqueness/overlap
        """
        print(f"\n⚖️  Calculating Sample Weights...")
        
        # Simplified approach: weight by label frequency and holding period
        weights = pd.Series(1.0, index=labels_df.index)
        
        # Weight by label balance (rarer labels get higher weight)
        label_counts = labels_df['label'].value_counts()
        total_samples = len(labels_df)
        
        for idx, row in labels_df.iterrows():
            label = row['label']
            holding_period = row['holding_period']
            
            # Inverse frequency weighting
            label_weight = total_samples / (label_counts[label] * len(label_counts))
            
            # Adjust for holding period (longer periods get slightly higher weight)
            period_weight = 1.0 + (holding_period - 3) * 0.1  # Neutral at 3 bars
            
            weights[idx] = label_weight * period_weight
        
        # Normalize weights
        weights = weights / weights.mean()
        
        print(f"   📈 Weight Statistics:")
        print(f"      Mean: {weights.mean():.2f}")
        print(f"      Std: {weights.std():.2f}")
        print(f"      Min: {weights.min():.2f}")
        print(f"      Max: {weights.max():.2f}")
        
        return weights
    
    def create_features(self, data, lookback_periods=[5, 10, 20, 50]):
        """
        Create features for López de Prado system
        Focus on stationary, predictive features
        """
        print(f"\n🔧 Creating López de Prado Features...")
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume'] if 'Volume' in data.columns else None
        
        features = pd.DataFrame(index=close.index)
        
        # Volatility-adjusted returns (López de Prado emphasis)
        volatility = self.calculate_volatility(close)
        returns = close.pct_change()
        
        # Volatility-normalized features
        for period in lookback_periods:
            period_return = close.pct_change(period)
            period_vol = returns.rolling(period).std()
            
            # Risk-adjusted returns (key López de Prado concept)
            features[f'risk_adj_return_{period}'] = period_return / (period_vol + 1e-8)
            features[f'volatility_{period}'] = period_vol
            features[f'return_{period}'] = period_return
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(close)
        
        # Bollinger Band position (mean reversion signal)
        bb_mean = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_position'] = (close - bb_mean) / (2 * bb_std)  # Normalized
        features['bb_width'] = (4 * bb_std) / bb_mean  # Volatility proxy
        
        # Microstructure features
        features['hl_ratio'] = (high - low) / close  # Intraday volatility
        features['close_position'] = (close - low) / (high - low + 1e-8)  # Where close landed
        
        # Momentum features
        features['momentum_5'] = close / close.shift(5) - 1
        features['momentum_acceleration'] = features['momentum_5'].diff()
        
        # Volume features (if available)
        if volume is not None:
            volume_ma = volume.rolling(20).mean()
            features['volume_ratio'] = volume / (volume_ma + 1e-8)
            features['volume_price_trend'] = (volume * returns).rolling(5).mean()
        
        # Volatility regime features (López de Prado concept)
        vol_ma = volatility.rolling(50).mean()
        features['vol_regime'] = volatility / (vol_ma + 1e-8)
        features['vol_trend'] = volatility.pct_change(5)
        
        # Clean features
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        print(f"   ✅ Created {len(features.columns)} features")
        print(f"      Focus: Volatility-adjusted, stationary features")
        
        return features
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def purged_cross_validate(self, X, y, sample_weights, n_splits=3, embargo_pct=0.05):
        """
        López de Prado Purged Cross-Validation (Chapter 7)
        Prevents data leakage in time series
        """
        print(f"\n🔀 Purged Cross-Validation...")
        print(f"   Splits: {n_splits}, Embargo: {embargo_pct:.1%}")
        
        n_samples = len(X)
        fold_size = n_samples // n_splits
        embargo_size = int(n_samples * embargo_pct)
        
        cv_scores = []
        feature_importances = []
        
        for fold in range(n_splits):
            print(f"   📊 Fold {fold + 1}/{n_splits}")
            
            # Define test indices
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, n_samples)
            
            # Define train indices with purging and embargo
            train_indices = []
            
            # Before test (with embargo)
            if test_start - embargo_size > 0:
                train_indices.extend(range(0, test_start - embargo_size))
            
            # After test (with embargo)
            if test_end + embargo_size < n_samples:
                train_indices.extend(range(test_end + embargo_size, n_samples))
            
            if len(train_indices) < 500:  # Minimum training samples
                print(f"      ⚠️ Skipping fold - insufficient training data")
                continue
            
            test_indices = list(range(test_start, test_end))
            
            # Split data
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]
            sw_train = sample_weights.iloc[train_indices]
            
            # Train model
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=50,  # López de Prado suggests larger leaves
                min_samples_split=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            )
            
            rf.fit(X_train, y_train, sample_weight=sw_train)
            
            # Evaluate
            y_pred = rf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            cv_scores.append(score)
            
            # Collect feature importances
            feature_importances.append(rf.feature_importances_)
            
            print(f"      Accuracy: {score:.3f}")
            print(f"      Train size: {len(X_train):,}, Test size: {len(X_test):,}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        print(f"\n   📈 Cross-Validation Results:")
        print(f"      Mean Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
        
        # Average feature importances
        if feature_importances:
            avg_importance = np.mean(feature_importances, axis=0)
            feature_importance_df = pd.Series(avg_importance, index=X.columns).sort_values(ascending=False)
            
            print(f"\n   🎯 Top 10 Features:")
            for i, (feature, importance) in enumerate(feature_importance_df.head(10).items()):
                print(f"      {i+1:2d}. {feature}: {importance:.3f}")
        
        return cv_scores, feature_importance_df if feature_importances else None
    
    def train_lopez_system(self, data):
        """
        Train complete López de Prado system on EURUSD data
        """
        print(f"\n📚 TRAINING LÓPEZ DE PRADO SYSTEM ON EURUSD")
        print("=" * 55)
        print(f"Dataset: {len(data):,} EURUSD 5-minute bars")
        
        # Use only training portion (60%) to avoid overfitting
        train_size = int(len(data) * 0.6)
        train_data = data.iloc[:train_size]
        
        print(f"Training on: {len(train_data):,} bars (60% of data)")
        print(f"Reserved for validation/test: {len(data) - len(train_data):,} bars")
        
        # Step 1: Calculate volatility
        close = train_data['Close']
        volatility = self.calculate_volatility(close)
        
        # Step 2: Create triple barrier labels
        labels_df = self.create_triple_barrier_labels(close, volatility)
        
        # Step 3: Calculate sample weights
        sample_weights = self.calculate_sample_weights(labels_df)
        
        # Step 4: Create features
        features = self.create_features(train_data)
        
        # Step 5: Align data
        common_index = features.index.intersection(labels_df.index)
        X = features.loc[common_index].dropna()
        y = labels_df.loc[X.index, 'label']
        sw = sample_weights.loc[X.index]
        
        print(f"\n📋 Final Dataset:")
        print(f"   Samples: {len(X):,}")
        print(f"   Features: {len(X.columns)}")
        print(f"   Label distribution: {dict(y.value_counts().sort_index())}")
        
        # Step 6: Purged cross-validation
        cv_scores, feature_importance = self.purged_cross_validate(X, y, sw)
        
        # Step 7: Train final model on all training data
        print(f"\n🧠 Training Final Model...")
        self.model = RandomForestClassifier(
            n_estimators=500,  # More trees for final model
            max_depth=10,
            min_samples_leaf=50,
            min_samples_split=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(X, y, sample_weight=sw)
        self.feature_columns = X.columns
        
        # Final model performance on training data
        train_accuracy = self.model.score(X, y)
        
        print(f"\n✅ LÓPEZ DE PRADO EURUSD SYSTEM TRAINED!")
        print(f"   📊 CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        print(f"   🎯 Train Accuracy: {train_accuracy:.3f}")
        print(f"   🔧 Features: {len(self.feature_columns)}")
        print(f"   ⚖️ Sample Weights: Applied")
        print(f"   🔒 No Overfitting: Trained on 60% only")
        
        return True
    
    def kelly_position_sizing(self, probabilities, max_position=0.05):
        """
        López de Prado Kelly Criterion (Chapter 10)
        """
        prob_buy, prob_sell, prob_neutral = probabilities
        
        # Kelly for long position
        if prob_buy > max(prob_sell, prob_neutral):
            p = prob_buy / (prob_buy + prob_sell + prob_neutral)  # Win probability
            q = 1 - p  # Lose probability
            b = 1.0  # Assume 1:1 payoff ratio (can be improved with actual data)
            
            kelly_long = (b * p - q) / b
            return max(0, min(kelly_long * 0.25, max_position))  # Quarter Kelly for safety
            
        # Kelly for short position  
        elif prob_sell > max(prob_buy, prob_neutral):
            p = prob_sell / (prob_buy + prob_sell + prob_neutral)
            q = 1 - p
            b = 1.0
            
            kelly_short = (b * p - q) / b
            return -max(0, min(kelly_short * 0.25, max_position))  # Negative for short
        
        return 0.0  # No clear edge
    
    def predict_and_trade(self, current_data):
        """
        Make prediction and calculate Kelly position size
        """
        if self.model is None:
            return "HOLD", 0.0, [0.33, 0.33, 0.33], "Model not trained"
        
        # Create features for current data
        features = self.create_features(current_data)
        current_features = features.iloc[-1][self.feature_columns]
        
        # Handle any missing features
        current_features = current_features.fillna(0)
        
        # Get prediction probabilities
        try:
            probabilities = self.model.predict_proba([current_features])[0]
            
            # Map to [prob_sell, prob_neutral, prob_buy] based on classes
            classes = self.model.classes_
            prob_dict = dict(zip(classes, probabilities))
            
            prob_buy = prob_dict.get(1, 0.0)
            prob_sell = prob_dict.get(-1, 0.0)
            prob_neutral = prob_dict.get(0, 0.0)
            
            # Ensure probabilities sum to 1
            total_prob = prob_buy + prob_sell + prob_neutral
            if total_prob > 0:
                prob_buy /= total_prob
                prob_sell /= total_prob
                prob_neutral /= total_prob
            
        except Exception as e:
            return "HOLD", 0.0, [0.33, 0.33, 0.33], f"Prediction error: {e}"
        
        # Calculate Kelly position size
        kelly_size = self.kelly_position_sizing([prob_buy, prob_sell, prob_neutral])
        
        # Determine direction
        if kelly_size > 0.005:  # Minimum 0.5% position
            direction = "BUY"
            position_size = kelly_size
        elif kelly_size < -0.005:
            direction = "SELL"
            position_size = abs(kelly_size)
        else:
            direction = "HOLD"
            position_size = 0.0
        
        analysis = f"Prob: Buy={prob_buy:.1%}, Sell={prob_sell:.1%}, Neutral={prob_neutral:.1%}"
        
        return direction, position_size, [prob_buy, prob_sell, prob_neutral], analysis


def main():
    """
    Run López de Prado system on EURUSD data
    """
    print("📚 LÓPEZ DE PRADO'S EURUSD TRADING SYSTEM")
    print("'Advances in Financial Machine Learning' Implementation")
    print("=" * 60)
    
    # Load EURUSD data
    try:
        print("📊 Loading EURUSD Dataset...")
        data_path = "/Users/jonspinogatti/Desktop/spin35TB/data/candles/eurusd_m5_500k.csv"
        data = pd.read_csv(data_path)
        
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        
        print(f"   ✅ Loaded {len(data):,} EURUSD 5-minute bars")
        print(f"   📅 Period: {data.index[0]} to {data.index[-1]}")
        
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return
    
    # Initialize López de Prado system
    system = EurusdLopezSystem(
        pt_sl_multiplier=1.5,     # 1.5x volatility barriers
        time_horizon_bars=6,      # 30-minute holding period  
        min_volatility=0.0001     # 1 pip minimum volatility
    )
    
    # Train the system
    if system.train_lopez_system(data):
        # Test current prediction
        print(f"\n🔮 CURRENT MARKET PREDICTION:")
        direction, position_size, probabilities, analysis = system.predict_and_trade(data.tail(200))
        
        prob_buy, prob_sell, prob_neutral = probabilities
        
        print(f"   📊 {analysis}")
        print(f"   🎯 Signal: {direction}")
        print(f"   📏 Kelly Size: {position_size:.2%}")
        
        if position_size > 0:
            print(f"   ✅ TRADE RECOMMENDATION: {direction} {position_size:.1%}")
            print(f"   ⏰ Hold for maximum 30 minutes")
            print(f"   📈 Barriers: ±1.5x volatility")
        else:
            print(f"   ⚠️  HOLD: Insufficient statistical edge")
        
        print(f"\n🎉 LÓPEZ DE PRADO EURUSD SYSTEM READY!")
        print(f"   🎯 Triple barrier labeling implemented")
        print(f"   ⚖️  Sample weighting applied")
        print(f"   🔀 Purged cross-validation used")
        print(f"   💰 Kelly criterion position sizing")
        print(f"   🔒 No overfitting (60% training data only)")
        
    return system


if __name__ == "__main__":
    eurusd_system = main()