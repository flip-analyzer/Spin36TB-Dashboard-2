#!/usr/bin/env python3
"""
Clustering Model Trainer for Hybrid System
Trains Carson's clustering model using your existing EURUSD data
Prepares pattern recognition for high-frequency trading
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import warnings

warnings.filterwarnings('ignore')

class ClusteringModelTrainer:
    """
    Train clustering model for pattern recognition using live EURUSD data
    """
    
    def __init__(self):
        # OANDA Configuration (use your existing setup)
        self.api_token = "fdf72f4b5166602e39e001d1c1cb38b0-524c6662baa3187260b34c006eeab1fd"
        self.base_url = "https://api-fxpractice.oanda.com"
        
        # Clustering configuration
        self.window_size = 30  # Carson's 30-candle patterns
        self.n_clusters = 8    # 8 clusters for pattern recognition
        self.n_pca_components = 20
        self.min_patterns = 500  # Need at least 500 patterns for training
        
        # Model components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_pca_components)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        
        # Training data
        self.patterns = []
        self.outcomes = []
        self.cluster_stats = {}
        
        print("🧠 CLUSTERING MODEL TRAINER INITIALIZED")
        print("=" * 50)
        print(f"📊 Target: {self.n_clusters} clusters from {self.window_size}-candle patterns")
        print(f"🎯 Minimum training patterns: {self.min_patterns}")
    
    def download_training_data(self, days_back=60):
        """Download historical EURUSD data for training"""
        print(f"📥 Downloading {days_back} days of EURUSD data...")
        
        try:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            
            # Calculate start time
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            params = {
                "from": start_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                "to": end_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                "granularity": "M5",
                "price": "M"
            }
            
            response = requests.get(
                f"{self.base_url}/v3/instruments/EUR_USD/candles",
                headers=headers,
                params=params,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                df_data = []
                
                for candle in data['candles']:
                    if candle['complete']:
                        df_data.append({
                            'time': pd.to_datetime(candle['time']),
                            'open': float(candle['mid']['o']),
                            'high': float(candle['mid']['h']),
                            'low': float(candle['mid']['l']),
                            'close': float(candle['mid']['c']),
                            'volume': int(candle.get('volume', 1))
                        })
                
                df = pd.DataFrame(df_data)
                df = df.sort_values('time').reset_index(drop=True)
                
                print(f"✅ Downloaded {len(df)} candles ({df['time'].min()} to {df['time'].max()})")
                return df
                
            else:
                print(f"❌ API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None
    
    def create_pattern_vector(self, candle_window):
        """Create 150-element pattern vector from 30 candles (Carson's method)"""
        if len(candle_window) != self.window_size:
            return None
            
        base_price = candle_window['close'].iloc[0]
        
        # Normalize prices to base
        opens = (candle_window['open'] / base_price).values
        highs = (candle_window['high'] / base_price).values
        lows = (candle_window['low'] / base_price).values
        closes = (candle_window['close'] / base_price).values
        
        # Create comprehensive pattern vector
        price_changes = closes[1:] - closes[:-1]  # 29 elements
        ranges = highs - lows  # 30 elements
        
        # Combine all features
        pattern_features = np.concatenate([
            opens,          # 30 features
            highs,          # 30 features  
            lows,           # 30 features
            closes,         # 30 features
            price_changes,  # 29 features
            ranges          # 30 features
        ])
        
        # Take first 149 elements for consistency
        return pattern_features[:149]
    
    def calculate_outcome(self, start_idx, market_data, lookahead_candles=6):
        """Calculate outcome for pattern (price change over next 6 candles = 30 minutes)"""
        if start_idx + self.window_size + lookahead_candles >= len(market_data):
            return None
        
        entry_price = market_data.iloc[start_idx + self.window_size]['close']
        exit_price = market_data.iloc[start_idx + self.window_size + lookahead_candles]['close']
        
        # Calculate percentage return
        return (exit_price - entry_price) / entry_price
    
    def extract_patterns_and_outcomes(self, market_data):
        """Extract all possible patterns and their outcomes from market data"""
        print("🔍 Extracting patterns and outcomes...")
        
        patterns = []
        outcomes = []
        
        # Extract patterns with sufficient lookahead
        max_start_idx = len(market_data) - self.window_size - 6
        
        for i in range(0, max_start_idx, 5):  # Sample every 5th candle for variety
            # Get pattern window
            pattern_window = market_data.iloc[i:i + self.window_size]
            
            if len(pattern_window) == self.window_size:
                # Create pattern vector
                pattern_vector = self.create_pattern_vector(pattern_window)
                
                if pattern_vector is not None:
                    # Calculate outcome
                    outcome = self.calculate_outcome(i, market_data)
                    
                    if outcome is not None:
                        patterns.append(pattern_vector)
                        outcomes.append(outcome)
        
        print(f"✅ Extracted {len(patterns)} patterns with outcomes")
        return np.array(patterns), np.array(outcomes)
    
    def train_clustering_model(self, patterns, outcomes):
        """Train the clustering model"""
        if len(patterns) < self.min_patterns:
            print(f"❌ Need at least {self.min_patterns} patterns, got {len(patterns)}")
            return False
        
        print(f"🧠 Training clustering model with {len(patterns)} patterns...")
        
        # Standardize patterns
        patterns_scaled = self.scaler.fit_transform(patterns)
        
        # Apply PCA for dimensionality reduction
        patterns_pca = self.pca.fit_transform(patterns_scaled)
        
        # Fit K-means clustering
        cluster_labels = self.kmeans.fit_predict(patterns_pca)
        
        # Calculate cluster statistics
        self.calculate_cluster_stats(cluster_labels, outcomes)
        
        print("✅ Clustering model training complete!")
        return True
    
    def calculate_cluster_stats(self, cluster_labels, outcomes):
        """Calculate predictive statistics for each cluster"""
        print("📊 Calculating cluster statistics...")
        
        self.cluster_stats = {}
        
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            cluster_outcomes = outcomes[mask]
            
            if len(cluster_outcomes) == 0:
                continue
            
            # Calculate directional probabilities
            strong_up_prob = (cluster_outcomes > 0.001).mean()    # >10 pips
            weak_up_prob = ((cluster_outcomes > 0.0005) & (cluster_outcomes <= 0.001)).mean()
            neutral_prob = ((cluster_outcomes >= -0.0005) & (cluster_outcomes <= 0.0005)).mean()
            weak_down_prob = ((cluster_outcomes < -0.0005) & (cluster_outcomes >= -0.001)).mean()
            strong_down_prob = (cluster_outcomes < -0.001).mean()  # <-10 pips
            
            # Overall directional bias
            up_prob = strong_up_prob + weak_up_prob
            down_prob = strong_down_prob + weak_down_prob
            
            # Expected return and risk metrics
            expected_return = cluster_outcomes.mean()
            volatility = cluster_outcomes.std()
            sharpe_like = expected_return / volatility if volatility > 0 else 0
            
            # Determine primary direction and confidence
            if up_prob > down_prob and up_prob > 0.55:
                direction = "BUY"
                confidence = up_prob
                edge = expected_return if expected_return > 0 else 0
            elif down_prob > up_prob and down_prob > 0.55:
                direction = "SELL"
                confidence = down_prob
                edge = abs(expected_return) if expected_return < 0 else 0
            else:
                direction = "HOLD"
                confidence = neutral_prob
                edge = 0
            
            self.cluster_stats[cluster_id] = {
                'size': mask.sum(),
                'direction': direction,
                'confidence': confidence,
                'edge': edge,
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_like': sharpe_like,
                'up_prob': up_prob,
                'down_prob': down_prob,
                'neutral_prob': neutral_prob,
                'strong_up_prob': strong_up_prob,
                'strong_down_prob': strong_down_prob
            }
            
            print(f"   Cluster {cluster_id}: {mask.sum():4d} patterns, "
                  f"{direction:4s}, conf={confidence:.2%}, edge={edge:.4f}")
    
    def save_model(self):
        """Save trained model components"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'kmeans': self.kmeans,
            'cluster_stats': self.cluster_stats,
            'config': {
                'window_size': self.window_size,
                'n_clusters': self.n_clusters,
                'n_pca_components': self.n_pca_components
            },
            'training_timestamp': timestamp
        }
        
        filename = f'/Users/jonspinogatti/Desktop/spin36TB/clustering_model_{timestamp}.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save as the default model
        default_filename = '/Users/jonspinogatti/Desktop/spin36TB/clustering_model.pkl'
        with open(default_filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to:")
        print(f"   {filename}")
        print(f"   {default_filename} (default)")
        
        return filename
    
    def validate_model(self, patterns, outcomes):
        """Quick validation of trained model"""
        print("🔍 Validating model performance...")
        
        # Transform test patterns
        patterns_scaled = self.scaler.transform(patterns)
        patterns_pca = self.pca.transform(patterns_scaled)
        predicted_clusters = self.kmeans.predict(patterns_pca)
        
        # Calculate accuracy metrics
        correct_predictions = 0
        total_predictions = 0
        
        for i, (cluster_id, actual_outcome) in enumerate(zip(predicted_clusters, outcomes)):
            if cluster_id in self.cluster_stats:
                stats = self.cluster_stats[cluster_id]
                predicted_direction = stats['direction']
                
                if predicted_direction == "BUY" and actual_outcome > 0:
                    correct_predictions += 1
                elif predicted_direction == "SELL" and actual_outcome < 0:
                    correct_predictions += 1
                elif predicted_direction == "HOLD" and abs(actual_outcome) <= 0.0005:
                    correct_predictions += 1
                
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"✅ Model accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
        
        return accuracy
    
    def run_training(self, days_back=60):
        """Complete training pipeline"""
        print("🚀 Starting clustering model training...")
        
        # Download data
        market_data = self.download_training_data(days_back)
        if market_data is None:
            return False
        
        # Extract patterns and outcomes
        patterns, outcomes = self.extract_patterns_and_outcomes(market_data)
        if len(patterns) == 0:
            return False
        
        # Split for training and validation
        split_idx = int(len(patterns) * 0.8)
        train_patterns = patterns[:split_idx]
        train_outcomes = outcomes[:split_idx]
        val_patterns = patterns[split_idx:]
        val_outcomes = outcomes[split_idx:]
        
        # Train model
        success = self.train_clustering_model(train_patterns, train_outcomes)
        if not success:
            return False
        
        # Validate model
        accuracy = self.validate_model(val_patterns, val_outcomes)
        
        # Save model
        model_file = self.save_model()
        
        print(f"\n🎉 CLUSTERING MODEL TRAINING COMPLETE!")
        print(f"=" * 50)
        print(f"📊 Training patterns: {len(train_patterns)}")
        print(f"🔍 Validation patterns: {len(val_patterns)}")
        print(f"✅ Model accuracy: {accuracy:.2%}")
        print(f"💾 Model saved: {model_file}")
        print(f"🚀 Ready for high-frequency trading!")
        
        return True

if __name__ == "__main__":
    print("🧠 Initializing Clustering Model Trainer...")
    
    trainer = ClusteringModelTrainer()
    
    try:
        # Train the model
        success = trainer.run_training(days_back=90)  # Use 90 days of data
        
        if success:
            print("\n✅ Training successful - Model ready for hybrid trading!")
        else:
            print("\n❌ Training failed - Check logs for details")
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()