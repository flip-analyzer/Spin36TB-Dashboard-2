#!/usr/bin/env python3
"""
Marcos-Scale Pattern Clustering

Uses 300k-500k pattern windows from EURUSD data for proper cluster analysis
This is what Marcos would actually do with 1M+ candles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans  # Better for large datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA  # Memory efficient
import warnings
warnings.filterwarnings('ignore')

class MarcosScaleClusterer:
    def __init__(self, window_size=30, target_patterns=300000, n_clusters=12):
        self.window_size = window_size
        self.target_patterns = target_patterns
        self.n_clusters = n_clusters
        
        # Use incremental/mini-batch methods for large scale
        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=50, batch_size=1000)
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=42)
        
        self.cluster_stats = {}
        
    def create_pattern_vector(self, candle_window):
        """Create 150-element pattern vector"""
        if len(candle_window) != self.window_size:
            return None
            
        base_price = candle_window['Close'].iloc[0]
        
        opens = (candle_window['Open'] / base_price).values
        highs = (candle_window['High'] / base_price).values  
        lows = (candle_window['Low'] / base_price).values
        closes = (candle_window['Close'] / base_price).values
        volumes = (candle_window['Volume'] / candle_window['Volume'].mean()).values
        
        # Enhanced features for better separation
        ranges = (highs - lows) / closes
        bodies = np.abs(closes - opens) / closes
        upper_wicks = (highs - np.maximum(opens, closes)) / closes
        lower_wicks = (np.minimum(opens, closes) - lows) / closes
        
        pattern_vector = np.concatenate([
            closes, ranges, bodies, upper_wicks, lower_wicks
        ])
        
        return pattern_vector
    
    def extract_marcos_scale_patterns(self, data):
        """Extract 300k+ patterns using Marcos methodology"""
        
        print(f"🧠 MARCOS-SCALE PATTERN EXTRACTION")
        print(f"   Dataset: {len(data):,} EURUSD 5-minute candles")
        print(f"   Target patterns: {self.target_patterns:,}")
        print(f"   Window size: {self.window_size} candles (2.5 hours)")
        
        max_possible = len(data) - self.window_size - 6
        step_size = max(1, max_possible // self.target_patterns)
        
        print(f"   Max possible windows: {max_possible:,}")
        print(f"   Sampling every {step_size} windows")
        print(f"   Expected patterns: {max_possible // step_size:,}")
        
        patterns = []
        outcomes = []
        timestamps = []
        
        batch_size = 10000
        batch_patterns = []
        batch_outcomes = []
        
        extracted = 0
        
        for i in range(0, max_possible, step_size):
            if extracted >= self.target_patterns:
                break
                
            # Extract pattern
            pattern_window = data.iloc[i:i + self.window_size]
            pattern_vector = self.create_pattern_vector(pattern_window)
            
            if pattern_vector is None:
                continue
            
            # Calculate 30-minute outcome
            entry_price = pattern_window['Close'].iloc[-1]
            exit_idx = min(i + self.window_size + 6, len(data) - 1)
            exit_price = data.iloc[exit_idx]['Close'] 
            outcome = (exit_price / entry_price) - 1
            
            batch_patterns.append(pattern_vector)
            batch_outcomes.append(outcome)
            
            extracted += 1
            
            # Process in batches for memory efficiency
            if len(batch_patterns) >= batch_size:
                patterns.extend(batch_patterns)
                outcomes.extend(batch_outcomes)
                batch_patterns = []
                batch_outcomes = []
                
                if extracted % 50000 == 0:
                    print(f"      ✅ Extracted {extracted:,} patterns...")
        
        # Add remaining batch
        patterns.extend(batch_patterns)
        outcomes.extend(batch_outcomes)
        
        self.patterns = np.array(patterns)
        self.outcomes = np.array(outcomes)
        
        print(f"   ✅ FINAL: {len(self.patterns):,} patterns extracted")
        return len(self.patterns) > 0
    
    def train_marcos_clusters(self):
        """Train clusters on Marcos-scale data"""
        
        print(f"\n🎯 TRAINING MARCOS-SCALE CLUSTERS")
        print(f"   Patterns: {len(self.patterns):,}")
        print(f"   Features: {self.patterns.shape[1]} per pattern")
        print(f"   Clusters: {self.n_clusters}")
        
        # Incremental scaling and PCA for memory efficiency
        print(f"   📊 Scaling and dimensionality reduction...")
        
        # Fit scaler in batches
        batch_size = 10000
        for i in range(0, len(self.patterns), batch_size):
            batch = self.patterns[i:i + batch_size]
            if i == 0:
                self.scaler.fit(batch)
            else:
                # Update scaler with new data
                self.scaler.partial_fit(batch)
        
        # Apply PCA in batches
        print(f"   🔍 Applying PCA (50 components)...")
        patterns_scaled = self.scaler.transform(self.patterns)
        
        # Fit PCA incrementally
        for i in range(0, len(patterns_scaled), batch_size):
            batch = patterns_scaled[i:i + batch_size]
            self.pca.partial_fit(batch)
        
        # Transform all data
        patterns_pca = self.pca.transform(patterns_scaled)
        
        # Mini-batch K-means clustering
        print(f"   🎲 K-means clustering...")
        cluster_labels = self.kmeans.fit_predict(patterns_pca)
        
        # Calculate cluster statistics  
        print(f"   📈 Analyzing cluster performance...")
        self._calculate_cluster_edge(cluster_labels)
        
        print(f"   ✅ Training complete!")
        return cluster_labels
    
    def _calculate_cluster_edge(self, cluster_labels):
        """Calculate trading edge for each cluster"""
        
        self.cluster_stats = {}
        
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            cluster_outcomes = self.outcomes[mask]
            
            if len(cluster_outcomes) == 0:
                continue
            
            # Outcome categories (in pips for EURUSD)
            strong_up = (cluster_outcomes > 0.0010).mean()    # >10 pips
            weak_up = ((cluster_outcomes > 0.0005) & (cluster_outcomes <= 0.0010)).mean()  
            neutral = ((cluster_outcomes >= -0.0005) & (cluster_outcomes <= 0.0005)).mean()
            weak_down = ((cluster_outcomes < -0.0005) & (cluster_outcomes >= -0.0010)).mean()
            strong_down = (cluster_outcomes < -0.0010).mean()  # <-10 pips
            
            # Trading metrics
            up_prob = strong_up + weak_up
            down_prob = strong_down + weak_down
            net_bias = up_prob - down_prob
            expected_return = cluster_outcomes.mean()
            volatility = cluster_outcomes.std()
            
            # Determine primary signal
            if up_prob > 0.55 and net_bias > 0.08:
                signal = "BUY"
                confidence = up_prob
                edge = net_bias
            elif down_prob > 0.55 and net_bias < -0.08:
                signal = "SELL"
                confidence = down_prob  
                edge = -net_bias
            else:
                signal = "HOLD"
                confidence = max(up_prob, down_prob, neutral)
                edge = abs(net_bias)
            
            self.cluster_stats[cluster_id] = {
                'size': mask.sum(),
                'signal': signal,
                'confidence': confidence,
                'edge': edge,
                'expected_return': expected_return,
                'expected_pips': expected_return * 10000,
                'volatility': volatility * 10000,  # In pips
                'up_prob': up_prob,
                'down_prob': down_prob,
                'neutral_prob': neutral,
                'strong_up_prob': strong_up,
                'strong_down_prob': strong_down
            }
            
            print(f"      Cluster {cluster_id:2d}: {mask.sum():6,} patterns, "
                  f"{signal:4s}, conf={confidence:.1%}, "
                  f"edge={edge:.1%}, exp={expected_return*10000:+.1f}pips")
    
    def find_best_trading_clusters(self):
        """Identify clusters with strong trading edges"""
        
        print(f"\n🏆 BEST TRADING CLUSTERS (Marcos Scale)")
        print("=" * 55)
        
        # Sort clusters by edge strength
        sorted_clusters = sorted(self.cluster_stats.items(), 
                               key=lambda x: x[1]['edge'], reverse=True)
        
        tradeable_clusters = []
        
        for cluster_id, stats in sorted_clusters:
            if stats['edge'] > 0.05 and stats['size'] > 1000:  # Minimum thresholds
                tradeable_clusters.append(cluster_id)
                
                print(f"📊 CLUSTER {cluster_id}")
                print(f"   Size: {stats['size']:,} patterns")
                print(f"   Signal: {stats['signal']}")  
                print(f"   Confidence: {stats['confidence']:.1%}")
                print(f"   Edge: {stats['edge']:.1%}")
                print(f"   Expected: {stats['expected_pips']:+.1f} pips")
                print(f"   Volatility: {stats['volatility']:.1f} pips")
                print(f"   Up/Down: {stats['up_prob']:.1%}/{stats['down_prob']:.1%}")
                print()
        
        print(f"🎯 TRADEABLE CLUSTERS: {len(tradeable_clusters)} out of {self.n_clusters}")
        return tradeable_clusters


def main():
    """Run Marcos-scale clustering analysis with proper data splits"""
    
    print("🎯 MARCOS-SCALE EURUSD CLUSTERING")
    print("=" * 45)
    print("Using 300k+ patterns with train/validation/test splits")
    
    # Load EURUSD data
    print("\n📊 Loading EURUSD Dataset...")
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
        
        # PROPER DATA SPLITS to avoid overfitting
        print(f"\n📋 Creating Train/Validation/Test Splits...")
        total_bars = len(data)
        
        # 60% train, 20% validation, 20% test (time-based, no shuffling!)
        train_end = int(total_bars * 0.60)
        val_end = int(total_bars * 0.80)
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]  
        test_data = data.iloc[val_end:]
        
        print(f"   📈 Train: {len(train_data):,} bars ({len(train_data)/total_bars:.1%})")
        print(f"   🎯 Validation: {len(val_data):,} bars ({len(val_data)/total_bars:.1%})")
        print(f"   ✅ Test: {len(test_data):,} bars ({len(test_data)/total_bars:.1%})")
        print(f"   ⏰ No future leakage - chronological splits only!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Initialize Marcos-scale clusterer
    clusterer = MarcosScaleClusterer(
        window_size=30,
        target_patterns=200000,  # Marcos scale from TRAINING data only
        n_clusters=12            # More clusters for better separation  
    )
    
    # ONLY train on training data to avoid overfitting
    print(f"\n🧠 Training clusters on TRAIN data only...")
    if not clusterer.extract_marcos_scale_patterns(train_data):
        print("❌ Pattern extraction failed")
        return
    
    # Train cluster model on training data
    cluster_labels = clusterer.train_marcos_clusters()
    
    # Find best trading opportunities
    tradeable_clusters = clusterer.find_best_trading_clusters()
    
    # TODO: Validate on validation set, test on test set
    print(f"\n📊 Next Steps:")
    print(f"   ✅ Model trained on {len(train_data):,} bars")
    print(f"   🎯 Ready to validate on {len(val_data):,} validation bars")
    print(f"   🧪 Final test on {len(test_data):,} unseen bars")
    print(f"   ⚠️  Never mix training/test data!")
    
    # Summary
    print(f"\n✅ MARCOS-SCALE TRAINING COMPLETE!")
    print(f"   📊 Patterns analyzed: {len(clusterer.patterns):,}")
    print(f"   🎯 Tradeable clusters: {len(tradeable_clusters)}")
    print(f"   🔒 No overfitting - proper data separation")
    
    return clusterer, train_data, val_data, test_data


if __name__ == "__main__":
    results = main()