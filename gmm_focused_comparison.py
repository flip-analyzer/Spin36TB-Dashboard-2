#!/usr/bin/env python3
"""
Focused GMM vs Current Methods Comparison
Efficient test of GMM variants vs K-Means and Agglomerative on large dataset
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def load_large_dataset():
    """Load substantial dataset efficiently"""
    try:
        df = pd.read_csv('/Users/jonspinogatti/Desktop/spin36TB/comprehensive_eurusd_data.csv')
        print(f"✅ Loaded {len(df):,} candles")
        
        # Use substantial portion for robust testing (50K+ samples)
        if len(df) > 10000:
            # Use recent data for relevance
            df = df.tail(15000).copy()  # 15K candles -> ~14.5K windows
            print(f"📊 Using recent {len(df):,} candles for robust testing")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

def create_efficient_features(data, window_size=30):
    """Create efficient feature vectors from 30-candle windows"""
    print(f"🔧 Creating feature vectors from {window_size}-candle windows...")
    
    features = []
    returns = []
    
    max_windows = len(data) - window_size - 6
    print(f"📊 Processing {max_windows:,} windows...")
    
    for i in range(0, max_windows, 2):  # Every 2nd window for efficiency
        try:
            # Extract window
            window = data[['open', 'high', 'low', 'close', 'volume']].iloc[i:i+window_size]
            
            if len(window) == window_size and window['close'].iloc[0] > 0:
                # Normalize prices
                first_close = window['close'].iloc[0]
                normalized = ((window / first_close) - 1) * 100
                
                # Create efficient feature vector
                closes = normalized['close'].values
                highs = normalized['high'].values
                lows = normalized['low'].values
                volumes = normalized['volume'].values
                
                # Key features for clustering
                feature_vector = np.concatenate([
                    # Price pattern (sampled)
                    closes[::5],  # Every 5th candle (6 values)
                    
                    # Statistical features
                    [np.mean(closes), np.std(closes), np.min(closes), np.max(closes)],
                    [np.mean(highs-lows), np.std(highs-lows)],
                    
                    # Trend features
                    [closes[-1] - closes[0],  # Total change
                     np.mean(np.diff(closes)),  # Avg change
                     closes[-5:].mean() - closes[:5].mean()],  # Momentum
                    
                    # Volatility features
                    [np.std(np.diff(closes)), np.max(highs) - np.min(lows)],
                    
                    # Volume features (if available)
                    [np.mean(volumes), np.std(volumes)] if np.any(volumes) else [0, 0]
                ])
                
                # Handle NaN/inf
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                
                if len(feature_vector) > 0 and not np.any(np.isinf(feature_vector)):
                    features.append(feature_vector)
                    
                    # Forward return
                    current_price = data['close'].iloc[i + window_size]
                    future_price = data['close'].iloc[i + window_size + 6]
                    forward_return = (future_price - current_price) / current_price
                    returns.append(forward_return)
                    
        except:
            continue
    
    if features:
        X = np.array(features)
        y = np.array(returns)
        print(f"✅ Created {len(features):,} feature vectors with {X.shape[1]} features each")
        return X, y
    else:
        print("❌ No features created")
        return None, None

def test_algorithm_efficient(name, algorithm, X_scaled, y):
    """Test algorithm efficiently with key metrics"""
    print(f"\n🔬 Testing {name}")
    print("-" * 40)
    
    try:
        # Fit algorithm
        if hasattr(algorithm, 'fit_predict'):
            labels = algorithm.fit_predict(X_scaled)
        else:
            labels = algorithm.fit(X_scaled).predict(X_scaled)
        
        # Handle noise
        valid_mask = labels != -1
        if valid_mask.sum() < 100:
            print(f"❌ Failed: Too few samples")
            return None
        
        X_valid = X_scaled[valid_mask]
        labels_valid = labels[valid_mask]
        y_valid = y[valid_mask]
        
        n_clusters = len(np.unique(labels_valid))
        if n_clusters < 2:
            print(f"❌ Failed: {n_clusters} clusters")
            return None
        
        # Key metrics
        silhouette = silhouette_score(X_valid, labels_valid)
        
        # Trading analysis
        cluster_performance = {}
        for cid in np.unique(labels_valid):
            mask = labels_valid == cid
            cluster_returns = y_valid[mask]
            
            if len(cluster_returns) >= 20:
                cluster_performance[cid] = {
                    'size': len(cluster_returns),
                    'win_rate': np.mean(cluster_returns > 0),
                    'mean_return': np.mean(cluster_returns),
                    'std_return': np.std(cluster_returns),
                    'sharpe': np.mean(cluster_returns) / np.std(cluster_returns) if np.std(cluster_returns) > 0 else 0
                }
        
        if not cluster_performance:
            print(f"❌ No viable clusters")
            return None
        
        # Best cluster
        best_cid = max(cluster_performance.keys(), key=lambda x: cluster_performance[x]['sharpe'])
        best_stats = cluster_performance[best_cid]
        
        # Performance metrics
        win_rates = [s['win_rate'] for s in cluster_performance.values()]
        returns_list = [s['mean_return'] for s in cluster_performance.values()]
        sharpes = [s['sharpe'] for s in cluster_performance.values()]
        
        win_rate_spread = max(win_rates) - min(win_rates)
        return_spread = max(returns_list) - min(returns_list)
        avg_sharpe = np.mean(sharpes)
        
        # Composite score
        score = (silhouette * 0.3 + win_rate_spread * 0.35 + 
                return_spread * 1000 * 0.2 + avg_sharpe * 0.15)
        
        result = {
            'name': name,
            'n_clusters': n_clusters,
            'samples': len(labels_valid),
            'silhouette': silhouette,
            'best_win_rate': best_stats['win_rate'],
            'best_return_pips': best_stats['mean_return'] * 10000,
            'best_sharpe': best_stats['sharpe'],
            'win_rate_spread': win_rate_spread,
            'return_spread': return_spread,
            'avg_sharpe': avg_sharpe,
            'performance_score': score,
            'viable_clusters': len(cluster_performance)
        }
        
        print(f"✅ Results:")
        print(f"   Clusters: {n_clusters} ({len(cluster_performance)} viable)")
        print(f"   Samples: {len(labels_valid):,}")
        print(f"   Silhouette: {silhouette:.4f}")
        print(f"   Best cluster: {best_stats['win_rate']:.1%} win rate, {best_stats['mean_return']*10000:.1f} pips")
        print(f"   Win rate spread: {win_rate_spread:.1%}")
        print(f"   Performance score: {score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

def main():
    """Run focused GMM comparison"""
    print("🎯 FOCUSED GMM vs CURRENT METHODS COMPARISON")
    print("=" * 60)
    print("📊 Testing GMM variants vs K-Means & Agglomerative")
    print("🎯 Focus: Large dataset, efficient processing, robust metrics")
    
    # Load data
    data = load_large_dataset()
    if data is None:
        return
    
    # Create features
    X, y = create_efficient_features(data)
    if X is None:
        return
    
    # Standardize
    print(f"\n📐 Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✅ Standardized: {X_scaled.shape}")
    
    # Test algorithms
    print(f"\n🧪 TESTING ALGORITHMS ON {len(X):,} SAMPLES")
    print("=" * 60)
    
    algorithms = [
        ('K-Means (Current)', KMeans(n_clusters=8, random_state=42, n_init=10)),
        ('Agglomerative (Ward)', AgglomerativeClustering(n_clusters=8, linkage='ward')),
        ('GMM Full Covariance', GaussianMixture(n_components=8, covariance_type='full', random_state=42)),
        ('GMM Diagonal Cov', GaussianMixture(n_components=8, covariance_type='diag', random_state=42)),
        ('GMM Tied Covariance', GaussianMixture(n_components=8, covariance_type='tied', random_state=42)),
        ('GMM Spherical', GaussianMixture(n_components=8, covariance_type='spherical', random_state=42)),
    ]
    
    results = []
    for name, algorithm in algorithms:
        result = test_algorithm_efficient(name, algorithm, X_scaled, y)
        if result:
            results.append(result)
    
    # Final comparison
    if results:
        print(f"\n🏆 FINAL COMPARISON RESULTS")
        print("=" * 80)
        
        results.sort(key=lambda x: x['performance_score'], reverse=True)
        
        print(f"{'Rank':<4} {'Algorithm':<22} {'Score':<8} {'Silhouette':<11} {'Best Win%':<10} {'WR Spread':<10} {'Sharpe':<8}")
        print("-" * 80)
        
        for i, r in enumerate(results, 1):
            print(f"{i:<4} {r['name']:<22} {r['performance_score']:<8.3f} {r['silhouette']:<11.4f} "
                  f"{r['best_win_rate']:<10.1%} {r['win_rate_spread']:<10.1%} {r['best_sharpe']:<8.2f}")
        
        # Winner details
        winner = results[0]
        print(f"\n🎉 WINNER: {winner['name'].upper()}")
        print(f"=" * 50)
        print(f"🏆 Performance Score: {winner['performance_score']:.3f}")
        print(f"📊 Silhouette: {winner['silhouette']:.4f}")
        print(f"🎯 Best Cluster: {winner['best_win_rate']:.1%} win rate, {winner['best_return_pips']:.1f} pips")
        print(f"📏 Win Rate Spread: {winner['win_rate_spread']:.1%}")
        print(f"📈 Avg Sharpe: {winner['avg_sharpe']:.2f}")
        print(f"👥 Samples: {winner['samples']:,}")
        print(f"🔢 Viable Clusters: {winner['viable_clusters']}/{winner['n_clusters']}")
        
        # Compare to K-Means
        kmeans = next((r for r in results if 'K-Means' in r['name']), None)
        if kmeans and winner != kmeans:
            improvement = (winner['performance_score'] - kmeans['performance_score']) / kmeans['performance_score'] * 100
            print(f"\n📈 IMPROVEMENT OVER K-MEANS: +{improvement:.1f}%")
            
        # GMM analysis
        gmm_results = [r for r in results if 'GMM' in r['name']]
        if gmm_results:
            best_gmm = max(gmm_results, key=lambda x: x['performance_score'])
            print(f"\n🧮 BEST GMM VARIANT: {best_gmm['name']}")
            print(f"   Score: {best_gmm['performance_score']:.3f}")
            if kmeans:
                gmm_improvement = (best_gmm['performance_score'] - kmeans['performance_score']) / kmeans['performance_score'] * 100
                print(f"   Improvement over K-Means: +{gmm_improvement:.1f}%")
        
        return winner
    else:
        print("❌ No successful results")
        return None

if __name__ == "__main__":
    winner = main()
    if winner:
        print(f"\n✅ RECOMMENDATION: Use {winner['name']} for optimal clustering")
        print(f"💡 Based on comprehensive testing with {winner['samples']:,} samples")
    else:
        print(f"\n❌ Testing failed - consider keeping current system")