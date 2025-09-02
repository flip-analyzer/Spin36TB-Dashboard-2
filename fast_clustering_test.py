#!/usr/bin/env python3
"""
Fast Clustering Algorithm Comparison
Quick test of the most promising clustering methods for financial time series
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare market data"""
    try:
        df = pd.read_csv('/Users/jonspinogatti/Desktop/spin36TB/comprehensive_eurusd_data.csv')
        print(f"✅ Loaded {len(df)} candles")
        
        # Use recent data only for speed (last 2000 candles)
        df = df.tail(2000).copy()
        print(f"📊 Using recent {len(df)} candles for testing")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None

def create_windows(data, window_size=30):
    """Create 30-candle windows with price movement vectors"""
    print(f"🔧 Creating {window_size}-candle windows...")
    
    features = ['open', 'high', 'low', 'close', 'volume']
    available = [f for f in features if f in data.columns]
    
    if not available:
        print("❌ No OHLC data found")
        return None, None
    
    windows = []
    forward_returns = []
    
    for i in range(len(data) - window_size - 6):
        # Get window data
        window = data[available].iloc[i:i+window_size].values
        
        # Normalize to percentage changes from first candle
        if len(window) == window_size and window[0].min() > 0:
            normalized = (window / window[0] - 1) * 100  # Convert to percentage
            windows.append(normalized.flatten())
            
            # Calculate 30-minute forward return
            current_price = data['close'].iloc[i + window_size]
            future_price = data['close'].iloc[i + window_size + 6]  # 6 periods = 30 min
            forward_return = (future_price - current_price) / current_price
            forward_returns.append(forward_return)
    
    print(f"✅ Created {len(windows)} windows, {len(windows[0]) if windows else 0} features each")
    return np.array(windows), np.array(forward_returns)

def test_algorithm(name, algorithm, X, y, X_scaled):
    """Test a single clustering algorithm"""
    print(f"\n🔬 Testing {name}...")
    try:
        if hasattr(algorithm, 'fit_predict'):
            labels = algorithm.fit_predict(X_scaled)
        else:
            labels = algorithm.fit(X_scaled).predict(X_scaled)
        
        # Handle noise points
        valid_mask = labels != -1
        if valid_mask.sum() < 2:
            print(f"   ❌ Failed: Too few valid clusters")
            return None
        
        n_clusters = len(np.unique(labels[valid_mask]))
        if n_clusters < 2:
            print(f"   ❌ Failed: Only {n_clusters} cluster(s)")
            return None
        
        # Calculate metrics
        X_valid = X_scaled[valid_mask]
        labels_valid = labels[valid_mask]
        y_valid = y[valid_mask]
        
        silhouette = silhouette_score(X_valid, labels_valid)
        
        # Trading performance
        cluster_returns = []
        cluster_win_rates = []
        
        for cluster_id in np.unique(labels_valid):
            mask = labels_valid == cluster_id
            cluster_y = y_valid[mask]
            
            if len(cluster_y) >= 10:  # Minimum samples for meaningful stats
                mean_return = np.mean(cluster_y)
                win_rate = np.mean(cluster_y > 0)
                cluster_returns.append(mean_return)
                cluster_win_rates.append(win_rate)
        
        if len(cluster_returns) < 2:
            print(f"   ❌ Failed: Not enough viable clusters")
            return None
        
        # Calculate performance metrics
        avg_return = np.mean(cluster_returns)
        avg_win_rate = np.mean(cluster_win_rates)
        return_spread = max(cluster_returns) - min(cluster_returns)
        win_rate_spread = max(cluster_win_rates) - min(cluster_win_rates)
        
        # Combined performance score
        performance_score = (avg_return * 1000 + avg_win_rate + return_spread * 500 + win_rate_spread * 2)
        
        result = {
            'name': name,
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'avg_return': avg_return,
            'avg_win_rate': avg_win_rate,
            'return_spread': return_spread,
            'win_rate_spread': win_rate_spread,
            'performance_score': performance_score,
            'noise_points': np.sum(labels == -1)
        }
        
        print(f"   ✅ Clusters: {n_clusters}, Silhouette: {silhouette:.4f}")
        print(f"   💰 Avg Return: {avg_return:.6f}, Win Rate: {avg_win_rate:.1%}")
        print(f"   📏 Spreads: Return {return_spread:.4f}, Win Rate {win_rate_spread:.1%}")
        print(f"   🎯 Score: {performance_score:.2f}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        return None

def main():
    print("🧪 FAST CLUSTERING COMPARISON")
    print("="*50)
    print("🎯 Testing 5 most promising algorithms on 30-candle windows")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Create windows
    X, y = create_windows(data, 30)
    if X is None:
        return
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n🚀 TESTING ALGORITHMS")
    print("="*50)
    
    # Test algorithms - focus on most promising ones
    algorithms = [
        ('1. K-Means (Current)', KMeans(n_clusters=8, random_state=42)),
        ('2. DBSCAN (Density)', DBSCAN(eps=0.5, min_samples=20)),
        ('3. Gaussian Mixture', GaussianMixture(n_components=8, random_state=42)),
        ('4. Agglomerative (Hierarchical)', AgglomerativeClustering(n_clusters=8, linkage='ward')),
        ('5. DBSCAN (Tight)', DBSCAN(eps=0.3, min_samples=10)),
    ]
    
    results = []
    for name, algorithm in algorithms:
        result = test_algorithm(name, algorithm, X, y, X_scaled)
        if result:
            results.append(result)
    
    # Rank results
    if results:
        print(f"\n📊 RESULTS RANKING")
        print("="*70)
        results.sort(key=lambda x: x['performance_score'], reverse=True)
        
        print(f"{'Rank':<4} {'Algorithm':<25} {'Clusters':<8} {'Silhouette':<11} {'Avg Return':<12} {'Win Rate':<10} {'Score':<8}")
        print("-" * 70)
        
        for i, r in enumerate(results, 1):
            print(f"{i:<4} {r['name']:<25} {r['n_clusters']:<8} {r['silhouette']:<11.4f} "
                  f"{r['avg_return']:<12.6f} {r['avg_win_rate']:<10.1%} {r['performance_score']:<8.1f}")
        
        # Best result details
        best = results[0]
        print(f"\n🏆 WINNER: {best['name']}")
        print("="*50)
        print(f"📊 Clusters: {best['n_clusters']}")
        print(f"📈 Silhouette: {best['silhouette']:.4f}")
        print(f"💰 Average Return: {best['avg_return']:.6f}")
        print(f"📊 Win Rate: {best['avg_win_rate']:.1%}")
        print(f"📏 Return Spread: {best['return_spread']:.4f}")
        print(f"📏 Win Rate Spread: {best['win_rate_spread']:.1%}")
        print(f"🎯 Performance Score: {best['performance_score']:.1f}")
        
        if best['noise_points'] > 0:
            print(f"🔇 Noise Points: {best['noise_points']}")
        
        # Comparison with current K-Means
        kmeans_result = next((r for r in results if 'K-Means' in r['name']), None)
        if kmeans_result and best != kmeans_result:
            improvement = ((best['performance_score'] - kmeans_result['performance_score']) / 
                          kmeans_result['performance_score'] * 100)
            print(f"\n📈 IMPROVEMENT OVER K-MEANS: {improvement:+.1f}%")
        
        return best['name']
    else:
        print("\n❌ No algorithms succeeded")
        return None

if __name__ == "__main__":
    winner = main()
    if winner:
        print(f"\n✅ RECOMMENDATION: Switch to {winner}")
        print("💡 This algorithm shows better performance for 30-candle patterns")
    else:
        print("\n⚠️  Consider keeping current K-Means approach")