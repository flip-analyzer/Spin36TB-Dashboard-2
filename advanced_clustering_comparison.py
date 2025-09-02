#!/usr/bin/env python3
"""
Advanced Clustering Comparison for Financial Time Series
Tests multiple clustering algorithms on 30-candle (150-minute) windows
"""

import numpy as np
import pandas as pd
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
    AffinityPropagation, OPTICS
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import dtw
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings('ignore')

class AdvancedClusteringComparison:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.results = {}
        print(f"🧪 ADVANCED CLUSTERING COMPARISON")
        print(f"==================================================")
        print(f"📊 Window size: {window_size} candles (150 minutes)")
        print(f"🔬 Testing 8 clustering algorithms")
        
    def load_market_data(self):
        """Load market data and create time series windows"""
        try:
            # Try comprehensive data first
            df = pd.read_csv('/Users/jonspinogatti/Desktop/spin36TB/comprehensive_eurusd_data.csv')
            print(f"✅ Loaded comprehensive market data: {len(df)} candles")
            
            # Ensure datetime index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            
            return df
        except Exception as e:
            try:
                # Fallback to oanda data
                df = pd.read_csv('/Users/jonspinogatti/Desktop/spin36TB/oanda_eurusd_data.csv')
                print(f"✅ Loaded OANDA market data: {len(df)} candles")
                
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                elif 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                
                return df
            except Exception as e2:
                print(f"❌ Could not load market data: {e2}")
                return None
    
    def create_time_series_windows(self, data):
        """Create overlapping 30-candle windows for clustering"""
        print(f"🔧 Creating {self.window_size}-candle windows...")
        
        # Use price movement patterns (OHLC normalized)
        features = ['open', 'high', 'low', 'close', 'volume']
        available_features = [f for f in features if f in data.columns]
        
        if not available_features:
            print("❌ No OHLC data found")
            return None, None
            
        print(f"📈 Using features: {available_features}")
        
        windows = []
        labels = []
        
        for i in range(len(data) - self.window_size - 5):  # -5 for forward returns
            window_data = data[available_features].iloc[i:i+self.window_size].values
            
            # Normalize each window (relative to first candle)
            if len(window_data) == self.window_size:
                normalized_window = window_data / window_data[0] - 1
                windows.append(normalized_window.flatten())
                
                # Calculate forward return (next 6 periods = 30 min)
                future_price = data['close'].iloc[i + self.window_size + 5]
                current_price = data['close'].iloc[i + self.window_size]
                forward_return = (future_price - current_price) / current_price
                labels.append(forward_return)
        
        windows = np.array(windows)
        labels = np.array(labels)
        
        print(f"✅ Created {len(windows)} windows")
        print(f"   Shape: {windows.shape}")
        print(f"   Features per window: {windows.shape[1]}")
        
        return windows, labels
    
    def test_clustering_algorithms(self, X, y_true):
        """Test multiple clustering algorithms"""
        print(f"\n🚀 TESTING CLUSTERING ALGORITHMS")
        print(f"==================================================")
        
        # Standardize features for traditional algorithms
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Prepare time series data for tslearn
        X_ts = X.reshape(X.shape[0], self.window_size, -1)
        
        algorithms = {
            '1. K-Means (Current)': KMeans(n_clusters=8, random_state=42),
            '2. Gaussian Mixture': GaussianMixture(n_components=8, random_state=42),
            '3. DBSCAN': DBSCAN(eps=0.3, min_samples=5),
            '4. OPTICS': OPTICS(min_samples=5),
            '5. Agglomerative': AgglomerativeClustering(n_clusters=8, linkage='ward'),
            '6. Spectral': SpectralClustering(n_clusters=8, random_state=42),
            '7. Affinity Propagation': AffinityPropagation(random_state=42),
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            print(f"\n🔬 Testing {name}...")
            try:
                if hasattr(algorithm, 'fit_predict'):
                    cluster_labels = algorithm.fit_predict(X_scaled)
                else:
                    cluster_labels = algorithm.fit(X_scaled).predict(X_scaled)
                
                # Handle noise points in DBSCAN/OPTICS
                if -1 in cluster_labels:
                    print(f"   Noise points detected: {np.sum(cluster_labels == -1)}")
                
                n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
                
                if n_clusters < 2:
                    print(f"   ❌ Failed: Only {n_clusters} cluster(s) found")
                    continue
                
                # Calculate metrics
                valid_mask = cluster_labels != -1
                X_valid = X_scaled[valid_mask]
                labels_valid = cluster_labels[valid_mask]
                
                if len(np.unique(labels_valid)) < 2:
                    print(f"   ❌ Failed: Insufficient clusters after noise removal")
                    continue
                
                silhouette = silhouette_score(X_valid, labels_valid)
                calinski = calinski_harabasz_score(X_valid, labels_valid)
                davies = davies_bouldin_score(X_valid, labels_valid)
                
                # Trading performance analysis
                trading_performance = self.analyze_trading_performance(
                    cluster_labels, y_true, valid_mask
                )
                
                results[name] = {
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies,
                    'noise_points': np.sum(cluster_labels == -1),
                    'trading_performance': trading_performance,
                    'labels': cluster_labels
                }
                
                print(f"   ✅ Clusters: {n_clusters}")
                print(f"   📊 Silhouette: {silhouette:.4f}")
                print(f"   💰 Avg Return: {trading_performance['avg_return']:.4f}")
                print(f"   📈 Win Rate: {trading_performance['win_rate']:.1%}")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
        
        # Test Time Series Specific Methods
        print(f"\n🕒 Testing Time Series Specific Methods...")
        
        # K-Shape clustering
        try:
            print(f"🔬 Testing K-Shape (DTW-based)...")
            ts_scaler = TimeSeriesScalerMeanVariance()
            X_ts_scaled = ts_scaler.fit_transform(X_ts)
            
            kshape = KShape(n_clusters=8, random_state=42)
            kshape_labels = kshape.fit_predict(X_ts_scaled)
            
            # Calculate metrics
            silhouette = silhouette_score(X_scaled, kshape_labels)
            calinski = calinski_harabasz_score(X_scaled, kshape_labels)
            davies = davies_bouldin_score(X_scaled, kshape_labels)
            
            trading_performance = self.analyze_trading_performance(
                kshape_labels, y_true, np.ones(len(kshape_labels), dtype=bool)
            )
            
            results['8. K-Shape (Time Series)'] = {
                'n_clusters': len(np.unique(kshape_labels)),
                'silhouette_score': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies,
                'noise_points': 0,
                'trading_performance': trading_performance,
                'labels': kshape_labels
            }
            
            print(f"   ✅ Clusters: {len(np.unique(kshape_labels))}")
            print(f"   📊 Silhouette: {silhouette:.4f}")
            print(f"   💰 Avg Return: {trading_performance['avg_return']:.4f}")
            print(f"   📈 Win Rate: {trading_performance['win_rate']:.1%}")
            
        except Exception as e:
            print(f"   ❌ K-Shape failed: {str(e)}")
        
        return results
    
    def analyze_trading_performance(self, cluster_labels, forward_returns, valid_mask):
        """Analyze trading performance for each clustering result"""
        valid_labels = cluster_labels[valid_mask]
        valid_returns = forward_returns[valid_mask]
        
        cluster_stats = {}
        for cluster_id in np.unique(valid_labels):
            if cluster_id == -1:  # Skip noise
                continue
                
            mask = valid_labels == cluster_id
            cluster_returns = valid_returns[mask]
            
            if len(cluster_returns) > 5:
                cluster_stats[cluster_id] = {
                    'size': len(cluster_returns),
                    'mean_return': np.mean(cluster_returns),
                    'std_return': np.std(cluster_returns),
                    'win_rate': np.mean(cluster_returns > 0),
                    'max_return': np.max(cluster_returns),
                    'min_return': np.min(cluster_returns),
                }
        
        # Overall performance
        if cluster_stats:
            all_returns = [stats['mean_return'] for stats in cluster_stats.values()]
            all_win_rates = [stats['win_rate'] for stats in cluster_stats.values()]
            sizes = [stats['size'] for stats in cluster_stats.values()]
            
            # Weighted averages
            total_samples = sum(sizes)
            weighted_return = sum(stats['mean_return'] * stats['size'] 
                                for stats in cluster_stats.values()) / total_samples
            weighted_win_rate = sum(stats['win_rate'] * stats['size'] 
                                  for stats in cluster_stats.values()) / total_samples
            
            return {
                'avg_return': weighted_return,
                'win_rate': weighted_win_rate,
                'return_spread': max(all_returns) - min(all_returns),
                'win_rate_spread': max(all_win_rates) - min(all_win_rates),
                'cluster_stats': cluster_stats,
                'n_viable_clusters': len(cluster_stats)
            }
        else:
            return {
                'avg_return': 0,
                'win_rate': 0,
                'return_spread': 0,
                'win_rate_spread': 0,
                'cluster_stats': {},
                'n_viable_clusters': 0
            }
    
    def create_comparison_report(self, results):
        """Create comprehensive comparison report"""
        print(f"\n📊 CLUSTERING ALGORITHM COMPARISON")
        print(f"==================================================")
        
        if not results:
            print("❌ No successful clustering results to compare")
            return
        
        # Sort by trading performance (combination of return and win rate)
        def score_algorithm(result):
            perf = result['trading_performance']
            # Combined score: weighted return * win_rate + return_spread * 0.5
            return (perf['avg_return'] * perf['win_rate'] + 
                   perf['return_spread'] * 0.5)
        
        sorted_results = sorted(results.items(), key=lambda x: score_algorithm(x[1]), reverse=True)
        
        print(f"📈 RANKING BY TRADING PERFORMANCE:")
        print(f"{'Rank':<4} {'Algorithm':<25} {'Clusters':<8} {'Silhouette':<11} {'Avg Return':<11} {'Win Rate':<10} {'R Spread':<10} {'Score':<8}")
        print(f"{'-'*4} {'-'*25} {'-'*8} {'-'*11} {'-'*11} {'-'*10} {'-'*10} {'-'*8}")
        
        for i, (name, result) in enumerate(sorted_results, 1):
            perf = result['trading_performance']
            score = score_algorithm(result)
            
            print(f"{i:<4} {name:<25} {result['n_clusters']:<8} {result['silhouette_score']:<11.4f} "
                  f"{perf['avg_return']:<11.6f} {perf['win_rate']:<10.1%} "
                  f"{perf['return_spread']:<10.4f} {score:<8.4f}")
        
        # Best algorithm analysis
        best_name, best_result = sorted_results[0]
        print(f"\n🏆 BEST ALGORITHM: {best_name}")
        print(f"==================================================")
        print(f"📊 Clusters: {best_result['n_clusters']}")
        print(f"📈 Silhouette Score: {best_result['silhouette_score']:.4f}")
        print(f"💰 Average Return: {best_result['trading_performance']['avg_return']:.6f}")
        print(f"📊 Win Rate: {best_result['trading_performance']['win_rate']:.1%}")
        print(f"📏 Return Spread: {best_result['trading_performance']['return_spread']:.4f}")
        print(f"🎯 Viable Clusters: {best_result['trading_performance']['n_viable_clusters']}")
        
        if best_result['noise_points'] > 0:
            print(f"🔇 Noise Points: {best_result['noise_points']}")
        
        # Detailed cluster analysis for best algorithm
        cluster_stats = best_result['trading_performance']['cluster_stats']
        if cluster_stats:
            print(f"\n📊 CLUSTER BREAKDOWN:")
            print(f"{'Cluster':<8} {'Size':<6} {'Mean Return':<12} {'Win Rate':<10} {'Std Return':<12}")
            print(f"{'-'*8} {'-'*6} {'-'*12} {'-'*10} {'-'*12}")
            
            for cluster_id, stats in sorted(cluster_stats.items(), 
                                          key=lambda x: x[1]['mean_return'], reverse=True):
                print(f"{cluster_id:<8} {stats['size']:<6} {stats['mean_return']:<12.6f} "
                      f"{stats['win_rate']:<10.1%} {stats['std_return']:<12.6f}")
        
        return best_name, best_result
    
    def run_comparison(self):
        """Run the complete clustering comparison"""
        # Load data
        market_data = self.load_market_data()
        if market_data is None:
            return None
        
        # Create time series windows
        X, y = self.create_time_series_windows(market_data)
        if X is None:
            return None
        
        # Test all algorithms
        results = self.test_clustering_algorithms(X, y)
        
        # Create comparison report
        best_algorithm, best_result = self.create_comparison_report(results)
        
        self.results = results
        return best_algorithm, best_result

def test_advanced_clustering():
    """Test the advanced clustering comparison"""
    comparison = AdvancedClusteringComparison(window_size=30)
    best_algo, best_result = comparison.run_comparison()
    
    if best_algo:
        print(f"\n✅ CLUSTERING COMPARISON COMPLETE!")
        print(f"🏆 Winner: {best_algo}")
        print(f"💡 Consider implementing this algorithm for better performance")
    else:
        print(f"\n❌ Clustering comparison failed")

if __name__ == "__main__":
    test_advanced_clustering()