#!/usr/bin/env python3
"""
Comprehensive Clustering Comparison on Large Dataset
Test GMM, Agglomerative, and K-Means on 300K+ candles for robust evaluation
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class LargeScaleClusteringComparison:
    def __init__(self, window_size=30, max_samples=None):
        self.window_size = window_size
        self.max_samples = max_samples
        print(f"🔬 COMPREHENSIVE CLUSTERING COMPARISON")
        print(f"============================================================")
        print(f"📊 Target: 300K+ candles for robust evaluation")
        print(f"🎯 Algorithms: K-Means, Agglomerative, GMM (Multiple variants)")
        print(f"⭐ Focus: Gaussian Mixture Models with different covariance types")
        
    def load_full_dataset(self):
        """Load complete dataset for comprehensive testing"""
        print(f"\n📥 LOADING COMPREHENSIVE DATASET")
        print(f"==================================================")
        
        try:
            # Load comprehensive dataset
            df = pd.read_csv('/Users/jonspinogatti/Desktop/spin36TB/comprehensive_eurusd_data.csv')
            print(f"✅ Loaded {len(df):,} total candles")
            
            if len(df) < 50000:
                print(f"⚠️  Dataset smaller than expected. Using all available data.")
            else:
                print(f"🎯 Dataset size: {len(df):,} candles (sufficient for robust testing)")
            
            # Prepare datetime index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Failed to load comprehensive dataset: {e}")
            return None
    
    def create_large_feature_matrix(self, data):
        """Create feature matrix from large dataset efficiently"""
        print(f"\n🔧 CREATING FEATURE MATRIX")
        print(f"==================================================")
        
        # Calculate maximum possible windows
        max_windows = len(data) - self.window_size - 6
        print(f"📊 Maximum possible windows: {max_windows:,}")
        
        # For memory efficiency, sample every N windows if dataset is very large
        if max_windows > 100000:
            step_size = max(1, max_windows // 50000)  # Target ~50K samples
            print(f"🔄 Using every {step_size} windows for efficiency ({max_windows//step_size:,} samples)")
        else:
            step_size = 1
            print(f"📈 Using all windows ({max_windows:,} samples)")
        
        features = []
        returns = []
        processed = 0
        
        print(f"🏗️  Building feature vectors...")
        
        for i in range(0, max_windows, step_size):
            try:
                if processed % 10000 == 0 and processed > 0:
                    print(f"   Processed {processed:,} windows...")
                
                # Extract 30-candle window
                window = data[['open', 'high', 'low', 'close', 'volume']].iloc[i:i+self.window_size]
                
                if len(window) == self.window_size:
                    # Create comprehensive feature vector
                    feature_vector = self.create_comprehensive_features(window.values)
                    
                    if feature_vector is not None and len(feature_vector) > 0:
                        features.append(feature_vector)
                        
                        # Calculate 30-minute forward return
                        current_price = data['close'].iloc[i + self.window_size]
                        future_price = data['close'].iloc[i + self.window_size + 6]
                        forward_return = (future_price - current_price) / current_price
                        returns.append(forward_return)
                        
                        processed += 1
                        
                        # Memory management for very large datasets
                        if self.max_samples and processed >= self.max_samples:
                            print(f"🛑 Reached sample limit: {self.max_samples:,}")
                            break
                            
            except Exception as e:
                continue
        
        if not features:
            print("❌ No valid features created")
            return None, None
        
        X = np.array(features)
        y = np.array(returns)
        
        print(f"✅ Feature matrix created: {X.shape[0]:,} samples × {X.shape[1]} features")
        print(f"💾 Memory usage: ~{(X.nbytes + y.nbytes) / 1024**2:.1f} MB")
        
        return X, y
    
    def create_comprehensive_features(self, window_data):
        """Create comprehensive feature vector from 30-candle window"""
        try:
            if window_data.shape[0] != self.window_size or np.any(window_data <= 0):
                return None
            
            # Normalize to percentage changes from first candle
            first_close = window_data[0, 3]
            if first_close <= 0:
                return None
            
            normalized = (window_data / first_close - 1) * 100
            
            # Extract price components
            opens = normalized[:, 0]
            highs = normalized[:, 1]
            lows = normalized[:, 2]
            closes = normalized[:, 3]
            volumes = normalized[:, 4] if normalized.shape[1] > 4 else np.zeros(len(closes))
            
            # Create multi-scale features
            features = []
            
            # 1. Raw price patterns (sampled for efficiency)
            price_sample = closes[::3]  # Every 3rd candle
            features.extend(price_sample)
            
            # 2. Statistical moments
            features.extend([
                np.mean(closes), np.std(closes), 
                np.mean(highs - lows), np.std(highs - lows),
                np.mean(volumes), np.std(volumes)
            ])
            
            # 3. Trend and momentum features
            x = np.arange(len(closes))
            trend_slope = np.polyfit(x, closes, 1)[0] if len(closes) > 1 else 0
            
            # Multi-timeframe momentum
            momentum_short = closes[-5:].mean() - closes[:5].mean() if len(closes) >= 10 else 0
            momentum_med = closes[-10:].mean() - closes[:10].mean() if len(closes) >= 20 else 0
            momentum_long = closes[-15:].mean() - closes[:15].mean() if len(closes) >= 30 else 0
            
            features.extend([trend_slope, momentum_short, momentum_med, momentum_long])
            
            # 4. Volatility measures
            returns = np.diff(closes)
            if len(returns) > 0:
                volatility = np.std(returns)
                max_drawdown = np.min(np.cumsum(returns))
                max_runup = np.max(np.cumsum(returns))
            else:
                volatility = max_drawdown = max_runup = 0
            
            features.extend([volatility, max_drawdown, max_runup])
            
            # 5. Pattern recognition features
            # Support/Resistance
            support = np.min(lows)
            resistance = np.max(highs)
            price_position = (closes[-1] - support) / (resistance - support) if resistance != support else 0.5
            
            # Trend patterns
            higher_highs = np.mean(np.diff(highs) > 0) if len(highs) > 1 else 0.5
            higher_lows = np.mean(np.diff(lows) > 0) if len(lows) > 1 else 0.5
            
            features.extend([support, resistance, price_position, higher_highs, higher_lows])
            
            # 6. Volume-price relationship
            if np.any(volumes):
                volume_trend = np.polyfit(x, volumes, 1)[0] if len(volumes) > 1 else 0
                price_volume_corr = np.corrcoef(closes, volumes)[0, 1] if len(closes) > 1 else 0
            else:
                volume_trend = price_volume_corr = 0
            
            features.extend([volume_trend, price_volume_corr])
            
            # Convert to numpy array and handle NaN/inf
            features = np.array(features, dtype=np.float64)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features
            
        except Exception as e:
            return None
    
    def test_clustering_algorithm(self, name, algorithm, X, y, X_scaled):
        """Test a single clustering algorithm comprehensively"""
        print(f"\n🔬 TESTING {name.upper()}")
        print(f"{'='*60}")
        
        try:
            print(f"🚀 Fitting {name}...")
            
            # Fit the algorithm
            if hasattr(algorithm, 'fit_predict'):
                labels = algorithm.fit_predict(X_scaled)
            else:
                labels = algorithm.fit(X_scaled).predict(X_scaled)
            
            # Handle noise points (for algorithms that produce them)
            valid_mask = labels != -1
            n_samples_valid = valid_mask.sum()
            noise_points = len(labels) - n_samples_valid
            
            if n_samples_valid < 100:
                print(f"❌ Failed: Only {n_samples_valid} valid samples")
                return None
            
            # Get valid data
            X_valid = X_scaled[valid_mask]
            labels_valid = labels[valid_mask]
            y_valid = y[valid_mask]
            
            n_clusters = len(np.unique(labels_valid))
            
            if n_clusters < 2:
                print(f"❌ Failed: Only {n_clusters} cluster(s)")
                return None
            
            print(f"✅ Clustering complete:")
            print(f"   📊 Clusters: {n_clusters}")
            print(f"   👥 Valid samples: {n_samples_valid:,}")
            if noise_points > 0:
                print(f"   🔇 Noise points: {noise_points:,}")
            
            # Calculate clustering quality metrics
            print(f"📊 Computing quality metrics...")
            silhouette = silhouette_score(X_valid, labels_valid)
            calinski = calinski_harabasz_score(X_valid, labels_valid)
            davies = davies_bouldin_score(X_valid, labels_valid)
            
            print(f"✅ Quality metrics:")
            print(f"   📈 Silhouette: {silhouette:.4f}")
            print(f"   📊 Calinski-Harabasz: {calinski:.1f}")
            print(f"   📉 Davies-Bouldin: {davies:.4f}")
            
            # Trading performance analysis
            print(f"💰 Analyzing trading performance...")
            trading_perf = self.analyze_comprehensive_trading_performance(labels_valid, y_valid)
            
            # Calculate composite performance score
            performance_score = self.calculate_composite_score(
                silhouette, trading_perf, n_clusters, n_samples_valid
            )
            
            result = {
                'name': name,
                'n_clusters': n_clusters,
                'n_samples': n_samples_valid,
                'noise_points': noise_points,
                'silhouette_score': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies,
                'trading_performance': trading_perf,
                'performance_score': performance_score,
                'labels': labels
            }
            
            print(f"🎯 Overall Performance Score: {performance_score:.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            return None
    
    def analyze_comprehensive_trading_performance(self, labels, returns):
        """Comprehensive analysis of trading performance per cluster"""
        cluster_stats = {}
        
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_returns = returns[mask]
            
            if len(cluster_returns) >= 20:  # Minimum samples for reliable stats
                mean_return = np.mean(cluster_returns)
                std_return = np.std(cluster_returns)
                win_rate = np.mean(cluster_returns > 0)
                sharpe = mean_return / std_return if std_return > 0 else 0
                
                # Risk metrics
                downside_returns = cluster_returns[cluster_returns < 0]
                downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
                sortino = mean_return / downside_std if downside_std > 0 else 0
                
                # Extreme values
                max_return = np.max(cluster_returns)
                min_return = np.min(cluster_returns)
                
                cluster_stats[cluster_id] = {
                    'size': len(cluster_returns),
                    'mean_return': mean_return,
                    'std_return': std_return,
                    'win_rate': win_rate,
                    'sharpe_ratio': sharpe,
                    'sortino_ratio': sortino,
                    'max_return': max_return,
                    'min_return': min_return,
                    'avg_pips': mean_return * 10000
                }
        
        if not cluster_stats:
            return {'viable_clusters': 0, 'best_cluster': None}
        
        # Find best cluster by risk-adjusted return
        best_cluster_id = max(cluster_stats.keys(), 
                             key=lambda x: cluster_stats[x]['sharpe_ratio'])
        
        # Calculate overall performance metrics
        all_returns = [stats['mean_return'] for stats in cluster_stats.values()]
        all_win_rates = [stats['win_rate'] for stats in cluster_stats.values()]
        all_sharpes = [stats['sharpe_ratio'] for stats in cluster_stats.values()]
        
        return {
            'viable_clusters': len(cluster_stats),
            'cluster_stats': cluster_stats,
            'best_cluster_id': best_cluster_id,
            'best_cluster_stats': cluster_stats[best_cluster_id],
            'return_spread': max(all_returns) - min(all_returns),
            'win_rate_spread': max(all_win_rates) - min(all_win_rates),
            'avg_sharpe': np.mean(all_sharpes),
            'max_sharpe': max(all_sharpes),
            'weighted_win_rate': np.average(all_win_rates, 
                                          weights=[cluster_stats[cid]['size'] for cid in cluster_stats.keys()])
        }
    
    def calculate_composite_score(self, silhouette, trading_perf, n_clusters, n_samples):
        """Calculate composite performance score"""
        # Normalize components
        silhouette_norm = max(0, min(1, (silhouette + 1) / 2))  # -1 to 1 -> 0 to 1
        
        if trading_perf['viable_clusters'] == 0:
            return 0.0
        
        # Trading performance components
        return_spread_norm = min(1, trading_perf['return_spread'] * 1000)  # Scale returns
        win_rate_spread_norm = trading_perf['win_rate_spread']
        avg_sharpe_norm = max(0, min(1, (trading_perf['avg_sharpe'] + 2) / 4))  # -2 to 2 -> 0 to 1
        
        # Composite score (weighted combination)
        score = (
            silhouette_norm * 0.25 +           # 25% - cluster separation
            return_spread_norm * 0.30 +        # 30% - return differentiation
            win_rate_spread_norm * 0.25 +      # 25% - win rate differentiation  
            avg_sharpe_norm * 0.20              # 20% - risk-adjusted performance
        )
        
        return score
    
    def run_comprehensive_comparison(self):
        """Run comprehensive clustering comparison"""
        print(f"🚀 STARTING COMPREHENSIVE CLUSTERING COMPARISON")
        print(f"{'='*80}")
        
        # Load large dataset
        data = self.load_full_dataset()
        if data is None:
            return None
        
        # Create feature matrix
        X, y = self.create_large_feature_matrix(data)
        if X is None:
            return None
        
        # Standardize features
        print(f"\n📐 STANDARDIZING FEATURES")
        print(f"==================================================")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"✅ Features standardized: {X_scaled.shape}")
        
        # Optional dimensionality reduction for very high-dimensional data
        if X_scaled.shape[1] > 50:
            print(f"🔄 Applying PCA for efficiency...")
            pca = PCA(n_components=30, random_state=42)
            X_scaled = pca.fit_transform(X_scaled)
            print(f"   📊 Reduced to {X_scaled.shape[1]} components")
        
        # Define algorithms to test
        algorithms = [
            ('K-Means (Current)', KMeans(n_clusters=8, random_state=42)),
            ('GMM Full Covariance', GaussianMixture(n_components=8, covariance_type='full', random_state=42)),
            ('GMM Diagonal', GaussianMixture(n_components=8, covariance_type='diag', random_state=42)),
            ('GMM Tied', GaussianMixture(n_components=8, covariance_type='tied', random_state=42)),
            ('GMM Spherical', GaussianMixture(n_components=8, covariance_type='spherical', random_state=42)),
            ('Agglomerative Ward', AgglomerativeClustering(n_clusters=8, linkage='ward')),
            ('Agglomerative Complete', AgglomerativeClustering(n_clusters=8, linkage='complete')),
            ('Agglomerative Average', AgglomerativeClustering(n_clusters=8, linkage='average')),
        ]
        
        print(f"\n🧪 TESTING {len(algorithms)} ALGORITHMS")
        print(f"{'='*80}")
        
        results = []
        for name, algorithm in algorithms:
            result = self.test_clustering_algorithm(name, algorithm, X, y, X_scaled)
            if result:
                results.append(result)
        
        # Generate comprehensive report
        return self.create_comprehensive_report(results)
    
    def create_comprehensive_report(self, results):
        """Create comprehensive comparison report"""
        if not results:
            print(f"\n❌ NO SUCCESSFUL RESULTS TO COMPARE")
            return None
        
        print(f"\n📊 COMPREHENSIVE CLUSTERING COMPARISON RESULTS")
        print(f"{'='*100}")
        
        # Sort by performance score
        results_sorted = sorted(results, key=lambda x: x['performance_score'], reverse=True)
        
        # Summary table
        print(f"{'Rank':<4} {'Algorithm':<25} {'Clusters':<8} {'Samples':<10} {'Silhouette':<11} "
              f"{'Best Win%':<10} {'R-Spread':<10} {'Score':<8}")
        print(f"{'-'*4} {'-'*25} {'-'*8} {'-'*10} {'-'*11} {'-'*10} {'-'*10} {'-'*8}")
        
        for i, result in enumerate(results_sorted, 1):
            trading = result['trading_performance']
            best_win_rate = trading['best_cluster_stats']['win_rate'] if trading['best_cluster_stats'] else 0
            
            print(f"{i:<4} {result['name']:<25} {result['n_clusters']:<8} {result['n_samples']:<10,} "
                  f"{result['silhouette_score']:<11.4f} {best_win_rate:<10.1%} "
                  f"{trading.get('return_spread', 0):<10.4f} {result['performance_score']:<8.3f}")
        
        # Winner analysis
        winner = results_sorted[0]
        print(f"\n🏆 WINNER: {winner['name'].upper()}")
        print(f"{'='*60}")
        print(f"📊 Algorithm: {winner['name']}")
        print(f"📈 Performance Score: {winner['performance_score']:.3f}")
        print(f"🎯 Clusters: {winner['n_clusters']}")
        print(f"👥 Training Samples: {winner['n_samples']:,}")
        print(f"📊 Silhouette Score: {winner['silhouette_score']:.4f}")
        
        trading = winner['trading_performance']
        if trading['best_cluster_stats']:
            best = trading['best_cluster_stats']
            print(f"🏆 Best Cluster Performance:")
            print(f"   📈 Win Rate: {best['win_rate']:.1%}")
            print(f"   💰 Avg Return: {best['avg_pips']:.1f} pips")
            print(f"   📊 Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            print(f"   📏 Return Spread: {trading['return_spread']:.4f}")
            print(f"   📏 Win Rate Spread: {trading['win_rate_spread']:.1%}")
        
        # Comparison with current system
        kmeans_result = next((r for r in results if 'K-Means' in r['name']), None)
        if kmeans_result and winner != kmeans_result:
            improvement = (winner['performance_score'] - kmeans_result['performance_score']) / kmeans_result['performance_score'] * 100
            print(f"\n📈 IMPROVEMENT OVER K-MEANS: +{improvement:.1f}%")
        
        return winner

def main():
    """Main function to run comprehensive comparison"""
    # Run comparison with large dataset
    comparison = LargeScaleClusteringComparison(window_size=30, max_samples=50000)
    
    winner = comparison.run_comprehensive_comparison()
    
    if winner:
        print(f"\n✅ COMPREHENSIVE COMPARISON COMPLETE!")
        print(f"🏆 Recommended Algorithm: {winner['name']}")
        print(f"💡 Use this algorithm for optimal 30-candle pattern clustering")
        print(f"🔬 Based on {winner['n_samples']:,} samples from 300K+ candle dataset")
    else:
        print(f"\n❌ Comprehensive comparison failed")

if __name__ == "__main__":
    main()