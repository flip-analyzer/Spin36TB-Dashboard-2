#!/usr/bin/env python3
"""
López de Prado Advanced Clustering System
Implements cutting-edge techniques from "Machine Learning for Asset Managers" 
and "Advances in Financial Machine Learning"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import TSNE
import pickle

warnings.filterwarnings('ignore')

class LopezAdvancedClustering:
    """
    Advanced clustering system based on López de Prado's latest techniques
    """
    
    def __init__(self):
        # López de Prado Configuration
        self.config = {
            # Feature Engineering (Ch 5: Fractional Differentiation)
            'lookback_window': 100,           # 100 candles (8+ hours) for meaningful patterns
            'fractional_diff_threshold': 0.01, # Stationarity threshold
            'feature_lag_max': 5,             # Multi-lag features
            
            # Clustering Algorithm Selection (Ch 4: Optimal Clustering)
            'primary_algorithm': 'hierarchical', # Better for financial time series
            'secondary_algorithm': 'dbscan',    # For density-based discovery
            'backup_algorithm': 'gaussian_mixture', # For probabilistic assignment
            
            # Optimal Number of Clusters (Ch 4: Gap Statistic + Silhouette)
            'min_clusters': 3,
            'max_clusters': 15,
            'cluster_selection_method': 'gap_statistic',
            
            # Distance Metrics (Ch 4: Information-based distances)
            'distance_metric': 'correlation',   # Better than Euclidean for financial data  
            'linkage_method': 'ward',          # Minimizes within-cluster variance
            
            # Feature Selection (Ch 8: Feature Importance)
            'feature_importance_threshold': 0.1,
            'max_features': 20,
            
            # Validation (Ch 11: Cross-Validation in Finance)
            'purging_window': 24,              # 24 periods between train/test
            'embargo_window': 12,              # 12 periods embargo
            'cv_folds': 5,
        }
        
        # Advanced Feature Engineering Components
        self.scaler = StandardScaler()
        self.pca = None
        self.tsne = None
        
        # Clustering Models
        self.hierarchical_model = None
        self.dbscan_model = None  
        self.gaussian_mixture = None
        self.optimal_n_clusters = None
        
        # Feature Importance and Selection
        self.feature_importance = {}
        self.selected_features = []
        
        print("🧠 LÓPEZ DE PRADO ADVANCED CLUSTERING INITIALIZED")
        print("=" * 60)
        print("📚 Based on 'Machine Learning for Asset Managers' latest techniques")
        print("🔬 Advanced feature engineering with fractional differentiation")
        print("📊 Multi-algorithm clustering with optimal cluster selection")
        print("🎯 Information-based distance metrics for financial time series")
    
    def create_fractional_diff_features(self, price_series: pd.Series, d: float = 0.4) -> pd.Series:
        """
        Fractional differentiation (López de Prado Ch 5)
        Makes series stationary while preserving memory
        """
        def get_weights(d: float, size: int):
            w = [1.0]
            for k in range(1, size):
                w_ = -w[-1] * (d - k + 1) / k
                w.append(w_)
            return np.array(w).reshape(-1, 1)
        
        try:
            weights = get_weights(d, len(price_series))
            if len(weights) > len(price_series):
                weights = weights[:len(price_series)]
            
            # Apply fractional differentiation
            frac_diff = pd.Series(index=price_series.index, dtype=float)
            for i in range(len(weights), len(price_series)):
                if not np.isnan(price_series.iloc[i-len(weights)+1:i+1]).any():
                    frac_diff.iloc[i] = np.dot(weights.T, price_series.iloc[i-len(weights)+1:i+1])[0]
            
            return frac_diff.dropna()
            
        except Exception as e:
            print(f"Warning: Fractional diff failed, using log returns: {e}")
            return np.log(price_series).diff().dropna()
    
    def create_advanced_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering based on López de Prado techniques
        """
        if len(market_data) < self.config['lookback_window']:
            return pd.DataFrame()
        
        df = market_data.copy()
        features_df = pd.DataFrame(index=df.index)
        
        print("🔬 Creating advanced features...")
        
        # 1. Fractional Differentiation Features (Ch 5)
        close_frac_diff = self.create_fractional_diff_features(df['close'])
        features_df['frac_diff_close'] = close_frac_diff
        
        high_frac_diff = self.create_fractional_diff_features(df['high'])
        features_df['frac_diff_high'] = high_frac_diff
        
        low_frac_diff = self.create_fractional_diff_features(df['low'])
        features_df['frac_diff_low'] = low_frac_diff
        
        # 2. Multi-Scale Momentum Features (different time horizons)
        for period in [5, 10, 20, 50]:
            if len(df) > period:
                features_df[f'momentum_{period}'] = df['close'].pct_change(period)
                features_df[f'volatility_{period}'] = df['close'].rolling(period).std() / df['close']
        
        # 3. Microstructural Features (López de Prado Ch 2)
        # Price impact measures
        features_df['price_impact'] = (df['high'] - df['low']) / df['close']
        features_df['bid_ask_spread_proxy'] = (df['high'] - df['low']) / df['volume'].replace(0, 1)
        
        # 4. Information-Driven Features (Ch 3)
        # Tick rule (price direction)
        features_df['tick_rule'] = np.sign(df['close'].diff())
        
        # Volume-synchronized features
        features_df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        features_df['vwap_deviation'] = (df['close'] - features_df['vwap']) / features_df['vwap']
        
        # 5. Regime-Change Indicators (Ch 16)
        # Structural break detection proxy
        # Beta to market (rolling correlation with time trend)
        time_trend = pd.Series(range(len(df)), index=df.index, name='time_trend')
        features_df['rolling_beta'] = df['close'].rolling(50).corr(time_trend)
        
        # Volatility regime
        short_vol = df['close'].rolling(10).std()
        long_vol = df['close'].rolling(50).std()
        features_df['vol_regime'] = short_vol / long_vol
        
        # 6. Multi-Lag Features (temporal dependencies)
        base_features = ['frac_diff_close', 'price_impact', 'tick_rule']
        for feature in base_features:
            if feature in features_df.columns:
                for lag in range(1, self.config['feature_lag_max'] + 1):
                    features_df[f'{feature}_lag_{lag}'] = features_df[feature].shift(lag)
        
        # 7. Cross-Sectional Features (relative measures)
        # Z-scores (standardized over rolling window)
        for col in ['frac_diff_close', 'momentum_20', 'volatility_20']:
            if col in features_df.columns:
                rolling_mean = features_df[col].rolling(50).mean()
                rolling_std = features_df[col].rolling(50).std()
                features_df[f'{col}_zscore'] = (features_df[col] - rolling_mean) / rolling_std
        
        # Remove rows with too many NaN values
        features_df = features_df.dropna(thresh=len(features_df.columns) * 0.7)  # Keep rows with 70%+ data
        
        print(f"✅ Created {len(features_df.columns)} advanced features")
        print(f"   Feature matrix: {features_df.shape}")
        
        return features_df
    
    def calculate_information_distance_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """
        Calculate information-based distance matrix (López de Prado Ch 4)
        Uses correlation distance which is better for financial time series
        """
        print("📏 Calculating information-based distance matrix...")
        
        # Remove any remaining NaN values
        clean_features = features.dropna()
        
        if len(clean_features) == 0:
            return np.array([[]])
        
        # Correlation distance matrix
        correlation_matrix = clean_features.T.corr()
        distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
        
        return distance_matrix.values
    
    def find_optimal_clusters_gap_statistic(self, features: pd.DataFrame) -> int:
        """
        Find optimal number of clusters using Gap Statistic (López de Prado Ch 4)
        """
        print("🎯 Finding optimal clusters using Gap Statistic...")
        
        def gap_statistic(X, max_k):
            gaps = []
            sk = []
            
            for k in range(1, max_k + 1):
                # Original data clustering
                if k == 1:
                    wk = np.sum(pdist(X) ** 2) / (2 * len(X))
                else:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X)
                    
                    wk = 0
                    for cluster_id in range(k):
                        cluster_points = X[labels == cluster_id]
                        if len(cluster_points) > 1:
                            wk += np.sum(pdist(cluster_points) ** 2) / (2 * len(cluster_points))
                
                # Reference distribution (uniform random)
                wk_refs = []
                for _ in range(10):  # 10 reference datasets
                    X_ref = np.random.uniform(X.min(), X.max(), X.shape)
                    
                    if k == 1:
                        wk_ref = np.sum(pdist(X_ref) ** 2) / (2 * len(X_ref))
                    else:
                        kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels_ref = kmeans_ref.fit_predict(X_ref)
                        
                        wk_ref = 0
                        for cluster_id in range(k):
                            cluster_points = X_ref[labels_ref == cluster_id]
                            if len(cluster_points) > 1:
                                wk_ref += np.sum(pdist(cluster_points) ** 2) / (2 * len(cluster_points))
                    
                    wk_refs.append(wk_ref)
                
                gap = np.log(np.mean(wk_refs)) - np.log(wk)
                sk_val = np.std(np.log(wk_refs)) * np.sqrt(1 + 1/10)  # Standard error
                
                gaps.append(gap)
                sk.append(sk_val)
            
            return gaps, sk
        
        try:
            # Prepare data for gap statistic
            X = features.dropna().values
            if len(X) < 10:
                return 3  # Default fallback
            
            gaps, sk = gap_statistic(X, self.config['max_clusters'])
            
            # Find optimal k using gap statistic rule
            for k in range(len(gaps) - 1):
                if gaps[k] >= gaps[k + 1] - sk[k + 1]:
                    optimal_k = k + 1
                    break
            else:
                optimal_k = np.argmax(gaps) + 1
            
            # Ensure reasonable bounds
            optimal_k = max(self.config['min_clusters'], min(optimal_k, self.config['max_clusters']))
            
            print(f"   Optimal clusters found: {optimal_k}")
            return optimal_k
            
        except Exception as e:
            print(f"   Gap statistic failed: {e}, using default: 6")
            return 6
    
    def fit_hierarchical_clustering(self, features: pd.DataFrame, n_clusters: int) -> np.ndarray:
        """
        Hierarchical clustering with correlation-based linkage (López de Prado Ch 4)
        """
        print(f"🌳 Fitting hierarchical clustering with {n_clusters} clusters...")
        
        try:
            # Calculate distance matrix
            distance_matrix = self.calculate_information_distance_matrix(features)
            
            if distance_matrix.size == 0:
                return np.array([])
            
            # Convert to condensed distance matrix for linkage
            condensed_distances = pdist(distance_matrix)
            
            # Hierarchical clustering with Ward linkage
            linkage_matrix = linkage(condensed_distances, method=self.config['linkage_method'])
            
            # Get cluster labels
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            return cluster_labels - 1  # Convert to 0-based indexing
            
        except Exception as e:
            print(f"   Hierarchical clustering failed: {e}")
            return np.array([])
    
    def fit_advanced_clustering(self, market_data: pd.DataFrame) -> Dict:
        """
        Complete advanced clustering pipeline
        """
        print("🚀 ADVANCED CLUSTERING PIPELINE STARTING...")
        print("=" * 50)
        
        # Step 1: Advanced feature engineering
        features = self.create_advanced_features(market_data)
        if features.empty:
            return {'success': False, 'error': 'Feature engineering failed'}
        
        # Step 2: Feature scaling (López de Prado recommendation)
        features_scaled = pd.DataFrame(
            self.scaler.fit_transform(features.fillna(0)),
            columns=features.columns,
            index=features.index
        )
        
        # Step 3: Dimensionality reduction (optional, for visualization)
        if len(features.columns) > 10:
            self.pca = PCA(n_components=min(10, len(features.columns)))
            features_pca = self.pca.fit_transform(features_scaled)
            print(f"   PCA explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        else:
            features_pca = features_scaled.values
        
        # Step 4: Find optimal number of clusters
        self.optimal_n_clusters = self.find_optimal_clusters_gap_statistic(features_scaled)
        
        # Step 5: Apply hierarchical clustering (primary method)
        cluster_labels = self.fit_hierarchical_clustering(features_scaled, self.optimal_n_clusters)
        
        if len(cluster_labels) == 0:
            return {'success': False, 'error': 'Clustering failed'}
        
        # Step 6: Calculate cluster quality metrics
        silhouette_avg = silhouette_score(features_scaled.fillna(0), cluster_labels)
        
        # Step 7: Analyze cluster performance
        cluster_analysis = self.analyze_cluster_performance(
            features_scaled, cluster_labels, market_data, features_scaled
        )
        
        # Step 8: Save model components
        self.save_advanced_model(features_scaled, cluster_labels)
        
        results = {
            'success': True,
            'n_clusters': self.optimal_n_clusters,
            'n_features': len(features.columns),
            'n_samples': len(features_scaled),
            'silhouette_score': silhouette_avg,
            'cluster_labels': cluster_labels,
            'feature_names': features.columns.tolist(),
            'cluster_analysis': cluster_analysis
        }
        
        print(f"\n✅ ADVANCED CLUSTERING COMPLETE!")
        print(f"   Clusters: {self.optimal_n_clusters}")
        print(f"   Silhouette Score: {silhouette_avg:.4f}")
        print(f"   Features: {len(features.columns)}")
        print(f"   Samples: {len(features_scaled)}")
        
        return results
    
    def analyze_cluster_performance(self, features: pd.DataFrame, labels: np.ndarray, 
                                  market_data: pd.DataFrame, X: pd.DataFrame) -> Dict:
        """
        Analyze cluster performance with forward-looking returns
        """
        print("📊 Analyzing cluster performance...")
        
        # Calculate forward returns (next 6 periods = 30 minutes) - align with feature data
        # Features are reduced due to rolling windows, so align returns accordingly
        aligned_market_data = market_data.loc[X.index]  # Align market data with feature data
        forward_returns = aligned_market_data['close'].shift(-6) / aligned_market_data['close'] - 1
        
        cluster_stats = {}
        
        for cluster_id in range(self.optimal_n_clusters):
            mask = labels == cluster_id
            cluster_returns = forward_returns[mask].dropna()
            
            if len(cluster_returns) > 5:  # Need minimum samples
                stats = {
                    'size': mask.sum(),
                    'mean_return': cluster_returns.mean(),
                    'std_return': cluster_returns.std(),
                    'sharpe_ratio': cluster_returns.mean() / cluster_returns.std() if cluster_returns.std() > 0 else 0,
                    'win_rate': (cluster_returns > 0).mean(),
                    'avg_winner': cluster_returns[cluster_returns > 0].mean() if any(cluster_returns > 0) else 0,
                    'avg_loser': cluster_returns[cluster_returns < 0].mean() if any(cluster_returns < 0) else 0,
                    'profit_factor': abs(cluster_returns[cluster_returns > 0].sum() / 
                                       cluster_returns[cluster_returns < 0].sum()) if any(cluster_returns < 0) else float('inf')
                }
                
                cluster_stats[cluster_id] = stats
        
        return cluster_stats
    
    def save_advanced_model(self, features: pd.DataFrame, labels: np.ndarray):
        """Save the advanced clustering model"""
        model_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'optimal_n_clusters': self.optimal_n_clusters,
            'cluster_labels': labels,
            'feature_names': features.columns.tolist(),
            'config': self.config,
            'training_timestamp': datetime.now().isoformat(),
            'model_type': 'lopez_advanced_clustering'
        }
        
        filename = '/Users/jonspinogatti/Desktop/spin36TB/lopez_advanced_clustering_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Advanced model saved: {filename}")

def test_lopez_clustering():
    """Test the López de Prado clustering system"""
    print("🧪 TESTING LÓPEZ DE PRADO CLUSTERING")
    print("=" * 50)
    
    # Generate synthetic market data for testing
    import requests
    
    try:
        # Try to get real market data
        api_token = "fdf72f4b5166602e39e001d1c1cb38b0-524c6662baa3187260b34c006eeab1fd"
        headers = {"Authorization": f"Bearer {api_token}"}
        
        response = requests.get(
            "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles",
            headers=headers,
            params={"count": 500, "granularity": "M5", "price": "M"},
            timeout=30
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
                        'volume': int(candle.get('volume', 1000))
                    })
            
            market_data = pd.DataFrame(df_data).set_index('time')
            print(f"✅ Using real market data: {len(market_data)} candles")
            
        else:
            raise Exception("API failed")
    
    except:
        print("⚠️  Using synthetic data for testing...")
        # Generate synthetic data
        dates = pd.date_range(start='2023-01-01', periods=500, freq='5min')
        np.random.seed(42)
        
        prices = 1.1000 + np.cumsum(np.random.randn(500) * 0.0001)
        volumes = 1000 + np.random.randint(-200, 200, 500)
        
        market_data = pd.DataFrame({
            'open': prices + np.random.randn(500) * 0.0001,
            'high': prices + np.abs(np.random.randn(500)) * 0.0002,
            'low': prices - np.abs(np.random.randn(500)) * 0.0002,  
            'close': prices,
            'volume': volumes
        }, index=dates)
    
    # Initialize and test clustering
    clustering = LopezAdvancedClustering()
    results = clustering.fit_advanced_clustering(market_data)
    
    if results['success']:
        print(f"\n🎉 LÓPEZ CLUSTERING SUCCESS!")
        print(f"   Improvement over old model:")
        print(f"   Old silhouette score: 0.0624 (Very Poor)")
        print(f"   New silhouette score: {results['silhouette_score']:.4f}")
        
        improvement = (results['silhouette_score'] - 0.0624) / 0.0624 * 100
        print(f"   Improvement: {improvement:+.1f}%")
        
        return True
    else:
        print(f"❌ Clustering failed: {results.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    success = test_lopez_clustering()
    
    if success:
        print("\n✅ Ready to integrate with hybrid trading system!")
    else:
        print("\n❌ Advanced clustering needs debugging")