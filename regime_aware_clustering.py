#!/usr/bin/env python3
"""
Regime-Aware Dynamic Clustering Enhancement for Spin36TB
Creates separate cluster models optimized for different market regimes
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
warnings.filterwarnings('ignore')

class RegimeDetector:
    """
    Advanced market regime detection using multiple indicators
    """
    
    def __init__(self, lookback_period=100):
        self.lookback_period = lookback_period
        self.regime_thresholds = {
            'volatility_high': 0.0008,  # 8 pips
            'volatility_low': 0.0003,   # 3 pips
            'momentum_strong': 0.6,
            'momentum_weak': 0.3,
            'trend_strength_high': 0.7,
            'trend_strength_low': 0.3
        }
        
    def calculate_regime_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for regime detection"""
        df = data.copy()
        
        # Volatility (ATR-based)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['volatility'] = df['tr'].rolling(14).mean()
        
        # Momentum (Rate of Change)
        df['momentum'] = abs(df['close'].pct_change(10))
        
        # Trend Strength (ADX-like indicator)
        df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                                df['high'] - df['high'].shift(1), 0)
        df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                                 df['low'].shift(1) - df['low'], 0)
        
        df['dm_plus'] = np.where(df['dm_plus'] < 0, 0, df['dm_plus'])
        df['dm_minus'] = np.where(df['dm_minus'] < 0, 0, df['dm_minus'])
        
        df['di_plus'] = 100 * (df['dm_plus'].rolling(14).mean() / df['tr'].rolling(14).mean())
        df['di_minus'] = 100 * (df['dm_minus'].rolling(14).mean() / df['tr'].rolling(14).mean())
        
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['trend_strength'] = df['dx'].rolling(14).mean() / 100
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price action
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        df['upper_wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'])
        df['lower_wick'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def detect_regime(self, data: pd.DataFrame, current_idx: int) -> str:
        """
        Detect current market regime based on recent data
        """
        if current_idx < self.lookback_period:
            return "MIXED_CONDITIONS"
        
        # Get recent data window
        recent_data = data.iloc[current_idx - self.lookback_period:current_idx + 1]
        
        # Calculate current regime characteristics
        current_vol = recent_data['volatility'].iloc[-1]
        avg_momentum = recent_data['momentum'].mean()
        trend_strength = recent_data['trend_strength'].iloc[-1]
        avg_volume_ratio = recent_data['volume_ratio'].mean()
        
        # Regime classification logic
        is_high_vol = current_vol > self.regime_thresholds['volatility_high']
        is_low_vol = current_vol < self.regime_thresholds['volatility_low']
        is_strong_momentum = avg_momentum > self.regime_thresholds['momentum_strong']
        is_weak_momentum = avg_momentum < self.regime_thresholds['momentum_weak']
        is_strong_trend = trend_strength > self.regime_thresholds['trend_strength_high']
        is_weak_trend = trend_strength < self.regime_thresholds['trend_strength_low']
        
        # High volume + strong trend = trending market
        if is_high_vol and is_strong_trend:
            return "HIGH_VOL_TRENDING"
        
        # Strong momentum regardless of volatility
        elif is_strong_momentum:
            return "HIGH_MOMENTUM"
        
        # Strong trend but not high volatility
        elif is_strong_trend and not is_high_vol:
            return "STRONG_TREND"
        
        # Low volatility + weak trend = ranging market
        elif is_low_vol and is_weak_trend:
            return "LOW_VOL_RANGING"
        
        # Everything else is mixed conditions
        else:
            return "MIXED_CONDITIONS"

class RegimeAwareClusteringSystem:
    """
    Enhanced clustering system that maintains separate models for each market regime
    """
    
    def __init__(self, n_clusters_per_regime=8, min_samples_per_regime=200):
        self.n_clusters_per_regime = n_clusters_per_regime
        self.min_samples_per_regime = min_samples_per_regime
        self.regime_detector = RegimeDetector()
        
        # Separate models for each regime
        self.regime_models = {}
        self.regime_scalers = {}
        self.regime_statistics = {}
        
        # Performance tracking per regime
        self.regime_performance = {}
        
        # Feature columns for clustering
        self.feature_columns = [
            'candles_to_peak', 'volatility', 'momentum', 'trend_strength',
            'rsi', 'macd', 'bb_position', 'body_size', 'wick_ratio',
            'volume_ratio', 'volume_trend', 'hour_sin', 'hour_cos'
        ]
        
    def prepare_excursion_features(self, excursions_df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare excursion features with regime information
        """
        print("🔍 Preparing excursion features with regime detection...")
        
        # Calculate regime indicators for market data
        market_with_regimes = self.regime_detector.calculate_regime_indicators(market_data)
        
        enhanced_excursions = []
        
        for idx, excursion in excursions_df.iterrows():
            # Find corresponding market data index
            excursion_start_time = excursion['timestamp']
            matching_indices = market_with_regimes[market_with_regimes.index <= excursion_start_time]
            
            if len(matching_indices) > 0:
                market_idx_position = len(matching_indices) - 1  # Get position, not timestamp
            else:
                market_idx_position = 0
            
            # Detect regime at excursion time
            regime = self.regime_detector.detect_regime(market_with_regimes, market_idx_position)
            
            # Add regime to excursion data
            enhanced_excursion = excursion.copy()
            enhanced_excursion['regime'] = regime
            enhanced_excursion['regime_volatility'] = market_with_regimes.iloc[market_idx_position]['volatility']
            enhanced_excursion['regime_momentum'] = market_with_regimes.iloc[market_idx_position]['momentum']
            enhanced_excursion['regime_trend_strength'] = market_with_regimes.iloc[market_idx_position]['trend_strength']
            
            enhanced_excursions.append(enhanced_excursion)
        
        enhanced_df = pd.DataFrame(enhanced_excursions)
        
        # Show regime distribution
        regime_counts = enhanced_df['regime'].value_counts()
        print(f"📊 Regime Distribution:")
        for regime, count in regime_counts.items():
            percentage = count / len(enhanced_df) * 100
            print(f"   {regime}: {count} excursions ({percentage:.1f}%)")
        
        return enhanced_df
    
    def train_regime_models(self, excursions_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Train separate clustering models for each market regime
        """
        print(f"🏋️ Training regime-specific clustering models...")
        
        regime_results = {}
        
        # Group excursions by regime
        regime_groups = excursions_df.groupby('regime')
        
        for regime, regime_data in regime_groups:
            print(f"\n🎯 Training {regime} model...")
            
            if len(regime_data) < self.min_samples_per_regime:
                print(f"   ⚠️  Insufficient data: {len(regime_data)} samples (min: {self.min_samples_per_regime})")
                print(f"   Using fallback clustering with reduced clusters...")
                n_clusters = max(3, min(6, len(regime_data) // 30))
            else:
                n_clusters = self.n_clusters_per_regime
                print(f"   ✅ Sufficient data: {len(regime_data)} samples")
            
            # Prepare features
            features = regime_data[self.feature_columns].copy()
            
            # Handle missing values
            features = features.fillna(features.mean())
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Try different clustering algorithms and choose the best
            best_score = -1
            best_model = None
            best_labels = None
            
            # K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(features_scaled)
            kmeans_score = silhouette_score(features_scaled, kmeans_labels) if n_clusters > 1 else 0
            
            if kmeans_score > best_score:
                best_score = kmeans_score
                best_model = kmeans
                best_labels = kmeans_labels
            
            # Gaussian Mixture Model (if enough data)
            if len(regime_data) >= 100:
                gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                gmm_labels = gmm.fit_predict(features_scaled)
                gmm_score = silhouette_score(features_scaled, gmm_labels) if n_clusters > 1 else 0
                
                if gmm_score > best_score:
                    best_score = gmm_score
                    best_model = gmm
                    best_labels = gmm_labels
            
            # Store the best model
            self.regime_models[regime] = best_model
            self.regime_scalers[regime] = scaler
            
            # Calculate cluster performance statistics
            regime_data_with_clusters = regime_data.copy()
            regime_data_with_clusters['cluster'] = best_labels
            
            cluster_stats = self.calculate_cluster_performance(regime_data_with_clusters)
            
            regime_results[regime] = {
                'model_type': type(best_model).__name__,
                'n_clusters': n_clusters,
                'silhouette_score': best_score,
                'cluster_stats': cluster_stats,
                'total_samples': len(regime_data)
            }
            
            print(f"   Model: {type(best_model).__name__}")
            print(f"   Clusters: {n_clusters}")
            print(f"   Silhouette Score: {best_score:.3f}")
            print(f"   Best Cluster Win Rate: {cluster_stats['best_cluster_win_rate']:.1%}")
            print(f"   Best Cluster Avg Pips: {cluster_stats['best_cluster_avg_pips']:+.1f}")
        
        self.regime_statistics = regime_results
        return regime_results
    
    def calculate_cluster_performance(self, data_with_clusters: pd.DataFrame) -> Dict:
        """Calculate performance metrics for clusters within a regime"""
        
        cluster_performance = {}
        
        for cluster_id in data_with_clusters['cluster'].unique():
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            
            # Basic performance metrics
            total_excursions = len(cluster_data)
            successful_excursions = len(cluster_data[cluster_data['successful'] == True])
            win_rate = successful_excursions / total_excursions if total_excursions > 0 else 0
            avg_pips = cluster_data['pips_captured'].mean() if total_excursions > 0 else 0
            
            cluster_performance[cluster_id] = {
                'total_excursions': total_excursions,
                'win_rate': win_rate,
                'avg_pips': avg_pips,
                'dominant_direction': cluster_data['direction'].mode().iloc[0] if len(cluster_data) > 0 else 'UP',
                'avg_candles_to_peak': cluster_data['candles_to_peak'].mean() if len(cluster_data) > 0 else 0
            }
        
        # Find best performing cluster
        if cluster_performance:
            best_cluster = max(cluster_performance.keys(), 
                             key=lambda k: cluster_performance[k]['win_rate'])
            
            return {
                'cluster_performance': cluster_performance,
                'best_cluster_id': best_cluster,
                'best_cluster_win_rate': cluster_performance[best_cluster]['win_rate'],
                'best_cluster_avg_pips': cluster_performance[best_cluster]['avg_pips'],
                'total_clusters': len(cluster_performance)
            }
        else:
            return {
                'cluster_performance': {},
                'best_cluster_id': 0,
                'best_cluster_win_rate': 0.5,
                'best_cluster_avg_pips': 0,
                'total_clusters': 0
            }
    
    def predict_regime_cluster(self, excursion_features: Dict, current_regime: str) -> Tuple[int, float, Dict]:
        """
        Predict cluster for new excursion based on its regime
        """
        # Use regime-specific model if available, otherwise fallback to best available
        if current_regime not in self.regime_models:
            # Find the regime with the most similar characteristics
            available_regimes = list(self.regime_models.keys())
            if not available_regimes:
                return 0, 0.5, {'error': 'No trained models available'}
            
            # Use the regime with best overall performance as fallback
            fallback_regime = max(available_regimes, 
                                key=lambda r: self.regime_statistics[r]['cluster_stats']['best_cluster_win_rate'])
            print(f"⚠️  Using fallback regime {fallback_regime} for prediction (requested: {current_regime})")
            current_regime = fallback_regime
        
        # Prepare features
        features = np.array([[excursion_features[col] for col in self.feature_columns]])
        
        # Scale features using regime-specific scaler
        features_scaled = self.regime_scalers[current_regime].transform(features)
        
        # Predict cluster
        model = self.regime_models[current_regime]
        cluster_id = model.predict(features_scaled)[0]
        
        # Get cluster confidence/probability
        if hasattr(model, 'predict_proba'):
            confidence = model.predict_proba(features_scaled)[0].max()
        elif hasattr(model, 'score_samples'):
            confidence = model.score_samples(features_scaled)[0]
        else:
            # For K-Means, use distance to centroid as inverse confidence
            distances = model.transform(features_scaled)[0]
            confidence = 1 / (1 + distances[cluster_id])
        
        # Get cluster statistics
        cluster_stats = self.regime_statistics[current_regime]['cluster_stats']['cluster_performance']
        cluster_info = cluster_stats.get(cluster_id, {
            'win_rate': 0.5,
            'avg_pips': 0,
            'total_excursions': 0
        })
        
        return cluster_id, confidence, cluster_info
    
    def get_regime_specific_trading_recommendation(self, excursion_features: Dict, 
                                                  current_regime: str) -> Dict:
        """
        Get enhanced trading recommendation based on regime-aware clustering
        """
        # Get cluster prediction
        cluster_id, confidence, cluster_info = self.predict_regime_cluster(
            excursion_features, current_regime
        )
        
        # Enhanced position sizing based on regime and cluster performance
        base_position_size = 0.04  # 4% base
        
        # Regime-based adjustments
        regime_multipliers = {
            'HIGH_VOL_TRENDING': 1.2,   # Increase in favorable regime
            'HIGH_MOMENTUM': 1.1,
            'STRONG_TREND': 1.0,
            'MIXED_CONDITIONS': 0.8,
            'LOW_VOL_RANGING': 0.6      # Reduce in unfavorable regime
        }
        
        regime_multiplier = regime_multipliers.get(current_regime, 1.0)
        
        # Cluster performance adjustments
        cluster_win_rate = cluster_info.get('win_rate', 0.5)
        if cluster_win_rate > 0.65:
            cluster_multiplier = 1.3    # Boost for high-performing clusters
        elif cluster_win_rate > 0.55:
            cluster_multiplier = 1.1
        elif cluster_win_rate < 0.45:
            cluster_multiplier = 0.7    # Reduce for poor-performing clusters
        else:
            cluster_multiplier = 1.0
        
        # Confidence adjustment
        confidence_multiplier = 0.7 + (0.6 * confidence)  # 0.7 to 1.3 range
        
        # Final position size
        enhanced_position_size = base_position_size * regime_multiplier * cluster_multiplier * confidence_multiplier
        enhanced_position_size = np.clip(enhanced_position_size, 0.01, 0.12)  # 1% to 12% limits
        
        return {
            'regime': current_regime,
            'cluster_id': cluster_id,
            'confidence': confidence,
            'cluster_win_rate': cluster_win_rate,
            'cluster_avg_pips': cluster_info.get('avg_pips', 0),
            'regime_multiplier': regime_multiplier,
            'cluster_multiplier': cluster_multiplier,
            'confidence_multiplier': confidence_multiplier,
            'recommended_position_size': enhanced_position_size,
            'should_trade': cluster_win_rate > 0.52 and confidence > 0.3,  # Enhanced filtering
            'reason': f"Regime: {current_regime}, Cluster {cluster_id} ({cluster_win_rate:.1%} WR)"
        }
    
    def save_models(self, filepath_base: str):
        """Save all regime models and scalers"""
        
        # Save models
        models_data = {
            'regime_models': {},
            'regime_scalers': {},
            'regime_statistics': self.regime_statistics,
            'feature_columns': self.feature_columns,
            'n_clusters_per_regime': self.n_clusters_per_regime
        }
        
        # Save each model separately (since some may not be pickle-able together)
        for regime in self.regime_models:
            model_file = f"{filepath_base}_{regime}_model.pkl"
            scaler_file = f"{filepath_base}_{regime}_scaler.pkl"
            
            with open(model_file, 'wb') as f:
                pickle.dump(self.regime_models[regime], f)
            
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.regime_scalers[regime], f)
            
            models_data['regime_models'][regime] = model_file
            models_data['regime_scalers'][regime] = scaler_file
        
        # Save metadata (convert numpy types to Python types)
        metadata_file = f"{filepath_base}_metadata.json"
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        models_data = convert_numpy(models_data)
        
        with open(metadata_file, 'w') as f:
            json.dump(models_data, f, indent=2, default=str)
        
        print(f"💾 Regime-aware models saved:")
        print(f"   • Metadata: {metadata_file}")
        for regime in self.regime_models:
            print(f"   • {regime} model and scaler")
    
    def load_models(self, filepath_base: str):
        """Load all regime models and scalers"""
        
        metadata_file = f"{filepath_base}_metadata.json"
        with open(metadata_file, 'r') as f:
            models_data = json.load(f)
        
        self.regime_statistics = models_data['regime_statistics']
        self.feature_columns = models_data['feature_columns']
        self.n_clusters_per_regime = models_data['n_clusters_per_regime']
        
        # Load each model
        for regime, model_file in models_data['regime_models'].items():
            with open(model_file, 'rb') as f:
                self.regime_models[regime] = pickle.load(f)
        
        for regime, scaler_file in models_data['regime_scalers'].items():
            with open(scaler_file, 'rb') as f:
                self.regime_scalers[regime] = pickle.load(f)
        
        print(f"✅ Loaded regime-aware models for {len(self.regime_models)} regimes")

def main():
    """Demo of regime-aware clustering system"""
    
    print("🎯 REGIME-AWARE DYNAMIC CLUSTERING SYSTEM")
    print("=" * 50)
    
    # Create sample data for demonstration
    print("📊 Generating sample excursion data...")
    
    # Generate sample market data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='5min')
    market_data = pd.DataFrame({
        'open': 1.0850 + np.random.normal(0, 0.001, len(dates)),
        'high': 1.0850 + np.random.normal(0, 0.001, len(dates)),
        'low': 1.0850 + np.random.normal(0, 0.001, len(dates)),
        'close': 1.0850 + np.random.normal(0, 0.001, len(dates)),
        'volume': np.random.uniform(800, 1200, len(dates))
    }, index=dates)
    
    # Ensure OHLC relationship
    for i in range(len(market_data)):
        market_data.iloc[i, market_data.columns.get_loc('high')] = max(
            market_data.iloc[i, market_data.columns.get_loc('open')],
            market_data.iloc[i, market_data.columns.get_loc('close')]
        ) + abs(np.random.normal(0, 0.0005))
        
        market_data.iloc[i, market_data.columns.get_loc('low')] = min(
            market_data.iloc[i, market_data.columns.get_loc('open')],
            market_data.iloc[i, market_data.columns.get_loc('close')]
        ) - abs(np.random.normal(0, 0.0005))
    
    # Generate sample excursions with features
    n_excursions = 1500
    excursions_data = []
    
    feature_columns = [
        'candles_to_peak', 'volatility', 'momentum', 'trend_strength',
        'rsi', 'macd', 'bb_position', 'body_size', 'wick_ratio',
        'volume_ratio', 'volume_trend', 'hour_sin', 'hour_cos'
    ]
    
    for i in range(n_excursions):
        # Random timestamp from market data
        timestamp = np.random.choice(dates)
        timestamp_dt = pd.to_datetime(timestamp)
        
        # Generate features
        excursion = {
            'timestamp': timestamp,
            'direction': np.random.choice(['UP', 'DOWN']),
            'candles_to_peak': np.random.randint(1, 31),
            'volatility': np.random.uniform(0.1, 1.5),
            'momentum': np.random.uniform(0.1, 1.8),
            'trend_strength': np.random.uniform(0.1, 2.0),
            'rsi': np.random.uniform(20, 80),
            'macd': np.random.uniform(-0.5, 0.5),
            'bb_position': np.random.uniform(0, 1),
            'body_size': np.random.uniform(0.3, 1.2),
            'wick_ratio': np.random.uniform(0.1, 0.8),
            'volume_ratio': np.random.uniform(0.8, 2.5),
            'volume_trend': np.random.uniform(-0.3, 0.3),
            'hour_sin': np.sin(2 * np.pi * timestamp_dt.hour / 24),
            'hour_cos': np.cos(2 * np.pi * timestamp_dt.hour / 24),
            'pips_captured': np.random.uniform(-20, 80),
            'successful': np.random.random() < 0.58  # 58% base win rate
        }
        
        excursions_data.append(excursion)
    
    excursions_df = pd.DataFrame(excursions_data)
    
    print(f"   Generated {len(excursions_df)} sample excursions")
    print(f"   Market data: {len(market_data)} candles")
    
    # Initialize regime-aware system
    regime_system = RegimeAwareClusteringSystem(n_clusters_per_regime=8, min_samples_per_regime=150)
    
    # Prepare features with regime detection
    enhanced_excursions = regime_system.prepare_excursion_features(excursions_df, market_data)
    
    # Train regime-specific models
    results = regime_system.train_regime_models(enhanced_excursions)
    
    # Display results
    print(f"\n📈 REGIME-AWARE CLUSTERING RESULTS")
    print("=" * 45)
    
    overall_best_win_rate = 0
    overall_best_regime = ""
    
    for regime, stats in results.items():
        print(f"\n🎯 {regime}:")
        print(f"   Samples: {stats['total_samples']}")
        print(f"   Model: {stats['model_type']}")
        print(f"   Clusters: {stats['n_clusters']}")
        print(f"   Silhouette Score: {stats['silhouette_score']:.3f}")
        print(f"   Best Cluster Win Rate: {stats['cluster_stats']['best_cluster_win_rate']:.1%}")
        print(f"   Best Cluster Avg Pips: {stats['cluster_stats']['best_cluster_avg_pips']:+.1f}")
        
        if stats['cluster_stats']['best_cluster_win_rate'] > overall_best_win_rate:
            overall_best_win_rate = stats['cluster_stats']['best_cluster_win_rate']
            overall_best_regime = regime
    
    print(f"\n🏆 OVERALL BEST PERFORMING REGIME:")
    print(f"   {overall_best_regime}: {overall_best_win_rate:.1%} win rate")
    
    # Test prediction on new data
    print(f"\n🔮 TESTING REGIME-AWARE PREDICTIONS")
    print("-" * 40)
    
    test_features = {
        'candles_to_peak': 5,
        'volatility': 0.8,
        'momentum': 1.2,
        'trend_strength': 0.9,
        'rsi': 65,
        'macd': 0.3,
        'bb_position': 0.7,
        'body_size': 0.8,
        'wick_ratio': 0.4,
        'volume_ratio': 1.5,
        'volume_trend': 0.1,
        'hour_sin': 0.5,
        'hour_cos': 0.8
    }
    
    for regime in results.keys():
        recommendation = regime_system.get_regime_specific_trading_recommendation(
            test_features, regime
        )
        
        print(f"\n{regime}:")
        print(f"   Cluster: {recommendation['cluster_id']}")
        print(f"   Win Rate: {recommendation['cluster_win_rate']:.1%}")
        print(f"   Position Size: {recommendation['recommended_position_size']:.1%}")
        print(f"   Should Trade: {recommendation['should_trade']}")
        print(f"   Reason: {recommendation['reason']}")
    
    # Save models
    print(f"\n💾 Saving regime-aware models...")
    regime_system.save_models('/Users/jonspinogatti/Desktop/spin36TB/regime_aware_clusters')
    
    print(f"\n✅ REGIME-AWARE CLUSTERING IMPLEMENTATION COMPLETE!")
    print(f"🎯 Key Benefits:")
    print(f"   • Separate optimized models for each market regime")
    print(f"   • Enhanced position sizing based on regime + cluster performance")
    print(f"   • Automatic fallback for unknown regimes")
    print(f"   • Comprehensive confidence scoring")
    
    return regime_system, results

if __name__ == "__main__":
    main()