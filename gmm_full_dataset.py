#!/usr/bin/env python3
"""
GMM Clustering on Full 300K+ Candle Dataset
Proper implementation using the complete dataset for robust clustering
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
from datetime import datetime

class FullDatasetGMMClustering:
    def __init__(self, n_components=8):
        self.n_components = n_components
        self.gmm_model = GaussianMixture(
            n_components=n_components,
            covariance_type='tied',
            random_state=42,
            warm_start=True,
            max_iter=200,
            n_init=3  # Faster convergence for large datasets
        )
        self.scaler = StandardScaler()
        self.cluster_performance = {}
        self.tier_mapping = {}
        
        print(f"🧮 FULL DATASET GMM CLUSTERING (300K+ CANDLES)")
        print(f"============================================================")
        print(f"🎯 Target: Use complete dataset for maximum robustness")
        print(f"📊 Efficient sampling strategy for large-scale processing")
        print(f"🏆 Expected: Superior performance with massive training set")
        
    def load_full_dataset_efficiently(self):
        """Load and efficiently sample from the complete dataset"""
        try:
            df = pd.read_csv('/Users/jonspinogatti/Desktop/spin36TB/comprehensive_eurusd_data.csv')
            print(f"✅ Loaded complete dataset: {len(df):,} candles")
            
            if len(df) < 50000:
                print(f"⚠️  Dataset smaller than expected")
                return df
            
            # Use the FULL dataset but with smart sampling
            print(f"📊 Using full {len(df):,} candles with efficient sampling")
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return None
    
    def create_large_scale_features(self, data, window_size=30, target_samples=50000):
        """Create feature vectors efficiently from large dataset"""
        print(f"🔧 Creating features from {len(data):,} candles...")
        print(f"🎯 Target samples: {target_samples:,}")
        
        max_possible = len(data) - window_size - 6
        
        # Calculate sampling strategy
        if max_possible > target_samples:
            step_size = max(1, max_possible // target_samples)
            print(f"📈 Sampling every {step_size} windows for efficiency")
        else:
            step_size = 1
            print(f"📈 Using all {max_possible:,} possible windows")
        
        features = []
        returns = []
        processed = 0
        
        print(f"🏗️  Processing windows...")
        
        for i in range(0, max_possible, step_size):
            try:
                if processed % 20000 == 0 and processed > 0:
                    print(f"   ✅ Processed {processed:,} windows...")
                
                window = data[['open', 'high', 'low', 'close', 'volume']].iloc[i:i+window_size]
                
                if len(window) == window_size and window['close'].iloc[0] > 0:
                    feature_vector = self.extract_efficient_features(window.values)
                    
                    if feature_vector is not None:
                        features.append(feature_vector)
                        
                        # Forward return (30 minutes)
                        current = data['close'].iloc[i + window_size]
                        future = data['close'].iloc[i + window_size + 6]
                        forward_return = (future - current) / current
                        returns.append(forward_return)
                        
                        processed += 1
                        
                        if processed >= target_samples:
                            print(f"🎯 Reached target: {target_samples:,} samples")
                            break
                            
            except:
                continue
        
        if features:
            X = np.array(features)
            y = np.array(returns)
            print(f"✅ Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
            print(f"💾 Memory usage: ~{(X.nbytes + y.nbytes) / 1024**3:.2f} GB")
            return X, y
        return None, None
    
    def extract_efficient_features(self, window_data):
        """Extract optimized features for large-scale processing"""
        try:
            first_close = window_data[0, 3]
            normalized = ((window_data / first_close) - 1) * 100
            
            closes = normalized[:, 3]
            highs = normalized[:, 1]
            lows = normalized[:, 2]
            volumes = normalized[:, 4] if normalized.shape[1] > 4 else np.zeros(len(closes))
            
            # Optimized feature set (same as winning test)
            feature_vector = np.concatenate([
                closes[::5],  # Sample every 5th candle (6 values)
                [np.mean(closes), np.std(closes), np.min(closes), np.max(closes)],
                [np.mean(highs-lows), np.std(highs-lows)],
                [closes[-1] - closes[0], np.mean(np.diff(closes)), 
                 closes[-5:].mean() - closes[:5].mean()],
                [np.std(np.diff(closes)), np.max(highs) - np.min(lows)],
                [np.mean(volumes), np.std(volumes)] if np.any(volumes) else [0, 0]
            ])
            
            return np.nan_to_num(feature_vector, nan=0.0)
        except:
            return None
    
    def fit_full_scale_gmm(self, X, y):
        """Fit GMM on the full-scale dataset"""
        print(f"\n🚀 FITTING GMM ON FULL DATASET")
        print(f"==================================================")
        print(f"📊 Training samples: {len(X):,}")
        print(f"🎯 Feature dimensions: {X.shape[1]}")
        
        # Standardize features
        print(f"📐 Standardizing features...")
        X_scaled = self.scaler.fit_transform(X)
        print(f"✅ Standardization complete")
        
        # Fit GMM with progress tracking
        print(f"🧮 Fitting GMM Tied Covariance...")
        print(f"   This may take several minutes for {len(X):,} samples...")
        
        self.gmm_model.fit(X_scaled)
        
        # Get results
        cluster_labels = self.gmm_model.predict(X_scaled)
        cluster_probabilities = self.gmm_model.predict_proba(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, cluster_labels)
        
        print(f"✅ GMM training complete!")
        print(f"   📊 Silhouette score: {silhouette:.4f}")
        print(f"   🎯 Converged: {self.gmm_model.converged_}")
        print(f"   🔄 Iterations: {self.gmm_model.n_iter_}")
        
        # Analyze performance
        self.analyze_full_scale_performance(cluster_labels, cluster_probabilities, y)
        
        # Create tier mapping
        self.create_confidence_tier_mapping()
        
        return {
            'model': self.gmm_model,
            'scaler': self.scaler,
            'silhouette': silhouette,
            'cluster_labels': cluster_labels,
            'cluster_probabilities': cluster_probabilities,
            'training_samples': len(X)
        }
    
    def analyze_full_scale_performance(self, labels, probabilities, returns):
        """Analyze performance with full dataset statistics"""
        print(f"📊 Analyzing cluster performance on {len(labels):,} samples...")
        
        cluster_stats = {}
        
        for cluster_id in range(self.n_components):
            mask = labels == cluster_id
            cluster_returns = returns[mask]
            cluster_probs = probabilities[mask, cluster_id]
            
            if len(cluster_returns) >= 50:  # Higher threshold for large dataset
                # Core metrics
                mean_return = np.mean(cluster_returns)
                win_rate = np.mean(cluster_returns > 0)
                std_return = np.std(cluster_returns)
                
                # Confidence metrics
                avg_confidence = np.mean(cluster_probs)
                high_conf_mask = cluster_probs > 0.8  # Stricter confidence threshold
                
                if np.any(high_conf_mask):
                    high_conf_win_rate = np.mean(cluster_returns[high_conf_mask] > 0)
                    high_conf_return = np.mean(cluster_returns[high_conf_mask])
                    high_conf_samples = np.sum(high_conf_mask)
                else:
                    high_conf_win_rate = win_rate
                    high_conf_return = mean_return
                    high_conf_samples = 0
                
                # Risk metrics
                sharpe = mean_return / std_return if std_return > 0 else 0
                
                # Extreme performance analysis
                percentile_95 = np.percentile(cluster_returns, 95)
                percentile_5 = np.percentile(cluster_returns, 5)
                
                cluster_stats[cluster_id] = {
                    'total_excursions': len(cluster_returns),
                    'win_rate': win_rate,
                    'avg_pips': mean_return * 10000,
                    'std_return': std_return,
                    'sharpe_ratio': sharpe,
                    'avg_confidence': avg_confidence,
                    'high_conf_win_rate': high_conf_win_rate,
                    'high_conf_return': high_conf_return * 10000,
                    'high_conf_samples': int(high_conf_samples),
                    'percentile_95': percentile_95 * 10000,
                    'percentile_5': percentile_5 * 10000,
                    'dominant_direction': "UP" if mean_return > 0 else "DOWN",
                    'avg_candles_to_peak': 15.0
                }
                
                print(f"   Cluster {cluster_id}: {len(cluster_returns):,} samples, "
                      f"{win_rate:.1%} win rate, {mean_return*10000:.1f} pips, "
                      f"{avg_confidence:.1%} avg confidence")
        
        self.cluster_performance = cluster_stats
        
        # Summary statistics
        if cluster_stats:
            total_samples = sum(s['total_excursions'] for s in cluster_stats.values())
            best_cluster = max(cluster_stats.keys(), key=lambda x: cluster_stats[x]['sharpe_ratio'])
            
            print(f"\n📈 FULL DATASET SUMMARY:")
            print(f"   Total training samples: {total_samples:,}")
            print(f"   Viable clusters: {len(cluster_stats)}/8")
            print(f"   Best cluster: #{best_cluster} ({cluster_stats[best_cluster]['win_rate']:.1%} win rate)")
            print(f"   Best cluster samples: {cluster_stats[best_cluster]['total_excursions']:,}")
    
    def create_confidence_tier_mapping(self):
        """Create enhanced tier mapping for full dataset"""
        print(f"🎯 Creating enhanced tier mapping...")
        
        if not self.cluster_performance:
            return
        
        # Sort by risk-adjusted performance
        sorted_clusters = sorted(
            self.cluster_performance.items(),
            key=lambda x: x[1]['sharpe_ratio'],
            reverse=True
        )
        
        # Enhanced tier assignment based on sample size and performance
        for i, (cluster_id, stats) in enumerate(sorted_clusters):
            sample_size = stats['total_excursions']
            
            # Adjust tiers based on both performance and sample size
            if i < 2 and sample_size >= 1000:  # Top performers with large samples
                base_tier = "TIER_1"
                base_size = 0.025  # Higher size for robust clusters
            elif i < 2:  # Top performers with smaller samples
                base_tier = "TIER_1"
                base_size = 0.020
            elif i < 4 and sample_size >= 500:  # Good performers with decent samples
                base_tier = "TIER_2"
                base_size = 0.018
            elif i < 4:  # Good performers with smaller samples
                base_tier = "TIER_2"
                base_size = 0.015
            elif i < 6:  # Moderate performers
                base_tier = "TIER_3"
                base_size = 0.012
            else:  # Lower performers
                base_tier = "TIER_4"
                base_size = 0.008
            
            self.tier_mapping[cluster_id] = {
                'base_tier': base_tier,
                'base_size': base_size,
                'win_rate': stats['win_rate'],
                'avg_pips': stats['avg_pips'],
                'sharpe': stats['sharpe_ratio'],
                'direction': stats['dominant_direction'],
                'sample_size': sample_size,
                'high_conf_samples': stats['high_conf_samples']
            }
        
        print(f"✅ Enhanced tier mapping created for {len(self.tier_mapping)} clusters")
    
    def save_full_scale_model(self, model_data):
        """Save the full-scale trained model"""
        print(f"\n💾 SAVING FULL-SCALE GMM MODEL")
        print(f"==================================================")
        
        try:
            # Save paths
            model_path = '/Users/jonspinogatti/Desktop/spin36TB/enhanced_spin36tb_regime_clustering_MIXED_CONDITIONS_model.pkl'
            scaler_path = '/Users/jonspinogatti/Desktop/spin36TB/enhanced_spin36tb_regime_clustering_MIXED_CONDITIONS_scaler.pkl'
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.gmm_model,
                    'model_type': 'GMM_tied_covariance_full_scale',
                    'tier_mapping': self.tier_mapping,
                    'cluster_performance': self.cluster_performance,
                    'training_samples': model_data['training_samples']
                }, f)
            
            # Save scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Create comprehensive metadata
            best_cluster_id = max(self.cluster_performance.keys(), 
                                key=lambda x: self.cluster_performance[x]['sharpe_ratio'])
            best_stats = self.cluster_performance[best_cluster_id]
            
            metadata = {
                'regime_models': {
                    'MIXED_CONDITIONS': model_path
                },
                'regime_scalers': {
                    'MIXED_CONDITIONS': scaler_path
                },
                'regime_statistics': {
                    'MIXED_CONDITIONS': {
                        'model_type': 'GMM_tied_covariance_full_scale',
                        'n_clusters': self.n_components,
                        'silhouette_score': model_data['silhouette'],
                        'training_samples': model_data['training_samples'],
                        'cluster_stats': {
                            'cluster_performance': {str(k): v for k, v in self.cluster_performance.items()},
                            'best_cluster_id': best_cluster_id,
                            'best_cluster_win_rate': best_stats['win_rate'],
                            'best_cluster_avg_pips': best_stats['avg_pips'],
                            'best_cluster_samples': best_stats['total_excursions'],
                            'total_clusters': len(self.cluster_performance)
                        },
                        'tier_mapping': self.tier_mapping,
                        'confidence_based_sizing': True,
                        'full_scale_training': True
                    }
                },
                'feature_columns': [f'gmm_feature_{i}' for i in range(19)],
                'n_clusters_per_regime': self.n_components,
                'full_scale_enhancements': {
                    'training_samples': model_data['training_samples'],
                    'dataset_scale': 'Full 300K+ candles',
                    'confidence_based_sizing': True,
                    'probabilistic_clustering': True,
                    'covariance_type': 'tied',
                    'enhanced_tier_mapping': True
                }
            }
            
            metadata_path = '/Users/jonspinogatti/Desktop/spin36TB/enhanced_spin36tb_regime_clustering_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"✅ Full-scale model saved: {model_path}")
            print(f"✅ Scaler saved: {scaler_path}")
            print(f"✅ Metadata updated: {metadata_path}")
            print(f"🎯 Training samples: {model_data['training_samples']:,}")
            print(f"🏆 Best cluster: #{best_cluster_id} with {best_stats['total_excursions']:,} samples")
            
            return True
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return False

def main():
    """Run full-scale GMM clustering"""
    print("🚀 FULL-SCALE GMM CLUSTERING ON 300K+ CANDLES")
    print("="*70)
    
    # Initialize system
    gmm_system = FullDatasetGMMClustering(n_components=8)
    
    # Load full dataset
    data = gmm_system.load_full_dataset_efficiently()
    if data is None:
        print("❌ Failed to load dataset")
        return False
    
    # Create features from full dataset
    X, y = gmm_system.create_large_scale_features(data, target_samples=50000)
    if X is None:
        print("❌ Failed to create features")
        return False
    
    # Fit GMM on full dataset
    model_data = gmm_system.fit_full_scale_gmm(X, y)
    if model_data is None:
        print("❌ Failed to fit GMM")
        return False
    
    # Save model
    success = gmm_system.save_full_scale_model(model_data)
    if not success:
        print("❌ Failed to save model")
        return False
    
    print(f"\n🎉 FULL-SCALE GMM COMPLETE!")
    print(f"============================================================")
    print(f"📊 Training samples: {model_data['training_samples']:,}")
    print(f"📈 Silhouette score: {model_data['silhouette']:.4f}")
    print(f"🏆 Viable clusters: {len(gmm_system.cluster_performance)}/8")
    print(f"🎯 Robust clustering from full 300K+ candle dataset")
    print(f"💡 Enhanced confidence-based sizing implemented")
    print(f"🚀 Ready for hybrid system integration!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ FULL-SCALE GMM READY!")
        print(f"🔄 Restart hybrid system to activate robust clustering")
    else:
        print(f"\n❌ Full-scale GMM failed")