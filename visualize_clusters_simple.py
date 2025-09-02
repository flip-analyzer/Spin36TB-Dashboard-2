#!/usr/bin/env python3
"""
Cluster Visualization using PCA and t-SNE
Shows Spin36TB's excursion pattern separation without UMAP dependency issues
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ClusterVisualizer:
    def __init__(self):
        self.excursions = []
        self.patterns = []
        self.scaler = StandardScaler()
        self.kmeans = None
        
    def detect_sample_excursions(self, data, n_samples=1000):
        """Detect sample excursions for visualization"""
        print(f"🔍 Detecting Sample Excursions...")
        
        close = data['Close']
        high = data['High'] 
        low = data['Low']
        window_size = 30
        min_excursion_pips = 20
        max_retrace_pct = 0.80
        
        valid_excursions = []
        
        # Sample every 15th candle for speed and variety
        sample_indices = range(0, len(close) - window_size, 15)
        
        for i in sample_indices:
            if len(valid_excursions) >= n_samples:
                break
                
            entry_price = close.iloc[i]
            entry_time = close.index[i]
            
            # Look ahead window
            future_window = data.iloc[i:i + window_size]
            future_highs = future_window['High']
            future_lows = future_window['Low']
            
            # Find maximum movements
            max_high = future_highs.max()
            min_low = future_lows.min()
            
            max_up_pips = (max_high - entry_price) * 10000
            max_down_pips = (entry_price - min_low) * 10000
            
            excursion_found = False
            excursion_direction = None
            peak_candle = None
            excursion_pips = 0
            
            # Check upward excursion
            if max_up_pips >= min_excursion_pips:
                peak_idx = future_highs.idxmax()
                peak_candle_num = list(future_window.index).index(peak_idx)
                
                # Check retracement
                if peak_candle_num < len(future_window) - 1:
                    post_peak_low = future_lows.iloc[peak_candle_num:].min()
                    retracement = (max_high - post_peak_low) / (max_high - entry_price)
                    
                    if retracement <= max_retrace_pct:
                        excursion_found = True
                        excursion_direction = "UP"
                        peak_candle = peak_candle_num
                        excursion_pips = max_up_pips
            
            # Check downward excursion
            elif max_down_pips >= min_excursion_pips:
                peak_idx = future_lows.idxmin()
                peak_candle_num = list(future_window.index).index(peak_idx)
                
                if peak_candle_num < len(future_window) - 1:
                    post_peak_high = future_highs.iloc[peak_candle_num:].max()
                    retracement = (post_peak_high - min_low) / (entry_price - min_low)
                    
                    if retracement <= max_retrace_pct:
                        excursion_found = True
                        excursion_direction = "DOWN"
                        peak_candle = peak_candle_num
                        excursion_pips = max_down_pips
            
            if excursion_found:
                # Create pattern
                pattern = self.create_pattern(future_window, peak_candle, excursion_direction)
                
                if pattern is not None:
                    # Timing tier
                    if peak_candle < 5:
                        tier = "Tier 3 (1-5)"
                        tier_color = 'darkred'
                    elif peak_candle < 15:
                        tier = "Tier 2 (6-15)"
                        tier_color = 'orange'
                    else:
                        tier = "Tier 1 (16-30)"
                        tier_color = 'lightblue'
                    
                    excursion_data = {
                        'entry_time': entry_time,
                        'direction': excursion_direction,
                        'peak_candle': peak_candle,
                        'excursion_pips': excursion_pips,
                        'tier': tier,
                        'tier_color': tier_color,
                        'pattern': pattern
                    }
                    
                    valid_excursions.append(excursion_data)
        
        self.excursions = valid_excursions
        
        # Statistics
        up_count = sum(1 for exc in valid_excursions if exc['direction'] == "UP")
        down_count = sum(1 for exc in valid_excursions if exc['direction'] == "DOWN")
        tier_counts = {}
        for exc in valid_excursions:
            tier = exc['tier']
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        print(f"   ✅ Found {len(valid_excursions)} excursions")
        print(f"      Up excursions: {up_count} ({up_count/len(valid_excursions):.1%})")
        print(f"      Down excursions: {down_count} ({down_count/len(valid_excursions):.1%})")
        for tier, count in tier_counts.items():
            print(f"      {tier}: {count} ({count/len(valid_excursions):.1%})")
        
        return len(valid_excursions) > 10
    
    def create_pattern(self, window_data, peak_candle, direction):
        """Create normalized pattern from 30-candle window"""
        close_prices = window_data['Close'].values
        
        if len(close_prices) != 30:
            return None
        
        # Normalize to entry price
        normalized_prices = close_prices / close_prices[0]
        
        # Spin36TB's insight: Add placeholders for early peaks
        if peak_candle < 5:
            pattern = normalized_prices.copy()
            peak_price = pattern[peak_candle]
            # Replace post-peak with linear decay
            for i in range(peak_candle + 1, 30):
                decay_factor = (i - peak_candle) / (30 - peak_candle)
                pattern[i] = peak_price + (1.0 - peak_price) * decay_factor * 0.5
        else:
            pattern = normalized_prices
        
        # Add metadata features
        extended_pattern = np.concatenate([
            pattern,  # 30 normalized prices
            [peak_candle / 30],  # Peak timing (0-1)
            [1.0 if direction == "UP" else -1.0],  # Direction
            [1.0 if peak_candle < 5 else 0.0]  # Early breakout flag
        ])
        
        return extended_pattern
    
    def cluster_patterns(self, n_clusters=10):
        """Cluster the excursion patterns"""
        print(f"\n🎯 Clustering {len(self.excursions)} Patterns...")
        
        if len(self.excursions) < n_clusters:
            n_clusters = max(3, len(self.excursions) // 5)  # Adjust clusters
            print(f"   Adjusting to {n_clusters} clusters")
        
        # Extract patterns
        patterns = np.array([exc['pattern'] for exc in self.excursions])
        
        # Scale patterns
        patterns_scaled = self.scaler.fit_transform(patterns)
        
        # K-means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(patterns_scaled)
        
        # Add cluster labels to excursions
        for i, excursion in enumerate(self.excursions):
            excursion['cluster'] = cluster_labels[i]
        
        # Cluster statistics
        print(f"   ✅ Clustered into {n_clusters} groups")
        
        for cluster_id in range(n_clusters):
            cluster_excursions = [exc for exc in self.excursions if exc['cluster'] == cluster_id]
            
            if len(cluster_excursions) > 0:
                up_count = sum(1 for exc in cluster_excursions if exc['direction'] == "UP")
                avg_pips = np.mean([exc['excursion_pips'] for exc in cluster_excursions])
                dominant_direction = "UP" if up_count > len(cluster_excursions)/2 else "DOWN"
                
                print(f"      Cluster {cluster_id}: {len(cluster_excursions)} patterns, "
                      f"{dominant_direction} bias ({up_count/len(cluster_excursions):.1%}), "
                      f"{avg_pips:.1f} pips avg")
        
        self.patterns = patterns_scaled
        return True
    
    def create_visualizations(self):
        """Create PCA and t-SNE visualizations"""
        print(f"\n📊 Creating Cluster Visualizations...")
        
        if len(self.patterns) == 0:
            print("   ❌ No patterns to visualize")
            return
        
        # PCA reduction
        print("   🔄 Running PCA...")
        pca = PCA(n_components=2, random_state=42)
        pca_embedding = pca.fit_transform(self.patterns)
        
        # t-SNE reduction (if enough samples)
        tsne_embedding = None
        if len(self.patterns) >= 30:
            print("   🔄 Running t-SNE...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.patterns)-1))
            tsne_embedding = tsne.fit_transform(self.patterns)
        
        # Create comprehensive visualization
        if tsne_embedding is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Spin36TB\'s Excursion Pattern Clusters - Dimensionality Reduction', fontsize=16, fontweight='bold')
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            fig.suptitle('Spin36TB\'s Excursion Pattern Clusters - PCA Analysis', fontsize=16, fontweight='bold')
        
        # Extract visualization data
        clusters = [exc['cluster'] for exc in self.excursions]
        directions = [exc['direction'] for exc in self.excursions]
        tiers = [exc['tier'] for exc in self.excursions]
        tier_colors = [exc['tier_color'] for exc in self.excursions]
        pips = [exc['excursion_pips'] for exc in self.excursions]
        
        # PCA Visualizations
        # 1. PCA - Clusters
        scatter1 = axes[0,0].scatter(pca_embedding[:, 0], pca_embedding[:, 1], 
                                   c=clusters, cmap='Set3', alpha=0.7, s=60)
        axes[0,0].set_title('PCA: Clusters by Pattern Similarity', fontweight='bold')
        axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter1, ax=axes[0,0], label='Cluster ID')
        
        # 2. PCA - Direction
        direction_colors = ['red' if d == 'DOWN' else 'green' for d in directions]
        axes[0,1].scatter(pca_embedding[:, 0], pca_embedding[:, 1], 
                         c=direction_colors, alpha=0.7, s=60)
        axes[0,1].set_title('PCA: Excursion Direction', fontweight='bold')
        axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='UP'),
                          Patch(facecolor='red', label='DOWN')]
        axes[0,1].legend(handles=legend_elements, loc='upper right')
        
        if tsne_embedding is not None:
            # t-SNE Visualizations
            # 3. t-SNE - Clusters
            scatter3 = axes[0,2].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                                       c=clusters, cmap='Set3', alpha=0.7, s=60)
            axes[0,2].set_title('t-SNE: Clusters by Pattern Similarity', fontweight='bold')
            axes[0,2].set_xlabel('t-SNE Dimension 1')
            axes[0,2].set_ylabel('t-SNE Dimension 2')
            plt.colorbar(scatter3, ax=axes[0,2], label='Cluster ID')
            
            # 4. t-SNE - Direction
            direction_colors = ['red' if d == 'DOWN' else 'green' for d in directions]
            axes[1,0].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                            c=direction_colors, alpha=0.7, s=60)
            axes[1,0].set_title('t-SNE: Excursion Direction', fontweight='bold')
            axes[1,0].set_xlabel('t-SNE Dimension 1')
            axes[1,0].set_ylabel('t-SNE Dimension 2')
            axes[1,0].legend(handles=legend_elements, loc='upper right')
            
            # 5. t-SNE - Timing Tiers
            axes[1,1].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                            c=tier_colors, alpha=0.7, s=60)
            axes[1,1].set_title('t-SNE: Spin36TB\'s Timing Tiers', fontweight='bold')
            axes[1,1].set_xlabel('t-SNE Dimension 1')
            axes[1,1].set_ylabel('t-SNE Dimension 2')
            
            # Tier legend
            tier_legend = [
                Patch(facecolor='darkred', label='Tier 3 (1-5 candles)'),
                Patch(facecolor='orange', label='Tier 2 (6-15 candles)'),
                Patch(facecolor='lightblue', label='Tier 1 (16-30 candles)')
            ]
            axes[1,1].legend(handles=tier_legend, loc='upper right')
            
            # 6. t-SNE - Pip Size
            scatter6 = axes[1,2].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                                       c=pips, cmap='viridis', alpha=0.7, s=60)
            axes[1,2].set_title('t-SNE: Excursion Size (Pips)', fontweight='bold')
            axes[1,2].set_xlabel('t-SNE Dimension 1')
            axes[1,2].set_ylabel('t-SNE Dimension 2')
            plt.colorbar(scatter6, ax=axes[1,2], label='Pips')
        
        else:
            # PCA only - use remaining subplots
            # 3. PCA - Timing Tiers
            axes[1,0].scatter(pca_embedding[:, 0], pca_embedding[:, 1], 
                            c=tier_colors, alpha=0.7, s=60)
            axes[1,0].set_title('PCA: Spin36TB\'s Timing Tiers', fontweight='bold')
            axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            
            # Tier legend
            tier_legend = [
                Patch(facecolor='darkred', label='Tier 3 (1-5 candles)'),
                Patch(facecolor='orange', label='Tier 2 (6-15 candles)'),
                Patch(facecolor='lightblue', label='Tier 1 (16-30 candles)')
            ]
            axes[1,0].legend(handles=tier_legend, loc='upper right')
            
            # 4. PCA - Pip Size
            scatter4 = axes[1,1].scatter(pca_embedding[:, 0], pca_embedding[:, 1], 
                                       c=pips, cmap='viridis', alpha=0.7, s=60)
            axes[1,1].set_title('PCA: Excursion Size (Pips)', fontweight='bold')
            axes[1,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[1,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.colorbar(scatter4, ax=axes[1,1], label='Pips')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = '/Users/jonspinogatti/Desktop/spin36TB/spin36TB_cluster_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualization saved to: {output_path}")
        
        # Analyze cluster quality
        self.analyze_clusters()
        
        plt.show()
    
    def analyze_clusters(self):
        """Analyze cluster quality and patterns"""
        print(f"\n📊 Cluster Quality Analysis:")
        
        clusters = [exc['cluster'] for exc in self.excursions]
        unique_clusters = list(set(clusters))
        
        print(f"   Number of clusters: {len(unique_clusters)}")
        print(f"   Total patterns: {len(self.excursions)}")
        print(f"   Average cluster size: {len(self.excursions) / len(unique_clusters):.1f}")
        
        # Direction consistency within clusters
        direction_consistency = []
        pip_consistency = []
        
        for cluster_id in unique_clusters:
            cluster_excursions = [exc for exc in self.excursions if exc['cluster'] == cluster_id]
            
            if len(cluster_excursions) > 1:
                # Direction consistency
                up_count = sum(1 for exc in cluster_excursions if exc['direction'] == "UP")
                direction_purity = max(up_count, len(cluster_excursions) - up_count) / len(cluster_excursions)
                direction_consistency.append(direction_purity)
                
                # Pip variance (lower is better)
                pips = [exc['excursion_pips'] for exc in cluster_excursions]
                pip_std = np.std(pips)
                pip_mean = np.mean(pips)
                pip_cv = pip_std / pip_mean if pip_mean > 0 else 1
                pip_consistency.append(1 - min(pip_cv, 1))  # Convert to consistency score
        
        if direction_consistency:
            avg_direction_consistency = np.mean(direction_consistency)
            avg_pip_consistency = np.mean(pip_consistency)
            
            print(f"   Direction consistency: {avg_direction_consistency:.1%} (higher = better)")
            print(f"   Pip size consistency: {avg_pip_consistency:.1%} (higher = better)")
            
            if avg_direction_consistency > 0.8:
                print("   ✅ EXCELLENT direction clustering!")
            elif avg_direction_consistency > 0.6:
                print("   ✅ GOOD direction clustering")
            else:
                print("   ⚠️  MODERATE direction clustering")
        
        # Show most distinct clusters
        print(f"\n   🎯 Most Distinct Clusters:")
        cluster_distinctiveness = []
        
        for cluster_id in unique_clusters:
            cluster_excursions = [exc for exc in self.excursions if exc['cluster'] == cluster_id]
            
            if len(cluster_excursions) >= 3:
                up_count = sum(1 for exc in cluster_excursions if exc['direction'] == "UP")
                direction_bias = up_count / len(cluster_excursions)
                avg_pips = np.mean([exc['excursion_pips'] for exc in cluster_excursions])
                tier_3_count = sum(1 for exc in cluster_excursions if exc['tier'] == "Tier 3 (1-5)")
                
                distinctiveness = abs(direction_bias - 0.5) + (tier_3_count / len(cluster_excursions))
                
                cluster_distinctiveness.append({
                    'cluster_id': cluster_id,
                    'size': len(cluster_excursions),
                    'direction_bias': direction_bias,
                    'avg_pips': avg_pips,
                    'tier_3_pct': tier_3_count / len(cluster_excursions),
                    'distinctiveness': distinctiveness
                })
        
        # Sort by distinctiveness
        cluster_distinctiveness.sort(key=lambda x: x['distinctiveness'], reverse=True)
        
        for i, cluster in enumerate(cluster_distinctiveness[:5]):
            direction = "UP" if cluster['direction_bias'] > 0.5 else "DOWN"
            print(f"      {i+1}. Cluster {cluster['cluster_id']}: {cluster['size']} patterns, "
                  f"{direction} bias ({cluster['direction_bias']:.1%}), "
                  f"{cluster['avg_pips']:.1f} pips, "
                  f"{cluster['tier_3_pct']:.1%} Tier 3")

def main():
    """Main visualization pipeline"""
    print("🎯 SPIN36TB'S EXCURSION PATTERN VISUALIZATION")
    print("=" * 50)
    
    # Load EURUSD data
    try:
        print("📊 Loading EURUSD Data...")
        data_path = "/Users/jonspinogatti/Desktop/spin35TB/data/candles/eurusd_m5_500k.csv"
        data = pd.read_csv(data_path)
        
        # Use larger subset for better variety
        data = data.head(100000)  # 100k bars
        
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        
        print(f"   ✅ Loaded {len(data):,} EURUSD 5-minute bars")
        
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return
    
    # Initialize visualizer
    visualizer = ClusterVisualizer()
    
    # Detect excursions
    if visualizer.detect_sample_excursions(data, n_samples=200):
        
        # Cluster patterns
        if visualizer.cluster_patterns(n_clusters=8):
            
            # Create visualizations
            visualizer.create_visualizations()
            
            print(f"\n🎉 CLUSTER VISUALIZATION COMPLETE!")
            print(f"   📊 Pattern clustering quality analyzed")
            print(f"   🎯 Visual separation of Spin36TB's excursion types")
            print(f"   💡 Use this to validate system performance")

if __name__ == "__main__":
    main()