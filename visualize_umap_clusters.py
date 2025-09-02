#!/usr/bin/env python3
"""
UMAP Visualization of Spin36TB's Excursion Clusters
Shows how well the 30-candle patterns separate into distinct groups
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP, install if not available
try:
    import umap
except ImportError:
    print("Installing UMAP...")
    import subprocess
    subprocess.check_call(["pip", "install", "umap-learn"])
    import umap

class UMAPClusterVisualizer:
    def __init__(self):
        self.excursions = []
        self.patterns = []
        self.scaler = StandardScaler()
        self.kmeans = None
        self.umap_reducer = None
        
    def detect_sample_excursions(self, data, n_samples=2000):
        """
        Detect a sample of excursions for visualization
        Using smaller sample for speed
        """
        print(f"🔍 Detecting Sample Excursions for Visualization...")
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        window_size = 30
        min_excursion_pips = 20
        max_retrace_pct = 0.80
        
        valid_excursions = []
        
        # Sample every 20th candle for speed
        sample_indices = range(0, len(close) - window_size, 20)
        
        for i in sample_indices[:n_samples]:
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
                    elif peak_candle < 15:
                        tier = "Tier 2 (6-15)"
                    else:
                        tier = "Tier 1 (16-30)"
                    
                    excursion_data = {
                        'entry_time': entry_time,
                        'direction': excursion_direction,
                        'peak_candle': peak_candle,
                        'excursion_pips': excursion_pips,
                        'tier': tier,
                        'pattern': pattern
                    }
                    
                    valid_excursions.append(excursion_data)
        
        self.excursions = valid_excursions
        
        print(f"   ✅ Found {len(valid_excursions)} sample excursions")
        
        # Statistics
        up_count = sum(1 for exc in valid_excursions if exc['direction'] == "UP")
        down_count = sum(1 for exc in valid_excursions if exc['direction'] == "DOWN")
        tier_counts = {}
        for exc in valid_excursions:
            tier = exc['tier']
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        print(f"      Up excursions: {up_count} ({up_count/len(valid_excursions):.1%})")
        print(f"      Down excursions: {down_count} ({down_count/len(valid_excursions):.1%})")
        for tier, count in tier_counts.items():
            print(f"      {tier}: {count} ({count/len(valid_excursions):.1%})")
        
        return len(valid_excursions) > 0
    
    def create_pattern(self, window_data, peak_candle, direction):
        """Create normalized pattern from 30-candle window"""
        close_prices = window_data['Close'].values
        
        if len(close_prices) != 30:
            return None
        
        # Normalize to entry price
        normalized_prices = close_prices / close_prices[0]
        
        # Add metadata features
        extended_pattern = np.concatenate([
            normalized_prices,  # 30 normalized prices
            [peak_candle / 30],  # Peak timing (0-1)
            [1.0 if direction == "UP" else -1.0],  # Direction
            [1.0 if peak_candle < 5 else 0.0]  # Early breakout flag
        ])
        
        return extended_pattern
    
    def cluster_patterns(self, n_clusters=8):
        """Cluster the excursion patterns"""
        print(f"\n🎯 Clustering {len(self.excursions)} Patterns...")
        
        if len(self.excursions) < n_clusters:
            print("   ❌ Not enough excursions for clustering")
            return False
        
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
        
        cluster_stats = {}
        for cluster_id in range(n_clusters):
            cluster_excursions = [exc for exc in self.excursions if exc['cluster'] == cluster_id]
            
            if len(cluster_excursions) > 0:
                up_count = sum(1 for exc in cluster_excursions if exc['direction'] == "UP")
                avg_pips = np.mean([exc['excursion_pips'] for exc in cluster_excursions])
                dominant_direction = "UP" if up_count > len(cluster_excursions)/2 else "DOWN"
                
                cluster_stats[cluster_id] = {
                    'size': len(cluster_excursions),
                    'up_pct': up_count / len(cluster_excursions),
                    'avg_pips': avg_pips,
                    'dominant_direction': dominant_direction
                }
                
                print(f"      Cluster {cluster_id}: {len(cluster_excursions)} excursions, "
                      f"{dominant_direction} bias ({up_count/len(cluster_excursions):.1%}), "
                      f"{avg_pips:.1f} pips avg")
        
        self.patterns = patterns_scaled
        return True
    
    def create_umap_visualization(self):
        """Create UMAP visualization of clusters"""
        print(f"\n📊 Creating UMAP Visualization...")
        
        if len(self.patterns) == 0:
            print("   ❌ No patterns to visualize")
            return
        
        # UMAP dimensionality reduction
        print("   🔄 Running UMAP dimensionality reduction...")
        self.umap_reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42
        )
        
        umap_embedding = self.umap_reducer.fit_transform(self.patterns)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Spin36TB\'s Excursion Pattern Clusters - UMAP Visualization', fontsize=16, fontweight='bold')
        
        # Extract visualization data
        clusters = [exc['cluster'] for exc in self.excursions]
        directions = [exc['direction'] for exc in self.excursions]
        tiers = [exc['tier'] for exc in self.excursions]
        pips = [exc['excursion_pips'] for exc in self.excursions]
        
        # 1. Clusters colored by cluster ID
        scatter1 = axes[0,0].scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                                   c=clusters, cmap='Set3', alpha=0.7, s=50)
        axes[0,0].set_title('Clusters by Pattern Similarity', fontweight='bold')
        axes[0,0].set_xlabel('UMAP Dimension 1')
        axes[0,0].set_ylabel('UMAP Dimension 2')
        cbar1 = plt.colorbar(scatter1, ax=axes[0,0])
        cbar1.set_label('Cluster ID')
        
        # 2. Colored by direction (UP/DOWN)
        direction_colors = ['red' if d == 'DOWN' else 'green' for d in directions]
        scatter2 = axes[0,1].scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                                   c=direction_colors, alpha=0.7, s=50)
        axes[0,1].set_title('Excursion Direction', fontweight='bold')
        axes[0,1].set_xlabel('UMAP Dimension 1')
        axes[0,1].set_ylabel('UMAP Dimension 2')
        
        # Add legend for directions
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='UP'),
                          Patch(facecolor='red', label='DOWN')]
        axes[0,1].legend(handles=legend_elements, loc='upper right')
        
        # 3. Colored by timing tier
        tier_color_map = {
            'Tier 3 (1-5)': 'darkred',    # Strongest breakouts
            'Tier 2 (6-15)': 'orange',    # Medium momentum
            'Tier 1 (16-30)': 'lightblue' # Late momentum
        }
        tier_colors = [tier_color_map[t] for t in tiers]
        scatter3 = axes[1,0].scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                                   c=tier_colors, alpha=0.7, s=50)
        axes[1,0].set_title('Spin36TB\'s Timing Tiers', fontweight='bold')
        axes[1,0].set_xlabel('UMAP Dimension 1')
        axes[1,0].set_ylabel('UMAP Dimension 2')
        
        # Add legend for tiers
        tier_legend = [Patch(facecolor=color, label=tier) 
                      for tier, color in tier_color_map.items()]
        axes[1,0].legend(handles=tier_legend, loc='upper right')
        
        # 4. Colored by pip size (intensity)
        scatter4 = axes[1,1].scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                                   c=pips, cmap='viridis', alpha=0.7, s=50)
        axes[1,1].set_title('Excursion Size (Pips)', fontweight='bold')
        axes[1,1].set_xlabel('UMAP Dimension 1')
        axes[1,1].set_ylabel('UMAP Dimension 2')
        cbar4 = plt.colorbar(scatter4, ax=axes[1,1])
        cbar4.set_label('Pips')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = '/Users/jonspinogatti/Desktop/spin36TB/spin36TB_umap_clusters.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ UMAP visualization saved to: {output_path}")
        
        # Show cluster separation quality
        self.analyze_cluster_separation(umap_embedding)
        
        plt.show()
    
    def analyze_cluster_separation(self, embedding):
        """Analyze how well clusters are separated in UMAP space"""
        print(f"\n📊 Cluster Separation Analysis:")
        
        clusters = [exc['cluster'] for exc in self.excursions]
        unique_clusters = list(set(clusters))
        
        # Calculate intra-cluster vs inter-cluster distances
        from sklearn.metrics import silhouette_score
        
        if len(unique_clusters) > 1:
            silhouette = silhouette_score(embedding, clusters)
            print(f"   Silhouette Score: {silhouette:.3f} (higher = better separation)")
            
            if silhouette > 0.5:
                print("   ✅ EXCELLENT cluster separation!")
            elif silhouette > 0.3:
                print("   ✅ GOOD cluster separation")
            elif silhouette > 0.1:
                print("   ⚠️  MODERATE cluster separation")
            else:
                print("   ❌ POOR cluster separation")
        
        # Calculate cluster centroids and distances
        cluster_centroids = {}
        for cluster_id in unique_clusters:
            cluster_points = [embedding[i] for i, exc in enumerate(self.excursions) 
                            if exc['cluster'] == cluster_id]
            if cluster_points:
                cluster_centroids[cluster_id] = np.mean(cluster_points, axis=0)
        
        print(f"   Number of distinct clusters: {len(cluster_centroids)}")
        print(f"   Average cluster size: {len(self.excursions) / len(cluster_centroids):.1f}")
        
        # Find closest cluster pairs (potential overlap)
        min_distance = float('inf')
        closest_pair = None
        
        for i, cluster_a in enumerate(unique_clusters):
            for cluster_b in unique_clusters[i+1:]:
                if cluster_a in cluster_centroids and cluster_b in cluster_centroids:
                    distance = np.linalg.norm(cluster_centroids[cluster_a] - cluster_centroids[cluster_b])
                    if distance < min_distance:
                        min_distance = distance
                        closest_pair = (cluster_a, cluster_b)
        
        if closest_pair:
            print(f"   Closest clusters: {closest_pair[0]} and {closest_pair[1]} (distance: {min_distance:.3f})")

def main():
    """Main visualization pipeline"""
    print("🎯 SPIN36TB'S EXCURSION PATTERN UMAP VISUALIZATION")
    print("=" * 55)
    
    # Load EURUSD data
    try:
        print("📊 Loading EURUSD Data...")
        data_path = "/Users/jonspinogatti/Desktop/spin35TB/data/candles/eurusd_m5_500k.csv"
        data = pd.read_csv(data_path)
        
        # Use subset for visualization (faster)
        data = data.head(50000)  # 50k bars for good sample
        
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
    visualizer = UMAPClusterVisualizer()
    
    # Detect excursions
    if visualizer.detect_sample_excursions(data, n_samples=1500):
        
        # Cluster patterns
        if visualizer.cluster_patterns(n_clusters=8):
            
            # Create UMAP visualization
            visualizer.create_umap_visualization()
            
            print(f"\n🎉 UMAP VISUALIZATION COMPLETE!")
            print(f"   📊 Pattern clustering quality analyzed")
            print(f"   🎯 Visual separation of Spin36TB's excursion types")
            print(f"   💡 Use this to validate cluster performance")

if __name__ == "__main__":
    main()