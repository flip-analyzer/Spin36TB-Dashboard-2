#!/usr/bin/env python3
"""
Visualize Pattern Clusters for EURUSD 5-minute Data

This shows the cluster separation we're getting from 30-candle pattern windows
using dimensionality reduction techniques to visualize the patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PatternClusterVisualizer:
    def __init__(self, window_size=30, max_patterns=5000):
        self.window_size = window_size
        self.max_patterns = max_patterns
        self.patterns = []
        self.outcomes = []
        self.timestamps = []
        
    def load_eurusd_data(self):
        """Load your EURUSD data"""
        print("📊 Loading EURUSD Data for Cluster Visualization...")
        
        try:
            data_path = "/Users/jonspinogatti/Desktop/spin35TB/data/candles/eurusd_m5_500k.csv"
            data = pd.read_csv(data_path)
            
            data['time'] = pd.to_datetime(data['time'])
            data = data.set_index('time')
            data = data.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume'
            })
            
            print(f"   ✅ Loaded {len(data):,} bars")
            return data
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
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
        
        ranges = (highs - lows) / closes
        bodies = np.abs(closes - opens) / closes
        upper_wicks = (highs - np.maximum(opens, closes)) / closes
        lower_wicks = (np.minimum(opens, closes) - lows) / closes
        
        pattern_vector = np.concatenate([
            closes, ranges, bodies, upper_wicks, lower_wicks
        ])
        
        return pattern_vector
    
    def extract_patterns_for_visualization(self, data):
        """Extract patterns for cluster visualization"""
        print(f"\n🧠 Extracting Patterns for Visualization...")
        print(f"   Target patterns: {self.max_patterns:,}")
        
        self.patterns = []
        self.outcomes = []
        self.timestamps = []
        
        total_possible = len(data) - self.window_size - 6
        step_size = max(1, total_possible // self.max_patterns)
        
        patterns_created = 0
        
        for i in range(0, total_possible, step_size):
            if patterns_created >= self.max_patterns:
                break
                
            pattern_start = i
            pattern_end = i + self.window_size
            pattern_window = data.iloc[pattern_start:pattern_end]
            
            pattern_vector = self.create_pattern_vector(pattern_window)
            if pattern_vector is None:
                continue
                
            # Calculate 30-minute outcome
            entry_price = pattern_window['Close'].iloc[-1]
            outcome_idx = pattern_end + 6 - 1  # 6 bars = 30 minutes
            
            if outcome_idx < len(data):
                exit_price = data.iloc[outcome_idx]['Close']
                outcome_return = (exit_price / entry_price) - 1
                
                self.patterns.append(pattern_vector)
                self.outcomes.append(outcome_return)
                self.timestamps.append(pattern_window.index[-1])
                patterns_created += 1
        
        self.patterns = np.array(self.patterns)
        self.outcomes = np.array(self.outcomes)
        
        print(f"   ✅ Extracted {len(self.patterns):,} patterns")
        return len(self.patterns) > 0
    
    def create_outcome_labels(self):
        """Create categorical labels based on outcomes"""
        labels = []
        
        for outcome in self.outcomes:
            if outcome > 0.001:  # >10 pips up
                labels.append('Strong Up')
            elif outcome > 0.0005:  # 5-10 pips up
                labels.append('Weak Up')
            elif outcome < -0.001:  # >10 pips down
                labels.append('Strong Down')
            elif outcome < -0.0005:  # 5-10 pips down
                labels.append('Weak Down')
            else:
                labels.append('Neutral')
        
        return np.array(labels)
    
    def visualize_clusters_pca(self):
        """Visualize patterns using PCA"""
        print(f"\n📊 Creating PCA Visualization...")
        
        # Standardize patterns
        scaler = StandardScaler()
        patterns_scaled = scaler.fit_transform(self.patterns)
        
        # Apply PCA
        pca = PCA(n_components=2)
        patterns_pca = pca.fit_transform(patterns_scaled)
        
        # Create outcome labels
        outcome_labels = self.create_outcome_labels()
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Color map for different outcomes
        colors = {
            'Strong Up': 'darkgreen',
            'Weak Up': 'lightgreen', 
            'Neutral': 'gray',
            'Weak Down': 'lightcoral',
            'Strong Down': 'darkred'
        }
        
        for label in colors.keys():
            mask = outcome_labels == label
            if mask.sum() > 0:
                plt.scatter(patterns_pca[mask, 0], patterns_pca[mask, 1], 
                           c=colors[label], label=f"{label} ({mask.sum()})", 
                           alpha=0.6, s=20)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('EURUSD Pattern Clusters (PCA Visualization)\n30-Candle Windows Colored by 30-Minute Outcome')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        print(f"   ✅ PCA explains {(pca.explained_variance_ratio_[:2].sum())*100:.1f}% of variance")
        
        return pca, patterns_pca
    
    def visualize_clusters_tsne(self, patterns_pca=None):
        """Visualize patterns using t-SNE"""
        print(f"\n🎯 Creating t-SNE Visualization...")
        
        # Use PCA-reduced data if available (faster)
        if patterns_pca is not None:
            print("   Using PCA-reduced data for t-SNE (faster)")
            input_data = patterns_pca
        else:
            scaler = StandardScaler()
            input_data = scaler.fit_transform(self.patterns)
        
        # Apply t-SNE with faster settings
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(input_data)//4), n_iter=300)
        patterns_tsne = tsne.fit_transform(input_data)
        
        # Create outcome labels
        outcome_labels = self.create_outcome_labels()
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        colors = {
            'Strong Up': 'darkgreen',
            'Weak Up': 'lightgreen',
            'Neutral': 'gray', 
            'Weak Down': 'lightcoral',
            'Strong Down': 'darkred'
        }
        
        for label in colors.keys():
            mask = outcome_labels == label
            if mask.sum() > 0:
                plt.scatter(patterns_tsne[mask, 0], patterns_tsne[mask, 1],
                           c=colors[label], label=f"{label} ({mask.sum()})",
                           alpha=0.6, s=20)
        
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2') 
        plt.title('EURUSD Pattern Clusters (t-SNE Visualization)\n30-Candle Windows Colored by 30-Minute Outcome')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return patterns_tsne
    
    def analyze_cluster_kmeans(self, n_clusters=8):
        """Analyze patterns using K-means clustering"""
        print(f"\n🔬 K-Means Cluster Analysis (k={n_clusters})...")
        
        # Standardize and apply PCA first
        scaler = StandardScaler()
        patterns_scaled = scaler.fit_transform(self.patterns)
        
        pca = PCA(n_components=20)  # Keep more components for clustering
        patterns_pca = pca.fit_transform(patterns_scaled)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(patterns_pca)
        
        # Analyze each cluster
        print(f"   📊 Cluster Analysis:")
        
        cluster_stats = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_outcomes = self.outcomes[cluster_mask]
            
            if len(cluster_outcomes) > 0:
                avg_outcome = cluster_outcomes.mean()
                std_outcome = cluster_outcomes.std()
                positive_rate = (cluster_outcomes > 0.0005).mean()  # >5 pips
                negative_rate = (cluster_outcomes < -0.0005).mean() # <-5 pips
                
                cluster_stats.append({
                    'cluster': i,
                    'size': cluster_mask.sum(),
                    'avg_outcome': avg_outcome,
                    'std_outcome': std_outcome,
                    'positive_rate': positive_rate,
                    'negative_rate': negative_rate
                })
                
                print(f"      Cluster {i}: {cluster_mask.sum():4d} patterns, "
                      f"avg={avg_outcome:+.4f}, pos={positive_rate:.2%}, neg={negative_rate:.2%}")
        
        # Visualize clusters
        pca_viz = PCA(n_components=2)
        patterns_viz = pca_viz.fit_transform(patterns_scaled)
        
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(patterns_viz[:, 0], patterns_viz[:, 1], 
                             c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'K-Means Clusters (k={n_clusters})\n30-Candle EURUSD Patterns')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return cluster_stats, cluster_labels
    
    def plot_sample_patterns(self, n_samples=6):
        """Plot sample pattern shapes"""
        print(f"\n📈 Plotting Sample Pattern Shapes...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Select diverse patterns based on outcomes
        outcome_labels = self.create_outcome_labels()
        unique_labels = np.unique(outcome_labels)
        
        sample_indices = []
        for label in unique_labels[:n_samples]:
            label_indices = np.where(outcome_labels == label)[0]
            if len(label_indices) > 0:
                # Pick a random sample from this category
                sample_idx = np.random.choice(label_indices)
                sample_indices.append(sample_idx)
        
        for i, idx in enumerate(sample_indices):
            if i >= n_samples:
                break
                
            # Extract the closes from the pattern (first 30 elements)
            pattern_closes = self.patterns[idx][:30]
            outcome = self.outcomes[idx]
            label = outcome_labels[idx]
            
            axes[i].plot(range(30), pattern_closes, 'b-', linewidth=2)
            axes[i].set_title(f'{label}\nOutcome: {outcome:+.4f} ({outcome*10000:+.1f} pips)')
            axes[i].set_xlabel('5-Minute Bars')
            axes[i].set_ylabel('Normalized Price')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(sample_indices), n_samples):
            axes[i].set_visible(False)
        
        plt.suptitle('Sample 30-Candle Pattern Shapes\n(2.5 Hours of EURUSD Price Action)', fontsize=14)
        plt.tight_layout()
        
    def create_outcome_distribution_plot(self):
        """Plot distribution of 30-minute outcomes"""
        print(f"\n📊 Creating Outcome Distribution Plot...")
        
        plt.figure(figsize=(12, 6))
        
        # Convert to pips for easier interpretation
        outcomes_pips = self.outcomes * 10000
        
        plt.subplot(1, 2, 1)
        plt.hist(outcomes_pips, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('30-Minute Outcome (Pips)')
        plt.ylabel('Frequency')
        plt.title('Distribution of 30-Minute Outcomes')
        plt.axvline(0, color='red', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_pips = outcomes_pips.mean()
        std_pips = outcomes_pips.std()
        plt.text(0.05, 0.95, f'Mean: {mean_pips:.1f} pips\nStd: {std_pips:.1f} pips', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.subplot(1, 2, 2)
        
        # Categorize outcomes
        outcome_labels = self.create_outcome_labels()
        label_counts = pd.Series(outcome_labels).value_counts()
        
        colors = ['darkgreen', 'lightgreen', 'gray', 'lightcoral', 'darkred']
        wedges, texts, autotexts = plt.pie(label_counts.values, labels=label_counts.index, 
                                          autopct='%1.1f%%', colors=colors[:len(label_counts)])
        plt.title('Outcome Categories\n(30-Minute Moves)')
        
        plt.tight_layout()


def main():
    """Run cluster visualization"""
    
    print("🎯 EURUSD PATTERN CLUSTER VISUALIZATION")
    print("=" * 50)
    print("Analyzing cluster separation of 30-candle patterns")
    
    # Initialize visualizer (reduced patterns for faster processing)
    viz = PatternClusterVisualizer(window_size=30, max_patterns=2000)
    
    # Load data
    data = viz.load_eurusd_data()
    if data is None:
        return
    
    # Extract patterns
    if not viz.extract_patterns_for_visualization(data):
        print("❌ Failed to extract patterns")
        return
    
    # Create all visualizations
    print(f"\n🎨 Creating Visualizations...")
    
    # 1. PCA visualization
    pca, patterns_pca = viz.visualize_clusters_pca()
    
    # 2. t-SNE visualization  
    patterns_tsne = viz.visualize_clusters_tsne(patterns_pca)
    
    # 3. K-means cluster analysis
    cluster_stats, cluster_labels = viz.analyze_cluster_kmeans(n_clusters=8)
    
    # 4. Sample pattern shapes
    viz.plot_sample_patterns(n_samples=6)
    
    # 5. Outcome distribution
    viz.create_outcome_distribution_plot()
    
    # Save individual plots
    print("   💾 Saving individual visualizations...")
    
    # Save PCA plot (figure should be active from visualize_clusters_pca)
    plt.figure(1)
    plt.savefig('/Users/jonspinogatti/Desktop/spin36TB/pca_clusters.png', dpi=300, bbox_inches='tight')
    
    # Save t-SNE plot  
    plt.figure(2)
    plt.savefig('/Users/jonspinogatti/Desktop/spin36TB/tsne_clusters.png', dpi=300, bbox_inches='tight')
    
    # Save K-means plot
    plt.figure(3)
    plt.savefig('/Users/jonspinogatti/Desktop/spin36TB/kmeans_clusters.png', dpi=300, bbox_inches='tight')
    
    # Save sample patterns
    plt.figure(4)
    plt.savefig('/Users/jonspinogatti/Desktop/spin36TB/sample_patterns.png', dpi=300, bbox_inches='tight')
    
    # Save outcome distribution (current plot)
    plt.savefig('/Users/jonspinogatti/Desktop/spin36TB/outcome_distribution.png', dpi=300, bbox_inches='tight')
    
    print("   💾 All visualizations saved as individual PNG files")
    
    print(f"\n✅ Cluster Visualization Complete!")
    print(f"   📊 {len(viz.patterns):,} patterns analyzed")
    print(f"   🎯 Check the plots to see cluster separation")
    
    # Summary statistics
    outcome_labels = viz.create_outcome_labels()
    unique, counts = np.unique(outcome_labels, return_counts=True)
    
    print(f"\n📈 Pattern Outcome Summary:")
    for label, count in zip(unique, counts):
        pct = count / len(outcome_labels) * 100
        print(f"   {label}: {count:,} patterns ({pct:.1f}%)")


if __name__ == "__main__":
    main()