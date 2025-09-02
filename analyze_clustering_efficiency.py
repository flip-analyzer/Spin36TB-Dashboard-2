#!/usr/bin/env python3
"""
Clustering Efficiency Analysis
Analyze the separation and performance of existing clustering models
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os

def load_clustering_model():
    """Load existing clustering model and analyze efficiency"""
    print("🔍 CLUSTERING EFFICIENCY ANALYSIS")
    print("=" * 50)
    
    # Load metadata
    metadata_file = '/Users/jonspinogatti/Desktop/spin36TB/enhanced_spin36tb_regime_clustering_metadata.json'
    
    if not os.path.exists(metadata_file):
        print("❌ No clustering metadata found")
        return False
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        regime_stats = metadata['regime_statistics']['MIXED_CONDITIONS']
        cluster_stats = regime_stats['cluster_stats']
        
        print(f"📊 CLUSTERING MODEL OVERVIEW:")
        print(f"   Model type: {regime_stats['model_type']}")
        print(f"   Number of clusters: {regime_stats['n_clusters']}")
        print(f"   Total samples: {regime_stats['total_samples']}")
        print(f"   Silhouette score: {regime_stats['silhouette_score']:.4f}")
        
        return analyze_cluster_performance(cluster_stats, regime_stats)
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def analyze_cluster_performance(cluster_stats, regime_stats):
    """Analyze cluster separation and performance efficiency"""
    
    print(f"\n📈 CLUSTER SEPARATION EFFICIENCY:")
    
    # Silhouette score interpretation
    silhouette = regime_stats['silhouette_score']
    if silhouette > 0.7:
        separation_quality = "Excellent"
        interpretation = "Very distinct, well-separated clusters"
    elif silhouette > 0.5:
        separation_quality = "Good" 
        interpretation = "Reasonable cluster separation"
    elif silhouette > 0.3:
        separation_quality = "Fair"
        interpretation = "Some cluster overlap, moderate separation"
    elif silhouette > 0.1:
        separation_quality = "Poor"
        interpretation = "Significant cluster overlap"
    else:
        separation_quality = "Very Poor"
        interpretation = "Clusters barely distinguishable"
    
    print(f"   Silhouette Score: {silhouette:.4f} ({separation_quality})")
    print(f"   Interpretation: {interpretation}")
    
    # Analyze individual cluster performance
    print(f"\n🎯 INDIVIDUAL CLUSTER ANALYSIS:")
    
    cluster_performance = cluster_stats['cluster_performance']
    clusters_data = []
    
    for cluster_id, stats in cluster_performance.items():
        clusters_data.append({
            'cluster_id': int(cluster_id),
            'samples': stats['total_excursions'],
            'win_rate': stats['win_rate'],
            'avg_pips': stats['avg_pips'],
            'direction': stats['dominant_direction'],
            'avg_candles': stats['avg_candles_to_peak']
        })
    
    # Sort by win rate
    clusters_data.sort(key=lambda x: x['win_rate'], reverse=True)
    
    print(f"   Cluster Performance (sorted by win rate):")
    print(f"   {'ID':<3} {'Samples':<8} {'Win Rate':<9} {'Avg Pips':<9} {'Direction':<9} {'Avg Hold':<8}")
    print(f"   {'-'*3} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*8}")
    
    for cluster in clusters_data:
        print(f"   {cluster['cluster_id']:<3} {cluster['samples']:<8} "
              f"{cluster['win_rate']:.1%}    {cluster['avg_pips']:<8.1f} "
              f"{cluster['direction']:<9} {cluster['avg_candles']:<7.1f}c")
    
    # Calculate efficiency metrics
    print(f"\n⚡ CLUSTERING EFFICIENCY METRICS:")
    
    win_rates = [c['win_rate'] for c in clusters_data]
    pip_returns = [c['avg_pips'] for c in clusters_data]
    sample_sizes = [c['samples'] for c in clusters_data]
    
    # Performance spread (indicates how well clusters differentiate performance)
    win_rate_spread = max(win_rates) - min(win_rates)
    pip_spread = max(pip_returns) - min(pip_returns)
    
    # Weighted average performance
    total_samples = sum(sample_sizes)
    weighted_win_rate = sum(c['win_rate'] * c['samples'] for c in clusters_data) / total_samples
    weighted_avg_pips = sum(c['avg_pips'] * c['samples'] for c in clusters_data) / total_samples
    
    print(f"   Win Rate Spread: {win_rate_spread:.1%} (higher = better differentiation)")
    print(f"   Pip Return Spread: {pip_spread:.1f} pips (higher = better differentiation)")
    print(f"   Weighted Avg Win Rate: {weighted_win_rate:.1%}")
    print(f"   Weighted Avg Pip Return: {weighted_avg_pips:.1f} pips")
    
    # Best vs worst cluster comparison
    best_cluster = clusters_data[0]
    worst_cluster = clusters_data[-1]
    
    print(f"\n🏆 BEST vs WORST CLUSTER COMPARISON:")
    print(f"   Best Cluster #{best_cluster['cluster_id']}:")
    print(f"      Win Rate: {best_cluster['win_rate']:.1%}")
    print(f"      Avg Return: {best_cluster['avg_pips']:.1f} pips")
    print(f"      Sample Size: {best_cluster['samples']} patterns")
    
    print(f"   Worst Cluster #{worst_cluster['cluster_id']}:")
    print(f"      Win Rate: {worst_cluster['win_rate']:.1%}")
    print(f"      Avg Return: {worst_cluster['avg_pips']:.1f} pips")
    print(f"      Sample Size: {worst_cluster['samples']} patterns")
    
    performance_improvement = (best_cluster['win_rate'] - worst_cluster['win_rate']) / worst_cluster['win_rate']
    print(f"   Performance Improvement: {performance_improvement:.1%} (best vs worst)")
    
    # Clustering quality assessment
    print(f"\n📋 CLUSTERING QUALITY ASSESSMENT:")
    
    # Balance check (are clusters reasonably sized?)
    min_size = min(sample_sizes)
    max_size = max(sample_sizes)
    size_balance = min_size / max_size
    
    print(f"   Cluster Size Balance: {size_balance:.2f} (1.0 = perfectly balanced)")
    print(f"   Size Range: {min_size} - {max_size} samples per cluster")
    
    # Directional diversity (do we have both UP and DOWN signals?)
    directions = [c['direction'] for c in clusters_data]
    up_clusters = directions.count('UP')
    down_clusters = directions.count('DOWN')
    
    print(f"   Directional Diversity:")
    print(f"      UP clusters: {up_clusters}")
    print(f"      DOWN clusters: {down_clusters}")
    print(f"      Balance: {min(up_clusters, down_clusters) / max(up_clusters, down_clusters):.2f}")
    
    # Overall efficiency score
    efficiency_score = calculate_overall_efficiency(silhouette, win_rate_spread, size_balance, 
                                                  weighted_win_rate, performance_improvement)
    
    print(f"\n🎯 OVERALL CLUSTERING EFFICIENCY:")
    print(f"   Composite Efficiency Score: {efficiency_score:.1%}")
    
    if efficiency_score > 0.8:
        overall_rating = "Excellent"
        recommendation = "Clusters are highly effective for trading signals"
    elif efficiency_score > 0.6:
        overall_rating = "Good"
        recommendation = "Clusters provide meaningful signal differentiation"
    elif efficiency_score > 0.4:
        overall_rating = "Fair"
        recommendation = "Clusters show some signal value but could be improved"
    else:
        overall_rating = "Poor"
        recommendation = "Consider re-clustering or different approach"
    
    print(f"   Overall Rating: {overall_rating}")
    print(f"   Recommendation: {recommendation}")
    
    return True

def calculate_overall_efficiency(silhouette, win_rate_spread, size_balance, 
                               weighted_win_rate, performance_improvement):
    """Calculate composite efficiency score"""
    
    # Normalize metrics to 0-1 scale
    silhouette_norm = min(1.0, max(0.0, silhouette * 5))  # 0.2 silhouette = 1.0 score
    spread_norm = min(1.0, win_rate_spread * 5)  # 20% spread = 1.0 score
    balance_norm = size_balance  # Already 0-1
    performance_norm = min(1.0, max(0.0, (weighted_win_rate - 0.5) * 2))  # 50% win rate = 0, 75% = 1.0
    improvement_norm = min(1.0, max(0.0, performance_improvement * 2))  # 50% improvement = 1.0
    
    # Weighted composite score
    efficiency = (
        silhouette_norm * 0.25 +      # 25% - cluster separation
        spread_norm * 0.25 +          # 25% - performance differentiation  
        balance_norm * 0.15 +         # 15% - cluster balance
        performance_norm * 0.20 +     # 20% - overall performance
        improvement_norm * 0.15       # 15% - best vs worst improvement
    )
    
    return efficiency

if __name__ == "__main__":
    success = load_clustering_model()
    
    if success:
        print(f"\n✅ Clustering analysis complete!")
    else:
        print(f"\n❌ Could not analyze clustering model")