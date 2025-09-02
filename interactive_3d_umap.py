#!/usr/bin/env python3
"""
Interactive 3D UMAP Visualization of Spin36TB Momentum Clusters
Allows you to rotate, zoom, and explore clusters in 3D space
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def create_sample_excursion_data(n_samples=2000):
    """Create realistic sample excursion data for visualization"""
    np.random.seed(42)
    
    data = []
    
    # Generate different types of excursions with distinct characteristics
    for i in range(n_samples):
        # Tier distribution (more Tier 3, fewer Tier 1)
        tier_prob = np.random.random()
        if tier_prob < 0.6:
            tier = 3
            candles_to_peak = np.random.randint(1, 6)
        elif tier_prob < 0.85:
            tier = 2
            candles_to_peak = np.random.randint(6, 16)
        else:
            tier = 1
            candles_to_peak = np.random.randint(16, 31)
        
        # Market regime influences
        regime = np.random.choice(['HIGH_VOL_TRENDING', 'HIGH_MOMENTUM', 'STRONG_TREND', 
                                 'MIXED_CONDITIONS', 'LOW_VOL_RANGING'], 
                                p=[0.25, 0.20, 0.15, 0.25, 0.15])
        
        # Generate features based on tier and regime
        if regime == 'HIGH_VOL_TRENDING':
            volatility = np.random.uniform(0.8, 1.5)
            momentum = np.random.uniform(0.6, 1.2)
            trend_strength = np.random.uniform(0.7, 1.3)
        elif regime == 'HIGH_MOMENTUM':
            volatility = np.random.uniform(0.5, 1.0)
            momentum = np.random.uniform(1.0, 1.8)
            trend_strength = np.random.uniform(0.8, 1.4)
        elif regime == 'STRONG_TREND':
            volatility = np.random.uniform(0.3, 0.8)
            momentum = np.random.uniform(0.4, 1.0)
            trend_strength = np.random.uniform(1.2, 2.0)
        elif regime == 'MIXED_CONDITIONS':
            volatility = np.random.uniform(0.4, 1.2)
            momentum = np.random.uniform(0.2, 0.8)
            trend_strength = np.random.uniform(0.3, 0.9)
        else:  # LOW_VOL_RANGING
            volatility = np.random.uniform(0.1, 0.4)
            momentum = np.random.uniform(0.1, 0.3)
            trend_strength = np.random.uniform(0.1, 0.4)
        
        # Technical indicators
        rsi = np.random.uniform(20, 80)
        macd = np.random.uniform(-0.5, 0.5)
        bb_position = np.random.uniform(0, 1)
        
        # Candle patterns (simplified)
        body_size = np.random.uniform(0.3, 1.2)
        wick_ratio = np.random.uniform(0.1, 0.8)
        
        # Volume characteristics
        volume_ratio = np.random.uniform(0.8, 2.5)
        volume_trend = np.random.uniform(-0.3, 0.3)
        
        # Time-based features
        hour_sin = np.sin(2 * np.pi * np.random.randint(0, 24) / 24)
        hour_cos = np.cos(2 * np.pi * np.random.randint(0, 24) / 24)
        
        # Success metrics
        direction = np.random.choice(['UP', 'DOWN'])
        pips_captured = np.random.uniform(20, 80) if np.random.random() < 0.58 else np.random.uniform(-20, -5)
        
        data.append({
            'tier': tier,
            'candles_to_peak': candles_to_peak,
            'regime': regime,
            'volatility': volatility,
            'momentum': momentum,
            'trend_strength': trend_strength,
            'rsi': rsi,
            'macd': macd,
            'bb_position': bb_position,
            'body_size': body_size,
            'wick_ratio': wick_ratio,
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'direction': direction,
            'pips_captured': pips_captured,
            'successful': pips_captured > 0
        })
    
    return pd.DataFrame(data)

def create_3d_umap_visualization():
    """Create interactive 3D UMAP visualization"""
    print("🔄 Generating sample excursion data...")
    df = create_sample_excursion_data(2000)
    
    # Prepare features for UMAP
    feature_columns = [
        'candles_to_peak', 'volatility', 'momentum', 'trend_strength',
        'rsi', 'macd', 'bb_position', 'body_size', 'wick_ratio',
        'volume_ratio', 'volume_trend', 'hour_sin', 'hour_cos'
    ]
    
    print("📊 Scaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[feature_columns])
    
    print("🔬 Running 3D UMAP dimensionality reduction...")
    # Use 3D UMAP for interactive visualization
    umap_3d = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    
    embedding_3d = umap_3d.fit_transform(features_scaled)
    
    print("🎯 Clustering with K-Means...")
    # Cluster the 3D embedding
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embedding_3d)
    
    # Add results to dataframe
    df['umap_x'] = embedding_3d[:, 0]
    df['umap_y'] = embedding_3d[:, 1]
    df['umap_z'] = embedding_3d[:, 2]
    df['cluster'] = clusters
    
    # Calculate cluster statistics
    cluster_stats = df.groupby('cluster').agg({
        'successful': ['count', 'sum', 'mean'],
        'pips_captured': 'mean',
        'tier': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean(),
        'regime': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'MIXED',
        'direction': lambda x: (x == 'UP').mean()
    }).round(3)
    
    cluster_stats.columns = ['total_count', 'successes', 'win_rate', 'avg_pips', 'dominant_tier', 'dominant_regime', 'up_bias']
    
    print("\n📈 CLUSTER ANALYSIS:")
    print("=" * 60)
    for cluster_id in range(8):
        stats = cluster_stats.loc[cluster_id]
        print(f"Cluster {cluster_id}:")
        print(f"   Size: {stats['total_count']:.0f} excursions")
        print(f"   Win Rate: {stats['win_rate']:.1%}")
        print(f"   Avg Pips: {stats['avg_pips']:+.1f}")
        print(f"   Dominant Tier: {stats['dominant_tier']:.0f}")
        print(f"   Dominant Regime: {stats['dominant_regime']}")
        print(f"   UP Bias: {stats['up_bias']:.1%}")
        print()
    
    # Create the interactive 3D plot
    print("🎨 Creating interactive 3D visualization...")
    
    # Color map for clusters
    colors = px.colors.qualitative.Set3
    
    fig = go.Figure()
    
    # Add each cluster as a separate trace
    for cluster_id in range(8):
        cluster_data = df[df['cluster'] == cluster_id]
        stats = cluster_stats.loc[cluster_id]
        
        # Create hover text
        hover_text = []
        for idx, row in cluster_data.iterrows():
            hover_text.append(
                f"Cluster: {cluster_id}<br>" +
                f"Tier: {row['tier']}<br>" +
                f"Regime: {row['regime']}<br>" +
                f"Direction: {row['direction']}<br>" +
                f"Pips: {row['pips_captured']:+.1f}<br>" +
                f"Candles to Peak: {row['candles_to_peak']}<br>" +
                f"Volatility: {row['volatility']:.2f}<br>" +
                f"Momentum: {row['momentum']:.2f}"
            )
        
        fig.add_trace(go.Scatter3d(
            x=cluster_data['umap_x'],
            y=cluster_data['umap_y'],
            z=cluster_data['umap_z'],
            mode='markers',
            marker=dict(
                size=4,
                color=colors[cluster_id % len(colors)],
                opacity=0.7,
                line=dict(width=0.5, color='black')
            ),
            name=f'Cluster {cluster_id} ({stats["win_rate"]:.1%} WR)',
            hovertext=hover_text,
            hoverinfo='text'
        ))
    
    # Add cluster centroids
    centroids_3d = kmeans.cluster_centers_
    fig.add_trace(go.Scatter3d(
        x=centroids_3d[:, 0],
        y=centroids_3d[:, 1],
        z=centroids_3d[:, 2],
        mode='markers',
        marker=dict(
            size=12,
            color='black',
            symbol='diamond',
            opacity=1.0,
            line=dict(width=2, color='white')
        ),
        name='Cluster Centers',
        hovertext=[f'Centroid {i}' for i in range(8)],
        hoverinfo='text'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': '🎯 Interactive 3D UMAP Visualization of Spin36TB Momentum Clusters',
            'x': 0.5,
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='rgba(0,0,0,0)',
            aspectmode='cube'
        ),
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Show the plot
    print("🚀 Launching interactive 3D visualization...")
    fig.show()
    
    # Save as HTML for sharing
    output_file = '/Users/jonspinogatti/Desktop/spin36TB/interactive_3d_clusters.html'
    fig.write_html(output_file)
    print(f"💾 Saved interactive plot to: {output_file}")
    
    return fig, df, cluster_stats

def create_cluster_performance_dashboard(df, cluster_stats):
    """Create additional performance analysis charts"""
    print("📊 Creating cluster performance dashboard...")
    
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Win Rate by Cluster',
            'Average Pips by Cluster', 
            'Cluster Size Distribution',
            'Tier Distribution by Cluster'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'pie'}, {'type': 'bar'}]]
    )
    
    # Win rate chart
    fig.add_trace(
        go.Bar(
            x=[f'Cluster {i}' for i in range(8)],
            y=cluster_stats['win_rate'] * 100,
            name='Win Rate %',
            marker_color='lightgreen'
        ),
        row=1, col=1
    )
    
    # Average pips chart
    fig.add_trace(
        go.Bar(
            x=[f'Cluster {i}' for i in range(8)],
            y=cluster_stats['avg_pips'],
            name='Avg Pips',
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    # Cluster size pie chart
    fig.add_trace(
        go.Pie(
            labels=[f'Cluster {i}' for i in range(8)],
            values=cluster_stats['total_count'],
            name='Size Distribution'
        ),
        row=2, col=1
    )
    
    # Tier distribution
    tier_data = df.groupby(['cluster', 'tier']).size().unstack(fill_value=0)
    for tier in [1, 2, 3]:
        if tier in tier_data.columns:
            fig.add_trace(
                go.Bar(
                    x=[f'Cluster {i}' for i in range(8)],
                    y=tier_data[tier] if tier in tier_data.columns else [0]*8,
                    name=f'Tier {tier}',
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        title='📈 Spin36TB Cluster Performance Analysis',
        height=800,
        showlegend=True
    )
    
    fig.show()
    
    # Save performance dashboard
    performance_file = '/Users/jonspinogatti/Desktop/spin36TB/cluster_performance_dashboard.html'
    fig.write_html(performance_file)
    print(f"💾 Saved performance dashboard to: {performance_file}")
    
    return fig

def main():
    """Main function to create 3D UMAP visualization"""
    print("🎯 SPIN36TB INTERACTIVE 3D CLUSTER VISUALIZATION")
    print("=" * 55)
    
    try:
        # Create 3D visualization
        fig_3d, df, cluster_stats = create_3d_umap_visualization()
        
        # Create performance dashboard
        fig_perf = create_cluster_performance_dashboard(df, cluster_stats)
        
        print("\n✅ VISUALIZATION COMPLETE!")
        print("🎮 Controls:")
        print("   • Click and drag to rotate")
        print("   • Scroll to zoom in/out") 
        print("   • Click legend items to show/hide clusters")
        print("   • Hover over points for details")
        print("\n📁 Files created:")
        print("   • interactive_3d_clusters.html")
        print("   • cluster_performance_dashboard.html")
        print("\n🚀 Both visualizations should have opened in your browser!")
        
        return fig_3d, fig_perf, df, cluster_stats
        
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("💡 Install missing packages with:")
        print("   pip install umap-learn plotly")
        return None
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return None

if __name__ == "__main__":
    main()