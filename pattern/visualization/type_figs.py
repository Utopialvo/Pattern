# Файл: visualization/type_figs.py
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from pathlib import Path
from collections import defaultdict

class BaseVisualizer:
    """Base class for visualizers with common save functionality."""
    def __init__(self, save_path=None):
        self.save_path = save_path
        self._validate_save_path()
    
    def _validate_save_path(self):
        """Create save directory if specified."""
        if self.save_path is not None:
            Path(self.save_path).mkdir(parents=True, exist_ok=True)
    
    def _save_figure(self, fig, name):
        """Save figure to specified path."""
        if self.save_path is not None:
            path = Path(self.save_path) / f"{name}.png"
            fig.savefig(path, bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig)

class FeaturesVisualizer(BaseVisualizer):
    """Visualizes feature distributions and relationships."""
    
    def pairplot(self, df, labels):
        """Pairwise feature relationships with cluster coloring."""
        df_plot = df.assign(cluster=labels.astype(str))
        g = sns.pairplot(df_plot, hue='cluster', palette='viridis', corner=True)
        g.fig.suptitle('Pairwise Feature Relationships', y=1.02)
        self._save_figure(g.fig, 'pairplot')
    
    def radar_chart(self, df, labels):
        """Radar chart of normalized cluster profiles."""
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df_scaled['cluster'] = labels
        
        cluster_means = df_scaled.groupby('cluster').mean().reset_index()
        features = df.columns.tolist()
        n_features = len(features)
        
        angles = np.linspace(0, 2*np.pi, n_features, endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})
        
        for _, row in cluster_means.iterrows():
            values = row[features].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=f'Cluster {int(row["cluster"])}')
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=8)
        ax.set_title('Radar Chart of Cluster Profiles', pad=20)
        ax.legend(bbox_to_anchor=(1.15, 1.1))
        self._save_figure(fig, 'radar_chart')
    
    def violin_plots(self, df, labels):
        """Violin plots per feature across clusters."""
        df_plot = df.assign(cluster=labels)
        n_features = len(df.columns)
        cols = 3
        rows = (n_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        axes = axes.ravel()
        
        for i, col in enumerate(df.columns):
            sns.violinplot(
                x='cluster', y=col, data=df_plot, 
                hue='cluster', legend=False, ax=axes[i]
            )
            axes[i].set_title(col)
        
        for j in range(i+1, len(axes)):
            axes[j].remove()
        
        plt.tight_layout()
        self._save_figure(fig, 'violin_plots')

class GraphVisualizer(BaseVisualizer):
    """Visualizes graph structures and connectivity patterns."""
    
    def force_directed(self, adj_matrix, labels):
        """Force-directed graph layout with cluster coloring."""
        G = nx.from_numpy_array(adj_matrix)
        
        if len(G) > 2000:
            sample_nodes = np.random.choice(G.nodes(), 2000, replace=False)
            G = G.subgraph(sample_nodes)
            labels = labels[sample_nodes]
        
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(15, 12))
        
        nx.draw_networkx_nodes(
            G, pos, node_color=labels, cmap='tab20',
            node_size=20, alpha=0.8, ax=ax
        )
        nx.draw_networkx_edges(
            G, pos, alpha=0.02, edge_color='grey',
            width=0.5, ax=ax
        )
        
        unique_labels = np.unique(labels)
        cmap = matplotlib.colormaps['tab20'].resampled(len(unique_labels))
        
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
            markerfacecolor=cmap(i), markersize=10,
            label=f'Cluster {label}')
            for i, label in enumerate(unique_labels)
        ]
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title('Force-Directed Graph Layout')
        self._save_figure(fig, 'force_directed')
    
    def adjacency_heatmap(self, adj_matrix, labels):
        """Sorted adjacency matrix heatmap."""
        order = np.argsort(labels)
        sorted_labels = labels[order]
        sorted_matrix = adj_matrix[np.ix_(order, order)]
    
        fig, ax = plt.subplots(figsize=(14, 12))
        
    
        sns.heatmap(
            sorted_matrix, cmap='Blues', 
            xticklabels=False, yticklabels=False,
            cbar_kws={'label': 'Connection Strength'}, 
            ax=ax
        )
        
        boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 1
        boundaries = np.concatenate([[0], boundaries, [len(labels)]])
        
        unique_clusters = np.unique(sorted_labels)
        cmap = matplotlib.colormaps['tab20'].resampled(len(unique_clusters))
        
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            cluster_color = cmap(i)
            
            rect = plt.Rectangle(
                (start, start), 
                end-start, end-start,
                fill=True, alpha=0.1,
                edgecolor='none', 
                facecolor=cluster_color,
                zorder=2
            )
            ax.add_patch(rect)
            
            if start > 0:
                ax.plot(
                    [start, start], [start, len(labels)],
                    color=cluster_color, 
                    linewidth=2, 
                    linestyle='--',
                    zorder=3
                )
                ax.plot(
                    [start, len(labels)], [start, start],
                    color=cluster_color, 
                    linewidth=2, 
                    linestyle='--',
                    zorder=3
                )
    
        legend_handles = [
            plt.Line2D([0], [0], 
                      color=cmap(i), 
                      lw=4,
                      label=f'Cluster {cluster}')
            for i, cluster in enumerate(unique_clusters)
        ]
        ax.legend(
            handles=legend_handles, 
            bbox_to_anchor=(1.33, 1), 
            title='Clusters', loc='upper right'
        )
        ax.set_title('Sorted Adjacency Matrix Heatmap', pad=20)
        self._save_figure(fig, 'adjacency_heatmap')
    
    def edge_bundling(self, adj_matrix, labels):
        """Hierarchical edge bundling between clusters."""
        G = nx.from_numpy_array(adj_matrix)
        communities = labels
        
        community_edges = defaultdict(float)
        for u, v, data in G.edges(data=True):
            c1, c2 = communities[u], communities[v]
            if c1 != c2:
                community_edges[(c1, c2)] += data.get('weight', 1.0)
        
        community_graph = nx.Graph()
        unique_labels = np.unique(labels)
        community_graph.add_nodes_from(unique_labels)
        
        for (c1, c2), weight in community_edges.items():
            community_graph.add_edge(c1, c2, weight=np.log1p(weight))
        
        fig, ax = plt.subplots(figsize=(12, 12))
        pos = nx.circular_layout(community_graph)
        weights = [d['weight'] for _, _, d in community_graph.edges(data=True)]
        
        edges = nx.draw_networkx_edges(
            community_graph, pos, width=np.array(weights)*0.5,
            edge_color=weights, edge_cmap=plt.cm.viridis,
            edge_vmin=min(weights, default=0), edge_vmax=max(weights, default=1),
            alpha=0.8, ax=ax
        )
        
        if weights:
            plt.colorbar(edges, ax=ax, shrink=0.8).set_label('Log-scaled Weight')
        
        nx.draw_networkx_nodes(
            community_graph, pos, node_size=1500,
            node_color='skyblue', edgecolors='darkblue',
            linewidths=2, ax=ax
        )
        nx.draw_networkx_labels(community_graph, pos, font_size=12, ax=ax)
        ax.set_title('Hierarchical Edge Bundling')
        self._save_figure(fig, 'edge_bundling')
    
    def aggregated_graph(self, adj_matrix, labels):
        """Aggregated cluster connectivity visualization."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        

        agg_matrix = np.zeros((n_clusters, n_clusters))
        for i, c1 in enumerate(unique_labels):
            for j, c2 in enumerate(unique_labels):
                mask = (labels == c1)[:, None] & (labels == c2)
                agg_matrix[i, j] = np.log1p(np.sum(adj_matrix[mask]))
        

        G = nx.from_numpy_array(agg_matrix)
        pos = nx.spring_layout(G, seed=42)
        

        edge_weights = MinMaxScaler().fit_transform(agg_matrix.reshape(-1,1)).flatten()
        node_sizes = MinMaxScaler().fit_transform(
            np.array([np.sum(labels == l) for l in unique_labels]).reshape(-1,1)
        ).flatten() * 2000 + 500
        
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = matplotlib.colormaps['tab20'].resampled(n_clusters)
        

        edges = nx.draw_networkx_edges(
            G, pos, width=edge_weights*6,
            edge_color=[cmap(i) for i in range(n_clusters)], edge_cmap=plt.cm.plasma,
            alpha=0.7, ax=ax
        )
        if not np.all(edge_weights == 0):
            plt.colorbar(edges, ax=ax, shrink=0.8).set_label('Normalized Weight')
        
        nx.draw_networkx_nodes(
            G, pos, node_size=node_sizes,
            node_color=[cmap(i) for i in range(n_clusters)],
            edgecolors='black', linewidths=1, ax=ax
        )
        nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
        

        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
            markerfacecolor=cmap(i), markersize=8,
            label=f'Cluster {label} (n={np.sum(labels == label)})')
            for i, label in enumerate(unique_labels)
        ]
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.15, 1))
        ax.set_title('Aggregated Cluster Graph')
        self._save_figure(fig, 'aggregated_graph')

class CombinedVisualizer(FeaturesVisualizer, GraphVisualizer):
    """Combined feature and graph visualization capabilities."""
    pass