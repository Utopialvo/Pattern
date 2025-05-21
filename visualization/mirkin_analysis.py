# Файл: visualization/mirkin_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import gridspec
from typing import Optional, Union, List
from pathlib import Path

class MirkinAnalysis:
    def __init__(self, X: Union[np.ndarray, pd.DataFrame, list], clusters: Union[np.ndarray, list], 
                 centroids: Optional[Union[np.ndarray, list]] = None, save_path: Optional[str] = None) -> None:
        """
        Perform cluster analysis using Mirkin's rule for feature importance interpretation.
        
        Parameters:
        X : Input data array or DataFrame of shape (n_samples, n_features)
        clusters : Cluster labels array of shape (n_samples,)
        centroids : Optional centroids array of shape (n_clusters, n_features)
        save_path : Optional directory path to save plots. If None, plots won't be saved.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            self.X = X.values
        else:
            self.X = np.asarray(X)
            self.feature_names = [f'Feature {i+1}' for i in range(self.X.shape[1])]
            
        self.clusters = np.asarray(clusters)
        self.n_clusters = len(np.unique(clusters))
        self.n_features = self.X.shape[1]
        self.save_path = save_path

        if centroids is not None:
            self.centroids = np.asarray(centroids)
        else:
            self.centroids = np.array([
                self.X[self.clusters == k].mean(axis=0) 
                for k in range(self.n_clusters)
            ])
            
        self.prepared = False
        self._prepare_directories()

    def _prepare_directories(self) -> None:
        """Create save directory if needed"""
        if self.save_path:
            Path(self.save_path).mkdir(parents=True, exist_ok=True)

    def prepare_data(self) -> None:
        """Compute global statistics and feature importance metrics"""
        self.global_means = np.mean(self.X, axis=0)
        self.global_stds = np.std(self.X, axis=0)
        self.deviations = (self.centroids - self.global_means) / self.global_stds
        self.importance = (self.centroids - self.global_means) / self.global_means
        self.prepared = True

    def _save_figure(self, fig: plt.Figure, base_name: str, page: int = None) -> None:
        """Helper method to save figures with pagination"""
        if not self.save_path:
            return
            
        suffix = f"_page{page+1}" if page is not None else ""
        filename = f"{base_name}{suffix}.png"
        save_path = os.path.join(self.save_path, filename)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def plot_feature_importance(self, top_features: Optional[int] = None, figsize: tuple = (15, 10)) -> None:
        """
        Visualize feature importance per cluster using sorted bar plots
        
        Parameters:
        top_features : Number of top features to display. If None, show all
        figsize : Figure size dimensions
        """
        if not self.prepared:
            self.prepare_data()
        
        # Calculate grid dimensions
        n_cols = 2
        n_rows = 2
        clusters_per_page = n_cols * n_rows
        
        # Split clusters into pages
        for page in range(0, self.n_clusters, clusters_per_page):
            current_clusters = range(page, min(page + clusters_per_page, self.n_clusters))
            num_current = len(current_clusters)
            
            if num_current == 0:
                continue
                
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
            
            actual_rows = (num_current + n_cols - 1) // n_cols
            actual_cols = min(num_current, n_cols)
            
            for i, cluster in enumerate(current_clusters):
                row = i // actual_cols
                col = i % actual_cols
                
                ax = fig.add_subplot(gs[row, col])
                importance = self.importance[cluster]
                
                sorted_idx = np.argsort(-np.abs(importance))
                if top_features is not None:
                    sorted_idx = sorted_idx[:top_features]
                
                sorted_names = [self.feature_names[i] for i in sorted_idx]
                colors = ['#d62728' if val > 0 else '#2ca02c' for val in importance[sorted_idx]]
                
                ax.bar(range(len(sorted_idx)), 
                      np.abs(importance[sorted_idx]), 
                      color=colors,
                      edgecolor='k',
                      linewidth=0.5)
                
                ax.set_title(f'Cluster {cluster}', pad=12, fontweight='semibold')
                ax.set_xlabel('Feature', labelpad=8)
                ax.set_ylabel('|d_kv| (Importance)', labelpad=10)
                ax.set_xticks(range(len(sorted_idx)))
                ax.set_xticklabels(sorted_names, rotation=45, ha='right')
                ax.grid(axis='y', linestyle=':', alpha=0.6)
                ax.set_axisbelow(True)
            
            for i in range(num_current, n_rows*n_cols):
                row = i // n_cols
                col = i % n_cols
                if row < n_rows and col < n_cols:
                    fig.delaxes(fig.add_subplot(gs[row, col]))
            
            plt.tight_layout()
            if self.save_path:
                self._save_figure(fig, "feature_importance", page//clusters_per_page)
                
            if not self.save_path:
                plt.show()

    def plot_feature_deviation(self, figsize: tuple = (15, 10)) -> None:
        """
        Visualize cluster deviations in standard deviations for each feature
        
        Parameters:
        figsize : Figure size dimensions
        """
        if not self.prepared:
            self.prepare_data()
            
        # Split features into groups of 4
        features_per_page = 4
        feature_groups = [self.feature_names[i:i+features_per_page] 
                        for i in range(0, len(self.feature_names), features_per_page)]

        for page, group in enumerate(feature_groups):
            fig = plt.figure(figsize=figsize)
            n_cols = 2
            n_rows = 2
            gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
            
            for idx, feat_name in enumerate(group):
                v = self.feature_names.index(feat_name)
                ax = fig.add_subplot(gs[idx//n_cols, idx%n_cols])
                feature_std = self.global_stds[v]
                
                ax.axhline(0, color='k', linestyle='--', linewidth=1, zorder=0)
                for k in range(self.n_clusters):
                    deviation = self.deviations[k, v]
                    ax.scatter(
                        k, deviation,
                        s=120,
                        zorder=10,
                        color=plt.cm.tab10(k),
                        edgecolor='k',
                        alpha=0.9
                    )
                    ax.text(
                        k, deviation + 0.1*np.sign(deviation),
                        f'{deviation:.2f}σ',
                        ha='center',
                        va='bottom' if deviation > 0 else 'top',
                        fontsize=9,
                        fontweight='bold'
                    )
                
                ax.set_title(
                    f'{feat_name}\n(μ={self.global_means[v]:.2f}, σ={feature_std:.2f})',
                    fontsize=10,
                    pad=12
                )
                ax.set_xlabel('Cluster ID', labelpad=8)
                ax.set_ylabel('Deviation (σ)', labelpad=10)
                ax.set_xticks(range(self.n_clusters))
                ax.grid(axis='y', linestyle=':', alpha=0.6)
                ax.set_axisbelow(True)

            plt.tight_layout()
            if self.save_path:
                self._save_figure(fig, "feature_deviation", page)
                
            if not self.save_path:
                plt.show()