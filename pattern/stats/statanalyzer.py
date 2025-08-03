# Файл: stats/statanalyzer.py
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.multivariate.manova import MANOVA
from sklearn.metrics import silhouette_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import mannwhitneyu
import itertools
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class StatisticsAnalyzer:
    """
    Class cluster statistical analysis
    """
    def __init__(self, save_path: str = None):
        self.results = {}
        self.save_path = save_path

    def analyze(self, df: pd.DataFrame, labels: np.ndarray, alpha: float = 0.05) -> None:
        """Execute complete analysis pipeline.
            Args:
                df: Input features DataFrame
                labels: Cluster labels array
                alpha: Significance threshold (default: 0.05)
        """
        self.df = df
        self.labels = labels
        self.alpha = alpha
        self.unique_clusters = np.unique(labels)
        self.n_clusters = len(self.unique_clusters)
        self.n_samples = len(df)

        
        self._run_multivariate_test()
        self._run_univariate_tests()
        self._run_silhouette_analysis()
        self._generate_interpretation()

    def _run_multivariate_test(self) -> None:
        """Perform MANOVA for multivariate cluster differences."""
        try:
            manova = MANOVA(endog=self.df, exog=pd.Series(self.labels, name='cluster'))
            manova_results = manova.mv_test()
            self.results['manova'] = {
                'p_value': manova_results.results['cluster']['stat'].iloc[0, 3],
                'significant': None
            }
        except Exception as e:
            self.results['manova'] = {'error': str(e)}

    def _run_univariate_tests(self) -> None:
        """Conduct feature-wise statistical analysis with parallel processing."""
        self.results['features'] = {}
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._analyze_feature, f): f 
                      for f in self.df.columns}
            
            for future in tqdm(futures, desc="Analyzing features"):
                feature, result = future.result()
                self.results['features'][feature] = result

    def _analyze_feature(self, feature: str) -> Tuple[str, Dict]:
        """Full analysis pipeline for a single feature."""
        data = self.df[feature]
        clusters = self.labels
        
        # Assumption checks
        normality = self._check_normality(data, clusters)
        equal_var = self._check_equal_variance(data, clusters)
        
        # Main statistical test
        test_type, p_value = self._select_test(data, clusters, normality, equal_var)
        
        # Post-hoc analysis
        posthoc = self._posthoc_analysis(data, clusters, test_type) if p_value < self.alpha else None
        
        # Multiple testing correction
        return feature, {
            'test_type': test_type,
            'p_value': p_value,
            'posthoc': posthoc,
            'assumptions': {
                'normality': normality,
                'equal_variance': equal_var
            }
        }

    def _check_normality(self, data: pd.Series, clusters: np.ndarray) -> List[bool]:
        """Shapiro-Wilk normality test for each cluster."""
        return [
            stats.shapiro(data[clusters == c])[1] > 0.05
            for c in self.unique_clusters
        ]

    def _check_equal_variance(self, data: pd.Series, clusters: np.ndarray) -> bool:
        """Levene's test for homogeneity of variances."""
        cluster_data = [data[clusters == c] for c in self.unique_clusters]
        return stats.levene(*cluster_data)[1] > self.alpha

    def _select_test(self, data: pd.Series, clusters: np.ndarray,
                    normality: List[bool], equal_var: bool) -> Tuple[str, float]:
        """Select statistical test based on data assumptions."""
        cluster_data = [data[clusters == c] for c in self.unique_clusters]
        
        if all(normality) and equal_var:
            return 'ANOVA', stats.f_oneway(*cluster_data)[1]
        return 'Kruskal-Wallis', stats.kruskal(*cluster_data)[1]

    def _posthoc_analysis(self, data: pd.Series, clusters: np.ndarray,
                         test_type: str) -> Dict[Tuple[int, int], float]:
        """Perform pairwise post-hoc tests with appropriate corrections."""
        pairs = list(itertools.combinations(self.unique_clusters, 2))
        results = {}
        
        if test_type == 'ANOVA':
            tukey = pairwise_tukeyhsd(data, clusters)
            for i, pair in enumerate(pairs):
                results[pair] = tukey.pvalues[i]
        else:
            # Bonferroni correction for non-parametric tests
            for pair in pairs:
                sample1 = data[clusters == pair[0]]
                sample2 = data[clusters == pair[1]]
                results[pair] = mannwhitneyu(sample1, sample2)[1] * len(pairs)
                
        return results

    def _run_silhouette_analysis(self) -> None:
        """Calculate adaptive silhouette score with optimized bootstrap."""
        # Configuration based on data size
        if self.n_samples > 5000:
            subsample_size = 2500
            n_actual = 10
            n_perms = 10
        else:
            subsample_size = None
            n_actual = 1
            n_perms = 100

        # Calculate actual score
        actual_scores = []
        for _ in range(n_actual):
            if subsample_size:
                idx = np.random.choice(self.n_samples, subsample_size, replace=False)
                sample_data = self.df.iloc[idx]
                sample_labels = self.labels[idx]
            else:
                sample_data = self.df
                sample_labels = self.labels
                
            actual_scores.append(silhouette_score(sample_data, sample_labels))
            
        actual_score = np.mean(actual_scores)

        # Permutation tests
        null_scores = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._permuted_silhouette, subsample_size)
                      for _ in range(n_perms)]
            null_scores = [f.result() for f in tqdm(futures, desc="Permutation tests")]

        # Calculate empirical p-value
        p_value = (np.sum(null_scores >= actual_score) + 1) / (n_perms + 1)
        
        self.results['silhouette'] = {
            'score': actual_score,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'subsampled': subsample_size is not None,
            'n_permutations': n_perms
        }

    def _permuted_silhouette(self, subsample_size: Optional[int]) -> float:
        """Calculate silhouette score with permuted labels."""
        if subsample_size:
            idx = np.random.choice(self.n_samples, subsample_size, replace=False)
            sample_data = self.df.iloc[idx]
            perm_labels = np.random.permutation(self.labels[idx])
        else:
            sample_data = self.df
            perm_labels = np.random.permutation(self.labels)
            
        return silhouette_score(sample_data, perm_labels)

    def _generate_interpretation(self) -> None:
        """Generate comprehensive analysis report."""
        interpretation = [
            "Cluster Significance Analysis Report",
            "=" * 50,
            f"Number of clusters: {self.n_clusters}",
            f"Total samples: {self.n_samples}",
            ""
        ]
    
        # MANOVA results
        manova = self.results.get('manova', {})
        if 'p_value' in manova:
            status = "significant" if manova['p_value'] < self.alpha else "not significant"
            interpretation.extend([
                "Multivariate Analysis (MANOVA):",
                f"- Status: {status}",
                f"- p-value: {manova['p_value']:.4f}",
                ""
            ])
        elif 'error' in manova:
            interpretation.extend([
                "MANOVA Results:",
                f"- Error: {manova['error']}",
                ""
            ])
    
        # Feature results
        sig_features = [f for f, d in self.results['features'].items() 
                       if d['p_value'] < self.alpha]
        interpretation.extend([
            "Feature Analysis:",
            f"Significant features: {len(sig_features)}/{len(self.df.columns)}",
            ""
        ])
    
        for feature in sig_features:
            data = self.results['features'][feature]
            feature_entry = [
                f"Feature: {feature}",
                f"- Test type: {data['test_type']}",
                f"- p-value: {data['p_value']:.4f}",
            ]
            
            if data['posthoc']:
                sig_pairs = [f"{k[0]}-{k[1]}" for k, v in data['posthoc'].items()
                            if v < self.alpha]
                if sig_pairs:
                    feature_entry.append("- Significant pairs: " + ", ".join(sig_pairs))
            
            interpretation.extend(feature_entry + [""])
    
        # Silhouette results
        sil = self.results.get('silhouette', {})
        if sil:
            method = "subsampled" if sil['subsampled'] else "full data"
            interpretation.extend([
                "Cluster Quality (Silhouette):",
                f"- Score: {sil['score']:.3f} ({method})",
                f"- Significance vs random: p = {sil['p_value']:.4f}",
                f"- Permutations: {sil['n_permutations']}",
                ""
            ])
        self.results['interpretation'] = interpretation

    def print_report(self) -> None:
        """Print formatted analysis report."""
        if 'interpretation' not in self.results:
            self._generate_interpretation()
        
        print("\n".join([
            "",
            "=" * 50,
            "ANALYSIS REPORT",
            "=" * 50,
        ]))
        
        for line in self.results['interpretation']:
            if line == "=" * 50:
                continue
            print(line)

    def take_results(self) -> pd.DataFrame:
        """
        Export results to DataFrame and save
        
        Returns:
            DataFrame with significant results
        """
        
        results = []
        
        # Add MANOVA results
        manova_row = {
            'feature': 'MANOVA',
            'test_type': 'Multivariate',
            'p_value': self.results['manova'].get('p_value', np.nan),
            'significant': self.results['manova'].get('p_value', np.nan) < self.alpha,
            'significant_clusters': 'All' if self.results['manova'].get('significant', False) else None,
            'assumptions': 'Multivariate normality checked' if 'p_value' in self.results['manova'] else self.results['manova'].get('error', '')
        }
        results.append(manova_row)
        
        # Add feature-wise results
        for feature, data in self.results['features'].items():
            # Format significant pairs
            if data['posthoc']:
                sig_pairs = [
                    f"{p[0]}-{p[1]}" 
                    for p, pval in data['posthoc'].items()
                    if pval < self.alpha
                ]
                sig_pairs_str = ", ".join(sig_pairs) if sig_pairs else "None"
            else:
                sig_pairs_str = "None"
                
            # Format normality assumptions
            norm_clusters = sum(data['assumptions']['normality'])
            total_clusters = len(data['assumptions']['normality'])
            norm_str = f"{norm_clusters}/{total_clusters} clusters normal"
            
            row = {
                'feature': feature,
                'test_type': data['test_type'],
                'p_value': data['p_value'],
                'significant': data['p_value'] < self.alpha,
                'significant_clusters': sig_pairs_str,
                'assumptions': f"{norm_str}, equal variance: {data['assumptions']['equal_variance']}"
            }
            results.append(row)
        
        # Add silhouette results
        sil = self.results.get('silhouette', {})
        results.append({
            'feature': 'SILHOUETTE',
            'test_type': 'Cluster quality',
            'p_value': sil.get('p_value', np.nan),
            'significant': sil.get('significant', False),
            'significant_clusters': f"Score: {sil.get('score', 0):.3f}",
            'assumptions': f"Subsampled: {sil.get('subsampled', False)}, Permutations: {sil.get('n_permutations', 0)}"
        })
        
        df = pd.DataFrame(results).sort_values(by=['significant', 'p_value'], ascending=[False, True])
        df = df[['feature', 'test_type', 'p_value', 'significant', 'significant_clusters','assumptions']]
        if isinstance(self.save_path, str):
            Path(self.save_path).mkdir(parents=True, exist_ok=True)
            path = Path(self.save_path) / f"result.parquet"
            df.to_parquet(path)
            print(f"Results saved to {path}")
        return df