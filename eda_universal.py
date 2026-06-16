"""
Professional Exploratory Data Analysis (EDA) - Universal Tool
Works with any dataset, with or without target column.

Usage:
    python eda_universal.py breast-cancer.csv --target diagnosis
    python eda_universal.py my_data.csv --target price
    python eda_universal.py my_data.csv  # unsupervised EDA (no target)

Features:
- Per-column rigorous analysis
- Parametric (Pearson, t-test) AND non-parametric (Spearman, Kendall, Mann-Whitney)
- Statistical advisor: recommends best method based on normality, outliers, completeness
- Target imbalance reporting with modeling suggestions
- Categorical analysis: Chi-square, Cramér's V, entropy, cardinality detection
"""

import argparse
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "EDA"
SUBDIRS = {
    "logs": "logs",
    "csv": "csv",
    "plots": "plots"
}

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Imbalance thresholds
SEVERE_IMBALANCE_RATIO = 3.0
MODERATE_IMBALANCE_RATIO = 1.5

# Cardinality thresholds
HIGH_CARDINALITY_THRESHOLD = 1000
WARNING_CARDINALITY_THRESHOLD = 500


# ============================================================================
# STATISTICAL ADVISOR CLASS
# ============================================================================

class StatisticalAdvisor:
    """
    Evaluates data characteristics and recommends appropriate statistical methods.
    
    Criteria:
    - Normality (Shapiro-Wilk or D'Agostino)
    - Outlier presence (IQR and Z-score)
    - Data completeness
    - Sample size
    
    Recommendations:
    - parametric: Pearson, t-test (requires normality, no severe outliers)
    - non_parametric: Spearman, Mann-Whitney (for non-normal or outliers present)
    - robust: median-based, bootstrapping (for severe outliers)
    """
    
    def __init__(self, series: pd.Series, column_name: str):
        self.series = series.dropna()
        self.name = column_name
        self.n = len(self.series)
        self._results = {}
        
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate all criteria and return recommendations."""
        # Check completeness
        self._check_completeness()
        
        # Check if numeric
        if not self.series.dtype.kind in 'iufc':
            self._results['is_numeric'] = False
            self._results['recommendation'] = {
                'preferred_method': 'categorical',
                'reasoning': 'Non-numeric column. Use categorical methods.',
                'suggested_correlation': None,
                'suggested_group_test': None
            }
            return self._results
        
        self._results['is_numeric'] = True
        
        # Check normality
        self._check_normality()
        
        # Check outliers
        self._check_outliers()
        
        # Generate recommendation
        self._generate_recommendation()
        
        return self._results
    
    def _check_completeness(self) -> None:
        """Evaluate data completeness."""
        self._results['completeness'] = {
            'n_original': self.n,
            'n_after_dropna': self.n,
            'pct_missing': 0.0
        }
    
    def _check_normality(self) -> None:
        """Test for normality and compute skewness/kurtosis."""
        if self.n < 3:
            self._results['normality'] = {
                'is_normal': False,
                'test': 'insufficient_data',
                'statistic': None,
                'p_value': None,
                'skewness': None,
                'kurtosis': None
            }
            return
        
        skewness = self.series.skew()
        kurtosis = self.series.kurtosis()
        
        if self.n < 5000:
            stat, p = stats.shapiro(self.series)
            test_name = "Shapiro-Wilk"
        else:
            stat, p = stats.normaltest(self.series)
            test_name = "D'Agostino-Pearson"
        
        self._results['normality'] = {
            'is_normal': p > 0.05,
            'test': test_name,
            'statistic': stat,
            'p_value': p,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'interpretation': 'Normal distribution' if p > 0.05 else 'Non-normal distribution'
        }
    
    def _check_outliers(self) -> None:
        """Detect outliers using IQR and Z-score methods."""
        if self.n < 3:
            self._results['outliers'] = {
                'has_outliers': False,
                'iqr_pct': 0.0,
                'zscore_pct': 0.0,
                'severity': 'none'
            }
            return
        
        q1 = self.series.quantile(0.25)
        q3 = self.series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = self.series[(self.series < lower_bound) | (self.series > upper_bound)]
        iqr_pct = (len(iqr_outliers) / self.n) * 100
        
        z_scores = np.abs(stats.zscore(self.series, nan_policy='omit'))
        z_outliers = self.series[z_scores > 3]
        z_pct = (len(z_outliers) / self.n) * 100
        
        has_outliers = iqr_pct > 0 or z_pct > 0
        
        if iqr_pct > 10 or z_pct > 5:
            severity = 'severe'
        elif iqr_pct > 5 or z_pct > 2:
            severity = 'moderate'
        elif has_outliers:
            severity = 'mild'
        else:
            severity = 'none'
        
        self._results['outliers'] = {
            'has_outliers': has_outliers,
            'iqr_pct': iqr_pct,
            'zscore_pct': z_pct,
            'severity': severity
        }
    
    def _generate_recommendation(self) -> None:
        """Generate statistical recommendations based on all criteria."""
        normality = self._results.get('normality', {})
        outliers = self._results.get('outliers', {})
        
        is_normal = normality.get('is_normal', False)
        outlier_severity = outliers.get('severity', 'none')
        
        if not is_normal or outlier_severity in ['moderate', 'severe']:
            preferred = 'non_parametric'
            reasoning_parts = []
            
            if not is_normal:
                reasoning_parts.append(f"non-normal distribution (p={normality.get('p_value', 0):.4f})")
            if outlier_severity in ['moderate', 'severe']:
                reasoning_parts.append(f"{outlier_severity} outliers ({outliers.get('iqr_pct', 0):.1f}% IQR)")
            
            reasoning = " + ".join(reasoning_parts) if reasoning_parts else "data characteristics"
            suggested_corr = 'spearman'
            suggested_test = 'mannwhitneyu'
            
        elif outlier_severity == 'mild':
            preferred = 'robust'
            reasoning = f"mild outliers present ({outliers.get('iqr_pct', 0):.1f}%) - consider both methods"
            suggested_corr = 'spearman'
            suggested_test = 'mannwhitneyu'
        else:
            preferred = 'parametric'
            reasoning = "normal distribution and no significant outliers"
            suggested_corr = 'pearson'
            suggested_test = 'ttest'
        
        self._results['recommendation'] = {
            'preferred_method': preferred,
            'reasoning': reasoning,
            'suggested_correlation': suggested_corr,
            'suggested_group_test': suggested_test
        }


# ============================================================================
# DATA LOADER CLASS
# ============================================================================

class DataLoader:
    """Handles data loading, initial validation, and basic info extraction."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.logger = logging.getLogger(__name__)
        
    def load(self) -> pd.DataFrame:
        """Load CSV and perform initial checks."""
        try:
            self.df = pd.read_csv(self.filepath)
            self.logger.info(f"Successfully loaded {self.filepath}")
            self.logger.info(f"Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def validate_columns(self, expected_target: str = None) -> None:
        """Ensure expected columns exist and log column info."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        self.logger.info(f"Total columns: {len(self.df.columns)}")
        self.logger.info(f"Column names: {list(self.df.columns)}")
        
        if expected_target and expected_target not in self.df.columns:
            self.logger.warning(f"Target column '{expected_target}' not found.")
        elif expected_target:
            self.logger.info(f"Target column '{expected_target}' present.")
    
    def get_basic_info(self, target_col: str = None) -> Dict[str, Any]:
        """Return basic dataset info as dict."""
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        target_dist = {}
        imbalance_warning = None
        modeling_suggestions = []
        
        if target_col and target_col in self.df.columns:
            target_counts = self.df[target_col].value_counts()
            target_dist = target_counts.to_dict()
            
            if len(target_counts) == 2:
                # CORREGIDO: values es atributo, no función
                counts = list(target_counts.values)
                ratio = max(counts) / min(counts)
                
                if ratio >= SEVERE_IMBALANCE_RATIO:
                    imbalance_warning = f"SEVERE imbalance: {target_counts.to_dict()} (ratio {ratio:.2f}:1)"
                    modeling_suggestions = [
                        "Use stratified sampling for train/test split",
                        "Consider SMOTE or ADASYN for oversampling minority class",
                        "Use class weights in models (e.g., 'balanced' in sklearn)",
                        "Consider evaluation metrics: F1-score, precision-recall AUC (not accuracy)"
                    ]
                elif ratio >= MODERATE_IMBALANCE_RATIO:
                    imbalance_warning = f"MODERATE imbalance: {target_counts.to_dict()} (ratio {ratio:.2f}:1)"
                    modeling_suggestions = [
                        "Consider stratified sampling",
                        "Class weights may help"
                    ]
                else:
                    imbalance_warning = "Well-balanced dataset"
                    modeling_suggestions = ["Standard modeling approaches should work well"]
        
        cardinality_summary = {}
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            cardinality_summary[col] = {
                'unique_values': unique_count,
                'high_cardinality': unique_count > HIGH_CARDINALITY_THRESHOLD,
                'cardinality_level': 'HIGH' if unique_count > HIGH_CARDINALITY_THRESHOLD else ('MEDIUM' if unique_count > WARNING_CARDINALITY_THRESHOLD else 'LOW')
            }
        
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
            "duplicated_rows": self.df.duplicated().sum(),
            "target_distribution": target_dist,
            "imbalance_status": imbalance_warning,
            "modeling_suggestions": modeling_suggestions,
            "missing_values_per_column": self.df.isna().sum().to_dict(),
            "cardinality_summary": cardinality_summary
        }


# ============================================================================
# COLUMN ANALYZER CLASS
# ============================================================================

class ColumnAnalyzer:
    """
    Performs rigorous analysis on a single column:
    - Missing values, zero values, constant columns
    - Descriptive statistics (parametric AND robust)
    - Outlier detection (IQR and Z-score)
    - Normality test
    - Correlations: Pearson (parametric) + Spearman + Kendall (non-parametric)
    - Group comparison: t-test (parametric) + Mann-Whitney U (non-parametric)
    - Statistical recommendations
    - Categorical analysis: Chi-square, Cramér's V, entropy, cardinality detection
    """
    
    def __init__(self, series: pd.Series, column_name: str, target_series: Optional[pd.Series] = None):
        self.series = series.dropna() if series.notna().any() else series
        self.original_series = series
        self.name = column_name
        self.target = target_series
        self.logger = logging.getLogger(__name__)
        
    def analyze(self) -> Dict[str, Any]:
        """Run complete analysis and return results dictionary."""
        results = {
            "column": self.name,
            "dtype": str(self.series.dtype),
            "n_total": len(self.original_series),
            "n_missing": self.original_series.isna().sum(),
            "pct_missing": self.original_series.isna().mean() * 100,
            "n_infinite": np.isinf(self.series).sum() if self.series.dtype.kind in 'fc' else 0,
            "n_unique": self.series.nunique(),
            "is_constant": self.series.nunique() == 1,
        }
        
        unique_count = self.series.nunique()
        results["cardinality"] = {
            "unique_values": unique_count,
            "is_high_cardinality": unique_count > HIGH_CARDINALITY_THRESHOLD,
            "cardinality_level": "HIGH" if unique_count > HIGH_CARDINALITY_THRESHOLD else ("MEDIUM" if unique_count > WARNING_CARDINALITY_THRESHOLD else "LOW"),
            "warning": f"⚠️ HIGH CARDINALITY: {unique_count} unique values - may cause performance issues" if unique_count > HIGH_CARDINALITY_THRESHOLD else None
        }
        
        if results["cardinality"]["is_high_cardinality"]:
            self.logger.warning(f"Column '{self.name}' has HIGH CARDINALITY: {unique_count} unique values")
        
        if self.series.dtype.kind in 'iufc':
            results["n_zeros"] = (self.series == 0).sum()
            results["pct_zeros"] = ((self.series == 0).mean() * 100)
            
            stats_dict = self._compute_numeric_stats()
            results.update(stats_dict)
            
            results["outliers_iqr"] = self._detect_outliers_iqr()
            results["outliers_zscore"] = self._detect_outliers_zscore()
            results["normality_test"] = self._test_normality()
            
            advisor = StatisticalAdvisor(self.series, self.name)
            advisor_results = advisor.evaluate()
            if 'completeness' in advisor_results:
                advisor_results['completeness']['pct_missing'] = results["pct_missing"]
                advisor_results['completeness']['n_original'] = results["n_total"]
            results["statistical_advisor"] = advisor_results
            
            if self.target is not None:
                results["target_comparison"] = self._compare_by_target()
                results["correlations_with_target"] = self._compute_correlations_with_target()
        
        elif self.series.dtype.kind in 'O' or pd.api.types.is_categorical_dtype(self.series):
            results["categorical_stats"] = self._analyze_categorical()
            
            if results["cardinality"]["is_high_cardinality"]:
                self.logger.warning(f"Categorical column '{self.name}' has high cardinality ({unique_count} values) - consider feature engineering")
        
        else:
            results["non_numeric_warning"] = f"Column type {self.series.dtype} not fully supported."
        
        return results
    
    def _compute_numeric_stats(self) -> Dict[str, Any]:
        """Compute descriptive statistics including robust measures."""
        s = self.series
        mad = np.median(np.abs(s - s.median()))
        
        return {
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "mad": mad,
            "min": s.min(),
            "max": s.max(),
            "range": s.max() - s.min(),
            "q1": s.quantile(0.25),
            "q3": s.quantile(0.75),
            "iqr": s.quantile(0.75) - s.quantile(0.25),
            "skewness": s.skew(),
            "kurtosis": s.kurtosis(),
        }
    
    def _detect_outliers_iqr(self, multiplier: float = 1.5) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        q1 = self.series.quantile(0.25)
        q3 = self.series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        outliers = self.series[(self.series < lower_bound) | (self.series > upper_bound)]
        
        return {
            "method": "IQR",
            "multiplier": multiplier,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_outliers": len(outliers),
            "pct_outliers": (len(outliers) / len(self.series)) * 100,
            "outlier_values": outliers.tolist()[:10]
        }
    
    def _detect_outliers_zscore(self, threshold: float = 3) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(self.series, nan_policy='omit'))
        outliers = self.series[z_scores > threshold]
        
        return {
            "method": "Z-score",
            "threshold": threshold,
            "n_outliers": len(outliers),
            "pct_outliers": (len(outliers) / len(self.series)) * 100,
            "outlier_values": outliers.tolist()[:10]
        }
    
    def _test_normality(self) -> Dict[str, Any]:
        """Test for normality using Shapiro-Wilk (n<5000) or D'Agostino."""
        n = len(self.series)
        if n < 3:
            return {"test": "Not enough samples", "statistic": None, "p_value": None, "is_normal": False}
        
        if n < 5000:
            stat, p = stats.shapiro(self.series)
            test_name = "Shapiro-Wilk"
        else:
            stat, p = stats.normaltest(self.series)
            test_name = "D'Agostino-Pearson"
        
        return {
            "test": test_name,
            "statistic": stat,
            "p_value": p,
            "is_normal": p > 0.05,
            "interpretation": "Normal distribution" if p > 0.05 else "Non-normal distribution"
        }
    
    def _compare_by_target(self) -> Dict[str, Any]:
        """Compare numeric distribution across target categories using BOTH parametric and non-parametric tests."""
        if self.target is None:
            return {}
        
        df_temp = pd.DataFrame({"value": self.original_series, "target": self.target})
        df_temp = df_temp.dropna()
        
        groups = df_temp.groupby("target")["value"]
        stats_by_target = {}
        for name, group in groups:
            stats_by_target[str(name)] = {
                "count": len(group),
                "mean": group.mean(),
                "median": group.median(),
                "std": group.std(),
                "mad": np.median(np.abs(group - group.median())),
                "min": group.min(),
                "max": group.max(),
                "q1": group.quantile(0.25),
                "q3": group.quantile(0.75)
            }
        
        ttest_result = None
        mannwhitney_result = None
        
        if len(groups) == 2:
            group_names = list(groups.groups.keys())
            g1 = df_temp[df_temp["target"] == group_names[0]]["value"]
            g2 = df_temp[df_temp["target"] == group_names[1]]["value"]
            
            t_stat, t_p_val = stats.ttest_ind(g1, g2, nan_policy='omit')
            ttest_result = {
                "test": "Independent t-test (parametric)",
                "group1": str(group_names[0]),
                "group2": str(group_names[1]),
                "statistic": t_stat,
                "p_value": t_p_val,
                "significant_difference": t_p_val < 0.05,
                "interpretation": "Significant difference between groups" if t_p_val < 0.05 else "No significant difference"
            }
            
            u_stat, mw_p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            mannwhitney_result = {
                "test": "Mann-Whitney U (non-parametric)",
                "group1": str(group_names[0]),
                "group2": str(group_names[1]),
                "statistic": u_stat,
                "p_value": mw_p_val,
                "significant_difference": mw_p_val < 0.05,
                "interpretation": "Significant difference between groups" if mw_p_val < 0.05 else "No significant difference"
            }
        
        return {
            "stats_by_target": stats_by_target,
            "parametric_test": ttest_result,
            "nonparametric_test": mannwhitney_result
        }
    
    def _compute_correlations_with_target(self) -> Dict[str, Any]:
        """Compute correlations with target using Pearson, Spearman, and Kendall."""
        if self.target is None:
            return {}
        
        if self.target.dtype.kind in 'O':
            unique_values = self.target.dropna().unique()
            if len(unique_values) == 2:
                target_encoded = self.target.map({unique_values[0]: 0, unique_values[1]: 1})
            else:
                return {"error": "Target is not binary, correlation not applicable"}
        else:
            target_encoded = self.target
        
        df_temp = pd.DataFrame({"value": self.original_series, "target_encoded": target_encoded})
        df_temp = df_temp.dropna()
        
        if len(df_temp) < 3:
            return {"error": "Insufficient data for correlation"}
        
        correlations = {}
        
        try:
            pearson_r, pearson_p = stats.pearsonr(df_temp["value"], df_temp["target_encoded"])
            correlations["pearson"] = {
                "method": "Pearson (parametric)",
                "correlation": pearson_r,
                "p_value": pearson_p,
                "interpretation": "Significant" if pearson_p < 0.05 else "Not significant"
            }
        except Exception as e:
            correlations["pearson"] = {"error": str(e)}
        
        try:
            spearman_r, spearman_p = stats.spearmanr(df_temp["value"], df_temp["target_encoded"])
            correlations["spearman"] = {
                "method": "Spearman (non-parametric)",
                "correlation": spearman_r,
                "p_value": spearman_p,
                "interpretation": "Significant" if spearman_p < 0.05 else "Not significant"
            }
        except Exception as e:
            correlations["spearman"] = {"error": str(e)}
        
        try:
            kendall_tau, kendall_p = stats.kendalltau(df_temp["value"], df_temp["target_encoded"])
            correlations["kendall"] = {
                "method": "Kendall (non-parametric)",
                "correlation": kendall_tau,
                "p_value": kendall_p,
                "interpretation": "Significant" if kendall_p < 0.05 else "Not significant"
            }
        except Exception as e:
            correlations["kendall"] = {"error": str(e)}
        
        advisor_results = self._get_advisor_recommendation()
        if advisor_results:
            correlations["recommended"] = advisor_results.get('suggested_correlation', 'spearman')
        
        return correlations
    
    def _get_advisor_recommendation(self) -> Optional[Dict]:
        """Get recommendation from StatisticalAdvisor."""
        advisor = StatisticalAdvisor(self.series, self.name)
        results = advisor.evaluate()
        return results.get('recommendation')
    
    def _analyze_categorical(self) -> Dict[str, Any]:
        """Analyze categorical column."""
        results = {
            "top_categories": self._get_top_categories(),
            "entropy": self._compute_entropy(),
            "chi_square_vs_target": None,
            "cramers_v": None,
            "cramers_v_interpretation": None
        }
        
        if self.target is not None and self.target.dtype.kind in 'O':
            chi2_result = self._chi_square_vs_target()
            results["chi_square_vs_target"] = chi2_result
            
            if chi2_result and chi2_result.get('chi2_statistic') is not None:
                cramers_v = self._compute_cramers_v(chi2_result.get('chi2_statistic', 0))
                results["cramers_v"] = cramers_v
                
                if cramers_v < 0.1:
                    results["cramers_v_interpretation"] = "Very weak association"
                elif cramers_v < 0.3:
                    results["cramers_v_interpretation"] = "Weak association"
                elif cramers_v < 0.5:
                    results["cramers_v_interpretation"] = "Moderate association"
                else:
                    results["cramers_v_interpretation"] = "Strong association"
        
        return results
    
    def _get_top_categories(self, top_n: int = 10) -> List[Dict]:
        """Get top N categories by frequency."""
        value_counts = self.series.value_counts().head(top_n)
        total = len(self.series)
        
        return [
            {
                "category": str(cat),
                "count": int(count),
                "percentage": float(count / total * 100)
            }
            for cat, count in value_counts.items()
        ]
    
    def _compute_entropy(self) -> float:
        """Compute Shannon entropy for categorical column."""
        value_counts = self.series.value_counts()
        probabilities = value_counts / len(self.series)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)
    
    def _chi_square_vs_target(self) -> Optional[Dict]:
        """Perform Chi-square test between categorical column and target."""
        if self.target is None:
            return None
        
        contingency = pd.crosstab(self.series, self.target)
        
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return {
                "error": "Insufficient categories for Chi-square test",
                "p_value": None,
                "chi2_statistic": None,
                "significant": False
            }
        
        try:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            return {
                "chi2_statistic": float(chi2),
                "p_value": float(p),
                "degrees_of_freedom": int(dof),
                "significant": p < 0.05,
                "interpretation": "Significant association with target" if p < 0.05 else "No significant association with target"
            }
        except Exception as e:
            self.logger.warning(f"Chi-square test failed for column '{self.name}': {e}")
            return {
                "error": str(e),
                "p_value": None,
                "chi2_statistic": None,
                "significant": False
            }
    
    def _compute_cramers_v(self, chi2: float) -> float:
        """Compute Cramér's V (effect size) from Chi-square statistic."""
        n = len(self.series)
        k = min(self.series.nunique(), self.target.nunique() if self.target is not None else 2)
        
        if n == 0 or k <= 1:
            return 0.0
        
        cramers_v = np.sqrt(chi2 / (n * (k - 1)))
        return float(cramers_v)


# ============================================================================
# VISUALIZER CLASS
# ============================================================================

class Visualizer:
    """Generate and save plots for EDA."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
    def plot_numeric_distribution(self, series: pd.Series, col_name: str, target: Optional[pd.Series] = None, target_name: str = 'target') -> str:
        """Create histogram + KDE, and optional boxplot by target."""
        fig, axes = plt.subplots(1, 2 if target is not None else 1, figsize=(12, 4))
        
        if target is not None:
            ax1, ax2 = axes
            df_temp = pd.DataFrame({"value": series, "target": target}).dropna()
            sns.boxplot(data=df_temp, x="target", y="value", ax=ax1)
            ax1.set_title(f'{col_name} - Boxplot by {target_name}')
            ax1.set_xlabel(target_name)
            ax1.set_ylabel(col_name)
            
            sns.histplot(data=df_temp, x="value", hue="target", kde=True, ax=ax2, alpha=0.6)
            ax2.set_title(f'{col_name} - Distribution by {target_name}')
            ax2.set_xlabel(col_name)
        else:
            ax = axes
            sns.histplot(series.dropna(), kde=True, ax=ax)
            ax.set_title(f'{col_name} - Distribution')
            ax.set_xlabel(col_name)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{col_name}_distribution.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved distribution plot: {filepath}")
        return filepath
    
    def plot_qq(self, series: pd.Series, col_name: str) -> str:
        """Generate Q-Q plot to assess normality."""
        fig, ax = plt.subplots(figsize=(6, 6))
        stats.probplot(series.dropna(), dist="norm", plot=ax)
        ax.set_title(f'{col_name} - Q-Q Plot')
        
        filepath = os.path.join(self.output_dir, f"{col_name}_qqplot.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved Q-Q plot: {filepath}")
        return filepath
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, target_col: str = None) -> str:
        """Plot correlation matrix including point-biserial correlation with target."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_col and target_col in df.columns:
            target_series = df[target_col]
            if target_series.dtype.kind in 'O':
                unique_values = target_series.dropna().unique()
                if len(unique_values) == 2:
                    target_encoded = target_series.map({unique_values[0]: 0, unique_values[1]: 1})
                    numeric_df = numeric_df.copy()
                    numeric_df[f'{target_col}_encoded'] = target_encoded
        
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0, 
                    annot=False, square=True, linewidths=0.5, 
                    cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix (Numeric Features)')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved correlation heatmap: {filepath}")
        return filepath
    
    def plot_top_correlations(self, df: pd.DataFrame, target_col: str = None, top_k: int = 10) -> str:
        """Plot top K features most correlated with target using Spearman (non-parametric)."""
        if not target_col or target_col not in df.columns:
            self.logger.warning("Target column not found for correlation plot.")
            return ""
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        target_series = df[target_col]
        if target_series.dtype.kind in 'O':
            unique_values = target_series.dropna().unique()
            if len(unique_values) == 2:
                target_encoded = target_series.map({unique_values[0]: 0, unique_values[1]: 1})
            else:
                self.logger.warning("Target is not binary, skipping correlation plot")
                return ""
        else:
            target_encoded = target_series
        
        correlations = {}
        for col in numeric_df.columns:
            try:
                corr, _ = stats.spearmanr(numeric_df[col], target_encoded)
                correlations[col] = abs(corr)
            except:
                correlations[col] = 0
        
        top_features = pd.Series(correlations).sort_values(ascending=False).head(top_k)
        
        plt.figure(figsize=(10, 6))
        top_features.plot(kind='barh')
        plt.xlabel('Absolute Spearman Correlation')
        plt.title(f'Top {top_k} Features Correlated with {target_col} (Non-parametric)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, "top_correlations_with_target.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved top correlations plot: {filepath}")
        return filepath
    
    def plot_categorical_distribution(self, series: pd.Series, col_name: str, target: Optional[pd.Series] = None, target_name: str = 'target') -> str:
        """Plot categorical distribution with optional target breakdown."""
        fig, axes = plt.subplots(1, 2 if target is not None else 1, figsize=(14, 5))
        
        top_categories = series.value_counts().head(15)
        
        if target is not None:
            ax1, ax2 = axes
            top_categories.plot(kind='barh', ax=ax1, color='steelblue')
            ax1.set_title(f'{col_name} - Top Categories')
            ax1.set_xlabel('Count')
            ax1.set_ylabel('Category')
            
            df_temp = pd.DataFrame({"category": series, "target": target}).dropna()
            top_cat_names = top_categories.index.tolist()
            df_top = df_temp[df_temp["category"].isin(top_cat_names)]
            cross_tab = pd.crosstab(df_top["category"], df_top["target"], normalize='index') * 100
            cross_tab.plot(kind='bar', ax=ax2, stacked=True, colormap='Set2')
            ax2.set_title(f'{col_name} - Target Distribution by Category')
            ax2.set_xlabel('Category')
            ax2.set_ylabel('Percentage (%)')
            ax2.legend(title=target_name)
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax = axes
            top_categories.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title(f'{col_name} - Top 15 Categories')
            ax.set_xlabel('Count')
            ax.set_ylabel('Category')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{col_name}_categorical.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved categorical distribution plot: {filepath}")
        return filepath


# ============================================================================
# EDA LOGGER & RESULTS SAVER
# ============================================================================

class EDAResultSaver:
    """Save analysis results to CSV files and log summaries."""
    
    def __init__(self, output_csv_dir: str, output_log_dir: str):
        self.csv_dir = output_csv_dir
        self.log_dir = output_log_dir
        self.logger = logging.getLogger(__name__)
        
    def save_results_to_csv(self, results: List[Dict[str, Any]], filename: str) -> str:
        """Save list of result dictionaries to CSV (flattened for readability)."""
        flat_results = []
        for res in results:
            flat = {}
            for key, value in res.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if not isinstance(subvalue, (dict, list)):
                            flat[f"{key}_{subkey}"] = subvalue
                elif not isinstance(value, (dict, list)):
                    flat[key] = value
            flat_results.append(flat)
        
        df_results = pd.DataFrame(flat_results)
        filepath = os.path.join(self.csv_dir, filename)
        df_results.to_csv(filepath, index=False)
        self.logger.info(f"Saved CSV: {filepath} ({len(results)} rows)")
        return filepath
    
    def save_detailed_correlations(self, results: List[Dict[str, Any]], filename: str = "detailed_correlations.csv") -> str:
        """Extract and save correlation data (Pearson, Spearman, Kendall) for each numeric column."""
        rows = []
        for res in results:
            if 'correlations_with_target' in res and isinstance(res['correlations_with_target'], dict):
                corrs = res['correlations_with_target']
                base_row = {'column': res['column']}
                
                if 'pearson' in corrs and isinstance(corrs['pearson'], dict):
                    base_row['pearson_correlation'] = corrs['pearson'].get('correlation', None)
                    base_row['pearson_p_value'] = corrs['pearson'].get('p_value', None)
                    base_row['pearson_significant'] = corrs['pearson'].get('interpretation', '')
                
                if 'spearman' in corrs and isinstance(corrs['spearman'], dict):
                    base_row['spearman_correlation'] = corrs['spearman'].get('correlation', None)
                    base_row['spearman_p_value'] = corrs['spearman'].get('p_value', None)
                    base_row['spearman_significant'] = corrs['spearman'].get('interpretation', '')
                
                if 'kendall' in corrs and isinstance(corrs['kendall'], dict):
                    base_row['kendall_correlation'] = corrs['kendall'].get('correlation', None)
                    base_row['kendall_p_value'] = corrs['kendall'].get('p_value', None)
                    base_row['kendall_significant'] = corrs['kendall'].get('interpretation', '')
                
                base_row['recommended_method'] = corrs.get('recommended', 'N/A')
                rows.append(base_row)
        
        if rows:
            df_corr = pd.DataFrame(rows)
            filepath = os.path.join(self.csv_dir, filename)
            df_corr.to_csv(filepath, index=False)
            self.logger.info(f"Saved detailed correlations CSV: {filepath}")
            return filepath
        return ""
    
    def save_outliers_summary(self, all_outliers: Dict[str, Dict], filename: str) -> str:
        """Save outliers summary to CSV."""
        rows = []
        for col, outlier_dict in all_outliers.items():
            if 'outliers_iqr' in outlier_dict:
                rows.append({
                    "column": col,
                    "outlier_method": outlier_dict['outliers_iqr']['method'],
                    "n_outliers": outlier_dict['outliers_iqr']['n_outliers'],
                    "pct_outliers": outlier_dict['outliers_iqr']['pct_outliers'],
                    "lower_bound": outlier_dict['outliers_iqr']['lower_bound'],
                    "upper_bound": outlier_dict['outliers_iqr']['upper_bound']
                })
            if 'outliers_zscore' in outlier_dict:
                rows.append({
                    "column": col,
                    "outlier_method": outlier_dict['outliers_zscore']['method'],
                    "n_outliers": outlier_dict['outliers_zscore']['n_outliers'],
                    "pct_outliers": outlier_dict['outliers_zscore']['pct_outliers'],
                    "threshold": outlier_dict['outliers_zscore']['threshold']
                })
        
        df_outliers = pd.DataFrame(rows)
        filepath = os.path.join(self.csv_dir, filename)
        df_outliers.to_csv(filepath, index=False)
        self.logger.info(f"Saved outliers CSV: {filepath}")
        return filepath
    
    def save_statistical_recommendations(self, results: List[Dict[str, Any]], filename: str = "statistical_recommendations.csv") -> str:
        """Extract and save statistical recommendations per column."""
        rows = []
        for res in results:
            if 'statistical_advisor' in res and 'recommendation' in res['statistical_advisor']:
                adv = res['statistical_advisor']
                rec = adv['recommendation']
                
                norm = adv.get('normality', {})
                outliers = adv.get('outliers', {})
                
                rows.append({
                    "column": res['column'],
                    "is_numeric": adv.get('is_numeric', False),
                    "is_normal": norm.get('is_normal', False),
                    "normality_p_value": norm.get('p_value', None),
                    "skewness": norm.get('skewness', None),
                    "kurtosis": norm.get('kurtosis', None),
                    "outlier_severity": outliers.get('severity', 'none'),
                    "outlier_pct_iqr": outliers.get('iqr_pct', 0),
                    "preferred_method": rec.get('preferred_method', 'N/A'),
                    "reasoning": rec.get('reasoning', 'N/A'),
                    "suggested_correlation": rec.get('suggested_correlation', 'N/A'),
                    "suggested_group_test": rec.get('suggested_group_test', 'N/A')
                })
        
        if rows:
            df_rec = pd.DataFrame(rows)
            filepath = os.path.join(self.csv_dir, filename)
            df_rec.to_csv(filepath, index=False)
            self.logger.info(f"Saved statistical recommendations CSV: {filepath}")
            return filepath
        return ""
    
    def save_cardinality_report(self, results: List[Dict[str, Any]], filename: str = "cardinality_report.csv") -> str:
        """Save cardinality report for all columns."""
        rows = []
        for res in results:
            if 'cardinality' in res:
                card = res['cardinality']
                rows.append({
                    "column": res['column'],
                    "dtype": res.get('dtype', 'unknown'),
                    "unique_values": card.get('unique_values', 0),
                    "cardinality_level": card.get('cardinality_level', 'UNKNOWN'),
                    "is_high_cardinality": card.get('is_high_cardinality', False),
                    "warning": card.get('warning', '')
                })
        
        if rows:
            df_card = pd.DataFrame(rows)
            df_card = df_card.sort_values('unique_values', ascending=False)
            filepath = os.path.join(self.csv_dir, filename)
            df_card.to_csv(filepath, index=False)
            self.logger.info(f"Saved cardinality report CSV: {filepath}")
            
            high_card_cols = df_card[df_card['is_high_cardinality'] == True]
            if not high_card_cols.empty:
                self.logger.warning(f"Found {len(high_card_cols)} columns with HIGH CARDINALITY: {high_card_cols['column'].tolist()}")
            
            return filepath
        return ""
    
    def save_categorical_analysis(self, results: List[Dict[str, Any]], filename: str = "categorical_analysis.csv") -> str:
        """Save categorical column analysis (Chi-square, Cramér's V, Entropy)."""
        rows = []
        for res in results:
            if 'categorical_stats' in res:
                cat = res['categorical_stats']
                card = res.get('cardinality', {})
                
                row = {
                    "column": res['column'],
                    "unique_values": card.get('unique_values', 0),
                    "cardinality_level": card.get('cardinality_level', 'UNKNOWN'),
                    "entropy": cat.get('entropy', None),
                }
                
                chi2 = cat.get('chi_square_vs_target', {})
                if chi2 and isinstance(chi2, dict):
                    row['chi2_statistic'] = chi2.get('chi2_statistic', None)
                    row['chi2_p_value'] = chi2.get('p_value', None)
                    row['chi2_significant'] = chi2.get('significant', False)
                    row['chi2_interpretation'] = chi2.get('interpretation', '')
                
                row['cramers_v'] = cat.get('cramers_v', None)
                row['cramers_v_interpretation'] = cat.get('cramers_v_interpretation', '')
                
                top_cats = cat.get('top_categories', [])
                if top_cats:
                    row['top_category'] = top_cats[0].get('category', '')
                    row['top_category_pct'] = top_cats[0].get('percentage', 0)
                
                rows.append(row)
        
        if rows:
            df_cat = pd.DataFrame(rows)
            filepath = os.path.join(self.csv_dir, filename)
            df_cat.to_csv(filepath, index=False)
            self.logger.info(f"Saved categorical analysis CSV: {filepath}")
            return filepath
        return ""


# ============================================================================
# MAIN EDA ORCHESTRATOR
# ============================================================================

class EDAReport:
    """
    Main orchestrator for the entire EDA process.
    Works with any dataset by specifying the target column.
    """
    
    def __init__(self, data_path: str, target_col: str = None, output_base_dir: str = OUTPUT_DIR):
        self.data_path = data_path
        self.target_col = target_col
        self.output_base_dir = output_base_dir
        self.df = None
        self.results = []
        
        self._setup_directories()
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        
    def _setup_directories(self) -> None:
        """Create output directories if they don't exist."""
        for subdir in SUBDIRS.values():
            path = os.path.join(self.output_base_dir, subdir)
            os.makedirs(path, exist_ok=True)
            
        self.plots_dir = os.path.join(self.output_base_dir, SUBDIRS["plots"])
        self.csv_dir = os.path.join(self.output_base_dir, SUBDIRS["csv"])
        self.log_dir = os.path.join(self.output_base_dir, SUBDIRS["logs"])
        
    def _setup_logging(self) -> None:
        """Configure logging to file and console."""
        log_file = os.path.join(self.log_dir, f"eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def run(self) -> None:
        """Execute full EDA pipeline."""
        self.logger.info("=" * 80)
        self.logger.info(f"PROFESSIONAL EDA - Dataset: {self.data_path}")
        if self.target_col:
            self.logger.info(f"Target column: {self.target_col}")
        else:
            self.logger.info("No target column specified - running unsupervised EDA")
        self.logger.info("=" * 80)
        
        # 1. Load data
        loader = DataLoader(self.data_path)
        self.df = loader.load()
        loader.validate_columns(self.target_col)
        
        # Get target series if target column exists
        target_series = None
        if self.target_col and self.target_col in self.df.columns:
            target_series = self.df[self.target_col]
        
        basic_info = loader.get_basic_info(self.target_col)
        
        self.logger.info(f"Dataset shape: {basic_info['rows']} rows x {basic_info['columns']} columns")
        
        if target_series is not None:
            self.logger.info(f"Target distribution: {basic_info['target_distribution']}")
            self.logger.info(f"Imbalance status: {basic_info['imbalance_status']}")
            for suggestion in basic_info['modeling_suggestions']:
                self.logger.info(f"  Modeling suggestion: {suggestion}")
        
        self.logger.info("-" * 40)
        self.logger.info("CARDINALITY DETECTION SUMMARY")
        card_summary = basic_info['cardinality_summary']
        high_card_cols = [col for col, info in card_summary.items() if info['high_cardinality']]
        if high_card_cols:
            self.logger.warning(f"⚠️ HIGH CARDINALITY columns ({len(high_card_cols)}): {high_card_cols}")
            for col in high_card_cols:
                self.logger.warning(f"  - {col}: {card_summary[col]['unique_values']} unique values")
        else:
            self.logger.info("✓ No high cardinality columns detected")
        
        missing_cols = {k: v for k, v in basic_info['missing_values_per_column'].items() if v > 0}
        if missing_cols:
            self.logger.warning(f"Columns with missing values: {missing_cols}")
        else:
            self.logger.info("No missing values detected in any column")
        
        # 2. Initialize visualizer and saver
        visualizer = Visualizer(self.plots_dir)
        saver = EDAResultSaver(self.csv_dir, self.log_dir)
        
        # 3. Analyze each column
        for col in self.df.columns:
            self.logger.info(f"Analyzing column: {col}")
            analyzer = ColumnAnalyzer(self.df[col], col, target_series)
            result = analyzer.analyze()
            self.results.append(result)
            
            # Generate plots based on column type
            if self.df[col].dtype.kind in 'iufc' and not result.get('is_constant', False):
                visualizer.plot_numeric_distribution(
                    self.df[col], col, target_series, 
                    target_name=self.target_col if self.target_col else 'target'
                )
                if len(self.df[col].dropna()) > 10:
                    visualizer.plot_qq(self.df[col], col)
            elif (self.df[col].dtype.kind in 'O' or pd.api.types.is_categorical_dtype(self.df[col])) and self.df[col].nunique() <= 50:
                visualizer.plot_categorical_distribution(
                    self.df[col], col, target_series,
                    target_name=self.target_col if self.target_col else 'target'
                )
            elif self.df[col].nunique() > 50:
                self.logger.info(f"Skipping categorical plot for '{col}' due to high cardinality ({self.df[col].nunique()} categories)")
        
        # 4. Global visualizations
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            visualizer.plot_correlation_heatmap(self.df, self.target_col)
            if self.target_col:
                visualizer.plot_top_correlations(self.df, self.target_col)
        
        # 5. Save results to CSV
        saver.save_results_to_csv(self.results, "eda_column_summary.csv")
        saver.save_detailed_correlations(self.results, "detailed_correlations.csv")
        saver.save_statistical_recommendations(self.results, "statistical_recommendations.csv")
        saver.save_cardinality_report(self.results, "cardinality_report.csv")
        saver.save_categorical_analysis(self.results, "categorical_analysis.csv")
        
        outliers_dict = {res['column']: res for res in self.results if 'outliers_iqr' in res}
        if outliers_dict:
            saver.save_outliers_summary(outliers_dict, "outliers_summary.csv")
        
        # 6. Final logging and summary
        self._print_summary(basic_info, target_series)
        
        self.logger.info("=" * 80)
        self.logger.info("EDA COMPLETED SUCCESSFULLY")
        self.logger.info(f"Results saved in: {self.output_base_dir}")
        self.logger.info("  - CSV summaries: csv/")
        self.logger.info("    * eda_column_summary.csv - Complete stats per column")
        self.logger.info("    * detailed_correlations.csv - Pearson, Spearman, Kendall correlations")
        self.logger.info("    * statistical_recommendations.csv - Which methods to trust")
        self.logger.info("    * outliers_summary.csv - Outlier detection results")
        self.logger.info("    * cardinality_report.csv - Cardinality analysis per column")
        self.logger.info("    * categorical_analysis.csv - Chi-square, Cramér's V, entropy")
        self.logger.info("  - Plots: plots/")
        self.logger.info("  - Logs: logs/")
        self.logger.info("=" * 80)
    
    def _print_summary(self, basic_info: Dict, target_series: Optional[pd.Series] = None) -> None:
        """Print a summary of key findings."""
        self.logger.info("-" * 40)
        self.logger.info("KEY FINDINGS SUMMARY")
        self.logger.info("-" * 40)
        
        if target_series is not None:
            self.logger.info(f"Target distribution: {basic_info['target_distribution']}")
            self.logger.info(f"Imbalance assessment: {basic_info['imbalance_status']}")
        
        card_summary = basic_info['cardinality_summary']
        high_card = sum(1 for info in card_summary.values() if info['high_cardinality'])
        self.logger.info(f"Columns with HIGH cardinality (> {HIGH_CARDINALITY_THRESHOLD}): {high_card}")
        
        numeric_cols = [r for r in self.results if r.get('dtype', '').startswith(('int', 'float'))]
        param_preferred = 0
        nonparam_preferred = 0
        robust_preferred = 0
        
        for r in numeric_cols:
            if 'statistical_advisor' in r and 'recommendation' in r['statistical_advisor']:
                pref = r['statistical_advisor']['recommendation'].get('preferred_method', '')
                if pref == 'parametric':
                    param_preferred += 1
                elif pref == 'non_parametric':
                    nonparam_preferred += 1
                elif pref == 'robust':
                    robust_preferred += 1
        
        self.logger.info(f"Statistical method recommendations ({len(numeric_cols)} numeric columns):")
        self.logger.info(f"  - Parametric (normal, no outliers): {param_preferred} columns")
        self.logger.info(f"  - Non-parametric (non-normal or outliers): {nonparam_preferred} columns")
        self.logger.info(f"  - Robust (mild outliers): {robust_preferred} columns")
        
        categorical_cols = [r for r in self.results if 'categorical_stats' in r]
        if categorical_cols:
            self.logger.info(f"Categorical columns analyzed: {len(categorical_cols)}")
            if target_series is not None:
                sig_assoc = 0
                for r in categorical_cols:
                    cat_stats = r.get('categorical_stats', {})
                    chi2_result = cat_stats.get('chi_square_vs_target')
                    if chi2_result and isinstance(chi2_result, dict):
                        if chi2_result.get('significant', False):
                            sig_assoc += 1
                self.logger.info(f"  - Columns with significant association with target: {sig_assoc}")
            else:
                self.logger.info("  - No target specified - skipping association analysis")
        
        self.logger.info("")
        self.logger.info("Correlation analysis includes:")
        self.logger.info("  - Pearson (parametric, assumes linearity & normality)")
        self.logger.info("  - Spearman (non-parametric, monotonic relationships)")
        self.logger.info("  - Kendall (non-parametric, robust for ties/small samples)")
        self.logger.info("")
        self.logger.info("For non-normal data or when outliers present, prefer Spearman or Kendall.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Professional EDA - Works with any dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eda_universal.py breast-cancer.csv --target diagnosis
  python eda_universal.py my_data.csv --target price
  python eda_universal.py my_data.csv  # unsupervised EDA (no target)
        """
    )
    parser.add_argument("input_csv", help="Ruta al archivo CSV de entrada")
    parser.add_argument("--target", "-t", help="Nombre de la columna objetivo (target)")
    parser.add_argument("--output", "-o", default="EDA", help="Directorio de salida (default: EDA)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"ERROR: No se encuentra el archivo {args.input_csv}")
        sys.exit(1)
    
    eda = EDAReport(
        data_path=args.input_csv,
        target_col=args.target,
        output_base_dir=args.output
    )
    eda.run()
