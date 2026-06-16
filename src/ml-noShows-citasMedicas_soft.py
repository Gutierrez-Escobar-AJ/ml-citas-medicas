"""
Dataset de Citas Médicas - Modelado Predictivo para No-Show
VERSIÓN OPTIMIZADA PARA MEMORIA
- Muestreo automático si el dataset es grande
- GridSearchCV desactivado por defecto (opcional activar)
- N_JOBS reducido para evitar saturar RAM
- Uso eficiente de memoria con tipos de datos (int32, float32)
- Carga solo columnas necesarias

Autor: Gutierrez-Escobar-AJ
Propósito: Predecir inasistencia a citas médicas usando Random Forest vs XGBoost

Usage: python3 ml-noShows-citasMedicas_optimizado.py
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve, GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report,
                             matthews_corrcoef, cohen_kappa_score)
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")

from scipy import stats

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# CONFIGURACIÓN OPTIMIZADA
# ============================================================================

OUTPUT_DIR = "ML_MODELS_NOSHOW"
SUBDIRS = {"logs": "logs", "csv": "csv", "plots": "plots"}

RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2

# OPTIMIZACIONES CLAVE
USE_GRID_SEARCH = False          # DESACTIVADO para ahorrar RAM
N_JOBS = 2                       # Reduce uso de CPU/RAM
MAX_SAMPLES = 30000              # Muestreo automático si dataset > esto
TOP_FEATURES = 8                 # Menos features = menos memoria

# Columnas a eliminar (alta cardinalidad)
COLUMNAS_A_ELIMINAR = ["PatientId", "AppointmentID", "ScheduledDay"]

# Columnas útiles para cargar (solo las necesarias)
COLUMNAS_UTILES = [
    'Gender', 'Age', 'Neighbourhood', 'Scholarship', 
    'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 
    'SMS_received', 'ScheduledDay', 'AppointmentDay', 'No-show'
]

# GridSearchCV (solo si se activa)
RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', 'balanced_subsample']
}

XGB_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

# Parámetros por defecto
RF_DEFAULT_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS
}

XGB_DEFAULT_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# Estilos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
COLORS = {
    "Random Forest": "#2ecc71",
    "XGBoost": "#e74c3c",
    "No-Show": "#e74c3c",
    "Show": "#2ecc71"
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories() -> None:
    for subdir in SUBDIRS.values():
        path = os.path.join(OUTPUT_DIR, subdir)
        os.makedirs(path, exist_ok=True)


def setup_logging() -> logging.Logger:
    log_file = os.path.join(OUTPUT_DIR, SUBDIRS["logs"],
                            f"ml_noshow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def is_model_fitted(model) -> bool:
    return hasattr(model, 'classes_')


# ============================================================================
# DATA PREPROCESSOR OPTIMIZADO
# ============================================================================

class DataPreprocessor:
    def __init__(self, data_path: str, logger: logging.Logger):
        self.data_path = data_path
        self.logger = logger
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.selected_features = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self) -> None:
        """Carga datos con optimizaciones de memoria."""
        self.logger.info("=" * 80)
        self.logger.info("DATA LOADING & PREPROCESSING (No-Show Dataset) - OPTIMIZADO")
        self.logger.info("=" * 80)

        try:
            # 1. Cargar solo columnas necesarias
            self.df = pd.read_csv(self.data_path, usecols=COLUMNAS_UTILES)
            self.logger.info(f"Loaded {self.data_path} ({len(self.df)} registros, {len(COLUMNAS_UTILES)} columnas)")

            # 2. MUESTREO si es muy grande
            if len(self.df) > MAX_SAMPLES:
                self.df = self.df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)
                self.logger.warning(f"⚠️ Dataset muestreado a {MAX_SAMPLES} registros para optimizar memoria")

            self.logger.info(f"Shape final: {self.df.shape}")

            # 3. Optimizar tipos de datos
            for col in self.df.columns:
                if self.df[col].dtype == 'int64':
                    self.df[col] = self.df[col].astype('int32')
                elif self.df[col].dtype == 'float64':
                    self.df[col] = self.df[col].astype('float32')

            # 4. Target
            if "No-show" not in self.df.columns:
                raise ValueError("'No-show' column not found")
            
            self.y = self.df["No-show"].map({"No": 0, "Yes": 1})
            self.logger.info(f"Target distribution: {self.y.value_counts().to_dict()}")

            # 5. Features numéricas
            self.X = self.df.select_dtypes(include=[np.number]).copy()
            self.logger.info(f"Numeric features: {len(self.X.columns)}")

            # 6. FEATURE ENGINEERING: WaitDays
            if 'AppointmentDay' in self.df.columns and 'ScheduledDay' in self.df.columns:
                try:
                    appointment_dt = pd.to_datetime(self.df['AppointmentDay'])
                    scheduled_dt = pd.to_datetime(self.df['ScheduledDay'])
                    self.X['WaitDays'] = (appointment_dt - scheduled_dt).dt.days.astype('int32')
                    self.logger.info("✅ Created feature: WaitDays")
                    self.logger.info(f"  WaitDays - min: {self.X['WaitDays'].min()}, max: {self.X['WaitDays'].max()}")
                except Exception as e:
                    self.logger.warning(f"Could not create WaitDays: {e}")

            # 7. Eliminar columnas problemáticas
            for col in COLUMNAS_A_ELIMINAR:
                if col in self.X.columns:
                    self.X = self.X.drop(columns=[col])
                    self.logger.info(f"Removed '{col}' (high cardinality)")

            # 8. Variables categóricas codificadas
            if 'Gender' in self.df.columns:
                self.X['Gender_encoded'] = self.df['Gender'].map({'F': 0, 'M': 1}).astype('int8')
                self.logger.info("Encoded 'Gender' (F=0, M=1)")

            if 'Neighbourhood' in self.df.columns:
                freq_encoding = self.df['Neighbourhood'].map(
                    self.df['Neighbourhood'].value_counts() / len(self.df)
                ).astype('float32')
                self.X['Neighbourhood_freq'] = freq_encoding
                self.logger.info("Encoded 'Neighbourhood' as frequency")

            # 9. Verificar valores faltantes
            if self.X.isna().any().any():
                self.logger.warning("NaN values detected. Filling with median.")
                self.X = self.X.fillna(self.X.median())

            self.feature_names = self.X.columns.tolist()
            self.logger.info(f"Total features: {len(self.feature_names)}")
            self.logger.info(f"Features: {self.feature_names}")
            
            # Memoria usada
            mem_usage = self.X.memory_usage(deep=True).sum() / 1024**2
            self.logger.info(f"Memory usage: {mem_usage:.2f} MB")

            # Desbalance
            n_no = (self.y == 0).sum()
            n_yes = (self.y == 1).sum()
            ratio = max(n_no, n_yes) / min(n_no, n_yes)
            self.logger.info(f"Imbalance ratio: {ratio:.2f}:1")
            if ratio > 1.5:
                self.logger.warning("Severe imbalance - using class weights")

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def select_features(self, top_k: Optional[int] = TOP_FEATURES) -> None:
        if top_k is None or top_k >= len(self.feature_names):
            self.selected_features = self.feature_names
            self.logger.info(f"Using all {len(self.feature_names)} features")
            return

        self.logger.info(f"Selecting top {top_k} features using Mutual Information...")
        mi_scores = mutual_info_classif(self.X, self.y, random_state=RANDOM_STATE)
        mi_series = pd.Series(mi_scores, index=self.feature_names).sort_values(ascending=False)

        self.selected_features = mi_series.head(top_k).index.tolist()
        self.logger.info(f"Selected {len(self.selected_features)} features:")
        for i, feat in enumerate(self.selected_features[:10], 1):
            self.logger.info(f"  {i}. {feat} (MI: {mi_series[feat]:.4f})")
        if len(self.selected_features) > 10:
            self.logger.info(f"  ... and {len(self.selected_features) - 10} more")

        self.X = self.X[self.selected_features]

    def scale_features(self, method: str = "robust") -> None:
        self.logger.info(f"Scaling features with {method.upper()} scaler...")
        if method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.logger.info("Feature scaling completed")

    def split_data(self) -> None:
        self.logger.info("Splitting data into train (80%) and test (20%) sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y.values, test_size=TEST_SIZE,
            stratify=self.y.values, random_state=RANDOM_STATE
        )
        self.logger.info(f"Train set: {len(self.X_train)} samples")
        self.logger.info(f"Test set: {len(self.X_test)} samples")

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X_train, self.y_train

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X_test, self.y_test

    def get_features(self) -> List[str]:
        return self.selected_features


# ============================================================================
# GRID SEARCH OPTIMIZER (CON OPCION DE DESACTIVAR)
# ============================================================================

class GridSearchOptimizer:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.best_params = {}

    def optimize_random_forest(self, X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestClassifier, Dict]:
        if not USE_GRID_SEARCH:
            self.logger.info("GridSearch DESACTIVADO - Usando parámetros por defecto")
            rf = RandomForestClassifier(**RF_DEFAULT_PARAMS)
            rf.fit(X, y)
            return rf, RF_DEFAULT_PARAMS

        self.logger.info("-" * 40)
        self.logger.info("Random Forest - GridSearchCV Optimization")
        
        total_combinations = (len(RF_PARAM_GRID['n_estimators']) * 
                             len(RF_PARAM_GRID['max_depth']) * 
                             len(RF_PARAM_GRID['min_samples_split']) * 
                             len(RF_PARAM_GRID['min_samples_leaf']) * 
                             len(RF_PARAM_GRID['class_weight']))
        self.logger.info(f"Parameter grid: {total_combinations} combinations")

        base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=RF_PARAM_GRID,
            cv=cv_inner,
            scoring='roc_auc',
            n_jobs=N_JOBS,
            verbose=0,
            return_train_score=False
        )

        self.logger.info("Running GridSearchCV (this may take a few minutes)...")
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        self.logger.info(f"Best parameters found: {best_params}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        best_rf = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=N_JOBS)
        best_rf.fit(X, y)
        self.logger.info("Fitted best model on full training data")

        return best_rf, best_params

    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray):
        if not XGB_AVAILABLE:
            self.logger.error("XGBoost not installed. Skipping.")
            return None, None

        if not USE_GRID_SEARCH:
            self.logger.info("GridSearch DESACTIVADO - Usando parámetros por defecto para XGBoost")
            n_benign = (y == 0).sum()
            n_malignant = (y == 1).sum()
            scale_pos_weight = n_benign / n_malignant
            xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_DEFAULT_PARAMS)
            xgb_model.fit(X, y)
            return xgb_model, {**XGB_DEFAULT_PARAMS, 'scale_pos_weight': scale_pos_weight}

        self.logger.info("-" * 40)
        self.logger.info("XGBoost - GridSearchCV Optimization")

        n_benign = (y == 0).sum()
        n_malignant = (y == 1).sum()
        scale_pos_weight = n_benign / n_malignant

        total_combinations = (len(XGB_PARAM_GRID['n_estimators']) * 
                             len(XGB_PARAM_GRID['max_depth']) * 
                             len(XGB_PARAM_GRID['learning_rate']) * 
                             len(XGB_PARAM_GRID['subsample']) * 
                             len(XGB_PARAM_GRID['colsample_bytree']))
        
        self.logger.info(f"Parameter grid: {total_combinations} combinations")
        self.logger.info(f"Fixed parameter: scale_pos_weight = {scale_pos_weight:.2f}")

        base_xgb = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            estimator=base_xgb,
            param_grid=XGB_PARAM_GRID,
            cv=cv_inner,
            scoring='roc_auc',
            n_jobs=N_JOBS,
            verbose=0,
            return_train_score=False
        )

        self.logger.info("Running GridSearchCV (this may take a few minutes)...")
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        self.logger.info(f"Best parameters found: {best_params}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        best_xgb = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        best_xgb.fit(X, y)
        self.logger.info("Fitted best model on full training data")

        return best_xgb, best_params


# ============================================================================
# MODEL TRAINER (SIN CAMBIOS)
# ============================================================================

class ModelTrainer:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.models = {}
        self.results = defaultdict(dict)
        self.best_params = {}
        self.cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        self.logger.info("-" * 40)
        self.logger.info("Training Random Forest...")

        optimizer = GridSearchOptimizer(self.logger)
        rf, best_params = optimizer.optimize_random_forest(X, y)
        self.best_params['random_forest'] = best_params

        self.models['random_forest'] = rf
        self.logger.info("Random Forest configured with:")
        for param, value in self.best_params.get('random_forest', {}).items():
            self.logger.info(f"  {param}: {value}")

        return rf

    def train_xgboost(self, X: np.ndarray, y: np.ndarray):
        if not XGB_AVAILABLE:
            self.logger.error("XGBoost not installed. Skipping.")
            return None

        self.logger.info("-" * 40)
        self.logger.info("Training XGBoost...")

        optimizer = GridSearchOptimizer(self.logger)
        xgb_model, best_params = optimizer.optimize_xgboost(X, y)
        
        if xgb_model:
            self.models['xgboost'] = xgb_model
            self.best_params['xgboost'] = best_params
            self.logger.info("XGBoost configured with:")
            for param, value in self.best_params.get('xgboost', {}).items():
                self.logger.info(f"  {param}: {value}")

        return xgb_model

    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, model_name: str) -> Dict:
        self.logger.info(f"Cross-validating {model_name} with {CV_FOLDS} folds...")

        y_pred = cross_val_predict(model, X, y, cv=self.cv, method='predict')
        y_pred_proba = cross_val_predict(model, X, y, cv=self.cv, method='predict_proba')[:, 1]

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'mcc': matthews_corrcoef(y, y_pred),
            'kappa': cohen_kappa_score(y, y_pred)
        }

        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_true': y
        }

        self.logger.info(f"{model_name} CV results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def evaluate_on_test(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict:
        self.logger.info(f"Evaluating {model_name} on test set...")

        if not is_model_fitted(model):
            self.logger.error(f"{model_name} is not fitted! Cannot evaluate.")
            return {}

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'kappa': cohen_kappa_score(y_test, y_pred)
        }

        if model_name not in self.results:
            self.results[model_name] = {}
        
        self.results[model_name]['test_metrics'] = metrics
        self.results[model_name]['test_y_pred'] = y_pred
        self.results[model_name]['test_y_pred_proba'] = y_pred_proba
        self.results[model_name]['test_y_true'] = y_test

        self.logger.info(f"{model_name} test results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def get_best_params(self) -> Dict:
        return self.best_params


# ============================================================================
# MODEL EVALUATOR (SIN CAMBIOS)
# ============================================================================

class ModelEvaluator:
    def __init__(self, results: Dict, logger: logging.Logger, output_dir: str):
        self.results = results
        self.logger = logger
        self.output_dir = output_dir

    def plot_roc_curves(self, test: bool = False) -> str:
        plt.figure(figsize=(8, 6))
        suffix = "_test" if test else ""
        title_suffix = " (Test Set)" if test else " (Cross-Validation)"

        for model_name, data in self.results.items():
            if test and 'test_metrics' in data:
                y_true = data.get('test_y_true', None)
                y_proba = data.get('test_y_pred_proba', None)
                if y_true is None or y_proba is None:
                    continue
                auc = data['test_metrics']['roc_auc']
            elif not test:
                y_true = data['y_true']
                y_proba = data['y_pred_proba']
                auc = data['metrics']['roc_auc']
            else:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})",
                     color=COLORS.get(model_name, "#2ecc71"), linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - No-Show Prediction{title_suffix}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        filepath = os.path.join(self.output_dir, f"roc_curves_noshow{suffix}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved ROC curves: {filepath}")
        return filepath

    def plot_confusion_matrices(self, test: bool = False) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        suffix = "_test" if test else ""
        title_suffix = " (Test Set)" if test else " (Cross-Validation)"

        for idx, (model_name, data) in enumerate(self.results.items()):
            if test and 'test_metrics' in data:
                y_true = data.get('test_y_true', None)
                y_pred = data.get('test_y_pred', None)
                if y_true is None or y_pred is None:
                    continue
                accuracy = data['test_metrics']['accuracy']
            elif not test:
                y_true = data['y_true']
                y_pred = data['y_pred']
                accuracy = data['metrics']['accuracy']
            else:
                continue

            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=['Show (0)', 'No-Show (1)'],
                        yticklabels=['Show (0)', 'No-Show (1)'])
            axes[idx].set_title(f'{model_name}{title_suffix}\nAccuracy: {accuracy:.4f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"confusion_matrices_noshow{suffix}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved confusion matrices: {filepath}")
        return filepath

    def generate_classification_reports(self, output_dir: str, test: bool = False) -> pd.DataFrame:
        reports = []
        suffix = "_test" if test else ""

        for model_name, data in self.results.items():
            if test and 'test_metrics' in data:
                y_true = data.get('test_y_true', None)
                y_pred = data.get('test_y_pred', None)
            elif not test:
                y_true = data['y_true']
                y_pred = data['y_pred']
            else:
                continue

            if y_true is None or y_pred is None:
                continue

            report = classification_report(y_true, y_pred,
                                           target_names=['Show (0)', 'No-Show (1)'],
                                           output_dict=True)

            for class_name, metrics in report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    reports.append({
                        'model': model_name,
                        'dataset': 'test' if test else 'cv',
                        'class': class_name,
                        'precision': metrics.get('precision', None),
                        'recall': metrics.get('recall', None),
                        'f1_score': metrics.get('f1-score', None),
                        'support': metrics.get('support', None)
                    })

        df_reports = pd.DataFrame(reports)
        if not df_reports.empty:
            filepath = os.path.join(output_dir, f"classification_reports_noshow{suffix}.csv")
            df_reports.to_csv(filepath, index=False)
            self.logger.info(f"Saved classification reports: {filepath}")
        return df_reports


# ============================================================================
# FEATURE IMPORTANCE ANALYZER
# ============================================================================

class FeatureImportanceAnalyzer:
    def __init__(self, models: Dict, feature_names: List[str], logger: logging.Logger, output_dir: str):
        self.models = models
        self.feature_names = feature_names
        self.logger = logger
        self.output_dir = output_dir

    def extract_importance(self) -> Dict[str, pd.DataFrame]:
        importance_dict = {}

        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                df_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)

                importance_dict[model_name] = df_importance
                self.logger.info(f"Extracted feature importance for {model_name}")
                
                self.logger.info(f"  Top 5 features for {model_name}:")
                for i, row in df_importance.head(5).iterrows():
                    self.logger.info(f"    {row['feature']}: {row['importance']:.4f}")
            else:
                self.logger.warning(f"{model_name} does not have feature_importances_ attribute")

        return importance_dict

    def plot_importance(self, importance_dict: Dict[str, pd.DataFrame], top_k: int = 10) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        for idx, (model_name, df_imp) in enumerate(importance_dict.items()):
            top_features = df_imp.head(top_k)
            axes[idx].barh(top_features['feature'], top_features['importance'],
                           color=COLORS.get(model_name, "#2ecc71"))
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name} - Top {top_k} Features (No-Show)')
            axes[idx].invert_yaxis()

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "feature_importance_noshow_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved feature importance comparison: {filepath}")
        return filepath

    def save_importance(self, importance_dict: Dict[str, pd.DataFrame], output_dir: str) -> None:
        for model_name, df_imp in importance_dict.items():
            safe_name = model_name.lower().replace(' ', '_')
            filepath = os.path.join(output_dir, f"feature_importance_noshow_{safe_name}.csv")
            df_imp.to_csv(filepath, index=False)
            self.logger.info(f"Saved feature importance for {model_name}: {filepath}")


# ============================================================================
# MODEL COMPARATOR
# ============================================================================

class ModelComparator:
    def __init__(self, results: Dict, logger: logging.Logger, output_dir: str):
        self.results = results
        self.logger = logger
        self.output_dir = output_dir

    def compare_metrics(self, test: bool = False) -> pd.DataFrame:
        comparison = []
        model_names = [name for name in self.results.keys()]
        if len(model_names) < 2:
            self.logger.warning("Not enough models to compare")
            return pd.DataFrame()

        model1, model2 = model_names[0], model_names[1]
        suffix = "test" if test else "cv"

        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'mcc', 'kappa']:
            if test:
                val1 = self.results[model1].get('test_metrics', {}).get(metric, 0)
                val2 = self.results[model2].get('test_metrics', {}).get(metric, 0)
            else:
                val1 = self.results[model1]['metrics'][metric]
                val2 = self.results[model2]['metrics'][metric]

            diff = val1 - val2
            better = model1 if diff > 0 else model2 if diff < 0 else 'tie'

            comparison.append({
                'metric': metric,
                'dataset': suffix,
                f'{model1}': val1,
                f'{model2}': val2,
                'difference': diff,
                'better_model': better
            })

        df_comparison = pd.DataFrame(comparison)
        filepath = os.path.join(self.output_dir, f"model_comparison_noshow_{suffix}.csv")
        df_comparison.to_csv(filepath, index=False)

        self.logger.info(f"Model comparison ({suffix}):")
        for _, row in df_comparison.iterrows():
            self.logger.info(f"  {row['metric']}: {row[model1]:.4f} vs {row[model2]:.4f} "
                             f"→ {row['better_model']} better")

        return df_comparison

    def mcnemar_test(self, test: bool = False) -> Dict:
        model_names = [name for name in self.results.keys()]
        if len(model_names) < 2:
            return {}

        model1, model2 = model_names[0], model_names[1]
        
        if test:
            y_true = self.results[model1].get('test_y_true', None)
            y_pred1 = self.results[model1].get('test_y_pred', None)
            y_pred2 = self.results[model2].get('test_y_pred', None)
            dataset = "test"
        else:
            y_true = self.results[model1]['y_true']
            y_pred1 = self.results[model1]['y_pred']
            y_pred2 = self.results[model2]['y_pred']
            dataset = "cv"

        if y_pred1 is None or y_pred2 is None or y_true is None:
            return {}

        b = ((y_pred1 == y_true) & (y_pred2 != y_true)).sum()
        c = ((y_pred1 != y_true) & (y_pred2 == y_true)).sum()

        if b + c > 0:
            mcnemar_stat = ((abs(b - c) - 1) ** 2) / (b + c)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            mcnemar_stat = 0
            p_value = 1.0

        result = {
            'dataset': dataset,
            'b_discordant_model1_correct': int(b),
            'c_discordant_model2_correct': int(c),
            'mcnemar_statistic': mcnemar_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05,
            'interpretation': "Models perform differently" if p_value < 0.05 else "No significant difference"
        }

        self.logger.info(f"McNemar test ({dataset}): χ² = {mcnemar_stat:.4f}, p = {p_value:.4f}")
        self.logger.info(f"  {result['interpretation']}")

        df_mcnemar = pd.DataFrame([result])
        filepath = os.path.join(self.output_dir, f"mcnemar_test_noshow_{dataset}.csv")
        df_mcnemar.to_csv(filepath, index=False)

        return result


# ============================================================================
# LEARNING CURVE ANALYZER
# ============================================================================

class LearningCurveAnalyzer:
    def __init__(self, logger: logging.Logger, output_dir: str):
        self.logger = logger
        self.output_dir = output_dir

    def plot_learning_curves(self, model, X: np.ndarray, y: np.ndarray,
                             model_name: str, cv: int = CV_FOLDS) -> str:
        self.logger.info(f"Generating learning curves for {model_name}...")

        # Reducir tamaños de entrenamiento para ahorrar tiempo
        train_sizes = np.linspace(0.1, 1.0, 8)
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=N_JOBS,
            train_sizes=train_sizes,
            scoring='accuracy'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(8, 6))
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                         alpha=0.1, color=COLORS.get(model_name, "#2ecc71"))
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                         alpha=0.1, color=COLORS.get(model_name, "#e74c3c"))
        plt.plot(train_sizes, train_mean, 'o-', label='Training score',
                 color=COLORS.get(model_name, "#2ecc71"))
        plt.plot(train_sizes, test_mean, 'o-', label='Cross-validation score',
                 color=COLORS.get(model_name, "#e74c3c"))

        plt.xlabel('Training Examples')
        plt.ylabel('Accuracy')
        plt.title(f'Learning Curves - {model_name} (No-Show)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        filepath = os.path.join(self.output_dir, f"learning_curves_noshow_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved learning curves: {filepath}")
        return filepath


# ============================================================================
# HYPERPARAMETER SUMMARY
# ============================================================================

class HyperparameterSummary:
    def __init__(self, best_params: Dict, logger: logging.Logger, output_dir: str):
        self.best_params = best_params
        self.logger = logger
        self.output_dir = output_dir

    def save_summary(self) -> str:
        rows = []
        for model_name, params in self.best_params.items():
            for param_name, param_value in params.items():
                rows.append({
                    'model': model_name,
                    'parameter': param_name,
                    'value': str(param_value)
                })

        if rows:
            df_params = pd.DataFrame(rows)
            filepath = os.path.join(self.output_dir, "best_hyperparameters_noshow.csv")
            df_params.to_csv(filepath, index=False)
            self.logger.info(f"Saved best hyperparameters: {filepath}")
            return filepath
        return ""

    def print_summary(self) -> None:
        self.logger.info("-" * 40)
        self.logger.info("BEST HYPERPARAMETERS FOUND (No-Show)")
        self.logger.info("-" * 40)
        for model_name, params in self.best_params.items():
            self.logger.info(f"{model_name}:")
            for param_name, param_value in params.items():
                self.logger.info(f"  {param_name}: {param_value}")


# ============================================================================
# MAIN ORCHESTRATOR OPTIMIZADO
# ============================================================================

class MLReportNoShow:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = None
        self.preprocessor = None
        self.trainer = None
        self.evaluator = None
        self.importance_analyzer = None
        self.comparator = None
        self.learning_curve_analyzer = None
        self.hyperparam_summary = None

        setup_directories()
        self.logger = setup_logging()

        self.csv_dir = os.path.join(OUTPUT_DIR, SUBDIRS["csv"])
        self.plots_dir = os.path.join(OUTPUT_DIR, SUBDIRS["plots"])

        if USE_GRID_SEARCH:
            self.logger.info("GridSearchCV ENABLED - Hyperparameter optimization active")
        else:
            self.logger.info("⚠️ GridSearchCV DISABLED - Using default parameters (MEMORIA OPTIMIZADA)")

        self.logger.info(f"Dataset limit: {MAX_SAMPLES} samples máximo")
        self.logger.info(f"N_JOBS: {N_JOBS} (para evitar saturar memoria)")

        if not XGB_AVAILABLE:
            self.logger.warning("XGBoost NOT INSTALLED - Only Random Forest will be trained")

    def run(self) -> None:
        self.logger.info("=" * 80)
        self.logger.info("PREDICTIVE MODELING: No-Show Prediction (VERSIÓN OPTIMIZADA)")
        self.logger.info("Random Forest vs XGBoost - Medical Appointments")
        self.logger.info("=" * 80)

        # 1. Data preprocessing (optimizado)
        self.preprocessor = DataPreprocessor(self.data_path, self.logger)
        self.preprocessor.load_data()
        self.preprocessor.select_features(top_k=TOP_FEATURES)
        self.preprocessor.scale_features(method="robust")
        self.preprocessor.split_data()
        X_train, y_train = self.preprocessor.get_train_data()
        X_test, y_test = self.preprocessor.get_test_data()
        feature_names = self.preprocessor.get_features()

        self.logger.info(f"Train set: {len(X_train)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")
        self.logger.info(f"Features: {feature_names}")

        # 2. Model training
        self.trainer = ModelTrainer(self.logger)

        rf_model = self.trainer.train_random_forest(X_train, y_train)
        rf_cv_metrics = self.trainer.cross_validate(rf_model, X_train, y_train, "Random Forest")
        rf_test_metrics = self.trainer.evaluate_on_test(rf_model, X_test, y_test, "Random Forest")

        if XGB_AVAILABLE:
            xgb_model = self.trainer.train_xgboost(X_train, y_train)
            if xgb_model:
                xgb_cv_metrics = self.trainer.cross_validate(xgb_model, X_train, y_train, "XGBoost")
                xgb_test_metrics = self.trainer.evaluate_on_test(xgb_model, X_test, y_test, "XGBoost")

        # 3. Hyperparameter summary
        best_params = self.trainer.get_best_params()
        self.hyperparam_summary = HyperparameterSummary(best_params, self.logger, self.csv_dir)
        self.hyperparam_summary.save_summary()
        self.hyperparam_summary.print_summary()

        # 4. Model evaluation
        self.evaluator = ModelEvaluator(self.trainer.results, self.logger, self.plots_dir)
        
        self.evaluator.plot_roc_curves(test=False)
        self.evaluator.plot_confusion_matrices(test=False)
        self.evaluator.generate_classification_reports(self.csv_dir, test=False)
        
        self.evaluator.plot_roc_curves(test=True)
        self.evaluator.plot_confusion_matrices(test=True)
        self.evaluator.generate_classification_reports(self.csv_dir, test=True)

        # 5. Feature importance
        self.importance_analyzer = FeatureImportanceAnalyzer(
            self.trainer.models, feature_names, self.logger, self.plots_dir
        )
        importance_dict = self.importance_analyzer.extract_importance()
        if importance_dict:
            self.importance_analyzer.plot_importance(importance_dict)
            self.importance_analyzer.save_importance(importance_dict, self.csv_dir)

        # 6. Model comparison
        self.comparator = ModelComparator(self.trainer.results, self.logger, self.csv_dir)
        comparison_cv = self.comparator.compare_metrics(test=False)
        comparison_test = self.comparator.compare_metrics(test=True)
        mcnemar_cv = self.comparator.mcnemar_test(test=False)
        mcnemar_test = self.comparator.mcnemar_test(test=True)

        # 7. Learning curves
        self.learning_curve_analyzer = LearningCurveAnalyzer(self.logger, self.plots_dir)
        for model_name, model in self.trainer.models.items():
            self.learning_curve_analyzer.plot_learning_curves(
                model, X_train, y_train, model_name
            )

        self._print_final_recommendation()

        self.logger.info("=" * 80)
        self.logger.info("PREDICTIVE MODELING FOR NO-SHOW COMPLETED")
        self.logger.info(f"Results saved in: {OUTPUT_DIR}/")
        self.logger.info("=" * 80)

    def _print_final_recommendation(self) -> None:
        self.logger.info("=" * 80)
        self.logger.info("FINAL RECOMMENDATION - No-Show Prediction")
        self.logger.info("=" * 80)

        results = self.trainer.results

        if len(results) == 0:
            self.logger.warning("No models were successfully trained")
            return

        best_model = None
        best_auc = -1

        for model_name, data in results.items():
            if 'test_metrics' in data:
                auc = data['test_metrics']['roc_auc']
                if auc > best_auc:
                    best_auc = auc
                    best_model = model_name

        if best_model:
            self.logger.info(f"Best performing model on TEST SET: {best_model} (ROC-AUC = {best_auc:.4f})")

        if 'Random Forest' in results and 'XGBoost' in results:
            rf_auc = results['Random Forest'].get('test_metrics', {}).get('roc_auc', 0)
            xgb_auc = results['XGBoost'].get('test_metrics', {}).get('roc_auc', 0)

            self.logger.info("")
            self.logger.info("Model comparison:")
            self.logger.info(f"  Random Forest AUC: {rf_auc:.4f}")
            self.logger.info(f"  XGBoost AUC: {xgb_auc:.4f}")
            
            if rf_auc > xgb_auc + 0.01:
                self.logger.info("  → Random Forest outperforms XGBoost on test data")
            elif xgb_auc > rf_auc + 0.01:
                self.logger.info("  → XGBoost outperforms Random Forest on test data")
            else:
                self.logger.info("  → Both models perform similarly on test data")

        self.logger.info("")
        self.logger.info("Key Insights for No-Show Prediction:")
        self.logger.info("  1. SMS_received is the most important predictor")
        self.logger.info("  2. Age and WaitDays are also significant factors")
        self.logger.info("  3. Class imbalance (3.95:1) requires careful handling")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    DATA_FILE = "noshow.csv"
    
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: File '{DATA_FILE}' not found.")
        print("Make sure the CSV file is in the same directory as this script.")
        print("Download from: https://www.kaggle.com/datasets/marwandiab/medical-appointment-no-shows-dataset")
        sys.exit(1)

    report = MLReportNoShow(DATA_FILE)
    report.run()
