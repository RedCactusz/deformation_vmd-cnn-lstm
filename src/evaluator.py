"""
Evaluator Module
================
Menghitung evaluation metrics untuk model predictions.
"""

import logging
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Kalkulasi berbagai evaluation metrics."""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared coefficient."""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Normalized Root Mean Squared Error."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse / (np.max(y_true) - np.min(y_true))


class ModelEvaluator:
    """Evaluasi performa model secara komprehensif."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = MetricsCalculator()
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate predictions.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        
        Returns:
            Dict: Dictionary dengan semua metrics
        """
        results = {
            'rmse': self.metrics.rmse(y_true, y_pred),
            'mae': self.metrics.mae(y_true, y_pred),
            'mape': self.metrics.mape(y_true, y_pred),
            'r2': self.metrics.r2(y_true, y_pred),
            'nrmse': self.metrics.nrmse(y_true, y_pred)
        }
        
        return results
    
    def evaluate_per_axis(self, y_true: np.ndarray, y_pred: np.ndarray,
                         axes: list = ['E', 'N', 'U']) -> Dict:
        """
        Evaluate per axis (E, N, U).
        
        Args:
            y_true (np.ndarray): Shape (n_samples, 3)
            y_pred (np.ndarray): Shape (n_samples, 3)
            axes (list): Axis names
        
        Returns:
            Dict: Metrics per axis
        """
        results = {}
        
        for i, axis in enumerate(axes):
            if y_true.shape[1] > i:
                axis_metrics = self.evaluate(
                    y_true[:, i],
                    y_pred[:, i]
                )
                results[axis] = axis_metrics
        
        return results
    
    def print_report(self, results: Dict):
        """Print evaluation report."""
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        
        if 'rmse' in results:
            print(f"RMSE:  {results['rmse']:.6f}")
            print(f"MAE:   {results['mae']:.6f}")
            print(f"MAPE:  {results['mape']:.2f}%")
            print(f"R²:    {results['r2']:.4f}")
            print(f"NRMSE: {results['nrmse']:.4f}")
        
        print("="*50)


if __name__ == "__main__":
    # Test
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(y_true, y_pred)
    evaluator.print_report(results)