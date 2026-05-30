"""
Preprocessor Module
===================
Outlier detection (MAD), temporal alignment, interpolasi data
untuk persiapan modeling.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.stats import zscore
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration untuk preprocessing."""
    outlier_method: str = "mad"        # 'mad' or 'zscore'
    outlier_threshold: float = 3.0     # threshold untuk outlier detection
    interpolation_method: str = "pchip"  # 'pchip' or 'spline'
    max_gap_days: float = 5.0          # max gap untuk interpolasi
    normalize: bool = True
    normalization_type: str = "minmax"  # 'minmax' or 'zscore'


class Preprocessor:
    """Preprocessing untuk GNSS time series data."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize Preprocessor.
        
        Args:
            config (PreprocessingConfig): Configuration object
        """
        self.config = config or PreprocessingConfig()
        logger.info(f"Preprocessor initialized with config: {self.config}")
    
    @staticmethod
    def detect_outliers_mad(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers menggunakan Median Absolute Deviation (MAD).
        
        Args:
            data (np.ndarray): Input data
            threshold (float): Threshold multiplier (default: 3.0)
        
        Returns:
            np.ndarray: Boolean mask untuk outliers
        """
        # Remove NaN first
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 3:
            return np.zeros(len(data), dtype=bool)
        
        median = np.median(data_clean)
        mad = np.median(np.abs(data_clean - median))
        
        # Avoid division by zero
        if mad == 0:
            mad = 1.0
        
        # Standardized MAD
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        return outlier_mask
    
    @staticmethod
    def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers menggunakan Z-score.
        
        Args:
            data (np.ndarray): Input data
            threshold (float): Threshold (default: 3.0 std)
        
        Returns:
            np.ndarray: Boolean mask untuk outliers
        """
        # Remove NaN
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 2:
            return np.zeros(len(data), dtype=bool)
        
        z_scores = np.abs(zscore(data_clean, nan_policy='omit'))
        outlier_indices = np.where(z_scores > threshold)[0]
        
        outlier_mask = np.zeros(len(data), dtype=bool)
        valid_indices = np.where(~np.isnan(data))[0]
        outlier_mask[valid_indices[outlier_indices]] = True
        
        return outlier_mask
    
    def remove_outliers(self, data: np.ndarray, axis: str = 'E') -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers dari time series.
        
        Args:
            data (np.ndarray): Time series data
            axis (str): Axis name untuk logging
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (cleaned_data, outlier_mask)
        """
        if self.config.outlier_method == "mad":
            outlier_mask = self.detect_outliers_mad(data, self.config.outlier_threshold)
        elif self.config.outlier_method == "zscore":
            outlier_mask = self.detect_outliers_zscore(data, self.config.outlier_threshold)
        else:
            raise ValueError(f"Unknown outlier method: {self.config.outlier_method}")
        
        n_outliers = np.sum(outlier_mask)
        if n_outliers > 0:
            logger.info(f"Axis '{axis}': Detected {n_outliers} outliers ({100*n_outliers/len(data):.2f}%)")
        
        # Mark outliers as NaN
        cleaned_data = data.copy()
        cleaned_data[outlier_mask] = np.nan
        
        return cleaned_data, outlier_mask
    
    def interpolate_gaps(self, time: np.ndarray, data: np.ndarray, 
                        axis: str = 'E') -> np.ndarray:
        """
        Interpolasi missing values (NaN) di data.
        
        Args:
            time (np.ndarray): Time array (MJD atau indeks)
            data (np.ndarray): Data dengan potential NaN
            axis (str): Axis name untuk logging
        
        Returns:
            np.ndarray: Interpolated data
        """
        # Find valid indices
        valid_mask = ~np.isnan(data)
        
        if np.sum(valid_mask) < 3:
            logger.warning(f"Axis '{axis}': Insufficient valid data for interpolation")
            return data
        
        valid_time = time[valid_mask]
        valid_data = data[valid_mask]
        
        # Check gaps
        time_diffs = np.diff(valid_time)
        large_gaps = time_diffs > self.config.max_gap_days
        
        n_gaps = np.sum(large_gaps)
        if n_gaps > 0:
            logger.info(f"Axis '{axis}': Found {n_gaps} gaps > {self.config.max_gap_days} days")
        
        # Interpolate
        try:
            if self.config.interpolation_method == "pchip":
                interpolator = PchipInterpolator(valid_time, valid_data)
            elif self.config.interpolation_method == "spline":
                interpolator = CubicSpline(valid_time, valid_data)
            else:
                raise ValueError(f"Unknown method: {self.config.interpolation_method}")
            
            interpolated = interpolator(time)
            
            # Handle extrapolation (set to NaN)
            interpolated[time < valid_time[0]] = np.nan
            interpolated[time > valid_time[-1]] = np.nan
            
            n_filled = np.sum(~valid_mask & ~np.isnan(interpolated))
            logger.info(f"Axis '{axis}': Filled {n_filled} missing values")
            
            return interpolated
            
        except Exception as e:
            logger.warning(f"Interpolation failed for axis '{axis}': {e}")
            return data
    
    def normalize_data(self, data: np.ndarray, axis: str = 'E') -> Tuple[np.ndarray, Dict]:
        """
        Normalize data (minmax atau zscore).
        
        Args:
            data (np.ndarray): Input data
            axis (str): Axis name
        
        Returns:
            Tuple[np.ndarray, Dict]: (normalized_data, normalization_params)
        """
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            logger.warning(f"Axis '{axis}': No valid data for normalization")
            return data, {}
        
        params = {}
        
        if self.config.normalization_type == "minmax":
            data_min = np.min(valid_data)
            data_max = np.max(valid_data)
            
            if data_max == data_min:
                normalized = np.zeros_like(data)
            else:
                normalized = (data - data_min) / (data_max - data_min)
            
            params = {"type": "minmax", "min": data_min, "max": data_max}
            
        elif self.config.normalization_type == "zscore":
            data_mean = np.mean(valid_data)
            data_std = np.std(valid_data)
            
            if data_std == 0:
                normalized = np.zeros_like(data)
            else:
                normalized = (data - data_mean) / data_std
            
            params = {"type": "zscore", "mean": data_mean, "std": data_std}
        
        else:
            raise ValueError(f"Unknown normalization: {self.config.normalization_type}")
        
        logger.info(f"Axis '{axis}': Normalized with {self.config.normalization_type}")
        return normalized, params
    
    def process_station_coordinates(self, 
                                   time: np.ndarray,
                                   coordinates: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Process koordinat untuk satu stasiun (outlier removal + interpolation + normalization).
        
        Args:
            time (np.ndarray): Time array
            coordinates (Dict): {axis: data_array} dengan axis='E','N','U'
        
        Returns:
            Tuple[Dict, Dict]: (processed_coordinates, normalization_params)
        """
        processed = {}
        norm_params = {}
        
        for axis, data in coordinates.items():
            logger.info(f"\nProcessing axis: {axis}")
            
            # Step 1: Outlier removal
            data_cleaned, _ = self.remove_outliers(data, axis)
            
            # Step 2: Interpolation
            data_interp = self.interpolate_gaps(time, data_cleaned, axis)
            
            # Step 3: Normalization
            if self.config.normalize:
                data_norm, params = self.normalize_data(data_interp, axis)
                norm_params[axis] = params
            else:
                data_norm = data_interp
                norm_params[axis] = {}
            
            processed[axis] = data_norm
        
        return processed, norm_params
    
    @staticmethod
    def denormalize_data(normalized: np.ndarray, params: Dict) -> np.ndarray:
        """
        Denormalize data back to original scale.
        
        Args:
            normalized (np.ndarray): Normalized data
            params (Dict): Normalization parameters
        
        Returns:
            np.ndarray: Denormalized data
        """
        if not params or "type" not in params:
            return normalized
        
        if params["type"] == "minmax":
            denorm = normalized * (params["max"] - params["min"]) + params["min"]
        elif params["type"] == "zscore":
            denorm = normalized * params["std"] + params["mean"]
        else:
            denorm = normalized
        
        return denorm


class PipelinePreprocessor:
    """Wrapper untuk processing multiple stations dalam pipeline."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize PipelinePreprocessor.
        
        Args:
            config (PreprocessingConfig): Configuration
        """
        self.preprocessor = Preprocessor(config)
        self.normalization_params = {}
    
    def process_all_stations(self, 
                            time_array: np.ndarray,
                            data_matrix: np.ndarray,
                            station_names: list) -> Tuple[np.ndarray, Dict]:
        """
        Process semua stations sekaligus.
        
        Args:
            time_array (np.ndarray): Time array (n_epochs,)
            data_matrix (np.ndarray): Data matrix (n_epochs, n_stations, 3)
            station_names (list): Station names
        
        Returns:
            Tuple[np.ndarray, Dict]: (processed_matrix, all_params)
        """
        n_epochs, n_stations, n_axes = data_matrix.shape
        processed_matrix = np.zeros_like(data_matrix)
        
        axes = ['E', 'N', 'U']
        
        logger.info(f"Processing {n_stations} stations...")
        
        for station_idx, station_name in enumerate(station_names):
            logger.info(f"\n[{station_idx+1}/{n_stations}] Processing {station_name}")
            
            station_data = data_matrix[:, station_idx, :]  # (n_epochs, 3)
            
            coords = {axes[i]: station_data[:, i] for i in range(3)}
            processed_coords, params = self.preprocessor.process_station_coordinates(
                time_array, coords
            )
            
            # Store normalization params
            self.normalization_params[station_name] = params
            
            # Store processed data
            for axis_idx, axis in enumerate(axes):
                processed_matrix[:, station_idx, axis_idx] = processed_coords[axis]
        
        return processed_matrix, self.normalization_params


def main_preprocess():
    """Example usage."""
    logger.info("Preprocessor module loaded successfully")


if __name__ == "__main__":
    main_preprocess()