"""
Preprocessor Module
===================
Outlier detection (MAD), temporal alignment, interpolasi data,
dan ekstraksi fitur geodetik (Strain, Displacement, Velocity Gradient)
untuk persiapan modeling.
"""

import logging
from typing import Dict, Tuple, Optional, List
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


class GeodeticFeatureExtractor:
    """
    Ekstraksi fitur fisik GNSS: Cumulative Displacement, Inter-station Strain,
    dan Velocity Gradient.
    """

    def __init__(self, config: Dict):
        self.config = config.get('feature_engineering', {})
        self.enabled = self.config.get('enabled', False)

    def calculate_cumulative_displacement(self, data_matrix: np.ndarray) -> np.ndarray:
        """
        Hitung pergeseran kumulatif sejak epoch pertama.
        Input: (n_epochs, n_stations, 3) -> Output: (n_epochs, n_stations, 3)
        """
        # Displacement = current_pos - initial_pos
        return data_matrix - data_matrix[0:1, :, :]

    def calculate_inter_station_strain(self, data_matrix: np.ndarray) -> np.ndarray:
        """
        Hitung strain relatif berdasarkan perubahan jarak antar semua pasangan stasiun.
        Input: (n_epochs, n_stations, 3) -> Output: (n_epochs, n_pairs)
        """
        n_epochs, n_stations, _ = data_matrix.shape

        # Hitung semua pasangan stasiun
        pairs = []
        for i in range(n_stations):
            for j in range(i + 1, n_stations):
                pairs.append((i, j))

        n_pairs = len(pairs)
        strain_matrix = np.zeros((n_epochs, n_pairs))

        for p_idx, (i, j) in enumerate(pairs):
            # Distansi Euclidean 3D: sqrt((Ei-Ej)^2 + (Ni-Nj)^2 + (Ui-Uj)^2)
            diff = data_matrix[:, i, :] - data_matrix[:, j, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))

            # Strain relatif: (D(t) - D(0)) / D(0)
            # Hindari division by zero
            d0 = dist[0] if dist[0] != 0 else 1e-6
            strain_matrix[:, p_idx] = (dist - d0) / d0

        return strain_matrix

    def calculate_velocity_gradient(self, data_matrix: np.ndarray, time_array: np.ndarray) -> np.ndarray:
        """
        Hitung gradien kecepatan (akselerasi) dari koordinat.
        Input: (n_epochs, n_stations, 3) -> Output: (n_epochs, n_stations, 3)
        """
        # time_array biasanya dalam decimal year, konversi ke hari untuk unit m/day
        dt = np.diff(time_array) * 365.25
        dt = np.where(dt == 0, 1e-6, dt) # avoid div by zero

        # Velocity V = dPos / dt
        v = np.diff(data_matrix, axis=0) / dt[:, np.newaxis, np.newaxis]

        # Gradient G = dV / dt
        g = np.diff(v, axis=0) / dt[:-1, np.newaxis, np.newaxis]

        # Padding agar panjangnya tetap n_epochs
        # Gunakan padding konstanta (0) di awal
        padding = np.zeros((1, data_matrix.shape[1], 3))
        return np.vstack([padding, g]) if g.shape[0] < data_matrix.shape[0] else g[:data_matrix.shape[0]]

    def extract_all_features(self, data_matrix: np.ndarray, time_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Ekstraksi semua fitur yang diaktifkan di config.
        """
        features = {}

        if not self.enabled:
            return features

        if self.config.get('calculate_cumulative_disp', False):
            features['cum_disp'] = self.calculate_cumulative_displacement(data_matrix)

        if self.config.get('calculate_strain', False):
            features['strain'] = self.calculate_inter_station_strain(data_matrix)

        if self.config.get('calculate_velocity_gradient', False):
            features['vel_grad'] = self.calculate_velocity_gradient(data_matrix, time_array)

        return features


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
        """
        data_clean = data[~np.isnan(data)]
        if len(data_clean) < 3:
            return np.zeros(len(data), dtype=bool)

        median = np.median(data_clean)
        mad = np.median(np.abs(data_clean - median))
        if mad == 0:
            mad = 1.0

        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        return outlier_mask

    @staticmethod
    def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers menggunakan Z-score.
        """
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

        cleaned_data = data.copy()
        cleaned_data[outlier_mask] = np.nan
        return cleaned_data, outlier_mask

    def interpolate_gaps(self, time: np.ndarray, data: np.ndarray,
                        axis: str = 'E') -> np.ndarray:
        """
        Interpolasi missing values (NaN) di data.
        """
        valid_mask = ~np.isnan(data)
        if np.sum(valid_mask) < 3:
            logger.warning(f"Axis '{axis}': Insufficient valid data for interpolation")
            return data

        valid_time = time[valid_mask]
        valid_data = data[valid_mask]

        time_diffs = np.diff(valid_time)
        large_gaps = time_diffs > self.config.max_gap_days

        n_gaps = np.sum(large_gaps)
        if n_gaps > 0:
            logger.info(f"Axis '{axis}': Found {n_gaps} gaps > {self.config.max_gap_days} days")

        try:
            if self.config.interpolation_method == "pchip":
                interpolator = PchipInterpolator(valid_time, valid_data)
            elif self.config.interpolation_method == "spline":
                interpolator = CubicSpline(valid_time, valid_data)
            else:
                raise ValueError(f"Unknown method: {self.config.interpolation_method}")

            interpolated = interpolator(time)
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
        """
        processed = {}
        norm_params = {}

        for axis, data in coordinates.items():
            logger.info(f"\nProcessing axis: {axis}")
            data_cleaned, _ = self.remove_outliers(data, axis)
            data_interp = self.interpolate_gaps(time, data_cleaned, axis)

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

    @staticmethod
    def normalize_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Normalize a feature matrix column-wise.
        Returns normalized matrix and a list of params for each column.
        """
        n_cols = matrix.shape[1]
        norm_matrix = np.zeros_like(matrix)
        all_params = []

        for i in range(n_cols):
            col = matrix[:, i]
            valid_data = col[~np.isnan(col)]
            if len(valid_data) == 0:
                all_params.append({})
                continue

            d_min = np.min(valid_data)
            d_max = np.max(valid_data)
            if d_max == d_min:
                norm_matrix[:, i] = 0
            else:
                norm_matrix[:, i] = (col - d_min) / (d_max - d_min)

            all_params.append({"type": "minmax", "min": d_min, "max": d_max})

        return norm_matrix, all_params


class PipelinePreprocessor:
    """Wrapper untuk processing multiple stations dalam pipeline."""

    def __init__(self, config: Optional[PreprocessingConfig] = None, full_config: Optional[Dict] = None):
        """
        Initialize PipelinePreprocessor.
        """
        self.preprocessor = Preprocessor(config)
        self.normalization_params = {}
        # Initialize Feature Extractor
        self.feature_extractor = GeodeticFeatureExtractor(full_config or {})

    def process_all_stations(self,
                            time_array: np.ndarray,
                            data_matrix: np.ndarray,
                            station_names: list) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Process semua stations sekaligus + Ekstraksi Fitur Geodetik.
        Two-pass outlier detection: Pass 1 pada data mentah (threshold 5.0),
        Pass 2 pada data ternormalisasi (threshold 3.0).

        Args:
            time_array (np.ndarray): Time array (n_epochs,)
            data_matrix (np.ndarray): Data matrix (n_epochs, n_stations, 3)
            station_names (list): Station names

        Returns:
            Tuple[np.ndarray, Dict, Dict]: (processed_matrix, all_params, geodetic_features)
        """
        n_epochs, n_stations, n_axes = data_matrix.shape
        processed_matrix = np.zeros_like(data_matrix)
        axes = ['E', 'N', 'U']

        # --- STEP A: Pass 1 — Cleaning pada data mentah (threshold tinggi) ---
        # Gunakan threshold 5.0 agar tidak membuang sinyal asli yang range-nya besar
        logger.info(f"Pass 1: Cleaning raw data (MAD threshold=5.0) for {n_stations} stations...")

        raw_preprocessor = Preprocessor(PreprocessingConfig(
            outlier_method=self.preprocessor.config.outlier_method,
            outlier_threshold=5.0,
            interpolation_method=self.preprocessor.config.interpolation_method,
            max_gap_days=self.preprocessor.config.max_gap_days,
            normalize=False
        ))

        cleaned_matrix = np.zeros_like(data_matrix)

        for station_idx, station_name in enumerate(station_names):
            station_data = data_matrix[:, station_idx, :]
            coords = {axes[i]: station_data[:, i] for i in range(3)}

            res_coords = {}
            for axis, data in coords.items():
                data_cleaned, _ = raw_preprocessor.remove_outliers(data, axis)
                data_interp = raw_preprocessor.interpolate_gaps(time_array, data_cleaned, axis)
                res_coords[axis] = data_interp

            for axis_idx, axis in enumerate(axes):
                cleaned_matrix[:, station_idx, axis_idx] = res_coords[axis]

        # --- STEP B: Geodetic Feature Extraction (HITUNG DALAM METER) ---
        logger.info("Extracting geodetic features (Strain, Cumulative Disp, Vel Grad)...")
        geodetic_features = self.feature_extractor.extract_all_features(cleaned_matrix, time_array)

        # --- STEP C: Normalization ---
        logger.info("Normalizing data matrix...")
        normalized_matrix = np.zeros_like(data_matrix)

        for station_idx, station_name in enumerate(station_names):
            station_data = cleaned_matrix[:, station_idx, :]
            coords = {axes[i]: station_data[:, i] for i in range(3)}

            processed_coords, params = self.preprocessor.process_station_coordinates(
                time_array, coords
            )
            self.normalization_params[station_name] = params

            for axis_idx, axis in enumerate(axes):
                normalized_matrix[:, station_idx, axis_idx] = processed_coords[axis]

        # --- STEP D: Pass 2 — Cleaning pada data ternormalisasi (threshold 3.0) ---
        # Tangkap outlier halus yang lolos pass 1
        logger.info("Pass 2: Cleaning normalized data (MAD threshold=3.0)...")

        norm_preprocessor = Preprocessor(PreprocessingConfig(
            outlier_method=self.preprocessor.config.outlier_method,
            outlier_threshold=3.0,
            interpolation_method=self.preprocessor.config.interpolation_method,
            max_gap_days=self.preprocessor.config.max_gap_days,
            normalize=False
        ))

        for station_idx, station_name in enumerate(station_names):
            for axis_idx, axis in enumerate(axes):
                col = normalized_matrix[:, station_idx, axis_idx]
                col_cleaned, mask = norm_preprocessor.remove_outliers(col, f"{station_name}/{axis}")
                col_interp = norm_preprocessor.interpolate_gaps(time_array, col_cleaned, f"{station_name}/{axis}")
                normalized_matrix[:, station_idx, axis_idx] = col_interp

        return normalized_matrix, self.normalization_params, geodetic_features
