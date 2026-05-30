"""
Pipeline Module
===============
Orkestrasi ringan untuk menjalankan step-step pipeline secara modular.
File ini melengkapi main.py dengan class Pipeline yang bisa dipanggil
secara parsial (berguna untuk debugging per-step).
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

from src.excel_extractor import ExcelExtractor, main_extract
from src.data_loader import DataLoader
from src.preprocessor import PipelinePreprocessor, PreprocessingConfig
from src.vmd_processor import PipelineVMD, VMDConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orkestrasi modular pipeline GNSS VMD-CNN-LSTM.

    Setiap step menyimpan hasilnya di atribut instance sehingga
    bisa di-inspect atau di-resume dari tengah.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.cfg = yaml.safe_load(f)

        # State per step
        self.extracted_files: Dict[str, str] = {}
        self.loader: Optional[DataLoader] = None
        self.time_array: Optional[np.ndarray] = None
        self.data_matrix: Optional[np.ndarray] = None
        self.station_names: List[str] = []
        self.processed_matrix: Optional[np.ndarray] = None
        self.norm_params: Dict = {}
        self.vmd_pipeline: Optional[PipelineVMD] = None
        self.feature_matrix: Optional[np.ndarray] = None

        logger.info(f"Pipeline loaded: {self.cfg['project']['name']}")

    # ------------------------------------------------------------------
    # Step 1 — Excel Extraction
    # ------------------------------------------------------------------
    def run_extraction(self) -> Dict[str, str]:
        """Step 1: Ekstrak Excel → TXT per stasiun."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: EXCEL EXTRACTION")
        logger.info("=" * 60)

        excel_path = self.cfg['data']['excel_master']
        out_dir = self.cfg['data']['txt_stations_dir']

        extractor = ExcelExtractor(excel_path, out_dir)
        self.extracted_files = extractor.extract_all_stations()
        summary = extractor.get_extraction_summary()

        logger.info(f"✓ {summary['total_stations']} stasiun diekstrak "
                    f"({summary['total_file_size_mb']:.2f} MB)")
        return self.extracted_files

    # ------------------------------------------------------------------
    # Step 2 — Data Loading
    # ------------------------------------------------------------------
    # def run_loading(self) -> Dict:
    #     """Step 2: Load TXT → GNSSStation objects, align ke common time."""
    #     logger.info("\n" + "=" * 60)
    #     logger.info("STEP 2: DATA LOADING")
    #     logger.info("=" * 60)

    #     self.loader = DataLoader(self.cfg['data']['txt_stations_dir'])
    #     self.loader.load_all_stations()
    #     self.loader.align_to_common_time()

    #     self.time_array, self.data_matrix, self.station_names = (
    #         self.loader.get_data_matrix()
    #     )

    #     stats = self.loader.get_statistics()
    #     logger.info(f"✓ {stats['n_stations']} stasiun, "
    #                 f"{stats['n_epochs']} epoch, "
    #                 f"data matrix {self.data_matrix.shape}")
    #     return stats
    
    def run_loading(self) -> Dict:
        """Step 2: Load TXT → GNSSStation objects, align ke common time."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: DATA LOADING")
        logger.info("=" * 60)

        self.loader = DataLoader(self.cfg['data']['txt_stations_dir'])
        
        # UBAH BARIS INI: Kirim konfigurasi stasiun dari yaml
        self.loader.load_all_stations(station_cfg=self.cfg.get('stations'))
        
        self.loader.align_to_common_time()

        self.time_array, self.data_matrix, self.station_names = (
            self.loader.get_data_matrix()
        )

        stats = self.loader.get_statistics()
        logger.info(f"✓ {stats['n_stations']} stasiun, "
                    f"{stats['n_epochs']} epoch, "
                    f"data matrix {self.data_matrix.shape}")
        return stats

    # ------------------------------------------------------------------
    # Step 3 — Preprocessing
    # ------------------------------------------------------------------
    def run_preprocessing(self) -> np.ndarray:
        """Step 3: Outlier removal (MAD) + PCHIP interpolasi + normalisasi."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: PREPROCESSING")
        logger.info("=" * 60)

        if self.data_matrix is None:
            raise RuntimeError("Jalankan run_loading() terlebih dahulu")

        pp_cfg = self.cfg['preprocessing']
        config = PreprocessingConfig(
            outlier_method=pp_cfg['outlier_detection']['method'],
            outlier_threshold=pp_cfg['outlier_detection']['threshold'],
            interpolation_method=pp_cfg['interpolation']['method'],
            max_gap_days=pp_cfg['interpolation']['max_gap_days'],
            normalize=True,
            normalization_type=pp_cfg['normalization']
        )

        pipeline_pp = PipelinePreprocessor(config)
        self.processed_matrix, self.norm_params = pipeline_pp.process_all_stations(
            self.time_array, self.data_matrix, self.station_names
        )

        logger.info(f"✓ Preprocessed matrix shape: {self.processed_matrix.shape}")
        return self.processed_matrix

    # ------------------------------------------------------------------
    # Step 4 — VMD Decomposition
    # ------------------------------------------------------------------
    def run_vmd(self) -> np.ndarray:
        """Step 4: Dekomposisi VMD → feature matrix untuk ML."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: VMD DECOMPOSITION")
        logger.info("=" * 60)

        if self.processed_matrix is None:
            raise RuntimeError("Jalankan run_preprocessing() terlebih dahulu")

        vmd_cfg = self.cfg['vmd']
        vmd_config = VMDConfig(
            n_modes=vmd_cfg['n_modes'],
            alpha=vmd_cfg['alpha'],
            tau=vmd_cfg['tau'],
            DC=vmd_cfg.get('DC', 0),
            init=vmd_cfg.get('init', 1),
            tol=vmd_cfg['tolerance'],
        )

        self.vmd_pipeline = PipelineVMD(vmd_config)
        self.vmd_pipeline.decompose_all_stations(
            self.processed_matrix, self.station_names
        )

        self.feature_matrix = self.vmd_pipeline.get_imf_feature_matrix()
        logger.info(f"✓ Feature matrix shape: {self.feature_matrix.shape}")
        return self.feature_matrix

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_epoch_window(self, decimal_year_start: float,
                          decimal_year_end: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Potong data ke jendela waktu tertentu.

        Returns:
            (time_slice, data_slice) dengan shape (T', n_stations, 3)
        """
        if self.time_array is None or self.processed_matrix is None:
            raise RuntimeError("Belum ada data. Jalankan run_preprocessing().")

        mask = (self.time_array >= decimal_year_start) & \
               (self.time_array <= decimal_year_end)
        return self.time_array[mask], self.processed_matrix[mask]

    def get_pre_seismic(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ambil slice data periode sebelum gempa."""
        eq = self.cfg['seismic_event']
        end = _parse_date_to_decimal(eq['earthquake_date'])
        start = end - eq['pre_seismic_days'] / 365.25
        return self.get_epoch_window(start, end)

    def get_co_seismic(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ambil slice data periode co-seismic."""
        eq = self.cfg['seismic_event']
        start = _parse_date_to_decimal(eq['earthquake_date'])
        end = start + eq['post_seismic_days'] / 365.25
        return self.get_epoch_window(start, end)


def _parse_date_to_decimal(date_str: str) -> float:
    """Konversi 'YYYY-MM-DD' ke decimal year."""
    from datetime import datetime
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year_start = datetime(dt.year, 1, 1)
    year_end = datetime(dt.year + 1, 1, 1)
    fraction = (dt - year_start).total_seconds() / (year_end - year_start).total_seconds()
    return dt.year + fraction


def main_pipeline():
    """Demo: jalankan pipeline sampai step VMD."""
    pipe = Pipeline("config/config.yaml")

    try:
        pipe.run_extraction()
    except FileNotFoundError as e:
        logger.warning(f"Ekstraksi dilewati: {e}")

    try:
        pipe.run_loading()
        pipe.run_preprocessing()
        pipe.run_vmd()
    except Exception as e:
        logger.error(f"Pipeline berhenti: {e}")
        raise


if __name__ == "__main__":
    main_pipeline()
