"""
GNSS Deformation VMD-CNN-LSTM Pipeline
========================================

Package untuk prediksi deformasi GNSS menggunakan hybrid model
Variational Mode Decomposition dan CNN-LSTM.

Modules:
--------
- excel_extractor: Ekstraksi data Excel → TXT per stasiun
- data_loader: Load dan parse TXT files
- preprocessor: Outlier removal, interpolasi, normalisasi
- vmd_processor: Variational Mode Decomposition
- models: CNN-LSTM architecture
- gmt_exporter: Export untuk GMT visualization
"""

__version__ = "1.0.0"
__author__ = "GNSS Pipeline Team"
__email__ = "info@gnss-pipeline.com"

# Import main components
from src.excel_extractor import ExcelExtractor
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor, PipelinePreprocessor
from src.vmd_processor import VMDProcessor, PipelineVMD
from src.models import CNNLSTMModel, ModelTrainer
from src import gmt_exporter

__all__ = [
    'ExcelExtractor',
    'DataLoader',
    'Preprocessor',
    'PipelinePreprocessor',
    'VMDProcessor',
    'PipelineVMD',
    'CNNLSTMModel',
    'ModelTrainer',
]

print(f"GNSS Deformation Pipeline v{__version__}")