"""
Quick Start Example
===================
Contoh penggunaan pipeline (untuk testing dan development).
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def example_excel_extraction():
    """Contoh: Extract data dari Excel."""
    print("\n" + "="*60)
    print("EXAMPLE 1: EXCEL EXTRACTION")
    print("="*60)
    
    from src.excel_extractor import ExcelExtractor
    
    excel_path = "data/raw/excel_master/gnss_master.xlsx"
    output_dir = "data/raw/txt_stations/"
    
    try:
        extractor = ExcelExtractor(excel_path, output_dir)
        
        # Get sheet names
        sheets = extractor.get_sheet_names()
        print(f"Found {len(sheets)} sheets: {sheets[:5]}...")
        
        # Extract all
        results = extractor.extract_all_stations()
        summary = extractor.get_extraction_summary()
        
        print(f"\n✓ Extracted {summary['total_stations']} stations")
        print(f"  Size: {summary['total_file_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_data_loading():
    """Contoh: Load TXT files."""
    print("\n" + "="*60)
    print("EXAMPLE 2: DATA LOADING")
    print("="*60)
    
    from src.data_loader import DataLoader
    
    txt_dir = "data/raw/txt_stations/"
    
    try:
        loader = DataLoader(txt_dir)
        stations = loader.load_all_stations()
        
        stats = loader.get_statistics()
        print(f"✓ Loaded {stats['n_stations']} stations")
        print(f"  Epochs: {stats['n_epochs']}")
        print(f"  Stations: {stats['station_names'][:3]}...")
        
        # Get data matrix
        data_matrix, stn_names = loader.get_data_matrix()
        print(f"  Data shape: {data_matrix.shape}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_preprocessing():
    """Contoh: Preprocessing."""
    print("\n" + "="*60)
    print("EXAMPLE 3: PREPROCESSING")
    print("="*60)
    
    from src.preprocessor import PipelinePreprocessor, PreprocessingConfig
    from src.data_loader import DataLoader
    
    try:
        # Load data first
        loader = DataLoader("data/raw/txt_stations/")
        loader.load_all_stations()
        
        data_matrix, station_names = loader.get_data_matrix()
        first_station = loader.get_station(station_names[0])
        time_array = first_station.get_epochs()
        
        # Preprocess
        config = PreprocessingConfig()
        preprocessor = PipelinePreprocessor(config)
        
        processed, params = preprocessor.process_all_stations(
            time_array, data_matrix, station_names
        )
        
        print(f"✓ Preprocessing completed")
        print(f"  Input shape: {data_matrix.shape}")
        print(f"  Output shape: {processed.shape}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_vmd_decomposition():
    """Contoh: VMD decomposition."""
    print("\n" + "="*60)
    print("EXAMPLE 4: VMD DECOMPOSITION")
    print("="*60)
    
    from src.vmd_processor import VMDProcessor, VMDConfig
    
    try:
        # Create synthetic signal
        t = np.linspace(0, 1, 1000)
        signal = (np.sin(2*np.pi*5*t) + 
                 np.sin(2*np.pi*10*t) + 
                 np.sin(2*np.pi*20*t) +
                 np.random.randn(1000) * 0.1)
        
        # VMD
        config = VMDConfig(n_modes=3)
        processor = VMDProcessor(config)
        
        decomp = processor.decompose_signal(signal)
        
        print(f"✓ VMD decomposition completed")
        print(f"  Input shape: {signal.shape}")
        print(f"  IMF shape: {decomp['imfs'].shape}")
        print(f"  Reconstruction error: {np.mean(np.abs(decomp['residual'])):.6f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_model_training():
    """Contoh: Model training."""
    print("\n" + "="*60)
    print("EXAMPLE 5: MODEL TRAINING")
    print("="*60)
    
    from src.models import CNNLSTMModel, ModelConfig, GNSSDataset
    from torch.utils.data import DataLoader as TorchDataLoader
    import torch
    
    try:
        # Create synthetic data
        n_epochs = 500
        n_features = 9  # 3 stations × 3 axes
        
        data = np.random.randn(n_epochs, n_features)
        
        # Create dataset
        dataset = GNSSDataset(data, window_size=30, horizon=7)
        
        if len(dataset) < 10:
            print("  Dataset too small for training")
            return
        
        loader = TorchDataLoader(dataset, batch_size=16, shuffle=True)
        
        # Create model
        config = ModelConfig(
            input_length=30,
            output_length=7,
            n_features=n_features
        )
        
        model = CNNLSTMModel(config)
        
        print(f"✓ Model created")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        x = torch.randn(16, 30, n_features)
        y = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_full_pipeline():
    """Contoh: Full pipeline."""
    print("\n" + "="*60)
    print("EXAMPLE 6: FULL PIPELINE")
    print("="*60)
    
    from main import GNSSPipeline
    
    try:
        pipeline = GNSSPipeline("config/config.yaml")
        
        print("✓ Pipeline initialized")
        print("  To run full pipeline: python main.py")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("GNSS DEFORMATION PIPELINE - QUICK START EXAMPLES")
    print("="*80)
    
    # Check if Excel file exists
    excel_path = Path("data/raw/excel_master/gnss_master.xlsx")
    
    if not excel_path.exists():
        print("\n⚠ Warning: Excel file not found at:")
        print(f"  {excel_path}")
        print("\nTo run extraction example, place your GNSS data Excel file at:")
        print(f"  {excel_path}")
        print("\nSkipping Excel examples...")
    
    # Run examples
    if excel_path.exists():
        try:
            example_excel_extraction()
            example_data_loading()
            example_preprocessing()
        except Exception as e:
            print(f"Some data-dependent examples skipped: {e}")
    
    example_vmd_decomposition()
    example_model_training()
    example_full_pipeline()
    
    print("\n" + "="*80)
    print("Examples completed! Check outputs for results.")
    print("="*80)


if __name__ == "__main__":
    main()