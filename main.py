"""
Main Pipeline Orchestrator
===========================
Entry point utama: Excel → Load → Preprocess → VMD → CNN-LSTM → GMT Export

Jalankan:
    python main.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

# Import modules
from src.excel_extractor import main_extract
from src.data_loader import DataLoader
from src.preprocessor import PipelinePreprocessor, PreprocessingConfig
from src.vmd_processor import PipelineVMD, VMDConfig
from src.models import CNNLSTMModel, ModelConfig, ModelTrainer, GNSSDataset
from src.evaluator import ModelEvaluator
from src.gmt_exporter import GMTExporter

# Setup logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/gnss_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GNSSPipeline:
    """Orkestrasi pipeline GNSS VMD-CNN-LSTM end-to-end."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config tidak ditemukan: {self.config_path}")

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Pipeline: {self.config['project']['name']} "
                    f"v{self.config['project']['version']}")

        self._create_directories()

        # State
        self.loader        = None
        self.time_array    = None
        self.data_matrix   = None
        self.station_names = []
        self.processed     = None
        self.norm_params   = {}
        self.vmd_pipeline  = None
        self.feature_mat   = None
        self.model         = None
        self.predictions   = None
        
        
    def setup_test_logger(self, level=logging.INFO, log_to_file: bool = True):
        """
        Mengonfigurasi logger ganda:
        1. Terminal (Console): Output minimalis, bersih, tidak banjir.
        2. File (pipeline_debug.log): Laporan lengkap terperinci untuk audit.
        """
        root_logger = logging.getLogger()
        
        # Bersihkan handler lama agar tidak duplikasi log
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # --- HANDLER 1: TERMINAL (CONSOLE) ---
        # Format super ringkas: Jam dan Pesan saja
        minimal_formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)-5s: %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(minimal_formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)
        
        # --- HANDLER 2: FILE LOG (Jika diaktifkan) ---
        if log_to_file:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True) # Buat folder 'logs' jika belum ada
            
            log_file_path = log_dir / "pipeline_debug.log"
            
            # Format di dalam file dibuat sangat lengkap untuk pelacakan error ilmiah
            detailed_formatter = logging.Formatter(
                fmt='%(asctime)s [%(levelname)s] %(name)s (Line: %(lineno)d): %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
            file_handler.setFormatter(detailed_formatter)
            file_handler.setLevel(logging.DEBUG) # File mencatat hingga level DEBUG terperinci
            root_logger.addHandler(file_handler)
            
            logger.info(f"💾 Laporan lengkap proses ini ditulis ke: {log_file_path}")
        
        # Set level utama untuk root
        root_logger.setLevel(logging.DEBUG if log_to_file else level)
        
        # Redam log eksternal dari library pihak ketiga yang tidak diperlukan
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('pandas').setLevel(logging.WARNING)
        logging.getLogger('numpy').setLevel(logging.WARNING)
        
        logger.info("🧪 Test Logger diaktifkan. Log terminal telah diringkas.")
        return self

    def _create_directories(self):
        dirs = [
            self.config['data']['txt_stations_dir'],
            self.config['data']['processed_dir'],
            self.config['data']['gmt_inputs_dir'],
            self.config['output']['save_dir'],
            self.config['checkpointing']['save_dir'],
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: Excel Extraction
    # ------------------------------------------------------------------
    def step1_extract_excel(self) -> dict:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: EXCEL EXTRACTION")
        logger.info("=" * 70)

        excel_path = self.config['data']['excel_master']
        out_dir = self.config['data']['txt_stations_dir']

        results, summary = main_extract(excel_path, out_dir)
        logger.info(f"✓ {summary['total_stations']} stasiun → {out_dir}")
        return summary

    # ------------------------------------------------------------------
    # STEP 2: Data Loading
    # ------------------------------------------------------------------
    # def step2_load_data(self) -> dict:
    #     logger.info("\n" + "=" * 70)
    #     logger.info("STEP 2: DATA LOADING")
    #     logger.info("=" * 70)

    #     txt_dir = self.config['data']['txt_stations_dir']
    #     self.loader = DataLoader(txt_dir)
    #     self.loader.load_all_stations()
    #     self.loader.align_to_common_time()

    #     self.time_array, self.data_matrix, self.station_names = (
    #         self.loader.get_data_matrix()
    #     )

    #     stats = self.loader.get_statistics()
    #     logger.info(f"✓ {stats['n_stations']} stasiun, "
    #                 f"{self.data_matrix.shape[0]} epoch, "
    #                 f"shape {self.data_matrix.shape}")
    #     return stats
    
    def step2_load_data(self) -> dict:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: DATA LOADING")
        logger.info("=" * 70)

        txt_dir = self.config['data']['txt_stations_dir']
        self.loader = DataLoader(txt_dir)
        
        # UBAH BARIS INI: Kirim konfigurasi stasiun dari yaml
        self.loader.load_all_stations(seismic_config=self.config.get('stations'))
        
        self.loader.align_to_common_time()

        self.time_array, self.data_matrix, self.station_names = (
            self.loader.get_data_matrix()
        )

        stats = self.loader.get_statistics()
        logger.info(f"✓ {stats['n_stations']} stasiun, "
                    f"{self.data_matrix.shape[0]} epoch, "
                    f"shape {self.data_matrix.shape}")
        return stats

    # ------------------------------------------------------------------
    # STEP 3: Preprocessing
    # ------------------------------------------------------------------
    def step3_preprocess(self) -> dict:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: PREPROCESSING (MAD + PCHIP + Normalisasi)")
        logger.info("=" * 70)

        pp = self.config['preprocessing']
        config = PreprocessingConfig(
            outlier_method=pp['outlier_detection']['method'],
            outlier_threshold=pp['outlier_detection']['threshold'],
            interpolation_method=pp['interpolation']['method'],
            max_gap_days=pp['interpolation']['max_gap_days'],
            normalize=True,
            normalization_type=pp['normalization']
        )

        pipeline_pp = PipelinePreprocessor(config)
        self.processed, self.norm_params = pipeline_pp.process_all_stations(
            self.time_array, self.data_matrix, self.station_names
        )

        logger.info(f"✓ Processed matrix shape: {self.processed.shape}")
        return {"shape": list(self.processed.shape), "n_stations": len(self.station_names)}

    # ------------------------------------------------------------------
    # STEP 4: VMD Decomposition
    # ------------------------------------------------------------------
    def step4_vmd_decomposition(self) -> dict:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: VMD DECOMPOSITION")
        logger.info("=" * 70)

        vmd_cfg = self.config['vmd']
        vmd_config = VMDConfig(
            n_modes=vmd_cfg['n_modes'],
            alpha=vmd_cfg['alpha'],
            tau=vmd_cfg['tau'],
            DC=vmd_cfg.get('DC', 0),
            init=vmd_cfg.get('init', 1),
            tol=vmd_cfg['tolerance'],
        )

        self.vmd_pipeline = PipelineVMD(vmd_config)
        self.vmd_pipeline.decompose_all_stations(self.processed, self.station_names)
        self.feature_mat = self.vmd_pipeline.get_imf_feature_matrix()

        logger.info(f"✓ Feature matrix: {self.feature_mat.shape}")
        return {"n_modes": vmd_cfg['n_modes'],
                "feature_shape": list(self.feature_mat.shape)}

    # ------------------------------------------------------------------
    # STEP 5: Model Training
    # ------------------------------------------------------------------
    def step5_train_model(self) -> dict:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: MODEL TRAINING (CNN-LSTM)")
        logger.info("=" * 70)

        tr_cfg = self.config['training']
        hw_cfg = self.config['hardware']

        # Pilih data input: feature matrix VMD (lebih kaya) atau processed langsung
        if self.feature_mat is not None:
            input_data = self.feature_mat  # (n_epochs, n_features)
        else:
            n_e, n_s, n_a = self.processed.shape
            input_data = self.processed.reshape(n_e, n_s * n_a)

        n_epochs, n_features = input_data.shape
        window_size = tr_cfg['window_size']          # ← benar, bukan batch_size
        pred_len = tr_cfg['prediction_horizon']

        device_str = "cuda" if (hw_cfg['use_gpu'] and torch.cuda.is_available()) else "cpu"

        # Dataset
        dataset = GNSSDataset(input_data, window_size=window_size, horizon=pred_len)

        if len(dataset) < 20:
            logger.warning(f"Dataset terlalu kecil ({len(dataset)} sampel), skip training")
            return {"status": "skipped", "reason": "insufficient_data"}

        # Train / val split
        n_val = max(1, int(len(dataset) * tr_cfg['validation_split']))
        n_train = len(dataset) - n_val
        # Temporal split: use the first part of the sequence for training and the end for validation
        train_ds = torch.utils.data.Subset(dataset, range(n_train))
        val_ds = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

        train_loader = TorchDataLoader(train_ds, batch_size=tr_cfg['batch_size'], shuffle=True)
        val_loader   = TorchDataLoader(val_ds,   batch_size=tr_cfg['batch_size'], shuffle=False)

        # Model config
        model_config = ModelConfig(
            input_length=window_size,
            output_length=pred_len,
            n_features=n_features,
            cnn_dropout=self.config['cnn']['dropout_rate'],
            lstm_hidden=self.config['lstm']['hidden_units'],
            lstm_layers=self.config['lstm']['num_layers'],
            lstm_bidirectional=self.config['lstm']['bidirectional'],
            learning_rate=tr_cfg['learning_rate'],
            weight_decay=tr_cfg['weight_decay'],
            device=device_str
        )

        self.model = CNNLSTMModel(model_config)
        trainer = ModelTrainer(self.model, model_config)

        logger.info(f"Device: {device_str} | Dataset: {len(dataset)} | "
                    f"Train: {n_train} | Val: {n_val}")
        logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Training loop
        epochs = tr_cfg['epochs']
        patience = tr_cfg['early_stopping']['patience']
        best_val = float('inf')
        no_improve = 0
        history = []

        for epoch in range(epochs):
            tr_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.evaluate(val_loader)
            history.append({'epoch': epoch + 1, 'train': tr_loss, 'val': val_loss})

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:4d}/{epochs} | "
                            f"Train: {tr_loss:.6f} | Val: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val - tr_cfg['early_stopping']['min_delta']:
                best_val = val_loss
                no_improve = 0
                # Save best
                ckpt_path = Path(self.config['checkpointing']['save_dir']) / "best_model.pt"
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                no_improve += 1
                if tr_cfg['early_stopping']['enabled'] and no_improve >= patience:
                    logger.info(f"Early stopping di epoch {epoch+1}")
                    break

        # Load best weights
        ckpt_path = Path(self.config['checkpointing']['save_dir']) / "best_model.pt"
        if ckpt_path.exists():
            self.model.load_state_dict(torch.load(ckpt_path, map_location=device_str))

        logger.info(f"✓ Training selesai | Best val loss: {best_val:.6f}")
        return {
            "epochs_done": len(history),
            "best_val_loss": float(best_val),
            "model_path": str(ckpt_path)
        }

    # ------------------------------------------------------------------
    # STEP 6: Predictions
    # ------------------------------------------------------------------
    def step6_make_predictions(self) -> dict:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: PREDICTIONS")
        logger.info("=" * 70)

        if self.model is None:
            logger.warning("Model belum ditraining, skip")
            return {"status": "skipped"}

        tr_cfg = self.config['training']
        window_size = tr_cfg['window_size']

        if self.feature_mat is not None:
            input_data = self.feature_mat
        else:
            n_e, n_s, n_a = self.processed.shape
            input_data = self.processed.reshape(n_e, n_s * n_a)

        # Prediksi dari window terakhir
        x_last = torch.FloatTensor(input_data[-window_size:]).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            pred_flat = self.model(x_last).cpu().numpy()  # (1, pred_len * n_features)

        # --- REKONSTRUKSI VMD KE KOORDINAT UTUH ---
        # pred_flat[0] shape: (pred_len * n_stations * 3 * K)
        pred_data = pred_flat[0]

        n_s = len(self.station_names)
        n_a = 3
        K = self.config['vmd']['n_modes']
        pred_len = self.config['training']['prediction_horizon']

        # 1. Reshape ke bentuk terstruktur: (horizon, station, axis, imf)
        # Urutan feature_mat adalah: [St1_E_K, St1_N_K, St1_U_K, St2_E_K, ...]
        pred_reshaped = pred_data.reshape(pred_len, n_s, n_a, K)

        # 2. Ambil prediksi step pertama (pred_len=0) dan rekonstruksi (sum IMFs)
        # Shape: (n_stations, 3)
        reconstructed_norm = np.sum(pred_reshaped[0], axis=-1)

        # 3. Denormalisasi menggunakan norm_params
        final_predictions = np.zeros((n_s, n_a))
        axes = ['E', 'N', 'U']

        from src.preprocessor import Preprocessor
        for s_idx, s_name in enumerate(self.station_names):
            for a_idx, axis in enumerate(axes):
                params = self.norm_params.get(s_name, {}).get(axis, {})
                val_norm = reconstructed_norm[s_idx, a_idx]
                final_predictions[s_idx, a_idx] = Preprocessor.denormalize_data(val_norm, params)

        self.predictions = final_predictions
        logger.info(f"✓ Prediksi direkonstruksi & denormalisasi shape: {self.predictions.shape}")
        return {"predictions_shape": list(self.predictions.shape)}

    # ------------------------------------------------------------------
    # STEP 7: GMT Export
    # ------------------------------------------------------------------
    def step7_gmt_export(self) -> dict:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 7: GMT EXPORT")
        logger.info("=" * 70)

        exporter = GMTExporter(self.config['data']['gmt_inputs_dir'])

        files = exporter.export_all(
            loader=self.loader,
            processed_matrix=self.processed,
            time_array=self.time_array,
            station_names=self.station_names,
            predictions=self.predictions,
            earthquake_cfg=self.config['seismic_event']
        )

        logger.info(f"✓ {len(files)} file GMT dibuat di {self.config['data']['gmt_inputs_dir']}")
        return files

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_full_pipeline(self) -> dict:
        logger.info("\n" + "=" * 80)
        logger.info("GNSS DEFORMATION VMD-CNN-LSTM PIPELINE — START")
        logger.info("=" * 80)
        start_time = datetime.now()
        summary = {}

        try:
            summary["step1"] = self.step1_extract_excel()
        except Exception as e:
            logger.error(f"Step 1 gagal: {e}")
            raise

        try:
            summary["step2"] = self.step2_load_data()
            summary["step3"] = self.step3_preprocess()
            summary["step4"] = self.step4_vmd_decomposition()
            summary["step5"] = self.step5_train_model()
            summary["step6"] = self.step6_make_predictions()
            summary["step7"] = self.step7_gmt_export()
        except Exception as e:
            logger.error(f"Pipeline berhenti: {e}")
            raise

        elapsed = (datetime.now() - start_time).total_seconds()
        summary["elapsed_seconds"] = elapsed
        summary["finished_at"] = datetime.now().isoformat()

        # Simpan ringkasan
        summary_path = "outputs/pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("\n" + "=" * 80)
        logger.info(f"PIPELINE SELESAI dalam {elapsed:.1f} detik")
        logger.info(f"Ringkasan: {summary_path}")
        logger.info("=" * 80)

        return summary


def main():
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config/config.yaml"

    pipeline = GNSSPipeline(config_path)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
