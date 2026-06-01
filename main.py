"""
Main Pipeline Orchestrator
===========================
Entry point utama untuk TRAINING dan PREPROCESING.
Tujuan: Menghasilkan model terbaik dan menyimpan state pipeline.
"""

import os
import sys
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

# Import modules
from src.excel_extractor import main_extract
from src.data_loader import DataLoader
from src.preprocessor import PipelinePreprocessor, PreprocessingConfig, Preprocessor
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

        # State yang harus disimpan untuk keperluan prediksi (Inference)
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

    def step1_extract_excel(self) -> dict:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: EXCEL EXTRACTION")
        logger.info("=" * 70)
        excel_path = self.config['data']['excel_master']
        out_dir = self.config['data']['txt_stations_dir']
        results, summary = main_extract(excel_path, out_dir)
        logger.info(f"✓ {summary['total_stations']} stasiun → {out_dir}")
        return summary

    def step2_load_data(self) -> dict:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: DATA LOADING")
        logger.info("=" * 70)
        txt_dir = self.config['data']['txt_stations_dir']
        self.loader = DataLoader(txt_dir)
        self.loader.load_all_stations(seismic_config=self.config.get('seismic_event'))
        self.loader.align_to_common_time()
        self.time_array, self.data_matrix, self.station_names = self.loader.get_data_matrix()
        stats = self.loader.get_statistics()
        logger.info(f"✓ {stats['n_stations']} stasiun, {self.data_matrix.shape[0]} epoch, shape {self.data_matrix.shape}")
        return stats

    def step3_preprocess(self) -> dict:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: PREPROCESSING & GEODETIC FEATURES")
        logger.info("=" * 70)
        pp_cfg = self.config['preprocessing']
        config = PreprocessingConfig(
            outlier_method=pp_cfg['outlier_detection']['method'],
            outlier_threshold=pp_cfg['outlier_detection']['threshold'],
            interpolation_method=pp_cfg['interpolation']['method'],
            max_gap_days=pp_cfg['interpolation']['max_gap_days'],
            normalize=True,
            normalization_type=pp_cfg['normalization']
        )

        pipeline_pp = PipelinePreprocessor(config, full_config=self.config)
        self.processed, self.norm_params, geod_feat = pipeline_pp.process_all_stations(
            self.time_array, self.data_matrix, self.station_names
        )

        base_feat = self.processed.reshape(self.processed.shape[0], -1)
        all_feat_list = [base_feat]
        for key, val in geod_feat.items():
            if len(val.shape) == 3:
                all_feat_list.append(val.reshape(val.shape[0], -1))
            else:
                all_feat_list.append(val)

        self.feature_mat = np.concatenate(all_feat_list, axis=1)

        # --- Normalize all features (including geodetic) ---
        # This prevents non-normalized features from exploding the gradients
        self.feature_mat, _ = Preprocessor.normalize_matrix(self.feature_mat)

        # Final NaN cleaning
        self.feature_mat = np.nan_to_num(self.feature_mat, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"✓ Final feature matrix shape: {self.feature_mat.shape}")
        return {"shape": list(self.feature_mat.shape), "n_stations": len(self.station_names)}

    def step4_vmd_decomposition(self) -> dict:
        if not self.config['vmd']['enabled']:
            return {"status": "disabled"}

        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: VMD DECOMPOSITION")
        logger.info("=" * 70)
        vmd_cfg = self.config['vmd']
        vmd_config = VMDConfig(
            n_modes=vmd_cfg['n_modes'], alpha=vmd_cfg['alpha'], tau=vmd_cfg['tau'],
            DC=vmd_cfg.get('DC', 0), init=vmd_cfg.get('init', 1), tol=vmd_cfg['tolerance'],
        )
        self.vmd_pipeline = PipelineVMD(vmd_config)
        self.vmd_pipeline.decompose_all_stations(self.processed, self.station_names)
        vmd_feat = self.vmd_pipeline.get_imf_feature_matrix()

        self.feature_mat = np.concatenate([self.feature_mat, vmd_feat], axis=1)
        logger.info(f"✓ Feature matrix with VMD: {self.feature_mat.shape}")
        return {"n_modes": vmd_cfg['n_modes'], "feature_shape": list(self.feature_mat.shape)}

    def step5_train_model(self) -> dict:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: MULTI-TASK MODEL TRAINING")
        logger.info("=" * 70)
        tr_cfg = self.config['training']
        hw_cfg = self.config['hardware']

        event_date_str = self.config['seismic_event']['earthquake_date']
        eq_dt = datetime.strptime(event_date_str, "%Y-%m-%d")
        eq_mjd = (eq_dt - datetime(1858, 11, 17)).days
        eq_dec = 1858.87759 + (eq_mjd / 365.25)

        train_until_dec = eq_dec + (7 / 365.25)
        train_mask = self.time_array <= train_until_dec
        val_mask = self.time_array > train_until_dec

        event_indices = np.where(np.abs(self.time_array - eq_dec) < (1/365.25))[0]
        device_str = "cuda" if (hw_cfg['use_gpu'] and torch.cuda.is_available()) else "cpu"

        n_coord_features = len(self.station_names) * 3

        full_dataset = GNSSDataset(
            self.feature_mat,
            event_indices=event_indices,
            window_size=tr_cfg['window_size'],
            horizon=tr_cfg['prediction_horizon'],
            event_window=self.config['model']['event_prediction_window'],
            n_coord_features=n_coord_features
        )

        all_indices = np.arange(len(full_dataset))
        np.random.shuffle(all_indices)
        split_point = int(len(all_indices) * 0.85)
        train_indices = all_indices[:split_point]
        val_indices = all_indices[split_point:]

        train_ds = torch.utils.data.Subset(full_dataset, train_indices)
        val_ds = torch.utils.data.Subset(full_dataset, val_indices)

        train_loader = TorchDataLoader(train_ds, batch_size=tr_cfg['batch_size'], shuffle=True)
        val_loader   = TorchDataLoader(val_ds,   batch_size=tr_cfg['batch_size'], shuffle=False)

        model_config = ModelConfig(
            input_length=tr_cfg['window_size'],
            output_length=tr_cfg['prediction_horizon'],
            n_features=self.feature_mat.shape[1],
            n_output_features=n_coord_features,
            cnn_dropout=self.config['model']['cnn']['dropout_rate'],
            lstm_hidden=self.config['model']['lstm']['hidden_units'],
            lstm_layers=self.config['model']['lstm']['num_layers'],
            lstm_bidirectional=self.config['model']['lstm']['bidirectional'],
            learning_rate=tr_cfg['learning_rate'],
            weight_decay=tr_cfg['weight_decay'],
            device=device_str
        )

        self.model = CNNLSTMModel(model_config)
        trainer = ModelTrainer(self.model, model_config)

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
                logger.info(f"Epoch {epoch+1:4d}/{epochs} | Train: {tr_loss:.6f} | Val: {val_loss:.6f}")

            if val_loss < best_val - tr_cfg['early_stopping']['min_delta']:
                best_val = val_loss
                no_improve = 0
                ckpt_path = Path(self.config['checkpointing']['save_dir']) / "best_model.pt"
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                no_improve += 1
                if tr_cfg['early_stopping']['enabled'] and no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        ckpt_path = Path(self.config['checkpointing']['save_dir']) / "best_model.pt"
        if ckpt_path.exists():
            self.model.load_state_dict(torch.load(ckpt_path, map_location=device_str))

        logger.info(f"✓ Training finished | Best val loss: {best_val:.6f}")
        return {"epochs_done": len(history), "best_val_loss": float(best_val), "model_path": str(ckpt_path)}

    def save_pipeline_state(self, filename: str = "outputs/pipeline_state.pkl"):
        """
        Sangat Penting: Simpan state pipeline agar bisa digunakan oleh script prediksi.
        Tanpa ini, script prediksi tidak akan tahu cara normalisasi data atau indeks waktu.
        """
        state = {
            "time_array": self.time_array,
            "station_names": self.station_names,
            "norm_params": self.norm_params,
            "feature_mat": self.feature_mat,
            "config": self.config
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"💾 Pipeline state saved to: {filename}")

    def run_training_pipeline(self) -> dict:
        """Hanya menjalankan proses hingga training selesai dan state disimpan."""
        logger.info("\n" + "=" * 80)
        logger.info("GNSS TRAINING PIPELINE — START")
        logger.info("=" * 80)
        start_time = datetime.now()
        summary = {}

        try:
            summary["step1"] = self.step1_extract_excel()
            summary["step2"] = self.step2_load_data()
            summary["step3"] = self.step3_preprocess()
            summary["step4"] = self.step4_vmd_decomposition()
            summary["step5"] = self.step5_train_model()

            # Simpan state pipeline untuk inference di script lain
            self.save_pipeline_state()

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

        elapsed = (datetime.now() - start_time).total_seconds()
        summary["elapsed_seconds"] = elapsed
        summary["finished_at"] = datetime.now().isoformat()

        summary_path = "outputs/pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("\n" + "=" * 80)
        logger.info(f"TRAINING PIPELINE SELESAI dalam {elapsed:.1f} detik")
        logger.info(f"Ringkasan: {summary_path}")
        logger.info("=" * 80)
        return summary

def main():
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config/config.yaml"

    pipeline = GNSSPipeline(config_path)
    pipeline.run_training_pipeline()

if __name__ == "__main__":
    main()
