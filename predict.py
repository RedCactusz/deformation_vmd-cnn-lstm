"""
Prediction Script
===================
Skrip terpisah untuk melakukan inferensi menggunakan model yang telah dilatih.
Menyediakan menu interaktif untuk prediksi deformasi dan deteksi event.
"""

import os
import sys
import pickle
import logging
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from src.models import CNNLSTMModel, ModelConfig
from src.preprocessor import Preprocessor

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class GNSSPredictor:
    """
    Sistem Inferensi GNSS.
    Memuat state pipeline dan model untuk memberikan prediksi.
    """
    def __init__(self, state_path: str = "outputs/pipeline_state.pkl",
                 model_path: str = "outputs/models/best_model.pt",
                 config_path: str = "config/config.yaml"):

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        if not Path(state_path).exists():
            raise FileNotFoundError(f"State file tidak ditemukan: {state_path}. Silakan jalankan main.py terlebih dahulu.")

        with open(state_path, 'rb') as f:
            state = pickle.load(f)
            self.time_array = state['time_array']
            self.station_names = state['station_names']
            self.norm_params = state['norm_params']
            self.feature_mat = state['feature_mat']

        tr_cfg = self.config['training']
        n_coord_features = len(self.station_names) * 3
        model_config = ModelConfig(
            input_length=tr_cfg['window_size'],
            output_length=tr_cfg['prediction_horizon'],
            n_features=self.feature_mat.shape[1],
            n_output_features=n_coord_features,
            cnn_dropout=self.config['model']['cnn']['dropout_rate'],
            lstm_hidden=self.config['model']['lstm']['hidden_units'],
            lstm_layers=self.config['model']['lstm']['num_layers'],
            lstm_bidirectional=self.config['model']['lstm']['bidirectional'],
            device=self.config['hardware'].get('device', 'cpu') if self.config['hardware'].get('use_gpu') else 'cpu'
        )

        self.model = CNNLSTMModel(model_config)
        self.device = torch.device(model_config.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.station_locations = self._load_station_locations()

        if HAS_PYPROJ:
            self._utm_transformer = Transformer.from_crs(
                "EPSG:4326", "EPSG:32654", always_xy=True
            )
        else:
            self._utm_transformer = None
            logger.warning("pyproj tidak tersedia. Konversi UTM dinonaktifkan.")

        logger.info("✓ Predictor initialized. Model and state loaded successfully.")

    def _load_station_locations(self) -> dict:
        txt_dir = Path(self.config['data']['txt_stations_dir'])
        locations = {}
        for name in self.station_names:
            txt_file = txt_dir / f"{name}.txt"
            if not txt_file.exists():
                continue
            try:
                with open(txt_file, 'r') as f:
                    header = f.readline().strip().split()
                    first_line = f.readline().strip().split()

                col_map = {h.lower().replace('_', ''): i for i, h in enumerate(header)}

                lat_idx = None
                lon_idx = None
                reflon_idx = None
                for key, idx in col_map.items():
                    if 'latitude' in key:
                        lat_idx = idx
                    if 'longitude' in key:
                        lon_idx = idx
                    if key == 'reflon':
                        reflon_idx = idx

                lat = float(first_line[lat_idx]) if lat_idx is not None else 0.0
                raw_lon = float(first_line[lon_idx]) if lon_idx is not None else 0.0
                lon = ((raw_lon + 180) % 360) - 180
                reflon = float(first_line[reflon_idx]) if reflon_idx is not None else lon

                locations[name] = {'lat': lat, 'lon': lon, 'reflon': reflon}
            except Exception as e:
                logger.warning(f"Gagal load lokasi stasiun {name}: {e}")
        return locations

    def _to_utm(self, station_name: str, e_topo: float, n_topo: float) -> tuple:
        loc = self.station_locations.get(station_name)
        if not loc or self._utm_transformer is None:
            return (None, None)
        lat = loc['lat']
        reflon = loc['reflon']
        lon_approx = reflon + (e_topo / 111320.0) / np.cos(np.radians(lat))
        utm_e, utm_n = self._utm_transformer.transform(lon_approx, lat)
        return (utm_e, utm_n)

    def _denormalize_coords(self, norm_coords: np.ndarray) -> np.ndarray:
        n_s = len(self.station_names)
        axes = ['E', 'N', 'U']
        result = np.zeros((n_s, 3))
        for s_idx, s_name in enumerate(self.station_names):
            for a_idx, axis in enumerate(axes):
                params = self.norm_params.get(s_name, {}).get(axis, {})
                result[s_idx, a_idx] = Preprocessor.denormalize_data(norm_coords[s_idx, a_idx], params)
        return result

    def _get_coords_at_dec(self, target_dec: float) -> np.ndarray:
        idx = np.argmin(np.abs(self.time_array - target_dec))
        n_s = len(self.station_names)
        norm_vals = self.feature_mat[idx, :n_s * 3].reshape(n_s, 3)
        return self._denormalize_coords(norm_vals)

    def _get_actual_coords(self, target_dec: float):
        idx = np.searchsorted(self.time_array, target_dec) - 1
        if idx < 0 or idx >= len(self.time_array):
            return None
        n_s = len(self.station_names)
        norm_vals = self.feature_mat[idx, :n_s * 3].reshape(n_s, 3)
        return self._denormalize_coords(norm_vals)

    def predict_deformation_at_date(self, target_date_str: str):
        try:
            target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
            target_mjd = (target_dt - datetime(1858, 11, 17)).days
            target_dec = 1858.87759 + (target_mjd / 365.25)

            idx = np.searchsorted(self.time_array, target_dec) - 1
            if idx < 0 or idx >= len(self.time_array):
                return f"Error: Tanggal {target_date_str} di luar rentang data tersedia."

            window_size = self.config['training']['window_size']
            if idx < window_size:
                return f"Error: Data tidak cukup untuk window size {window_size} sebelum tanggal {target_date_str}."

            x_input = self.feature_mat[idx - window_size + 1 : idx + 1]
            x_tensor = torch.FloatTensor(x_input).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred_coords, _ = self.model(x_tensor)

            n_s = len(self.station_names)
            coord_feat_count = n_s * 3
            coords_flat = pred_coords[0, :coord_feat_count].cpu().numpy()
            reconstructed_norm = coords_flat.reshape(n_s, 3)
            predicted = self._denormalize_coords(reconstructed_norm)

            eq_date_str = self.config['seismic_event']['earthquake_date']
            eq_dt = datetime.strptime(eq_date_str, "%Y-%m-%d")
            eq_mjd = (eq_dt - datetime(1858, 11, 17)).days
            eq_dec = 1858.87759 + (eq_mjd / 365.25)
            earthquake_coords = self._get_coords_at_dec(eq_dec)

            displacement = earthquake_coords - predicted

            actual_coords = self._get_actual_coords(target_dec)

            return {
                'predicted': predicted,
                'earthquake_coords': earthquake_coords,
                'displacement': displacement,
                'actual_coords': actual_coords,
                'target_date': target_date_str,
                'earthquake_date': eq_date_str
            }

        except Exception as e:
            return f"Error processing date: {e}"

    def predict_next_event(self):
        window_size = self.config['training']['window_size']
        x_last = self.feature_mat[-window_size:]
        x_tensor = torch.FloatTensor(x_last).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_coords, pred_event = self.model(x_tensor)

        prob = torch.sigmoid(pred_event).item()

        coords_flat = pred_coords[0, :len(self.station_names)*3].cpu().numpy()
        norms = np.linalg.norm(coords_flat.reshape(len(self.station_names), 3), axis=1)
        max_idx = np.argmax(norms)
        critical_station = self.station_names[max_idx]

        return {
            "probability": prob,
            "estimated_window": f"{self.config['model']['event_prediction_window']} days from now",
            "critical_area": critical_station
        }

def print_prediction_report(result: dict, predictor: GNSSPredictor):
    n_s = len(predictor.station_names)
    axes = ['E', 'N', 'U']

    print(f"\n{'='*70}")
    print(f" PREDIKSI DEFORMASI: {result['target_date']}")
    print(f" Tanggal Gempa Referensi: {result['earthquake_date']}")
    print(f"{'='*70}")

    print(f"\n--- Koordinat Toposentrik & UTM ---")
    has_utm = HAS_PYPROJ and predictor._utm_transformer is not None
    if has_utm:
        print(f"{'Stasiun':<8} | {'East(m)':>12} | {'North(m)':>14} | {'Up(m)':>10} | {'UTM E(m)':>12} | {'UTM N(m)':>14}")
        print("-" * 85)
    else:
        print(f"{'Stasiun':<8} | {'East(m)':>12} | {'North(m)':>14} | {'Up(m)':>10}")
        print("-" * 52)

    for i, name in enumerate(predictor.station_names):
        e, n, u = result['predicted'][i]
        if has_utm:
            utm_e, utm_n = predictor._to_utm(name, e, n)
            if utm_e is not None:
                print(f"{name:<8} | {e:12.4f} | {n:14.4f} | {u:10.4f} | {utm_e:12.2f} | {utm_n:14.2f}")
            else:
                print(f"{name:<8} | {e:12.4f} | {n:14.4f} | {u:10.4f} | {'N/A':>12} | {'N/A':>14}")
        else:
            print(f"{name:<8} | {e:12.4f} | {n:14.4f} | {u:10.4f}")

    print(f"\n--- Pergeseran dari Gempa (Gempa - Prediksi) ---")
    print(f"{'Stasiun':<8} | {'ΔE(m)':>10} | {'ΔN(m)':>10} | {'ΔU(m)':>10} | {'Total(m)':>10}")
    print("-" * 56)
    for i, name in enumerate(predictor.station_names):
        de, dn, du = result['displacement'][i]
        total = np.sqrt(de**2 + dn**2 + du**2)
        print(f"{name:<8} | {de:+10.6f} | {dn:+10.6f} | {du:+10.6f} | {total:10.6f}")

    actual = result['actual_coords']
    if actual is not None:
        errors = result['predicted'] - actual
        rmses = np.sqrt(np.mean(errors**2, axis=1))
        avg_rmse = np.mean(rmses)

        print(f"\n--- Validasi vs Data Aktual ---")
        print(f"{'Stasiun':<8} | {'Err_E(m)':>10} | {'Err_N(m)':>10} | {'Err_U(m)':>10} | {'RMSE(m)':>10}")
        print("-" * 56)
        for i, name in enumerate(predictor.station_names):
            print(f"{name:<8} | {errors[i,0]:+10.6f} | {errors[i,1]:+10.6f} | {errors[i,2]:+10.6f} | {rmses[i]:10.6f}")
        print(f"\nRMSE Rata-rata: {avg_rmse:.6f} m")
    else:
        print(f"\n--- Validasi ---")
        print("Data aktual tidak tersedia di dataset untuk tanggal ini.")

def main():
    print("\n" + "="*50)
    print(" GNSS EARLY WARNING & MONITORING SYSTEM ")
    print("="*50)

    try:
        predictor = GNSSPredictor()
    except Exception as e:
        print(f"FAILED TO INITIALIZE: {e}")
        return

    while True:
        print("\nMENU PREDIKSI:")
        print("1. Prediksi Deformasi pada Tanggal Spesifik")
        print("2. Analisis Potensi Gempa Mendatang (Kapan & Di mana)")
        print("3. Keluar")

        choice = input("\nPilih opsi (1/2/3): ")

        if choice == '1':
            date_str = input("Masukkan tanggal (YYYY-MM-DD): ")
            res = predictor.predict_deformation_at_date(date_str)
            if isinstance(res, str):
                print(f"\n{res}")
            else:
                print_prediction_report(res, predictor)

        elif choice == '2':
            res = predictor.predict_next_event()
            print("\n--- Analisis Potensi Gempa ---")
            print(f"Probabilitas Kejadian : {res['probability']*100:.2f}%")
            print(f"Estimasi Jendela Waktu: {res['estimated_window']}")
            print(f"Area Kritis (Station) : {res['critical_area']}")
            print("\nCatatan: Analisis berdasarkan akumulasi strain dan pola temporal terbaru.")

        elif choice == '3':
            print("Keluar dari sistem. Sampai jumpa!")
            break
        else:
            print("Pilihan tidak valid.")

if __name__ == "__main__":
    main()
