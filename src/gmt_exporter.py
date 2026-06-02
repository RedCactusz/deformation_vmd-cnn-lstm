"""
GMT Exporter Module
===================
Export data GNSS dan hasil prediksi CNN-LSTM ke format siap pakai
Generic Mapping Tools (GMT 6).

Format output utama:
  - stations_velocity.gmt   → untuk `gmt velo` (Lon Lat VeloE VeloN SigE SigN CorEN Name)
  - stations_coords.gmt     → untuk `gmt plot` (Lon Lat Name)
  - timeseries_<STA>.txt    → time series per stasiun (MJD E N U)
  - earthquake_event.gmt    → marker episenter gempa
  - predictions_coords.gmt  → koordinat hasil prediksi

Semua koordinat dalam satuan METER untuk E/N/U.
Longitude tenv3 (negatif = Barat) dikonversi ke 0–360 atau -180–180 sesuai region.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GMTExporter:
    """Export data ke format GMT."""

    def __init__(self, output_dir: str = "data/gmt_inputs/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"GMTExporter → {self.output_dir}")

    @staticmethod
    def mjd_to_datetime(mjd: float) -> datetime:
        """Konversi MJD ke datetime (epoch MJD: 17 Nov 1858)."""
        return datetime(1858, 11, 17) + timedelta(days=float(mjd))

    @staticmethod
    def normalize_longitude(lon: float) -> float:
        """Konversi longitude ke range -180..180."""
        while lon > 180:
            lon -= 360
        while lon < -180:
            lon += 360
        return lon

    # ------------------------------------------------------------------
    # Velocity file (untuk gmt velo)
    # ------------------------------------------------------------------
    def create_velocity_file(self,
                              stations: Dict[str, Dict],
                              period: str = "pre",
                              output_file: Optional[str] = None) -> str:
        """
        Buat file vektor kecepatan untuk `gmt velo`.

        Format: Lon Lat VeloE VeloN SigmaE SigmaN CorEN StationName
        Units: mm/yr (standard geodesi)

        Args:
            stations: {name: {lon, lat, vE, vN, sigE, sigN, corEN}}  (vE/vN in m/yr)
            period:   'pre', 'co', atau 'predicted'
            output_file: nama file output (auto-generate jika None)
        """
        if output_file is None:
            output_file = f"velocity_{period}_seismic.gmt"
        filepath = self.output_dir / output_file

        with open(filepath, 'w') as f:
            f.write(f"# GMT velo format — {period}-seismic\n")
            f.write("# Lon Lat VeloE(mm/yr) VeloN(mm/yr) SigmaE SigmaN CorEN StationName\n")

            for name, d in stations.items():
                lon = self.normalize_longitude(d.get('lon', 110.0))
                lat = d.get('lat', -7.5)
                vE  = d.get('vE', 0.0) * 1000
                vN  = d.get('vN', 0.0) * 1000
                sE  = d.get('sigE', 0.001) * 1000
                sN  = d.get('sigN', 0.001) * 1000
                cor = d.get('corEN', 0.0)
                f.write(f"{lon:10.5f} {lat:9.5f} {vE:10.6f} {vN:10.6f} "
                        f"{sE:10.6f} {sN:10.6f} {cor:7.4f} {name}\n")

        logger.info(f"Velocity file: {filepath} ({len(stations)} stasiun)")
        return str(filepath)

    # ------------------------------------------------------------------
    # Station coordinates (untuk gmt plot + label)
    # ------------------------------------------------------------------
    def create_coordinates_file(self,
                                  stations: Dict[str, Dict],
                                  output_file: str = "stations_coords.gmt") -> str:
        """
        Buat file koordinat stasiun.

        Format: Lon Lat StationName
        """
        filepath = self.output_dir / output_file

        with open(filepath, 'w') as f:
            f.write("# Koordinat stasiun GNSS\n")
            f.write("# Lon Lat StationName\n")

            for name, d in stations.items():
                lon = self.normalize_longitude(d.get('lon', 110.0))
                lat = d.get('lat', -7.5)
                f.write(f"{lon:10.5f} {lat:9.5f} {name}\n")

        logger.info(f"Coordinates file: {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Time series per stasiun
    # ------------------------------------------------------------------
    def create_time_series_file(self,
                                  station_name: str,
                                  time_array: np.ndarray,
                                  coordinates: Dict[str, np.ndarray],
                                  unit_mm: bool = True,
                                  output_file: Optional[str] = None) -> str:
        """
        Buat file time series untuk satu stasiun.

        Format: DecimalYear East North Up  (dalam mm jika unit_mm=True)
        """
        if output_file is None:
            output_file = f"timeseries_{station_name}.txt"
        filepath = self.output_dir / output_file

        scale = 1000.0 if unit_mm else 1.0
        unit_label = "mm" if unit_mm else "m"

        with open(filepath, 'w') as f:
            f.write(f"# Time series stasiun {station_name}\n")
            f.write(f"# DecimalYear East({unit_label}) North({unit_label}) Up({unit_label})\n")

            n = len(time_array)
            for i in range(n):
                E = coordinates.get('E', np.zeros(n))[i] * scale
                N = coordinates.get('N', np.zeros(n))[i] * scale
                U = coordinates.get('U', np.zeros(n))[i] * scale
                f.write(f"{time_array[i]:.6f} {E:12.4f} {N:12.4f} {U:12.4f}\n")

        logger.info(f"Time series file: {filepath} ({n} epoch)")
        return str(filepath)

    # ------------------------------------------------------------------
    # Event marker (episenter gempa)
    # ------------------------------------------------------------------
    def create_event_markers(self,
                               earthquake_date: str,
                               event_location: Tuple[float, float],
                               magnitude: float = 6.5,
                               output_file: str = "earthquake_event.gmt") -> str:
        """
        Buat file marker episenter gempa untuk `gmt plot`.

        Format: Lon Lat Magnitude  (untuk -Sa simbol bintang)
        """
        filepath = self.output_dir / output_file
        lon, lat = event_location
        lon = self.normalize_longitude(lon)

        with open(filepath, 'w') as f:
            f.write(f"# Marker episenter gempa\n")
            f.write(f"# Tanggal: {earthquake_date}\n")
            f.write(f"# Lon Lat Magnitude\n")
            f.write(f"{lon:10.5f} {lat:9.5f} {magnitude:.1f}\n")

        logger.info(f"Event marker file: {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Predictions file
    # ------------------------------------------------------------------
    def create_predictions_file(self,
                                  station_names: List[str],
                                  station_locations: Dict[str, Tuple[float, float]],
                                  predictions: np.ndarray,
                                  output_file: str = "predictions_coords.gmt") -> str:
        """
        Buat file koordinat prediksi CNN-LSTM untuk plotting.

        Format: Lon Lat dE(mm) dN(mm) dU(mm) StationName

        Args:
            station_names    : list nama stasiun
            station_locations: {name: (lat, lon)}
            predictions      : (n_stations, 3) nilai prediksi [E, N, U] dalam meter
        """
        filepath = self.output_dir / output_file

        with open(filepath, 'w') as f:
            f.write("# Hasil prediksi CNN-LSTM\n")
            f.write("# Lon Lat dEast(mm) dNorth(mm) dUp(mm) StationName\n")

            for i, name in enumerate(station_names):
                lat, lon = station_locations.get(name, (-7.5, 110.0))
                lon = self.normalize_longitude(lon)

                if i < predictions.shape[0]:
                    dE = float(predictions[i, 0]) * 1000
                    dN = float(predictions[i, 1]) * 1000
                    dU = float(predictions[i, 2]) * 1000
                else:
                    dE = dN = dU = 0.0

                f.write(f"{lon:10.5f} {lat:9.5f} {dE:10.4f} {dN:10.4f} {dU:10.4f} {name}\n")

        logger.info(f"Predictions file: {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Export semua sekaligus dari output pipeline
    # ------------------------------------------------------------------
    def export_all(self,
                    loader,               # DataLoader instance
                    processed_matrix: np.ndarray,
                    time_array: np.ndarray,
                    station_names: List[str],
                    predictions: Optional[np.ndarray] = None,
                    earthquake_cfg: Optional[Dict] = None) -> Dict[str, str]:
        """
        Export semua file GMT sekaligus.

        Returns dict berisi path ke semua file yang dibuat.
        """
        files = {}

        # Lokasi stasiun
        locations = loader.get_station_locations()  # {name: (lat, lon)}

        # Bangun data stasiun untuk velocity/coord files
        stations_data = {}
        for i, name in enumerate(station_names):
            lat, lon = locations.get(name, (-7.5, 110.0))
            coords = processed_matrix[:, i, :]  # (n_epochs, 3)

            # Hitung velocity sederhana (slope linear terhadap waktu)
            dt = np.gradient(time_array)
            dE = np.gradient(coords[:, 0]) / dt if len(time_array) > 1 else np.zeros(1)
            dN = np.gradient(coords[:, 1]) / dt if len(time_array) > 1 else np.zeros(1)

            stations_data[name] = {
                'lat': lat, 'lon': lon,
                'vE': float(np.nanmedian(dE)),
                'vN': float(np.nanmedian(dN)),
                'sigE': 0.001, 'sigN': 0.001, 'corEN': 0.0,
                'E': float(np.nanmean(coords[:, 0])),
                'N': float(np.nanmean(coords[:, 1])),
                'U': float(np.nanmean(coords[:, 2])),
            }

        # Velocity file
        files['velocity'] = self.create_velocity_file(stations_data, period='pre')
        # Coordinates file
        files['coordinates'] = self.create_coordinates_file(stations_data)

        # Time series per stasiun
        for i, name in enumerate(station_names):
            coords = {
                'E': processed_matrix[:, i, 0],
                'N': processed_matrix[:, i, 1],
                'U': processed_matrix[:, i, 2],
            }
            files[f'ts_{name}'] = self.create_time_series_file(name, time_array, coords)

        # Event marker
        if earthquake_cfg:
            eq_date = earthquake_cfg.get('earthquake_date', '2024-01-01')
            eq_loc = earthquake_cfg.get('epicenter', (110.0, -7.5))  # (lon, lat)
            files['event'] = self.create_event_markers(eq_date, eq_loc)

        # Predictions
        if predictions is not None:
            files['predictions'] = self.create_predictions_file(
                station_names, locations, predictions
            )

        logger.info(f"GMT export selesai: {len(files)} file dibuat")
        return files


    # ------------------------------------------------------------------
    # Predicted vs Actual displacement vectors
    # ------------------------------------------------------------------
    def create_predicted_vs_actual_file(self,
                                         station_names: List[str],
                                         station_locations: Dict[str, Dict],
                                         predicted: np.ndarray,
                                         actual: np.ndarray,
                                         target_date: str = "") -> Tuple[str, str]:
        """
        Buat file vektor prediksi dan aktual untuk `gmt velo -Se`.

        Format: Lon Lat dE(mm) dN(mm) SigE SigN Cor

        Args:
            station_names    : list nama stasiun
            station_locations: {name: {lat, lon, ...}}
            predicted        : (n_stations, 3) prediksi [E, N, U] dalam meter
            actual           : (n_stations, 3) aktual [E, N, U] dalam meter
            target_date      : tanggal untuk judul

        Returns:
            (path_predicted, path_actual)
        """
        pred_file = self.output_dir / "predicted_vectors.gmt"
        actual_file = self.output_dir / "actual_vectors.gmt"

        with open(pred_file, 'w') as fp, open(actual_file, 'w') as fa:
            fp.write(f"# Predicted displacement vectors — {target_date}\n")
            fp.write("# Lon Lat dE(cm) dN(cm) SigE SigN Cor\n")
            fa.write(f"# Actual displacement vectors — {target_date}\n")
            fa.write("# Lon Lat dE(cm) dN(cm) SigE SigN Cor\n")

            for i, name in enumerate(station_names):
                loc = station_locations.get(name, {})
                lat = loc.get('lat', 0.0)
                lon = self.normalize_longitude(loc.get('lon', 0.0))

                if i < predicted.shape[0]:
                    pE = float(predicted[i, 0]) * 100
                    pN = float(predicted[i, 1]) * 100
                else:
                    pE = pN = 0.0

                if i < actual.shape[0]:
                    aE = float(actual[i, 0]) * 100
                    aN = float(actual[i, 1]) * 100
                else:
                    aE = aN = 0.0

                fp.write(f"{lon:10.5f} {lat:9.5f} {pE:10.4f} {pN:10.4f} 0.5 0.5 0.0\n")
                fa.write(f"{lon:10.5f} {lat:9.5f} {aE:10.4f} {aN:10.4f} 0.5 0.5 0.0\n")

        logger.info(f"Predicted vectors: {pred_file}")
        logger.info(f"Actual vectors: {actual_file}")
        return str(pred_file), str(actual_file)

    # ------------------------------------------------------------------
    # Co-seismic displacement vectors
    # ------------------------------------------------------------------
    def create_co_seismic_displacement_file(self,
                                             station_names: List[str],
                                             station_locations: Dict[str, Dict],
                                             displacement: np.ndarray,
                                             earthquake_date: str = "") -> str:
        """
        Buat file vektor displacement co-seismic untuk `gmt velo -Se`.

        Format: Lon Lat dE(mm) dN(mm) SigE SigN Cor

        Args:
            displacement: (n_stations, 3) displacement [E, N, U] dalam meter
        """
        filepath = self.output_dir / "co_seismic_disp.gmt"

        with open(filepath, 'w') as f:
            f.write(f"# Co-seismic displacement vectors — {earthquake_date}\n")
            f.write("# Lon Lat dE(mm) dN(mm) SigE SigN Cor\n")

            for i, name in enumerate(station_names):
                loc = station_locations.get(name, {})
                lat = loc.get('lat', 0.0)
                lon = self.normalize_longitude(loc.get('lon', 0.0))

                if i < displacement.shape[0]:
                    dE = float(displacement[i, 0]) * 1000
                    dN = float(displacement[i, 1]) * 1000
                else:
                    dE = dN = 0.0

                f.write(f"{lon:10.5f} {lat:9.5f} {dE:10.4f} {dN:10.4f} 0.5 0.5 0.0\n")

        logger.info(f"Co-seismic displacement: {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Station coordinates for GMT (updated)
    # ------------------------------------------------------------------
    def create_station_coords_file(self,
                                    station_names: List[str],
                                    station_locations: Dict[str, Dict]) -> str:
        """
        Buat file koordinat stasiun dari data pipeline aktual.

        Format: Lon Lat StationName
        """
        filepath = self.output_dir / "stations_coords.gmt"

        with open(filepath, 'w') as f:
            f.write("# Koordinat stasiun GNSS\n")
            f.write("# Lon Lat StationName\n")

            for name in station_names:
                loc = station_locations.get(name, {})
                lat = loc.get('lat', 0.0)
                lon = self.normalize_longitude(loc.get('lon', 0.0))
                f.write(f"{lon:10.5f} {lat:9.5f} {name}\n")

        logger.info(f"Station coords: {filepath}")
        return str(filepath)


def main_gmt():
    """Test GMT exporter dengan data dummy."""
    exporter = GMTExporter("data/gmt_inputs/")

    stations = {
        'UKU1': {'lon': 145.9, 'lat': 44.03, 'vE': 0.001, 'vN': 0.002,
                  'sigE': 0.0001, 'sigN': 0.0001, 'corEN': 0.0},
        'COVE': {'lon': 146.2, 'lat': 44.10, 'vE': -0.001, 'vN': 0.001,
                  'sigE': 0.0001, 'sigN': 0.0001, 'corEN': 0.0},
    }

    vel = exporter.create_velocity_file(stations)
    coord = exporter.create_coordinates_file(stations)
    print(f"Velocity: {vel}")
    print(f"Coords  : {coord}")


if __name__ == "__main__":
    main_gmt()
