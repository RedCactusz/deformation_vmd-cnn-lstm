"""
Data Loader Module
==================
Load file TXT (tenv3 format) dari semua stasiun dan mengorganisir
data untuk preprocessing dan modeling.

Kolom tenv3 yang dipakai:
  yyyy.yyyy  → decimal year (time axis utama)
  __MJD      → Modified Julian Date
  _e0(m) + __east(m)   → koordinat East absolut = e0 + east
  ____n0(m) + _north(m) → koordinat North absolut = n0 + north
  u0(m) + ____up(m)    → koordinat Up absolut = u0 + up
  sig_e/n/u  → sigma (uncertainty)
  _latitude, _longitude → posisi geografis stasiun
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GNSSStation:
    """Data satu stasiun GNSS."""
    name: str
    header: List[str]
    data: pd.DataFrame

    # Atribut yang di-derive otomatis
    lat: float = 0.0
    lon: float = 0.0

    def __post_init__(self):
        if self.data.empty:
            raise ValueError(f"Station {self.name} tidak memiliki data")
        self._derive_position()

    def _col(self, *candidates) -> Optional[str]:
        """Cari kolom pertama yang cocok (case-insensitive, partial match)."""
        for cand in candidates:
            for col in self.data.columns:
                if cand.lower().strip('_') in col.lower().replace('_', ''):
                    return col
        return None

    def _derive_position(self):
        """Ambil posisi geografis stasiun dari kolom latitude/longitude."""
        lat_col = self._col('latitude', 'lat')
        lon_col = self._col('longitude', 'lon')
        if lat_col:
            self.lat = float(self.data[lat_col].dropna().iloc[0])
        if lon_col:
            raw_lon = float(self.data[lon_col].dropna().iloc[0])
            # tenv3 memakai longitude negatif untuk barat; konversi ke -180..180
            self.lon = ((raw_lon + 180) % 360) - 180

    def get_time(self) -> np.ndarray:
        """
        Ambil time axis dalam decimal year (yyyy.yyyy).
        Fallback ke MJD jika tidak ada.
        """
        col = self._col('yyyy.yyyy', 'decimalyear', 'year')
        if col:
            return self.data[col].values.astype(float)

        col = self._col('MJD', 'mjd')
        if col:
            mjd = self.data[col].values.astype(float)
            # Konversi MJD → decimal year (approx)
            return 1858.87759 + mjd / 365.25
        raise KeyError(f"Tidak ditemukan kolom time di station {self.name}")

    def get_mjd(self) -> np.ndarray:
        """Ambil MJD."""
        col = self._col('MJD', 'mjd')
        if col:
            return self.data[col].values.astype(float)
        # Konversi dari decimal year
        t = self.get_time()
        return (t - 1858.87759) * 365.25

    def get_coordinates(self) -> Dict[str, np.ndarray]:
        """
        Ambil koordinat toposentrik E, N, U dalam meter.

        tenv3: koordinat absolut = integer_part + fractional_part
          East  = _e0(m)    + __east(m)
          North = ____n0(m) + _north(m)
          Up    = u0(m)     + ____up(m)
        """
        e0_col = self._col('e0', '_e0')
        east_col = self._col('east', '__east')
        n0_col = self._col('n0', 'n0')
        north_col = self._col('north', '_north')
        u0_col = self._col('u0', 'u0')
        up_col = self._col('up', '____up', '__up')

        def _combine(int_col, frac_col, label) -> np.ndarray:
            if int_col and frac_col:
                return (self.data[int_col].values.astype(float) +
                        self.data[frac_col].values.astype(float))
            elif frac_col:
                return self.data[frac_col].values.astype(float)
            elif int_col:
                return self.data[int_col].values.astype(float)
            else:
                logger.warning(f"Station {self.name}: kolom {label} tidak ditemukan, pakai 0")
                return np.zeros(len(self.data))

        return {
            'E': _combine(e0_col, east_col, 'East'),
            'N': _combine(n0_col, north_col, 'North'),
            'U': _combine(u0_col, up_col, 'Up'),
        }

    def get_sigma(self) -> Dict[str, np.ndarray]:
        """Ambil sigma (uncertainty) E, N, U."""
        result = {}
        for axis, candidates in [('E', ['sige', 'sig_e']),
                                   ('N', ['sign', 'sig_n']),
                                   ('U', ['sigu', 'sig_u'])]:
            col = self._col(*candidates)
            result[axis] = (self.data[col].values.astype(float)
                            if col else np.ones(len(self.data)) * 0.001)
        return result

    def n_epochs(self) -> int:
        return len(self.data)


class DataLoader:
    """Load dan manage data dari multiple GNSS stations."""

    def __init__(self, txt_stations_dir: str):
        """
        Args:
            txt_stations_dir: Directory berisi file TXT per stasiun
        """
        self.txt_dir = Path(txt_stations_dir)
        if not self.txt_dir.exists():
            raise FileNotFoundError(f"Directory tidak ditemukan: {self.txt_dir}")
        self.stations: Dict[str, GNSSStation] = {}
        logger.info(f"DataLoader → {self.txt_dir}")

    def load_single_station(self, filepath: Path) -> GNSSStation:
        """
        Load satu file TXT stasiun.

        Mendukung format:
         - Space/tab-separated dengan baris header pertama
         - Kolom pertama bisa string (nama stasiun) atau numerik
        """
        station_name = filepath.stem

        # Baca header
        with open(filepath, 'r') as f:
            header_line = f.readline().strip()
        header = header_line.split()

        # Baca data dengan pandas (lebih robust untuk kolom campuran)
        try:
            df = pd.read_csv(
                filepath,
                sep=r'\s+',
                skiprows=1,
                header=None,
                names=header,
                na_values=['NaN', 'nan', 'NA', '-'],
                low_memory=False
            )
        except Exception:
            # Fallback: numpy loadtxt (hanya kolom numerik)
            data_num = np.loadtxt(filepath, skiprows=1)
            # Cek apakah kolom pertama itu string stasiun
            with open(filepath, 'r') as f:
                f.readline()  # skip header
                first_data = f.readline().split()
            try:
                float(first_data[0])
                df = pd.DataFrame(data_num, columns=header[:data_num.shape[1]])
            except ValueError:
                # Kolom 0 adalah string (nama site), skip
                df = pd.DataFrame(data_num, columns=header[1:1 + data_num.shape[1]])

        df = df.dropna(how='all').reset_index(drop=True)

        station = GNSSStation(name=station_name, header=header, data=df)
        logger.info(f"  Loaded '{station_name}': {station.n_epochs()} epoch, "
                    f"lat={station.lat:.4f}, lon={station.lon:.4f}")
        return station

    # def load_all_stations(self, pattern: str = "*.txt") -> Dict[str, GNSSStation]:
    #     """Load semua stasiun dari directory."""
    #     txt_files = sorted(self.txt_dir.glob(pattern))
    #     if not txt_files:
    #         raise FileNotFoundError(f"Tidak ada file {pattern} di {self.txt_dir}")

    #     logger.info(f"Ditemukan {len(txt_files)} file stasiun")

    #     for i, fp in enumerate(txt_files, 1):
    #         try:
    #             station = self.load_single_station(fp)
    #             self.stations[station.name] = station
    #         except Exception as e:
    #             logger.warning(f"[{i}] Dilewati {fp.name}: {e}")

    #     logger.info(f"Berhasil load {len(self.stations)} stasiun")
    #     return self.stations
    
    def load_all_stations(self, pattern: str = "*.txt", seismic_config: Optional[Dict] = None) -> Dict[str, GNSSStation]:
        """
        Load semua stasiun dari directory dengan filter otomatis berbasis ketersediaan data
        pada periode kejadian gempa jika konfigurasi gempa diberikan.
        """
        txt_files = sorted(self.txt_dir.glob(pattern))
        if not txt_files:
            raise FileNotFoundError(f"Tidak ada file {pattern} di {self.txt_dir}")

        logger.info(f"Ditemukan {len(txt_files)} file stasiun")

        # --- Logika Pra-Perhitungan Jendela Waktu Gempa dalam Desimal ---
        target_start_dec = None
        target_end_dec = None
        
        
        if seismic_config:
            try:
                eq_date_str = seismic_config.get('earthquake_date')
                pre_days = seismic_config.get('pre_seismic_days', 0)
                post_days = seismic_config.get('post_seismic_days', 0)
                
                # Parsing tanggal gempa ke objek datetime
                eq_dt = datetime.strptime(eq_date_str, "%Y-%m-%d")
                
                # Hitung Epoch batas bawah dan batas atas dalam satuan Modified Julian Date (MJD)
                # Rumus konversi datetime ke MJD internal Python (epoch astronomis)
                def dt_to_mjd(dt_obj):
                    return (dt_obj - datetime(1858, 11, 17)).days
                
                eq_mjd = dt_to_mjd(eq_dt)
                start_mjd = eq_mjd - pre_days
                end_mjd = eq_mjd + post_days
                
                # Konversi MJD kembali ke koordinat Decimal Year (Pendekatan Teoretis Geodesi)
                # Sesuai formula fallback di fungsi get_time(): t = 1858.87759 + mjd / 365.25
                target_start_dec = 1858.87759 + (start_mjd / 365.25)
                target_end_dec = 1858.87759 + (end_mjd / 365.25)
                
                logger.info(f"Filter Gempa Aktif: Mencari stasiun dengan data eksis antara "
                            f"{target_start_dec:.4f} dan {target_end_dec:.4f} (MJD {start_mjd} - {end_mjd})")
            except Exception as conf_err:
                logger.error(f"Gagal memproses konfigurasi seismic_event, filter diabaikan: {conf_err}")
                seismic_config = None

        # --- Proses Loading dan Penyaringan Stasiun ---
        for i, fp in enumerate(txt_files, 1):
            try:
                # Load struktur data awal stasiun
                station = self.load_single_station(fp)
                
                # Jika filter gempa aktif, uji ketersediaan rentang waktu secara ketat
                if seismic_config and target_start_dec and target_end_dec:
                    station_times = station.get_time()
                    
                    # Cek irisan data perekaman stasiun terhadap rentang jendela gempa
                    has_data_in_window = np.any((station_times >= target_start_dec) & 
                                                (station_times <= target_end_dec))
                    
                    if not has_data_in_window:
                        logger.warning(f"[{i}] Dilewati {fp.name}: Tidak memiliki data pada jendela waktu gempa Tohoku 2011.")
                        continue
                
                # Masukkan ke dictionary jika lolos seleksi
                self.stations[station.name] = station
                
            except Exception as e:
                logger.warning(f"[{i}] Dilewati {fp.name}: {e}")

        # Proteksi fungsional jika tidak ada stasiun yang memenuhi syarat ketersediaan data
        if not self.stations:
            raise ValueError(
                f"Kesalahan Fatal: Tidak ada satu pun stasiun yang memiliki rekaman data aktif "
                f"pada rentang waktu kejadian gempa yang ditentukan di config.yaml ({eq_date_str})."
            )

        logger.info(f"Berhasil mengamankan {len(self.stations)} stasiun untuk analisis selanjutnya")
        return self.stations
    

    def get_station(self, name: str) -> GNSSStation:
        if name not in self.stations:
            raise KeyError(f"Stasiun '{name}' tidak ditemukan. "
                           f"Tersedia: {list(self.stations.keys())}")
        return self.stations[name]

    def get_station_names(self) -> List[str]:
        return list(self.stations.keys())

    def get_common_time_axis(self) -> np.ndarray:
        """
        Buat common time axis (union semua epoch, diurutkan).
        Pakai union (bukan intersection) karena data GNSS jarang 100% sinkron;
        gap akan diisi saat preprocessing dengan interpolasi.
        """
        all_times = set()
        for station in self.stations.values():
            try:
                times = np.round(station.get_time(), 6)
                all_times.update(times.tolist())
            except Exception as e:
                logger.warning(f"Gagal ambil time dari {station.name}: {e}")

        common = np.sort(np.array(list(all_times)))
        logger.info(f"Common time axis: {len(common)} epoch dari "
                    f"{len(self.stations)} stasiun")
        return common

    # def align_to_common_time(self) -> Dict[str, GNSSStation]:
    #     """Potong data tiap stasiun ke common time axis (intersection)."""
    #     # Untuk alignment ketat, pakai intersection
    #     all_time_sets = []
    #     for station in self.stations.values():
    #         try:
    #             t = set(np.round(station.get_time(), 6).tolist())
    #             all_time_sets.append(t)
    #         except Exception:
    #             pass

    #     if not all_time_sets:
    #         return self.stations

    #     common_set = all_time_sets[0]
    #     for t_set in all_time_sets[1:]:
    #         common_set &= t_set
    #     common_arr = np.sort(np.array(list(common_set)))

    #     logger.info(f"Intersection time axis: {len(common_arr)} epoch")

    #     aligned = {}
    #     for name, station in self.stations.items():
    #         try:
    #             t = np.round(station.get_time(), 6)
    #             mask = np.isin(t, common_arr)
    #             aligned_df = station.data[mask].reset_index(drop=True)
    #             aligned_station = GNSSStation(
    #                 name=name, header=station.header, data=aligned_df
    #             )
    #             aligned[name] = aligned_station
    #             logger.info(f"  {name}: {mask.sum()} epoch retained")
    #         except Exception as e:
    #             logger.warning(f"  {name}: alignment gagal — {e}")

    #     self.stations = aligned
    #     return aligned
    
    def align_to_common_time(self) -> Dict[str, GNSSStation]:
        """Potong data tiap stasiun ke common time axis menggunakan UNION (bukan intersection)."""
        
        # GANTI LOGIKA: Gunakan fungsi get_common_time_axis() yang sudah berbasis UNION
        common_arr = self.get_common_time_axis()

        if len(common_arr) == 0:
            logger.error("Gagal membentuk time axis. Periksa format kolom waktu pada file TXT Anda.")
            return self.stations

        logger.info(f"Hasil Union time axis: {len(common_arr)} total epoch terkombinasi")

        aligned = {}
        for name, station in self.stations.items():
            try:
                # Ambil data waktu stasiun saat ini
                t_station = np.round(station.get_time(), 6)
                
                # Buat DataFrame baru dengan indeks berupa common_arr agar sinkron secara spasial
                # Menggunakan sinkronisasi berbasis pandas reindex untuk akurasi koordinat geodesi
                station_mjd = np.round(station.get_mjd(), 6)
                
                # Membuat mapper sementara untuk memetakan data asli ke common time axis
                df_temp = station.data.copy()
                
                # Cari kolom desimal year di dataframe asli, set sebagai index sementara
                time_col = station._col('yyyy.yyyy', 'decimalyear', 'year')
                df_temp[time_col] = np.round(df_temp[time_col].values.astype(float), 6)
                df_temp = df_temp.set_index(time_col)
                
                # Reindex ke common_arr (mengisi epoch yang kosong dengan NaN secara aman)
                # Langkah ini mencegah hilangnya data stasiun akibat strict intersection
                aligned_df = df_temp.reindex(common_arr).reset_index()
                
                # Kembalikan nama kolom agar konsisten dengan format tenv3 awal
                aligned_df = aligned_df.rename(columns={'index': time_col})

                aligned_station = GNSSStation(
                    name=name, header=station.header, data=aligned_df
                )
                aligned[name] = aligned_station
                logger.info(f"  {name}: {len(aligned_df)} epoch aligned (termasuk NaNs)")
            except Exception as e:
                logger.warning(f"  {name}: alignment gagal — {e}")

        self.stations = aligned
        return aligned

    def get_data_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Buat matriks data gabungan.

        Returns:
            time_array : (n_epochs,)  decimal year
            data_matrix: (n_epochs, n_stations, 3)  [E, N, U] dalam meter
            station_names: List nama stasiun
        """
        names = self.get_station_names()
        if not names:
            raise ValueError("Tidak ada stasiun yang di-load")

        # Pakai time axis dari stasiun pertama sebagai referensi
        ref_station = self.get_station(names[0])
        time_array = ref_station.get_time()
        n_epochs = len(time_array)

        data_matrix = np.full((n_epochs, len(names), 3), np.nan)

        for i, name in enumerate(names):
            station = self.get_station(name)
            try:
                coords = station.get_coordinates()
                n = min(n_epochs, len(coords['E']))
                data_matrix[:n, i, 0] = coords['E'][:n]
                data_matrix[:n, i, 1] = coords['N'][:n]
                data_matrix[:n, i, 2] = coords['U'][:n]
            except Exception as e:
                logger.warning(f"Gagal ambil koordinat {name}: {e}")

        return time_array, data_matrix, names

    def get_station_locations(self) -> Dict[str, Tuple[float, float]]:
        """Return {station_name: (lat, lon)} untuk semua stasiun."""
        return {name: (s.lat, s.lon) for name, s in self.stations.items()}

    def get_statistics(self) -> Dict:
        names = self.get_station_names()
        stats = {
            "n_stations": len(self.stations),
            "station_names": names,
            "n_epochs": (self.get_station(names[0]).n_epochs()
                         if names else 0),
            "timestamp": datetime.now().isoformat(),
            "per_station": {}
        }
        for name, station in self.stations.items():
            stats["per_station"][name] = {
                "n_epochs": station.n_epochs(),
                "lat": station.lat,
                "lon": station.lon,
                "columns": station.header[:6]
            }
        return stats

    def save_to_cache(self, cache_file: str):
        with open(cache_file, 'wb') as f:
            pickle.dump(self.stations, f)
        logger.info(f"Cache disimpan: {cache_file}")

    def load_from_cache(self, cache_file: str):
        with open(cache_file, 'rb') as f:
            self.stations = pickle.load(f)
        logger.info(f"Cache dimuat: {cache_file}")


def main_load(txt_dir: str = "data/raw/txt_stations/"):
    loader = DataLoader(txt_dir)
    loader.load_all_stations()
    stats = loader.get_statistics()

    print("\n" + "=" * 60)
    print("STATISTIK DATA LOADING")
    print("=" * 60)
    print(f"Stasiun berhasil dimuat : {stats['n_stations']}")
    print(f"Epoch per stasiun (ref) : {stats['n_epochs']}")
    print(f"Nama stasiun            : {', '.join(stats['station_names'])}")
    print("=" * 60)

    return loader


if __name__ == "__main__":
    main_load()
