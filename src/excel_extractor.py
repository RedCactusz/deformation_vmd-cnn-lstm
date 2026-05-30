# """
# Excel Extractor Module
# ======================
# Membaca file Excel master multi-sheet dan mengekstrak data tenv3
# ke individual file TXT per stasiun.

# Format tenv3 (kolom utama):
#   site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m)
#   _north(m) u0(m) ____up(m) _ant(m) sig_e(m) sig_n(m) sig_u(m)
#   __corr_en __corr_eu __corr_nu _latitude(deg) _longitude(deg) __height(m)

# Data dimulai dari baris A2 (A1 = header).
# """

# import logging
# from pathlib import Path
# from typing import Dict, List, Tuple
# import pandas as pd
# import numpy as np
# from datetime import datetime

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # Kolom tenv3 yang wajib ada (subset minimal)
# TENV3_REQUIRED_COLS = ['site', 'yyyy.yyyy', '__MJD']

# # Kolom koordinat tenv3 yang dipakai pipeline
# TENV3_COORD_COLS = {
#     'decimal_year': 'yyyy.yyyy',
#     'mjd':          '__MJD',
#     'e0':           '_e0(m)',
#     'east':         '__east(m)',
#     'n0':           '____n0(m)',
#     'north':        '_north(m)',
#     'u0':           'u0(m)',
#     'up':           '____up(m)',
#     'sig_e':        'sig_e(m)',
#     'sig_n':        'sig_n(m)',
#     'sig_u':        'sig_u(m)',
#     'lat':          '_latitude(deg)',
#     'lon':          '_longitude(deg)',
# }


# class ExcelExtractor:
#     """
#     Ekstrak data tenv3 dari Excel multi-sheet dan export ke TXT individual.
#     Setiap sheet = satu stasiun GNSS.
#     """

#     def __init__(self, excel_path: str, output_dir: str):
#         """
#         Args:
#             excel_path: Path ke file Excel master (.xlsx)
#             output_dir: Output directory untuk file TXT per stasiun
#         """
#         self.excel_path = Path(excel_path)
#         self.output_dir = Path(output_dir)

#         if not self.excel_path.exists():
#             raise FileNotFoundError(f"File Excel tidak ditemukan: {self.excel_path}")

#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         logger.info(f"ExcelExtractor: {self.excel_path} → {self.output_dir}")

#     def get_sheet_names(self) -> List[str]:
#         """Ambil semua nama sheet dari file Excel."""
#         xls = pd.ExcelFile(self.excel_path, engine='openpyxl')
#         sheets = xls.sheet_names
#         logger.info(f"Ditemukan {len(sheets)} sheet: {sheets}")
#         return sheets

#     def read_sheet_data(self, sheet_name: str) -> Tuple[List[str], pd.DataFrame]:
#         """
#         Baca data dari satu sheet mulai baris A2.

#         Returns:
#             (header_list, DataFrame)
#         """
#         # Baca header dari baris 1 (index 0)
#         header_row = pd.read_excel(
#             self.excel_path,
#             sheet_name=sheet_name,
#             header=None,
#             nrows=1,
#             engine='openpyxl'
#         )
#         raw_header = header_row.iloc[0].tolist()
#         header = [str(h).strip() for h in raw_header if pd.notna(h) and str(h).strip() != '']

#         # Baca data mulai baris 2 (skiprows=1)
#         data = pd.read_excel(
#             self.excel_path,
#             sheet_name=sheet_name,
#             header=None,
#             skiprows=1,
#             engine='openpyxl'
#         )
#         data = data.dropna(how='all')

#         # Potong kolom sesuai jumlah header
#         if len(header) < data.shape[1]:
#             data = data.iloc[:, :len(header)]
#         data.columns = header[:data.shape[1]]

#         logger.info(f"Sheet '{sheet_name}': {len(data)} baris, kolom: {list(data.columns)}")
#         return header, data

#     def validate_tenv3_format(self, header: List[str], data: pd.DataFrame) -> bool:
#         """
#         Validasi apakah data sesuai format tenv3.
#         Cek keberadaan kolom wajib (case-insensitive partial match).
#         """
#         if data.empty or len(data) < 5:
#             logger.warning("Data terlalu sedikit untuk diproses")
#             return False

#         header_lower = [str(h).lower().strip() for h in header]

#         # Cek kolom minimal: ada 'mjd' atau 'yyyy' dan ada nilai numerik
#         has_mjd = any('mjd' in h for h in header_lower)
#         has_year = any('yyyy' in h for h in header_lower)

#         if not (has_mjd or has_year):
#             logger.warning(f"Tidak ditemukan kolom MJD/yyyy di header: {header[:5]}...")
#             return False

#         return True

#     def export_to_txt(self, station_name: str, header: List[str],
#                       data: pd.DataFrame) -> str:
#         """
#         Export data ke file TXT berformat tenv3 (space-separated).

#         Format output:
#             baris 1: header kolom
#             baris 2+: data numerik/string

#         Returns:
#             Path ke file TXT yang dibuat
#         """
#         output_file = self.output_dir / f"{station_name}.txt"

#         with open(output_file, 'w') as f:
#             # Tulis header
#             f.write(" ".join(str(h) for h in data.columns) + "\n")

#             # Tulis data baris per baris
#             for _, row in data.iterrows():
#                 parts = []
#                 for val in row:
#                     if pd.isna(val):
#                         parts.append('NaN')
#                     elif isinstance(val, float):
#                         parts.append(f"{val:.10g}")
#                     else:
#                         parts.append(str(val))
#                 f.write(" ".join(parts) + "\n")

#         logger.info(f"Exported: {output_file} ({len(data)} baris)")
#         return str(output_file)

#     def extract_all_stations(self) -> Dict[str, str]:
#         """
#         Extract semua stasiun dari file Excel.

#         Returns:
#             {station_name: output_file_path}
#         """
#         sheet_names = self.get_sheet_names()
#         results = {}

#         logger.info("=" * 60)
#         logger.info(f"MULAI EKSTRAKSI: {len(sheet_names)} sheet")
#         logger.info("=" * 60)

#         for i, sheet_name in enumerate(sheet_names, 1):
#             logger.info(f"[{i}/{len(sheet_names)}] Memproses: {sheet_name}")
#             try:
#                 header, data = self.read_sheet_data(sheet_name)

#                 if not self.validate_tenv3_format(header, data):
#                     logger.warning(f"  → Dilewati (format tidak valid)")
#                     continue

#                 output_file = self.export_to_txt(sheet_name, header, data)
#                 results[sheet_name] = output_file

#             except Exception as e:
#                 logger.error(f"  → Gagal: {e}")
#                 continue

#         logger.info("=" * 60)
#         logger.info(f"SELESAI: {len(results)}/{len(sheet_names)} stasiun berhasil")
#         logger.info("=" * 60)
#         return results

#     def get_extraction_summary(self) -> Dict:
#         """Ringkasan hasil ekstraksi."""
#         txt_files = list(self.output_dir.glob("*.txt"))
#         return {
#             "total_stations": len(txt_files),
#             "total_file_size_mb": sum(f.stat().st_size for f in txt_files) / 1024 / 1024,
#             "output_directory": str(self.output_dir),
#             "extraction_date": datetime.now().isoformat(),
#             "files": {f.stem: f.stat().st_size for f in txt_files}
#         }


# def main_extract(excel_path: str, output_dir: str = "data/raw/txt_stations/"):
#     """
#     Entry point untuk ekstraksi Excel → TXT.

#     Args:
#         excel_path: Path ke file Excel master
#         output_dir: Output directory untuk TXT files
#     """
#     extractor = ExcelExtractor(excel_path, output_dir)
#     results = extractor.extract_all_stations()
#     summary = extractor.get_extraction_summary()

#     print("\n" + "=" * 60)
#     print("RINGKASAN EKSTRAKSI")
#     print("=" * 60)
#     print(f"Total Stasiun : {summary['total_stations']}")
#     print(f"Total Ukuran  : {summary['total_file_size_mb']:.2f} MB")
#     print(f"Output Dir    : {summary['output_directory']}")
#     print("=" * 60)

#     return results, summary


# if __name__ == "__main__":
#     excel_file = "data/raw/excel_master/tenv3.xlsx"
#     output_directory = "data/raw/txt_stations/"

#     if Path(excel_file).exists():
#         main_extract(excel_file, output_directory)
#     else:
#         print(f"File tidak ditemukan: {excel_file}")



"""
Excel Extractor Module (Revised for Single-String Column Format)
==============================================================
Membaca file Excel master multi-sheet di mana data tenv3 menumpuk 
sebagai string panjang di Kolom A, lalu mengekstraknya menjadi 
individual file TXT per stasiun.

Struktur Excel Aktual:
  - Kolom A: Data tenv3 (String panjang space-separated)
  - Kolom B: Data xyz (Diabaikan)
  - Baris 1 (Index 0): Judul Format ("tenv3", "xyz")
  - Baris 2 (Index 1): Header Kolom internal tenv3
  - Baris 3+ (Index 2+): Baris data koordinat per epoch
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExcelExtractor:
    """
    Ekstrak data tenv3 dari Kolom A file Excel multi-sheet 
    dan export ke TXT individual (space-separated) per stasiun.
    """

    def __init__(self, excel_path: str, output_dir: str):
        """
        Args:
            excel_path: Path ke file Excel master (.xlsx)
            output_dir: Output directory untuk file TXT per stasiun
        """
        self.excel_path = Path(excel_path)
        self.output_dir = Path(output_dir)

        if not self.excel_path.exists():
            raise FileNotFoundError(f"File Excel tidak ditemukan: {self.excel_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ExcelExtractor: {self.excel_path} → {self.output_dir}")

    def get_sheet_names(self) -> List[str]:
        """Ambil semua nama sheet dari file Excel."""
        xls = pd.ExcelFile(self.excel_path, engine='openpyxl')
        sheets = xls.sheet_names
        logger.info(f"Ditemukan {len(sheets)} sheet: {sheets}")
        return sheets

    # def parse_sheet_string_data(self, sheet_name: str) -> List[str]:
    #     """
    #     Membaca Kolom A pada sheet, mengekstrak header (Baris 2) 
    #     dan seluruh baris data (Baris 3 ke bawah).

    #     Returns:
    #         List[str]: Kumpulan baris string yang siap ditulis ke berkas .txt
    #     """
    #     # Baca seluruh sheet tanpa header otomatis (header=None) untuk kendali penuh
    #     df_raw = pd.read_excel(
    #         self.excel_path,
    #         sheet_name=sheet_name,
    #         header=None,
    #         engine='openpyxl'
    #     )

    #     if df_raw.empty or df_raw.shape[0] < 3:
    #         logger.warning(f"Sheet '{sheet_name}' kosong atau kekurangan baris data.")
    #         return []

    #     cleaned_lines = []

    #     # 1. Ambil Header Tenv3 (Ada di Baris ke-2 / Index 1, Kolom A / Index 0)
    #     tenv3_header_string = str(df_raw.iloc[1, 0]).strip()
        
    #     # Bersihkan spasi ganda dalam string header agar menjadi single-space separated
    #     header_line = " ".join(tenv3_header_string.split())
    #     cleaned_lines.append(header_line)

    #     # 2. Ambil Data Epoch (Mulai Baris ke-3 / Index 2 ke bawah, Kolom A / Index 0)
    #     data_rows = df_raw.iloc[2:, 0]
        
    #     for idx, val in data_rows.items():
    #         if pd.isna(val):
    #             continue
            
    #         row_str = str(val).strip()
    #         if row_str == "":
    #             continue
                
    #         # Normalisasi pemisah karakter: memecah string berdasarkan spasi berlebih
    #         # dan menggabungkannya kembali hanya dengan spasi tunggal (" ")
    #         normalized_line = " ".join(row_str.split())
    #         cleaned_lines.append(normalized_line)

    #     logger.info(f"Sheet '{sheet_name}': Berhasil memproses {len(cleaned_lines) - 1} epoch data.")
    #     return cleaned_lines

    def parse_sheet_string_data(self, sheet_name: str) -> List[str]:
        """
        Membaca Kolom A pada sheet, mengekstrak header (Baris 2) 
        dan seluruh baris data (Baris 3 ke bawah) dengan proteksi kolom ujung.
        """
        # Baca seluruh sheet sebagai string murni untuk mencegah pemotongan otomatis
        df_raw = pd.read_excel(
            self.excel_path,
            sheet_name=sheet_name,
            header=None,
            dtype=str,  # memaksa semua sel dibaca sebagai string utuh
            engine='openpyxl'
        )

        if df_raw.empty or df_raw.shape[0] < 3:
            logger.warning(f"Sheet '{sheet_name}' kosong atau kekurangan baris data.")
            return []

        cleaned_lines = []

        # 1. Ambil Header Tenv3 (Baris ke-2 / Index 1, Kolom A / Index 0)
        tenv3_header_string = str(df_raw.iloc[1, 0]).strip()
        header_line = " ".join(tenv3_header_string.split())
        cleaned_lines.append(header_line)

        # 2. Ambil Data Epoch (Mulai Baris ke-3 / Index 2 ke bawah, Kolom A / Index 0)
        data_rows = df_raw.iloc[2:, 0]
        
        for idx, val in data_rows.items():
            if pd.isna(val):
                continue
            
            # Mengonversi ke string, menghilangkan karakter enter (\n atau \r), lalu menghapus spasi ujung
            row_str = str(val).replace('\n', ' ').replace('\r', ' ').strip()
            if row_str == "":
                continue
                
            # Memecah berdasarkan spasi berapapun jumlahnya dan menggabungkannya kembali
            tokens = row_str.split()
            
            # Proteksi: Pastikan token data tidak kosong
            if len(tokens) > 0:
                normalized_line = " ".join(tokens)
                cleaned_lines.append(normalized_line)

        logger.info(f"Sheet '{sheet_name}': Berhasil memproses {len(cleaned_lines) - 1} epoch data (Total {len(cleaned_lines)} baris teks).")
        return cleaned_lines


    def export_parsed_to_txt(self, station_name: str, text_lines: List[str]) -> str:
        """
        Menulis baris teks yang sudah dibersihkan langsung ke file TXT.
        """
        if not text_lines:
            raise ValueError(f"Tidak ada data untuk diekspor pada stasiun {station_name}")

        output_file = self.output_dir / f"{station_name}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            for line in text_lines:
                f.write(line + "\n")

        logger.info(f"Exported: {output_file} ({len(text_lines) - 1} baris data)")
        return str(output_file)

    def extract_all_stations(self) -> Dict[str, str]:
        """
        Extract semua stasiun dari file Excel berdasarkan kolom kontainer string.
        """
        sheet_names = self.get_sheet_names()
        results = {}

        logger.info("=" * 60)
        logger.info(f"MULAI EKSTRAKSI (FORMAT STRING KOLOM A): {len(sheet_names)} sheet")
        logger.info("=" * 60)

        for i, sheet_name in enumerate(sheet_names, 1):
            logger.info(f"[{i}/{len(sheet_names)}] Memproses stasiun: {sheet_name}")
            try:
                # Proses pemecahan string per baris
                text_lines = self.parse_sheet_string_data(sheet_name)

                if len(text_lines) <= 1: # Hanya berisi header atau kosong
                    logger.warning(f"  → Dilewati (Data tidak valid atau kosong)")
                    continue

                # Ekspor langsung ke berkas teks space-separated standar geodesi
                output_file = self.export_parsed_to_txt(sheet_name, text_lines)
                results[sheet_name] = output_file

            except Exception as e:
                logger.error(f"  → Gagal mengekstrak stasiun {sheet_name}: {e}")
                continue

        logger.info("=" * 60)
        logger.info(f"SELESAI: {len(results)}/{len(sheet_names)} stasiun berhasil diekstrak")
        logger.info("=" * 60)
        return results

    def get_extraction_summary(self) -> Dict:
        """Ringkasan hasil ekstraksi file teks."""
        txt_files = list(self.output_dir.glob("*.txt"))
        return {
            "total_stations": len(txt_files),
            "total_file_size_mb": sum(f.stat().st_size for f in txt_files) / 1024 / 1024,
            "output_directory": str(self.output_dir),
            "extraction_date": datetime.now().isoformat(),
            "files": {f.stem: f.stat().st_size for f in txt_files}
        }


def main_extract(excel_path: str, output_dir: str = "data/raw/txt_stations/"):
    """
    Entry point otomatisasi ekstraksi Excel → TXT.
    """
    extractor = ExcelExtractor(excel_path, output_dir)
    results = extractor.extract_all_stations()
    summary = extractor.get_extraction_summary()

    print("\n" + "=" * 60)
    print("RINGKASAN EKSTRAKSI (SINKRONISASI KOLOM STRING)")
    print("=" * 60)
    print(f"Total Stasiun Berhasil : {summary['total_stations']}")
    print(f"Total Ukuran Berkas     : {summary['total_file_size_mb']:.4f} MB")
    print(f"Direktori Output       : {summary['output_directory']}")
    print("=" * 60)

    return results, summary


if __name__ == "__main__":
    # Sesuaikan path berkas master Excel Anda di sini
    excel_file = "data/raw/excel_master/tenv3.xlsx"
    output_directory = "data/raw/txt_stations/"

    if Path(excel_file).exists():
        main_extract(excel_file, output_directory)
    else:
        print(f"File master Excel tidak ditemukan di: {excel_file}")