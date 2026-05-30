# GNSS Deformation VMD-CNN-LSTM Pipeline

## 📋 Deskripsi Proyek

Pipeline end-to-end untuk prediksi deformasi GNSS menggunakan hybrid model **Variational Mode Decomposition (VMD)** dan **Convolutional Neural Network - Long Short-Term Memory (CNN-LSTM)**.

**Tujuan Utama:**
- Ekstraksi data GNSS dari file Excel multi-sheet
- Preprocessing data geodetis (outlier removal, alignment, interpolasi)
- Dekomposisi sinyal menggunakan VMD
- Prediksi koordinat toposentrik (East, North, Up) dengan CNN-LSTM
- Visualisasi hasil dengan Generic Mapping Tools (GMT)

---

## 📁 Struktur Proyek

```
gnss_deformation_vmd_cnn_lstm/
│
├── config/
│   └── config.yaml                 # Konfigurasi utama pipeline
│
├── data/
│   ├── raw/
│   │   ├── excel_master/          # File Excel master (.xlsx)
│   │   └── txt_stations/          # Hasil ekstraksi TXT per stasiun
│   ├── processed/                 # Data setelah preprocessing
│   └── gmt_inputs/                # File untuk GMT visualization
│
├── gmt_scripts/                   # Shell scripts untuk GMT
│   ├── plot_pre_seismic.sh
│   ├── plot_co_seismic.sh
│   └── plot_predicted.sh
│
├── src/                           # Source code Python
│   ├── __init__.py
│   ├── excel_extractor.py         # Baca Excel → export TXT
│   ├── data_loader.py             # Load & parse TXT files
│   ├── preprocessor.py            # Outlier removal, interpolasi
│   ├── vmd_processor.py           # Variational Mode Decomposition
│   ├── models.py                  # CNN-LSTM architecture
│   ├── trainer.py                 # Training logic
│   ├── evaluator.py               # Evaluation metrics
│   └── gmt_exporter.py            # Export untuk GMT
│
├── outputs/
│   ├── models/                    # Trained model weights
│   └── plots/                     # Output peta GMT (PDF/PNG)
│
├── requirements.txt               # Python dependencies
├── main.py                        # Pipeline orchestrator
└── README.md                      # Dokumentasi ini
```

---

## 🚀 Quick Start

### 1. Instalasi

```bash
# Clone atau extract project
cd gnss_deformation_vmd_cnn_lstm

# Install dependencies
pip install -r requirements.txt

# Install GMT 6 (Linux/macOS)
# Ubuntu/Debian:
sudo apt-get install gmt

# macOS:
brew install gmt

# Windows:
# Download dari: https://www.generic-mapping-tools.org/download/
```

### 2. Persiapan Data

```bash
# 1. Copy file Excel master ke folder
cp your_gnss_data.xlsx data/raw/excel_master/gnss_master.xlsx

# 2. Konfigurasi parameter di config/config.yaml
# Edit: project, seismic_event, preprocessing, dll
```

### 3. Jalankan Pipeline

```bash
# Jalankan full pipeline (semua step otomatis)
python main.py

# Atau jalankan step per step (untuk debugging):
python main.py --step extraction
python main.py --step loading
python main.py --step preprocessing
python main.py --step vmd
python main.py --step training
python main.py --step predictions
python main.py --step gmt_export
```

### 4. Generate Peta GMT

```bash
# Jalankan GMT scripts
chmod +x gmt_scripts/*.sh

bash gmt_scripts/plot_pre_seismic.sh
bash gmt_scripts/plot_co_seismic.sh
bash gmt_scripts/plot_predicted.sh

# Output: outputs/plots/map_*.pdf dan map_*.png
```

---

## 📊 Alur Kerja Pipeline

### STEP 1: Excel Extraction
- **Input:** File Excel multi-sheet (`gnss_master.xlsx`)
- **Proses:** Baca setiap sheet (nama sheet = nama stasiun), ekstrak data dari baris A2 ke bawah
- **Output:** Individual file TXT per stasiun di `data/raw/txt_stations/`

```python
from src.excel_extractor import main_extract
results, summary = main_extract("data/raw/excel_master/gnss_master.xlsx")
```

### STEP 2: Data Loading
- **Input:** TXT files dari semua stasiun
- **Proses:** Load, parse tenv3 format, align ke common time axis
- **Output:** Data structure siap preprocessing

```python
from src.data_loader import DataLoader
loader = DataLoader("data/raw/txt_stations/")
stations = loader.load_all_stations()
loader.align_to_common_time()
```

### STEP 3: Preprocessing
- **Outlier Detection:** MAD (Median Absolute Deviation) dengan threshold 3σ
- **Interpolation:** PCHIP untuk mengisi gap data
- **Normalization:** Min-Max atau Z-score normalization

```python
from src.preprocessor import PipelinePreprocessor
preprocessor = PipelinePreprocessor(config)
processed_matrix, norm_params = preprocessor.process_all_stations(time_array, data_matrix, station_names)
```

### STEP 4: VMD Decomposition
- **Metode:** Variational Mode Decomposition
- **Output:** Intrinsic Mode Functions (IMFs) per axis (E, N, U)
- **Parameter:** 5 modes, α=2000, max iterations=1000

```python
from src.vmd_processor import PipelineVMD
vmd = PipelineVMD(config)
decompositions = vmd.decompose_all_stations(processed_matrix, station_names)
```

### STEP 5: Model Training
- **Arsitektur:** CNN-LSTM hybrid
  - **CNN:** Multiple kernel sizes (3, 5, 7), extract spatial features
  - **LSTM:** Bidirectional, 2 layers, capture temporal dependencies
  - **Dense:** 128 → 64 units, output = prediction horizon

```python
from src.models import CNNLSTMModel, ModelTrainer
model = CNNLSTMModel(config)
trainer = ModelTrainer(model, config)
# Training loop...
```

### STEP 6: Predictions
- **Input:** Last 30 epochs dari setiap stasiun
- **Output:** Prediksi koordinat untuk 7 hari ke depan
- **Format:** (n_stations, n_axes=3)

### STEP 7: GMT Visualization
- **Format:** GMT velo (velocity) format untuk `gmt velo`
- **Output:** Peta PDF/PNG dengan:
  - **Pre-seismic:** Vektor sebelum gempa
  - **Co-seismic:** Pergeseran sesaat setelah gempa
  - **Predicted:** Hasil prediksi model

---

## ⚙️ Konfigurasi

Edit `config/config.yaml` untuk menyesuaikan:

```yaml
# Seismic Event
seismic_event:
  earthquake_date: "2024-01-15"
  pre_seismic_days: 30
  post_seismic_days: 30
  prediction_horizon: 60

# Preprocessing
preprocessing:
  outlier_threshold: 3.0           # σ multiplier
  interpolation_method: "pchip"
  max_gap_days: 5

# VMD
vmd:
  n_modes: 5
  alpha: 2000
  tolerance: 1e-5

# LSTM
lstm:
  hidden_units: 64
  num_layers: 2
  bidirectional: true

# Training
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  early_stopping:
    patience: 15
```

---

## 📈 Expected Outputs

### 1. Data Products
- `data/processed/`: Cleaned and preprocessed data (NumPy arrays)
- `data/gmt_inputs/`: GMT-formatted files (*.gmt)

### 2. Models
- `outputs/models/cnn_lstm_model.pt`: Trained model weights

### 3. Visualizations
- `outputs/plots/map_pre_seismic.pdf|png`: Pre-seismic deformation map
- `outputs/plots/map_co_seismic.pdf|png`: Co-seismic displacement map
- `outputs/plots/map_predicted.pdf|png`: Predicted deformation map

### 4. Reports
- `outputs/pipeline_summary.json`: Complete execution summary

---

## 🔧 Modul Descriptions

### `excel_extractor.py`
Mengekstrak data tenv3 dari Excel multi-sheet.

**Fungsi Utama:**
- `ExcelExtractor.get_sheet_names()`: Ambil nama semua sheet
- `ExcelExtractor.read_sheet_data()`: Baca data sheet specific
- `ExcelExtractor.extract_all_stations()`: Extract semua ke TXT

### `data_loader.py`
Load dan manage data dari multiple stations.

**Fungsi Utama:**
- `DataLoader.load_all_stations()`: Load semua TXT files
- `DataLoader.align_to_common_time()`: Sinkronisasi time axis
- `DataLoader.get_data_matrix()`: Get consolidated data matrix

### `preprocessor.py`
Cleaning dan normalisasi data.

**Fungsi Utama:**
- `Preprocessor.remove_outliers()`: Deteksi outlier dengan MAD
- `Preprocessor.interpolate_gaps()`: Interpolasi missing values
- `Preprocessor.normalize_data()`: Normalize ke range 0-1 atau zscore

### `vmd_processor.py`
Variational Mode Decomposition untuk dekomposisi sinyal.

**Fungsi Utama:**
- `VMDProcessor.vmd()`: Core VMD algorithm
- `VMDProcessor.decompose_signal()`: Decompose dengan residual
- `VMDProcessor.select_significant_imfs()`: Filter IMFs by energy

### `models.py`
CNN-LSTM architecture untuk prediksi.

**Arsitektur:**
```
Input (30 epochs × 3 axes)
    ↓
CNN (kernel 3,5,7) → Concatenate → MaxPool
    ↓
LSTM (Bidirectional, 2 layers)
    ↓
Dense (128 → 64 → output)
```

### `gmt_exporter.py`
Export data dalam format GMT.

**Fungsi Utama:**
- `GMTExporter.create_velocity_file()`: GMT velo format
- `GMTExporter.create_coordinates_file()`: Koordinat format
- `GMTExporter.create_time_series_file()`: Time series per station

---

## 📚 Format Data

### Tenv3 Format (Input)
```
MJD         X(m)      Xsig      Y(m)      Ysig      Z(m)      Zsig      ...
58849.123   1234.567  0.001    2345.678  0.001    3456.789  0.001     ...
58850.123   1234.568  0.001    2345.679  0.001    3456.790  0.001     ...
```

### GMT Velocity Format (Output)
```
# GMT velo format
# Lon Lat VeloE VeloN SigmaE SigmaN CorEN StationName
110.5000  -7.5000  0.00100  0.00200  0.00010  0.00010  0.0000 COVE
111.0000  -7.8000 -0.00100  0.00100  0.00010  0.00010  0.0000 J027
```

---

## 🐛 Troubleshooting

### Issue: "Excel file not found"
```bash
# Check path
ls data/raw/excel_master/
# Ensure file is named: gnss_master.xlsx
```

### Issue: "No valid stations found"
```bash
# Check TXT files generated correctly
ls data/raw/txt_stations/
# Should see: COVE.txt, J027.txt, etc.
```

### Issue: "GMT command not found"
```bash
# Install GMT
sudo apt-get install gmt
# or
brew install gmt
```

### Issue: "CUDA out of memory"
Edit `config.yaml`:
```yaml
hardware:
  use_gpu: false  # Use CPU instead
```

---

## 📖 References

1. **VMD:** K. Dragomiretskiy & D. Zosso, "Variational Mode Decomposition" (2013)
2. **CNN-LSTM:** Hybrid deep learning for time series prediction
3. **GMT:** Generic Mapping Tools (https://www.generic-mapping-tools.org/)
4. **GNSS:** Global Navigation Satellite System data processing

---

## 📝 License

MIT License - Lihat LICENSE file

---

## 👥 Authors

Project developed for GNSS deformation monitoring and seismic hazard assessment.

---

## 📞 Support

Untuk issues atau pertanyaan:
1. Check documentation di README
2. Review configuration di `config/config.yaml`
3. Enable verbose logging (set `logging.level: DEBUG`)
4. Check output di `outputs/pipeline_summary.json`

---

**Last Updated:** 2024