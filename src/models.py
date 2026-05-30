"""
Models Module
=============
Implementasi CNN-LSTM hybrid architecture untuk prediksi koordinat GNSS.
"""

import logging
import numpy as np
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration untuk CNN-LSTM model."""
    # Input/Output
    input_length: int = 30           # Lookback window (L)
    output_length: int = 7           # Prediction horizon
    n_features: int = 3              # E, N, U (atau disesuaikan dengan fitur VMD)
    
    # CNN parameters
    cnn_enabled: bool = True
    cnn_kernels: Tuple = (3, 5, 7)
    cnn_filters: Tuple = (32, 64, 128)
    cnn_dropout: float = 0.3
    
    # LSTM parameters
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = True
    
    # Dense layers
    dense_units: Tuple = (128, 64)
    dense_activation: str = "relu"
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GNSSDataset(Dataset):
    """Dataset untuk GNSS time series prediction."""
    
    def __init__(self, data: np.ndarray, window_size: int, horizon: int):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        self.horizon = horizon
        self.n_samples = len(data) - window_size - horizon + 1
    
    def __len__(self) -> int:
        return max(0, self.n_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size:idx + self.window_size + self.horizon]
        return x, y


class CNNLayer(nn.Module):
    """1D CNN layer dengan multiple kernel sizes."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_sizes: Tuple = (3, 5, 7), dropout: float = 0.3):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass CNN.
        Args:
            x (torch.Tensor): (batch, in_channels, length)
        Returns:
            torch.Tensor: (batch, len(kernel_sizes)*out_channels, length // 2)
        """
        conv_outs = [self.relu(conv(x)) for conv in self.convs]
        concat = torch.cat(conv_outs, dim=1)
        concat = self.dropout(concat)
        out = self.pool(concat)
        return out


class LSTMLayer(nn.Module):
    """LSTM layer dengan optional bidirectional."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass LSTM.
        Args:
            x (torch.Tensor): (batch, length, input_size)
        """
        output, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            # Menggabungkan hidden state terakhir dari arah forward dan backward
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        
        return output, last_hidden


class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model untuk GNSS prediction."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 1. Inisialisasi Blok CNN
        if config.cnn_enabled:
            self.cnn = CNNLayer(
                in_channels=config.n_features,
                out_channels=config.cnn_filters[0],
                kernel_sizes=config.cnn_kernels,
                dropout=config.cnn_dropout
            )
            # Menghitung output channels spasial hasil penggabungan multi-kernel
            lstm_input_size = len(config.cnn_kernels) * config.cnn_filters[0]
        else:
            self.cnn = None
            lstm_input_size = config.n_features
        
        # 2. Inisialisasi Blok LSTM
        self.lstm = LSTMLayer(
            input_size=lstm_input_size,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            dropout=config.lstm_dropout,
            bidirectional=config.lstm_bidirectional
        )
        
        # 3. Inisialisasi Lapisan Dense
        lstm_output_dim = config.lstm_hidden * (2 if config.lstm_bidirectional else 1)
        dense_input = lstm_output_dim
        
        self.dense_layers = nn.ModuleList()
        for hidden_units in config.dense_units:
            self.dense_layers.append(nn.Linear(dense_input, hidden_units))
            self.dense_layers.append(nn.ReLU())
            self.dense_layers.append(nn.Dropout(0.3))
            dense_input = hidden_units
        
        # Output layer (Multi-step output flattened)
        self.output_layer = nn.Linear(dense_input, config.output_length * config.n_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass hybrid model.
        Input x shape: [Batch_size, Sequence_length, N_features]
        """
        # --- PROSESI CNN ---
        if self.cnn is not None:
            # Pastikan format tensor sesuai untuk Conv1D: [Batch, Channels, Length]
            if x.shape[1] == self.config.n_features and x.shape[2] != self.config.n_features:
                x_cnn_in = x
            else:
                x_cnn_in = x.transpose(1, 2)
            
            # Ekstraksi Fitur Spasial via Multi-Kernel
            x_cnn_out = self.cnn(x_cnn_in)  # Output: [Batch, New_Channels, New_Length]
            
            # Permutasi dimensi kembali ke sekuensial sejati: [Batch, New_Length, New_Channels]
            x_lstm_in = x_cnn_out.permute(0, 2, 1)
        else:
            x_lstm_in = x
        
        # --- PROSESI LSTM ---
        _, lstm_out = self.lstm(x_lstm_in)  # lstm_out: [Batch, Hidden_Combined]
        
        # --- PROSESI DENSE & PROYEKSI OUTPUT ---
        out = lstm_out
        for layer in self.dense_layers:
            out = layer(out)
        
        out = self.output_layer(out)
        return out


class ModelTrainer:
    """Trainer untuk CNN-LSTM model."""
    
    def __init__(self, model: CNNLSTMModel, config: ModelConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.MSELoss()
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y = y.reshape(y.shape[0], -1)  # Flatten target sekuensial [B, L_out * C]
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def evaluate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y = y.reshape(y.shape[0], -1)
                
                pred = self.model(x)
                loss = self.criterion(pred, y)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_tensor = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            pred = self.model(x_tensor)
        return pred.cpu().numpy()


def main_models():
    """Unit test lokal untuk verifikasi tensor flow."""
    config = ModelConfig()
    model = CNNLSTMModel(config)
    
    # Simulasi data sekuens GNSS [Batch=32, Window=30, Features=3]
    x = torch.randn(32, config.input_length, config.n_features)
    y = model(x)
    
    logger.info(f"Input tensor shape : {x.shape}")
    logger.info(f"Output tensor shape: {y.shape}")
    logger.info(f"Target yang diharapkan: (32, {config.output_length * config.n_features})")


if __name__ == "__main__":
    main_models()