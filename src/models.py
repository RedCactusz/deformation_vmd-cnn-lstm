"""
Models Module
===================
Implementasi Multi-Task CNN-LSTM-Attention hybrid architecture untuk prediksi koordinat GNSS
dan deteksi dini kejadian gempa.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration untuk Multi-Task CNN-LSTM-Attention model."""
    # Input/Output
    input_length: int = 30           # Lookback window (L)
    output_length: int = 1           # Prediction horizon for coordinates (T=1 for specific date)
    n_features: int = 3              # Total input features (coords + geodetic features)

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

    # Attention parameters
    attention_enabled: bool = True

    # Dense layers
    dense_units: Tuple = (128, 64)
    dense_activation: str = "relu"

    # Multi-Task Specifics
    model_type: str = "multi_task"   # 'single_output' or 'multi_task'
    event_window: int = 30           # Window for event classification
    n_output_features: Optional[int] = None  # Output dim for regression (None = n_features)

    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GNSSDataset(Dataset):
    """
    Dataset untuk Multi-Task GNSS prediction.
    Target:
    1. Regression: Koordinat [E, N, U] untuk semua stasiun pada t + output_length.
    2. Classification: Probabilitas gempa terjadi dalam window tertentu.
    """
    def __init__(self,
                 data_matrix: np.ndarray,
                 event_indices: Optional[np.ndarray] = None,
                 window_size: int = 30,
                 horizon: int = 1,
                 event_window: int = 30,
                 n_coord_features: Optional[int] = None):
        self.data = torch.FloatTensor(data_matrix)
        self.window_size = window_size
        self.horizon = horizon
        self.event_window = event_window
        self.n_coord_features = n_coord_features if n_coord_features is not None else data_matrix.shape[1]

        self.n_samples = len(data_matrix) - window_size - max(horizon, event_window)

        self.event_labels = None
        if event_indices is not None:
            self.event_labels = np.zeros(len(data_matrix))
            for idx in event_indices:
                start = max(0, idx - event_window)
                self.event_labels[start:idx] = 1.0
            self.event_labels = torch.FloatTensor(self.event_labels)

    def __len__(self) -> int:
        return max(0, self.n_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.window_size]
        y_reg = self.data[idx + self.window_size + self.horizon - 1, :self.n_coord_features]
        y_cls = torch.tensor(0.0)
        if self.event_labels is not None:
            look_ahead_start = idx + self.window_size
            look_ahead_end = look_ahead_start + self.event_window
            if look_ahead_end < len(self.event_labels):
                if torch.any(self.event_labels[look_ahead_start:look_ahead_end] == 1.0):
                    y_cls = torch.tensor(1.0)
        return x, y_reg, y_cls


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
        output, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        return output, last_hidden


class AttentionLayer(nn.Module):
    """
    Self-Attention mechanism untuk memberikan bobot lebih pada epoch kritis.
    Berdasarkan pendekatan 'Attention' pada deret waktu geodetik.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq_len, Hidden_dim]
        weights = self.attention(x) # [Batch, Seq_len, 1]
        # Weighted sum of LSTM outputs
        context = torch.sum(weights * x, dim=1) # [Batch, Hidden_dim]
        return context


class CNNLSTMModel(nn.Module):
    """
    Multi-Task CNN-LSTM-Attention hybrid model.
    Output:
    1. Coordinate Regression (E, N, U per station)
    2. Event Classification (Probabilitas Gempa)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 1. CNN Block
        if config.cnn_enabled:
            self.cnn = CNNLayer(
                in_channels=config.n_features,
                out_channels=config.cnn_filters[0],
                kernel_sizes=config.cnn_kernels,
                dropout=config.cnn_dropout
            )
            lstm_input_size = len(config.cnn_kernels) * config.cnn_filters[0]
        else:
            self.cnn = None
            lstm_input_size = config.n_features

        # 2. LSTM Block
        self.lstm = LSTMLayer(
            input_size=lstm_input_size,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            dropout=config.lstm_dropout,
            bidirectional=config.lstm_bidirectional
        )

        # 3. Attention Block
        lstm_out_dim = config.lstm_hidden * (2 if config.lstm_bidirectional else 1)
        if config.attention_enabled:
            self.attention = AttentionLayer(lstm_out_dim)
        else:
            self.attention = None

        # 4. Shared Dense Layers
        dense_input = lstm_out_dim
        dense_layers = []
        for hidden_units in config.dense_units:
            dense_layers.append(nn.Linear(dense_input, hidden_units))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(0.3))
            dense_input = hidden_units
        self.shared_dense = nn.Sequential(*dense_layers)

        # --- TASK SPECIFIC HEADS ---
        n_out = config.n_output_features if config.n_output_features is not None else config.n_features
        self.reg_head = nn.Linear(dense_input, n_out)
        self.cls_head = nn.Sequential(
            nn.Linear(dense_input, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input x: [Batch, Sequence, Features]
        if self.cnn is not None:
            x_cnn_in = x.transpose(1, 2)
            x_cnn_out = self.cnn(x_cnn_in)
            x_lstm_in = x_cnn_out.permute(0, 2, 1)
        else:
            x_lstm_in = x

        # LSTM
        lstm_seq, _ = self.lstm(x_lstm_in) # lstm_seq: [Batch, Seq_len, Hidden_Combined]

        # Attention
        if self.attention is not None:
            shared = self.attention(lstm_seq) # [Batch, Hidden_Combined]
        else:
            # Fallback to last hidden state if attention disabled
            _, last_hidden = self.lstm(x_lstm_in)
            shared = last_hidden

        # Shared Dense
        out = self.shared_dense(shared)

        # Outputs
        coords = self.reg_head(out)
        event_prob = self.cls_head(out)

        return coords, event_prob


class ModelTrainer:
    """Multi-Task Trainer for CNN-LSTM."""
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

        self.reg_criterion = nn.MSELoss()
        self.cls_criterion = nn.BCEWithLogitsLoss()

        logger.info(f"Multi-Task Trainer initialized on device: {self.device}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y_reg, y_cls in train_loader:
            x = x.to(self.device)
            y_reg = y_reg.to(self.device)
            y_cls = y_cls.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            pred_reg, pred_cls = self.model(x)

            loss_reg = self.reg_criterion(pred_reg, y_reg)
            loss_cls = self.cls_criterion(pred_cls, y_cls)

            loss = loss_reg + loss_cls
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
            for x, y_reg, y_cls in val_loader:
                x = x.to(self.device)
                y_reg = y_reg.to(self.device)
                y_cls = y_cls.to(self.device).unsqueeze(1)

                pred_reg, pred_cls = self.model(x)

                loss_reg = self.reg_criterion(pred_reg, y_reg)
                loss_cls = self.cls_criterion(pred_cls, y_cls)

                loss = loss_reg + loss_cls
                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def predict_coordinates(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_tensor = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            pred_reg, _ = self.model(x_tensor)
        return pred_reg.cpu().numpy()

    def predict_event(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_tensor = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            _, pred_cls = self.model(x_tensor)
            pred_cls = torch.sigmoid(pred_cls)
        return pred_cls.cpu().numpy()
