"""
Trainer Module
==============
Advanced training dengan early stopping, learning rate scheduling, checkpointing.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping untuk mencegah overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0001,
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Jumlah epochs tanpa improvement sebelum stop
            min_delta (float): Minimum change untuk dianggap improvement
            restore_best_weights (bool): Restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Model to save weights
        
        Returns:
            bool: True if should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict()
        elif val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict()
            logger.info(f"Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                return True
        
        return False
    
    def restore_best_weights_to_model(self, model: nn.Module):
        """Restore best weights ke model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best weights from training")


class AdvancedModelTrainer:
    """Advanced trainer dengan scheduling dan checkpointing."""
    
    def __init__(self, model: nn.Module, config: Dict, device: str = "cpu"):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): Model to train
            config (Dict): Configuration dictionary
            device (str): 'cuda' atau 'cpu'
        """
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        ) if config.get('scheduler_enabled', False) else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 15),
            min_delta=config.get('early_stopping_min_delta', 0.0001)
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_epoch = 0
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train satu epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
        
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Flatten y jika perlu
            if y.dim() > 2:
                y = y.reshape(y.shape[0], -1)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping untuk stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Batch {batch_idx+1}: loss={loss.item():.6f}")
        
        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate pada validation set.
        
        Args:
            val_loader (DataLoader): Validation data loader
        
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                if y.dim() > 2:
                    y = y.reshape(y.shape[0], -1)
                
                pred = self.model(x)
                loss = self.criterion(pred, y)
                
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, checkpoint_dir: Optional[str] = None) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of epochs
            checkpoint_dir (str): Directory untuk checkpoints
        
        Returns:
            Dict: Training history
        """
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            
            # Log
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | "
                           f"Train Loss: {train_loss:.6f} | "
                           f"Val Loss: {val_loss:.6f}")
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Checkpointing
            if checkpoint_dir and (epoch + 1) % 20 == 0:
                ckpt_path = Path(checkpoint_dir) / f"model_epoch_{epoch+1}.pt"
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Stopped at epoch {epoch+1}")
                self.best_epoch = epoch + 1 - self.early_stopping.patience
                break
        
        # Restore best weights
        if self.early_stopping.restore_best_weights:
            self.early_stopping.restore_best_weights_to_model(self.model)
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_epoch': self.best_epoch if self.best_epoch else epochs,
            'best_loss': min(self.val_losses) if self.val_losses else float('inf')
        }
        
        logger.info("Training completed")
        logger.info(f"Best loss: {history['best_loss']:.6f} at epoch {history['best_epoch']}")
        
        return history
    
    def predict_on_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on batch.
        
        Args:
            x (np.ndarray): Input data
        
        Returns:
            np.ndarray: Predictions
        """
        self.model.eval()
        
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        with torch.no_grad():
            pred = self.model(x_tensor)
        
        return pred.cpu().numpy()


def main_trainer():
    """Test trainer module."""
    logger.info("Trainer module loaded successfully")


if __name__ == "__main__":
    main_trainer()