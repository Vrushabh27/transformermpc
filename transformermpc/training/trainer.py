"""
Model trainer module.

This module provides classes for training the transformer models.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from ..models.constraint_predictor import ConstraintPredictor
from ..models.warm_start_predictor import WarmStartPredictor
from ..utils.visualization import plot_training_history, plot_constraint_prediction_metrics, plot_warm_start_metrics


class ModelTrainer:
    """
    Trainer class for training transformer models.
    
    This class handles the training loop, validation, logging, and checkpointing
    for the transformer models.
    """
    
    def __init__(self,
                model: Union[ConstraintPredictor, WarmStartPredictor],
                train_dataset: Any,
                val_dataset: Any,
                batch_size: int = 32,
                learning_rate: float = 1e-4,
                weight_decay: float = 1e-5,
                num_epochs: int = 2000,
                patience: int = 50,
                checkpoint_dir: Optional[str] = None,
                device: Optional[torch.device] = None):
        """
        Initialize the trainer.
        
        Parameters:
        -----------
        model : ConstraintPredictor or WarmStartPredictor
            Model to train
        train_dataset : Dataset
            Training dataset
        val_dataset : Dataset
            Validation dataset
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        weight_decay : float
            Weight decay for optimizer
        num_epochs : int
            Number of epochs to train for
        patience : int
            Number of epochs to wait for improvement before early stopping
        checkpoint_dir : str or None
            Directory to save checkpoints
        device : torch.device or None
            Device to use for training (defaults to GPU if available)
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Move model to device
        self.model.to(self.device)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            verbose=True
        )
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Determine model type
        self.is_constraint_predictor = isinstance(model, ConstraintPredictor)
        
        # Add model-specific metrics to history
        if self.is_constraint_predictor:
            self.history.update({
                'train_accuracy': [],
                'val_accuracy': [],
                'val_precision': [],
                'val_recall': [],
                'val_f1': []
            })
        else:
            self.history.update({
                'train_mse': [],
                'val_mse': [],
                'train_mae': [],
                'val_mae': [],
                'train_relative_error': [],
                'val_relative_error': []
            })
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ModelTrainer')
        
    def train(self, log_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Parameters:
        -----------
        log_dir : str or None
            Directory to save tensorboard logs
            
        Returns:
        --------
        history : Dict[str, List[float]]
            Training history
        """
        # Set up tensorboard if available
        if TENSORBOARD_AVAILABLE and log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
            self.logger.info(f"Tensorboard logs will be saved to {log_dir}")
        else:
            writer = None
            if log_dir is not None:
                self.logger.warning("Tensorboard not available, logs will not be saved")
        
        # Initialize best validation loss and patience counter
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        self.logger.info(f"Model type: {'Constraint Predictor' if self.is_constraint_predictor else 'Warm Start Predictor'}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training dataset size: {len(self.train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(self.val_dataset)}")
        
        # Training loop
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_metrics = self._train_epoch()
            
            # Validate
            val_metrics = self._validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Update history
            for k, v in train_metrics.items():
                if f'train_{k}' in self.history:
                    self.history[f'train_{k}'].append(v)
            
            for k, v in val_metrics.items():
                if f'val_{k}' in self.history:
                    self.history[f'val_{k}'].append(v)
            
            # Log metrics
            epoch_time = time.time() - start_time
            self._log_metrics(epoch, train_metrics, val_metrics, epoch_time, writer)
            
            # Check for improvement
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                best_epoch = epoch
                
                # Save best model
                if self.checkpoint_dir is not None:
                    self._save_checkpoint(f"best_model.pt")
                    
            else:
                patience_counter += 1
                
            # Save checkpoint periodically
            if self.checkpoint_dir is not None and (epoch + 1) % 100 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            # Early stopping
            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1} as validation loss hasn't improved for {self.patience} epochs")
                break
                
        # Log final results
        self.logger.info(f"Training finished. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
        
        # Load best model
        if self.checkpoint_dir is not None:
            self._load_checkpoint(f"best_model.pt")
            
        # Close tensorboard writer
        if writer is not None:
            writer.close()
            
        # Plot training history
        self._plot_history()
            
        return self.history
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
        --------
        metrics : Dict[str, float]
            Training metrics for the epoch
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize metrics
        metrics_sum = {}
        
        # Training loop
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            # Extract features and targets
            features = batch['features'].to(self.device)
            
            if self.is_constraint_predictor:
                targets = batch['active_constraints'].to(self.device)
            else:
                targets = batch['solution'].to(self.device)
            
            # Train step
            batch_metrics = self.model.train_step(features, targets, self.optimizer)
            
            # Update metrics sum
            for k, v in batch_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v
        
        # Compute average metrics
        metrics = {k: v / len(self.train_loader) for k, v in metrics_sum.items()}
        
        return metrics
    
    def _validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
        --------
        metrics : Dict[str, float]
            Validation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        metrics_sum = {}
        
        # Validation loop
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Extract features and targets
                features = batch['features'].to(self.device)
                
                if self.is_constraint_predictor:
                    targets = batch['active_constraints'].to(self.device)
                else:
                    targets = batch['solution'].to(self.device)
                
                # Validate
                batch_metrics = self.model.validate(features, targets)
                
                # Update metrics sum
                for k, v in batch_metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0) + v
        
        # Compute average metrics
        metrics = {k: v / len(self.val_loader) for k, v in metrics_sum.items()}
        
        return metrics
    
    def _log_metrics(self, 
                    epoch: int, 
                    train_metrics: Dict[str, float],
                    val_metrics: Dict[str, float],
                    epoch_time: float,
                    writer: Optional[Any] = None) -> None:
        """
        Log metrics for the epoch.
        
        Parameters:
        -----------
        epoch : int
            Current epoch
        train_metrics : Dict[str, float]
            Training metrics
        val_metrics : Dict[str, float]
            Validation metrics
        epoch_time : float
            Time taken for the epoch
        writer : SummaryWriter or None
            Tensorboard writer
        """
        # Log to console
        log_str = f"Epoch {epoch+1}/{self.num_epochs} ({epoch_time:.2f}s) - "
        log_str += f"Train Loss: {train_metrics['loss']:.6f}, "
        log_str += f"Val Loss: {val_metrics['loss']:.6f}"
        
        if self.is_constraint_predictor:
            log_str += f", Train Acc: {train_metrics['accuracy']:.4f}, "
            log_str += f"Val Acc: {val_metrics['accuracy']:.4f}, "
            log_str += f"Val F1: {val_metrics['f1']:.4f}"
        else:
            log_str += f", Train MSE: {train_metrics['mse']:.6f}, "
            log_str += f"Val MSE: {val_metrics['mse']:.6f}, "
            log_str += f"Val Rel Err: {val_metrics['relative_error']:.4f}"
            
        self.logger.info(log_str)
        
        # Log to tensorboard
        if writer is not None:
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
                
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)
                
            writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)
    
    def _save_checkpoint(self, filename: str) -> None:
        """
        Save a checkpoint.
        
        Parameters:
        -----------
        filename : str
            Checkpoint filename
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, filepath)
        
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def _load_checkpoint(self, filename: str) -> None:
        """
        Load a checkpoint.
        
        Parameters:
        -----------
        filename : str
            Checkpoint filename
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(filepath):
            self.logger.warning(f"Checkpoint {filepath} does not exist")
            return
            
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def _plot_history(self) -> None:
        """
        Plot training history.
        """
        # Plot general training history
        plot_training_history(self.history)
        
        # Plot model-specific metrics
        epochs = list(range(1, len(self.history['train_loss']) + 1))
        
        if self.is_constraint_predictor:
            plot_constraint_prediction_metrics(
                self.history['val_precision'],
                self.history['val_recall'],
                self.history['val_f1'],
                self.history['val_accuracy'],
                epochs
            )
        else:
            plot_warm_start_metrics(
                self.history['val_mse'],
                self.history['val_mae'],
                self.history['val_relative_error'],
                epochs
            )
    
    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        test_loader : DataLoader or None
            Test data loader. If None, use validation loader.
            
        Returns:
        --------
        metrics : Dict[str, float]
            Evaluation metrics
        """
        # Use validation loader if test loader is not provided
        if test_loader is None:
            test_loader = self.val_loader
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        metrics_sum = {}
        
        # Evaluation loop
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation"):
                # Extract features and targets
                features = batch['features'].to(self.device)
                
                if self.is_constraint_predictor:
                    targets = batch['active_constraints'].to(self.device)
                else:
                    targets = batch['solution'].to(self.device)
                
                # Validate
                batch_metrics = self.model.validate(features, targets)
                
                # Update metrics sum
                for k, v in batch_metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0) + v
        
        # Compute average metrics
        metrics = {k: v / len(test_loader) for k, v in metrics_sum.items()}
        
        # Log metrics
        log_str = "Evaluation metrics: "
        for k, v in metrics.items():
            log_str += f"{k}: {v:.6f}, "
        self.logger.info(log_str[:-2])  # Remove last comma and space
        
        return metrics
