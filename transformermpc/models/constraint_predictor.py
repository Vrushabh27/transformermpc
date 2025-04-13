"""
Constraint predictor model.

This module defines the transformer-based model for predicting
which constraints are active in QP problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Optional, Tuple, Union, List, Any


class TransformerEncoder(nn.Module):
    """Vanilla Transformer Encoder implementation."""
    
    def __init__(self, 
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize the transformer encoder.
        
        Parameters:
        -----------
        hidden_dim : int
            Dimension of hidden layers
        num_layers : int
            Number of transformer layers
        num_heads : int
            Number of attention heads
        dropout : float
            Dropout probability
        """
        super().__init__()
        
        # Make sure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Create encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        
        # Create encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=hidden_dim,
            dropout=dropout,
            max_len=100
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
        --------
        output : torch.Tensor
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder
        output = self.encoder(x)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        """
        Initialize the positional encoding.
        
        Parameters:
        -----------
        d_model : int
            Dimension of the model
        dropout : float
            Dropout probability
        max_len : int
            Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
        --------
        output : torch.Tensor
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ConstraintPredictor(nn.Module):
    """
    Transformer-based model for predicting active constraints.
    
    This model takes QP problem features as input and outputs a binary vector
    indicating which constraints are active.
    """
    
    def __init__(self, 
                 input_dim: int = 50,
                 hidden_dim: int = 128,
                 num_constraints: int = 100,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize the constraint predictor model.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        hidden_dim : int
            Dimension of hidden layers
        num_constraints : int
            Maximum number of constraints to predict
        num_layers : int
            Number of transformer layers
        num_heads : int
            Number of attention heads
        dropout : float
            Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_constraints = num_constraints
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, num_constraints)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns:
        --------
        output : torch.Tensor
            Output tensor of shape (batch_size, num_constraints)
            containing probabilities of constraints being active
        """
        # Input projection
        x = self.input_projection(x)
        
        # Reshape for transformer: [batch_size, seq_len, hidden_dim]
        # Here we use a sequence length of 1
        x = x.unsqueeze(1)
        
        # Pass through transformer
        transformer_output = self.transformer(x)
        
        # Extract first token output
        first_token = transformer_output[:, 0, :]
        
        # Output projection
        logits = self.output_projection(first_token)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        return probs
    
    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Make prediction on input data.
        
        Parameters:
        -----------
        x : torch.Tensor or numpy.ndarray
            Input features
            
        Returns:
        --------
        prediction : numpy.ndarray
            Binary prediction of active constraints
        """
        # Convert to tensor if numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
            
        # Make sure the input has the right shape
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Set model to evaluation mode
        self.eval()
        
        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            probs = self(x)
            
            # Convert to binary predictions
            binary_pred = (probs > 0.5).float()
            
            # Convert to numpy
            binary_pred = binary_pred.cpu().numpy()
            
        return binary_pred
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_constraints': self.num_constraints,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout
        }, filepath)
        
    @classmethod
    def load(cls, filepath: Optional[str] = None) -> 'ConstraintPredictor':
        """
        Load model from file.
        
        Parameters:
        -----------
        filepath : str or None
            Path to load the model from, or None to create a new model
            
        Returns:
        --------
        model : ConstraintPredictor
            Loaded model
        """
        if filepath is None or not os.path.exists(filepath):
            # Return a new model with default parameters
            return cls()
        
        # Load model state
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Create model with saved parameters
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_constraints=checkpoint['num_constraints'],
            num_layers=checkpoint.get('num_layers', 4),
            num_heads=checkpoint.get('num_heads', 8),
            dropout=checkpoint.get('dropout', 0.1)
        )
        
        # Load state dictionary
        model.load_state_dict(checkpoint['state_dict'])
        
        return model
    
    def train_step(self, 
                  x: torch.Tensor, 
                  y: torch.Tensor, 
                  optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
        y : torch.Tensor
            Target tensor of shape (batch_size, num_constraints)
        optimizer : torch.optim.Optimizer
            Optimizer to use for the step
            
        Returns:
        --------
        metrics : Dict[str, float]
            Dictionary containing loss and accuracy
        """
        # Set model to training mode
        self.train()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        probs = self(x)
        
        # Compute loss (binary cross entropy)
        loss = F.binary_cross_entropy(probs, y)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Compute accuracy
        binary_pred = (probs > 0.5).float()
        accuracy = (binary_pred == y).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def validate(self, 
                x: torch.Tensor, 
                y: torch.Tensor) -> Dict[str, float]:
        """
        Validate the model on validation data.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
        y : torch.Tensor
            Target tensor of shape (batch_size, num_constraints)
            
        Returns:
        --------
        metrics : Dict[str, float]
            Dictionary containing loss and accuracy
        """
        # Set model to evaluation mode
        self.eval()
        
        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            probs = self(x)
            
            # Compute loss (binary cross entropy)
            loss = F.binary_cross_entropy(probs, y)
            
            # Compute accuracy
            binary_pred = (probs > 0.5).float()
            accuracy = (binary_pred == y).float().mean().item()
            
            # Compute precision, recall, and F1 score
            binary_pred_flat = binary_pred.flatten().cpu().numpy()
            y_flat = y.flatten().cpu().numpy()
            
            # Compute TP, FP, TN, FN
            tp = ((binary_pred_flat == 1) & (y_flat == 1)).sum()
            fp = ((binary_pred_flat == 1) & (y_flat == 0)).sum()
            fn = ((binary_pred_flat == 0) & (y_flat == 1)).sum()
            
            # Compute precision, recall, and F1
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
