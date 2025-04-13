"""
Warm start predictor model.

This module defines the transformer-based model for predicting
warm start solutions for QP problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Optional, Tuple, Union, List, Any


class TransformerEncoderDecoder(nn.Module):
    """Vanilla Transformer Encoder-Decoder implementation."""
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 output_dim: int = 20,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize the transformer encoder-decoder.
        
        Parameters:
        -----------
        hidden_dim : int
            Dimension of hidden layers
        output_dim : int
            Dimension of output
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
        
        # Create decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        
        # Create decoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=hidden_dim,
            dropout=dropout,
            max_len=100
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        src : torch.Tensor
            Source sequence tensor of shape (batch_size, src_seq_len, hidden_dim)
        tgt : torch.Tensor, optional
            Target sequence tensor of shape (batch_size, tgt_seq_len, hidden_dim)
            If None, the source is used as target
            
        Returns:
        --------
        output : torch.Tensor
            Output tensor of shape (batch_size, tgt_seq_len, hidden_dim)
        """
        # If no target is provided, use source
        if tgt is None:
            tgt = src
            
        # Add positional encoding
        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)
        
        # Pass through encoder
        memory = self.encoder(src)
        
        # Pass through decoder
        output = self.decoder(tgt, memory)
        
        # Project to output dimension
        output = self.output_projection(output)
        
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


class WarmStartPredictor(nn.Module):
    """
    Transformer-based model for predicting warm start solutions.
    
    This model takes QP problem features as input and outputs an approximate
    solution that can be used to warm start the QP solver.
    """
    
    def __init__(self, 
                 input_dim: int = 50,
                 hidden_dim: int = 256,
                 output_dim: int = 20,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize the warm start predictor model.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        hidden_dim : int
            Dimension of hidden layers
        output_dim : int
            Dimension of output solution vector
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
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer model
        self.transformer = TransformerEncoderDecoder(
            hidden_dim=hidden_dim,
            output_dim=1,  # We'll handle the output projection separately
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output layers with residual connections
        self.output_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
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
            Output tensor of shape (batch_size, output_dim)
            containing approximate solution vector
        """
        # Input projection
        x = self.input_projection(x)
        
        # Reshape for transformer: [batch_size, seq_len, hidden_dim]
        # Here we use a sequence length of 1
        x = x.unsqueeze(1)
        
        # Pass through transformer
        transformer_output = self.transformer(x)
        
        # Extract first token of the output sequence
        first_token = transformer_output[:, 0, :]
        
        # Output layers with residual connections
        output = F.relu(self.output_layer1(first_token))
        output = output + first_token  # Residual connection
        output = F.relu(self.output_layer2(output))
        output = output + first_token  # Residual connection
        
        # Final output projection
        solution = self.output_projection(output)
        
        return solution
    
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
            Predicted solution vector
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
            solution = self(x)
            
            # Convert to numpy
            solution = solution.cpu().numpy()
            
        return solution
    
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
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout
        }, filepath)
        
    @classmethod
    def load(cls, filepath: Optional[str] = None) -> 'WarmStartPredictor':
        """
        Load model from file.
        
        Parameters:
        -----------
        filepath : str or None
            Path to load the model from, or None to create a new model
            
        Returns:
        --------
        model : WarmStartPredictor
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
            output_dim=checkpoint['output_dim'],
            num_layers=checkpoint.get('num_layers', 6),
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
            Target tensor of shape (batch_size, output_dim)
        optimizer : torch.optim.Optimizer
            Optimizer to use for the step
            
        Returns:
        --------
        metrics : Dict[str, float]
            Dictionary containing loss and error metrics
        """
        # Set model to training mode
        self.train()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        solution = self(x)
        
        # Compute loss (mean squared error)
        mse_loss = F.mse_loss(solution, y)
        
        # Add L1 regularization for sparsity
        l1_loss = torch.mean(torch.abs(solution))
        
        # Combined loss
        loss = mse_loss + 0.01 * l1_loss
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Compute additional metrics
        mae = F.l1_loss(solution, y).item()
        
        # Compute relative error
        rel_error = torch.norm(solution - y, dim=1) / (torch.norm(y, dim=1) + 1e-8)
        rel_error = torch.mean(rel_error).item()
        
        return {
            'loss': loss.item(),
            'mse': mse_loss.item(),
            'mae': mae,
            'relative_error': rel_error
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
            Target tensor of shape (batch_size, output_dim)
            
        Returns:
        --------
        metrics : Dict[str, float]
            Dictionary containing loss and error metrics
        """
        # Set model to evaluation mode
        self.eval()
        
        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            solution = self(x)
            
            # Compute MSE loss
            mse_loss = F.mse_loss(solution, y)
            
            # Compute additional metrics
            mae = F.l1_loss(solution, y).item()
            
            # Compute relative error
            rel_error = torch.norm(solution - y, dim=1) / (torch.norm(y, dim=1) + 1e-8)
            rel_error = torch.mean(rel_error).item()
            
        return {
            'loss': mse_loss.item(),
            'mse': mse_loss.item(),
            'mae': mae,
            'relative_error': rel_error
        }
