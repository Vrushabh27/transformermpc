"""
Warm start predictor model.

This module defines the transformer-based model for predicting
warm start solutions for QP problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import numpy as np
import os
from typing import Dict, Optional, Tuple, Union, List, Any


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
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer configuration
        self.transformer_config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=4 * hidden_dim,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        
        # Transformer model
        self.transformer = BertModel(self.transformer_config)
        
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
        transformer_output = self.transformer(inputs_embeds=x).last_hidden_state
        
        # Extract CLS token output (first token of sequence)
        cls_output = transformer_output[:, 0, :]
        
        # Output layers with residual connections
        output = F.relu(self.output_layer1(cls_output))
        output = output + cls_output  # Residual connection
        output = F.relu(self.output_layer2(output))
        output = output + cls_output  # Residual connection
        
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
            'transformer_config': self.transformer_config
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
            num_layers=checkpoint['transformer_config'].num_hidden_layers,
            num_heads=checkpoint['transformer_config'].num_attention_heads,
            dropout=checkpoint['transformer_config'].hidden_dropout_prob
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
