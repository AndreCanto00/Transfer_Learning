import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, device: torch.device):
        """
        Initialize the model trainer.
        
        Args:
            device: The device to use for training (CPU/GPU)
        """
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def get_optimizer(self, 
                     optimizer_name: str, 
                     model: nn.Module, 
                     lr: float, 
                     weight_decay: float) -> optim.Optimizer:
        """
        Get the specified optimizer.
        
        Args:
            optimizer_name: Name of the optimizer ('Adam' or 'SGD')
            model: The model to optimize
            lr: Learning rate
            weight_decay: Weight decay factor
        
        Returns:
            The configured optimizer
        """
        if optimizer_name == 'Adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def train_epoch(self, 
                    model: nn.Module, 
                    train_loader: DataLoader, 
                    optimizer: optim.Optimizer) -> float:
        """
        Train the model for one epoch.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            optimizer: The optimizer to use
        
        Returns:
            Average loss for the epoch
        """
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(train_loader)
    
    def evaluate(self, 
                model: nn.Module, 
                data_loader: DataLoader) -> float:
        """
        Evaluate the model on a dataset.
        
        Args:
            model: The model to evaluate
            data_loader: DataLoader for evaluation data
        
        Returns:
            Accuracy score
        """
        model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def train_model(self, 
                   model: nn.Module, 
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   lr: float, 
                   weight_decay: float, 
                   optimizer_name: str,
                   num_epochs: int = 10) -> Tuple[float, List[Dict[str, float]]]:
        """
        Train the model with the given parameters.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            lr: Learning rate
            weight_decay: Weight decay factor
            optimizer_name: Name of the optimizer
            num_epochs: Number of epochs to train
        
        Returns:
            Tuple of (final validation accuracy, training history)
        """
        model = model.to(self.device)
        optimizer = self.get_optimizer(optimizer_name, model, lr, weight_decay)
        history = []
        
        for epoch in range(num_epochs):
            # Train
            epoch_loss = self.train_epoch(model, train_loader, optimizer)
            
            # Evaluate
            val_accuracy = self.evaluate(model, val_loader)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
                       f"Validation Accuracy: {val_accuracy*100:.2f}%")
            
            # Store metrics
            history.append({
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'val_accuracy': val_accuracy
            })
        
        return val_accuracy, history