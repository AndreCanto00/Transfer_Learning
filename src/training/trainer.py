import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, device: torch.device, num_epochs: int = 10):
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
    
    def get_optimizer(self, 
                     optimizer_name: str, 
                     model: nn.Module, 
                     lr: float, 
                     weight_decay: float) -> optim.Optimizer:
        """
        Crea l'ottimizzatore specificato con i parametri dati.
        """
        optimizers = {
            'Adam': lambda: optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            ),
            'SGD': lambda: optim.SGD(
                model.parameters(), 
                lr=lr, 
                momentum=0.9, 
                weight_decay=weight_decay
            )
        }
        
        if optimizer_name not in optimizers:
            raise ValueError(f"Ottimizzatore non supportato: {optimizer_name}")
            
        return optimizers[optimizer_name]()

    def train_epoch(self, 
                   model: nn.Module, 
                   train_loader: torch.utils.data.DataLoader,
                   optimizer: optim.Optimizer) -> float:
        """
        Esegue un'epoca di training.
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
                data_loader: torch.utils.data.DataLoader) -> float:
        """
        Valuta il modello sul dataset fornito.
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

    def train(self, 
             model: nn.Module, 
             train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader,
             lr: float, 
             weight_decay: float, 
             optimizer_name: str,
             num_epochs: int = None) -> Tuple[float, Dict]:
        """
        Esegue il training completo del modello.
        """
        if num_epochs is None:
            num_epochs = self.num_epochs
        
        model = model.to(self.device)
        optimizer = self.get_optimizer(optimizer_name, model, lr, weight_decay)
        best_val_acc = 0.0
        training_history = []
        
        for epoch in range(num_epochs):
            # Training
            epoch_loss = self.train_epoch(model, train_loader, optimizer)
            
            # Validation
            val_accuracy = self.evaluate(model, val_loader)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} "
                f"Loss: {epoch_loss:.4f} "
                f"Val Acc: {val_accuracy*100:.2f}%"
            )
            
            # Save best accuracy
            best_val_acc = max(best_val_acc, val_accuracy)
            
            # Save history
            training_history.append({
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'val_accuracy': val_accuracy
            })
            
        return best_val_acc, {
            'history': training_history,
            'best_val_accuracy': best_val_acc
        }