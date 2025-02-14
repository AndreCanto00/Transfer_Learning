from typing import Dict, List, Any
import logging
from torch import nn
from torch.utils.data import DataLoader
from .trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, trainer: ModelTrainer):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            trainer: ModelTrainer instance
        """
        self.trainer = trainer
    
    def grid_search(self,
                   model_class: Any,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   param_grid: Dict[str, List[Any]],
                   num_classes: int,
                   **model_kwargs) -> Dict[str, Any]:
        """
        Perform grid search over hyperparameters.
        
        Args:
            model_class: The model class to instantiate
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            param_grid: Dictionary of parameters to search
            num_classes: Number of classes for the model
            **model_kwargs: Additional arguments for model instantiation
        
        Returns:
            Dictionary containing best hyperparameters and results
        """
        best_accuracy = 0.0
        best_hyperparameters = {}
        best_model = None
        all_results = []
        
        # Extract parameters from grid
        learning_rates = param_grid.get('learning_rates', [0.001])
        weight_decays = param_grid.get('weight_decays', [0.0001])
        optimizer_names = param_grid.get('optimizer_names', ['Adam'])
        num_epochs = param_grid.get('num_epochs', 10)
        
        total_combinations = (len(learning_rates) * 
                            len(weight_decays) * 
                            len(optimizer_names))
        
        logger.info(f"Starting grid search with {total_combinations} combinations")
        
        for lr in learning_rates:
            for weight_decay in weight_decays:
                for optimizer_name in optimizer_names:
                    logger.info(f"\nTesting: LR={lr}, "
                              f"Weight Decay={weight_decay}, "
                              f"Optimizer={optimizer_name}")
                    
                    # Initialize model
                    model = model_class(num_classes=num_classes, **model_kwargs)
                    
                    # Train model
                    accuracy_val, history = self.trainer.train_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        lr=lr,
                        weight_decay=weight_decay,
                        optimizer_name=optimizer_name,
                        num_epochs=num_epochs
                    )
                    
                    # Store results
                    result = {
                        'lr': lr,
                        'weight_decay': weight_decay,
                        'optimizer': optimizer_name,
                        'val_accuracy': accuracy_val,
                        'history': history
                    }
                    all_results.append(result)
                    
                    # Update best model if necessary
                    if accuracy_val > best_accuracy:
                        best_accuracy = accuracy_val
                        best_hyperparameters = {
                            'lr': lr,
                            'weight_decay': weight_decay,
                            'optimizer': optimizer_name
                        }
                        best_model = model
        
        logger.info("\nGrid Search Results:")
        logger.info(f"Best Hyperparameters: {best_hyperparameters}")
        logger.info(f"Best Validation Accuracy: {best_accuracy*100:.2f}%")
        
        return {
            'best_hyperparameters': best_hyperparameters,
            'best_accuracy': best_accuracy,
            'best_model': best_model,
            'all_results': all_results
        }