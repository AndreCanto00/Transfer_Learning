import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
import logging
from .model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
    
    def grid_search(self,
                   model_class: Any,
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader,
                   hyperparameters: Dict[str, List[Any]],
                   model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Esegue una ricerca a griglia sugli iperparametri.
        """
        best_accuracy = 0.0
        best_hyperparameters = {}
        best_model = None
        results = []
        
        # Estrazione parametri
        learning_rates = hyperparameters.get('learning_rates', [0.001])
        weight_decays = hyperparameters.get('weight_decays', [0.0001])
        optimizer_names = hyperparameters.get('optimizer_names', ['Adam'])
        
        total_combinations = (
            len(learning_rates) * 
            len(weight_decays) * 
            len(optimizer_names)
        )
        
        logger.info(f"Inizio grid search con {total_combinations} combinazioni")
        
        for lr in learning_rates:
            for weight_decay in weight_decays:
                for optimizer_name in optimizer_names:
                    logger.info(
                        f"\nTest con: LR={lr}, "
                        f"Weight Decay={weight_decay}, "
                        f"Optimizer={optimizer_name}"
                    )
                    
                    # Inizializzazione modello
                    model = model_class(**model_kwargs)
                    
                    # Training
                    accuracy, train_info = self.trainer.train(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        lr=lr,
                        weight_decay=weight_decay,
                        optimizer_name=optimizer_name
                    )
                    
                    # Salvataggio risultati
                    current_result = {
                        'hyperparameters': {
                            'lr': lr,
                            'weight_decay': weight_decay,
                            'optimizer': optimizer_name
                        },
                        'accuracy': accuracy,
                        'training_info': train_info
                    }
                    results.append(current_result)
                    
                    # Aggiornamento migliori iperparametri
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_hyperparameters = current_result['hyperparameters']
                        best_model = model
        
        logger.info("\nRisultati Grid Search:")
        logger.info(f"Migliori Iperparametri: {best_hyperparameters}")
        logger.info(f"Miglior Accuracy: {best_accuracy*100:.2f}%")
        
        return {
            'best_model': best_model,
            'best_hyperparameters': best_hyperparameters,
            'best_accuracy': best_accuracy,
            'all_results': results
        }

    def evaluate_best_model(self,
                          model: nn.Module,
                          test_loader: torch.utils.data.DataLoader) -> float:
        """
        Valuta il modello migliore sul test set.
        """
        test_accuracy = self.trainer.evaluate(model, test_loader)
        logger.info(f"Test Accuracy: {test_accuracy*100:.2f}%")
        return test_accuracy