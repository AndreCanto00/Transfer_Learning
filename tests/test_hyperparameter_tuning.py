import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.training.hyperparameter_tuning import HyperparameterTuner
from src.training.trainer import ModelTrainer

class SimpleModel(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    
    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def mock_data():
    x = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=4)

@pytest.fixture
def tuner():
    device = torch.device("cpu")
    trainer = ModelTrainer(device)
    return HyperparameterTuner(trainer)

def test_grid_search(tuner, mock_data):
    # Define a small parameter grid for testing
    param_grid = {
        'learning_rates': [0.001, 0.01],
        'weight_decays': [0.0001],
        'optimizer_names': ['Adam', 'SGD'],
        'num_epochs': 2
    }
    
    results = tuner.grid_search(
        model_class=SimpleModel,
        train_loader=mock_data,
        val_loader=mock_data,
        param_grid=param_grid,
        num_classes=2
    )
    
    # Check that all expected keys are present in results
    assert 'best_hyperparameters' in results
    assert 'best_accuracy' in results
    assert 'best_model' in results
    assert 'all_results' in results
    
    # Check best_hyperparameters structure
    best_params = results['best_hyperparameters']
    assert 'lr' in best_params
    assert 'weight_decay' in best_params
    assert 'optimizer' in best_params
    
    # Check accuracy is valid
    assert 0 <= results['best_accuracy'] <= 1
    
    # Check model instance
    assert isinstance(results['best_model'], SimpleModel)
    
    # Check all_results
    assert len(results['all_results']) == 4  # 2 learning rates * 1 weight decay * 2 optimizers
    
    for result in results['all_results']:
        assert 'lr' in result
        assert 'weight_decay' in result
        assert 'optimizer' in result
        assert 'val_accuracy' in result
        assert 'history' in result
        assert len(result['history']) == 2  # num_epochs = 2

def test_grid_search_empty_grid(tuner, mock_data):
    # Test with empty parameter grid (should use defaults)
    param_grid = {}
    
    results = tuner.grid_search(
        model_class=SimpleModel,
        train_loader=mock_data,
        val_loader=mock_data,
        param_grid=param_grid,
        num_classes=2
    )
    
    # Should still work with default values
    assert isinstance(results['best_model'], SimpleModel)
    assert len(results['all_results']) == 1  # Only default combination

def test_grid_search_model_kwargs(tuner, mock_data):
    # Test passing additional kwargs to model
    param_grid = {
        'learning_rates': [0.001],
        'weight_decays': [0.0001],
        'optimizer_names': ['Adam'],
        'num_epochs': 1
    }
    
    extra_kwarg = {'extra_param': 42}
    
    results = tuner.grid_search(
        model_class=SimpleModel,
        train_loader=mock_data,
        val_loader=mock_data,
        param_grid=param_grid,
        num_classes=2,
        **extra_kwarg
    )
    
    assert isinstance(results['best_model'], SimpleModel)