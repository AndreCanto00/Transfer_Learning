import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.training.trainer import ModelTrainer

class SimpleModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    
    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def trainer(device):
    return ModelTrainer(device)

@pytest.fixture
def mock_data():
    # Create simple mock data
    x = torch.randn(20, 10)  # 20 samples, 10 features
    y = torch.randint(0, 2, (20,))  # Binary classification
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=4)

def test_get_optimizer(trainer):
    model = SimpleModel()
    
    # Test Adam optimizer
    adam = trainer.get_optimizer('Adam', model, lr=0.001, weight_decay=0.01)
    assert isinstance(adam, torch.optim.Adam)
    assert adam.defaults['lr'] == 0.001
    assert adam.defaults['weight_decay'] == 0.01
    
    # Test SGD optimizer
    sgd = trainer.get_optimizer('SGD', model, lr=0.001, weight_decay=0.01)
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.defaults['lr'] == 0.001
    assert sgd.defaults['momentum'] == 0.9
    
    # Test invalid optimizer
    with pytest.raises(ValueError):
        trainer.get_optimizer('InvalidOptimizer', model, lr=0.001, weight_decay=0.01)

def test_train_epoch(trainer, mock_data):
    model = SimpleModel()
    optimizer = trainer.get_optimizer('Adam', model, lr=0.001, weight_decay=0.01)
    
    loss = trainer.train_epoch(model, mock_data, optimizer)
    assert isinstance(loss, float)
    assert loss > 0  # Loss should be positive

def test_evaluate(trainer, mock_data):
    model = SimpleModel()
    accuracy = trainer.evaluate(model, mock_data)
    
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1  # Accuracy should be between 0 and 1

def test_train_model(trainer, mock_data):
    model = SimpleModel()
    val_accuracy, result = trainer.train(
        model=model,
        train_loader=mock_data,
        val_loader=mock_data,
        lr=0.001,
        weight_decay=0.01,
        optimizer_name='Adam',
        num_epochs=2
    )
    
    assert isinstance(val_accuracy, float)
    assert 0 <= val_accuracy <= 1
    assert len(result['history']) == 2  # Should have metrics for 2 epochs
    
    # Check history structure
    for epoch_data in result['history']:
        assert 'epoch' in epoch_data
        assert 'loss' in epoch_data
        assert 'val_accuracy' in epoch_data