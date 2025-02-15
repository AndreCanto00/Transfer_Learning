# Transfer_Learning

[![Run Python Tests](https://github.com/AndreCanto00/Transfer_Learning/actions/workflows/test.yml/badge.svg)](https://github.com/AndreCanto00/Transfer_Learning/actions/workflows/test.yml)

project_root/
│
├── .github/
│   └── workflows/
│       └── python-tests.yml
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_utils.py     # Gestione e split del dataset
│   │   └── data_loader.py       # DataLoader e trasformazioni
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── file_operations.py   # Operazioni sui file
│   │
│   └── training/
│       ├── __init__.py
│       ├── trainer.py           # Logica di training
│       └── hyperparameter_tuning.py  # Ricerca iperparametri
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Fixture condivise
│   ├── test_dataset_utils.py   # Test per dataset_utils
│   ├── test_file_operations.py # Test per file_operations
│   ├── test_data_loader.py     # Test per data_loader
│   ├── test_trainer.py         # Test per trainer
│   └── test_hyperparameter_tuning.py  # Test per hyperparameter_tuning
│
├── notebooks/
│   └── training.ipynb          # Notebook principale
│
├── requirements.txt            # Dipendenze del progetto
└── README.md                   # Documentazione del progetto
