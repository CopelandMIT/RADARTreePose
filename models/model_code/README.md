### README for `models/model_code`

#### Purpose
This directory contains reusable code components related to model definitions, hyperparameters, and utilities for the overall model architecture and training processes. The scripts and classes in this directory are designed to support experimentation and final model implementations.

#### Scope
The code in `model_code` is modular and not specific to any single experiment or dataset. Instead, it provides general-purpose components, such as model architecture classes, utility functions, and hyperparameter configurations, that can be imported and used across various training and evaluation scripts.

#### Usage
- Use `architecture_classes/` to define or import model architectures for training.
- Use `hyperparameters/` for standardized configurations for experiments and final training runs.
- Use `utilities/` for helper functions and modules that assist in data processing, model evaluation, and other common tasks.

#### Directory Structure
- `architecture_classes/`: Contains Python scripts that define model classes (e.g., neural networks) used in training and experimentation.
- `hyperparameters/`: Stores configuration files for hyperparameters, making it easy to adjust training parameters in one place.
- `utilities/`: Contains utility functions and helper classes that support model training, evaluation, and data handling.

#### Examples
- A user can import a specific model from `architecture_classes/` and define its hyperparameters by referring to `hyperparameters/`.
- `utilities/` might contain functions for logging, data augmentation, or metrics calculation that are widely used across multiple scripts.

#### Types of Code/Files
- Python classes for model architectures.
- JSON or YAML files in `hyperparameters/` for hyperparameter configurations.
- Utility scripts in Python for various helper functions.

#### Interfaces With
- **model_experiments/** and **final_model_workflows/** for importing model architectures and hyperparameters.
- **data/data_code/** and **data/dataloaders/** for processing and loading data.
- **model_data/** for saving trained model artifacts.

