### README for `models/architecture_classes`

#### Purpose
This directory contains model architecture classes that define the structure and forward pass of various deep learning models used in the project.

#### Scope
The code here is focused on implementing the structure of each model (e.g., CNN, LSTM). Each file typically contains one or more model classes that can be imported into training scripts.

#### Usage
- These architecture classes are intended to be imported and instantiated in training pipelines or scripts.
- The models defined here can be used for both training and inference.

#### Directory Structure
- `COPSpeedPredictionByRDMFrameCNNLSTMModel.py`: An example model combining CNN and LSTM layers for predicting COP speed.
- `RdmClassifier.py`: A classifier model for radar data.
- `RdmSingleVariablePrediction.py`: A model for predicting a single variable from radar data.

#### Examples
- `COPSpeedPredictionByRDMFrameCNNLSTMModel.py`: Contains the `COPSpeedPredictionByRDMFrameCNNLSTMModel` class, which combines CNN and LSTM layers.
- Other files define different architectures for tasks like classification or regression.

#### Types of Code/Files
- Python files defining PyTorch model classes, including layers, forward pass methods, and configuration parameters.

#### Interfaces With
- **models/training_pipelines/**: The training scripts in this directory import model architectures from `architecture_classes`.
- **scripts/training/**: End-to-end training scripts import these model classes for executing training workflows.

#### Gitignore
- Not included in `.gitignore`. The code here is essential to the project and should be version-controlled.
