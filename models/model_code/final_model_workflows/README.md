
### README for `models/model_code/final_model_workflows`

#### Purpose
This directory contains final_model_workflows that execute the complete training process from data loading to model evaluation. These scripts are designed for production-ready workflows rather than experimental fine-tuning.

#### Scope
Each script in this directory is a standardized, repeatable pipeline that automates the entire training process. These pipelines are designed to be run with minimal modification and can handle different configurations and models.

#### Usage
- Run these scripts to execute a full training workflow, including data loading, model selection, training, and evaluation.
- Can be used to produce final models or evaluate model performance on specific datasets.

#### Directory Structure
- `train_full_pipeline.py`: Executes the entire training process for one or more models, automating data loading, training, and evaluation.
- `train_evaluation_pipeline.py`: A script dedicated to evaluating trained models on validation or test data.

#### Examples
- `train_full_pipeline.py`: Imports data from `data/dataloaders` and models from `models/architecture_classes` to run a complete training workflow.
- `train_evaluation_pipeline.py`: Uses the trained models for inference on validation or test datasets and logs performance metrics.

#### Types of Code/Files
- Python scripts that configure and run training workflows. These scripts are modular, with clear sections for loading data, setting up models, configuring training parameters, and handling evaluation.

#### Interfaces With
- **models/architecture_classes/**: Imports model architectures for training.
- **data/dataloaders/**: Loads data for training using custom dataset classes.
- **models/training_pipelines/**: Some configurations or hyperparameters may be derived from experiments in `training_pipelines`.

#### Gitignore
- Should not be included in `.gitignore` since these files are essential for running and tracking training workflows.