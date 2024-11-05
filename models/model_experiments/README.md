### README for `models/model_experiments`

#### Purpose
This directory, `model_experiments`, contains scripts and notebooks focused on running experimental training configurations for individual models. These files serve as an experimental playground for trying out various model architectures, data structures, preprocessing steps, and hyperparameter tuning. It is intended for exploring different approaches, not for finalized workflows.

#### Scope
Each file here is focused on a specific experiment or aspect of model development, such as testing data handling techniques, fine-tuning hyperparameters, or validating model performance on different datasets. These scripts are exploratory and support refining model configurations, which may later be incorporated into more standardized workflows found in `final_model_workflows`.

#### Usage
- Use these scripts to test and refine various model configurations and techniques.
- Run independently for experimenting with different training setups, validation methods, and feature engineering steps.

#### Directory Structure
- `main_Find_GOUP_DOWN2.ipynb`: A notebook dedicated to experimenting with the GOUP_DOWN prediction task, including fine-tuning and validation.
- `main_LOO_CV_FineTune.ipynb`: Implements Leave-One-Out Cross-Validation for a specific model architecture, focusing on model robustness.
- `main_stability_prediction_preprocessing.ipynb`: Preprocessing steps aimed specifically at stability prediction tasks.

#### Examples
- Each `.ipynb` notebook corresponds to a specific experimental setup, often targeting a unique combination of data, model, or preprocessing technique.
- These notebooks are designed to be modular and adaptable for testing a variety of experimental configurations.

#### Types of Code/Files
- Jupyter notebooks and Python scripts aimed at model experimentation, validation, and hyperparameter tuning.
- Code often includes data loading, model configuration, and training logic that may vary from one experiment to another.

#### Interfaces With
- **model_code/architecture_classes/**: Imports model classes defined in `architecture_classes` for various experiments.
- **data/data_code/dataloaders/**: Uses data loaders and preprocessing utilities to prepare data for experimentation.
- **models/final_model_workflows/**: Insights and configurations from experiments here may inform the final workflows in this directory.

#### Tracking Training Projects and Trials
To ensure all experiments, hyperparameters, results, and outputs are tracked, consider the following solutions:

- **Option 1: Use a Tracking Library (Recommended)**:
  - **MLflow** or **Weights & Biases (W&B)**: These tools provide advanced tracking for experiments, including hyperparameters, metrics, and output artifacts (e.g., plots, model checkpoints). They support comparing results across experiments, aiding in identifying the best configurations.
  - **Implementation**: 
    - Log hyperparameters and metrics (e.g., accuracy, loss) for each experiment.
    - Save artifacts such as plots and checkpoints to the dashboard for accessible visualization.
    - Link each experiment back to the notebook/script used for reproducibility.

- **Option 2: Manual Logging with Versioned Folders**:
  - If not using an external tool, create a logging structure within `model_data/experiment_records` to record outcomes for each experiment.
  - Suggested structure: Use date-based subdirectories like `experiment_YYYY_MM_DD/` to save:
    - `config.json`: Contains hyperparameters.
    - `results.json` or `.csv`: Logs metrics.
    - `plots/`: Stores generated visualizations.
  - Although more manual, this approach can work for small-scale projects without needing external tools.

#### Gitignore
- This directory should generally be tracked in Git. However, consider adding large artifacts such as model checkpoints, plot files, or intermediate data to an external storage location if they exceed Git’s storage limits.