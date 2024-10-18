# RADARTreePose

**RADARTreePose** is a research project focused on utilizing radar and motion capture (MoCap) data to analyze human movement and balance. This project integrates multiple sensor types including force plates (FP), radar, and MoCap data, leveraging neural networks such as CNNs and LSTMs for data classification and analysis.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Data Preprocessing and Pipelines](#data-preprocessing-and-pipelines)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

## Project Overview

This repository is designed to collect, process, and analyze multi-sensor data to assess human motion and balance. The core objective is to combine radar and motion capture data using machine learning models for tasks such as velocity estimation, 3D pose analysis, and human pose classification during exercises.

The project also includes tools for managing large datasets, performing time-series alignment, and comparing data across multiple sensors. The final goal is to build a comprehensive and synchronized data pipeline that can handle the complex analysis involved in human movement assessment.

## Directory Structure

Here’s an overview of the updated directory structure:

```
├── RADARTreePose/
│   ├── envs/                  # Environment YAML files
│   │   └── radartreepose_env.yml
│   ├── data/                  # Contains raw and processed datasets
│   ├── preprocessing/          # Scripts related to data preprocessing
│   │   └── scripts/             
│   │        └── pipelines/    # Data pipelines for FP, RADAR, and MoCap data
│   ├── models/                # Machine Learning model definitions
│   ├── notebooks/             # Jupyter Notebooks for data exploration and analysis
│   ├── archived/              # Archived or old files that are not used anymore
│   └── README.md              # Project documentation
```

### Key Directories

- `envs/`: Contains environment YAML files needed to replicate the environment using `conda` or `virtualenv`. 
- `data/`: Holds both raw and processed data, including CSV and `.tsv` files from radar, MoCap, and force plates.
- `preprocessing/`: Contains scripts for preprocessing data from multiple sensors.
  - `pipelines/`: Houses scripts that manage the preprocessing workflows for different sensor data.
- `models/`: Contains neural network models such as CNNs and LSTMs.
- `notebooks/`: Jupyter notebooks for data exploration, pipeline testing, and early-stage analyses.
- `archived/`: Old or deprecated scripts that are no longer used but preserved for reference.

## Data Preprocessing and Pipelines

- **Force Plate Data**: Scripts are located in the `pipelines/` folder, responsible for force plate data preprocessing.
- **Radar Data**: The radar data preprocessing scripts ensure radar timestamps are synced with other sensors.
- **MoCap Data**: Scripts to handle 3D pose extraction and comparison against other data modalities (like radar and force plates).

Scripts are structured to:
- Read raw `.tsv` files, preprocess them (filter, clean, compute angles, velocities).
- Synchronize the data across multiple sensors.
- Push the data through a processing pipeline for model training or analysis.

You can configure each pipeline to match the structure of your datasets and the specific analysis tasks.

## Requirements

To run this project, the following dependencies are required:

- Python 3.8 or higher
- PyTorch
- Pandas
- NumPy
- Matplotlib
- TorchVision
- Other dependencies listed in `requirements.txt`

To install these dependencies, use the provided `radartreepose_env.yml` file.

```bash
conda env create -f envs/radartreepose_env.yml
conda activate radartreepose_env
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/CopelandMIT/RADARTreePose.git
   cd RADARTreePose
   ```

2. **Set up the environment**:
   ```bash
   conda env create -f envs/radartreepose_env.yml
   conda activate radartreepose_env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up local data directories**:
   Ensure that your `data/` directory is set up correctly with necessary subdirectories (`raw`, `processed`) and contains the datasets required for processing.

## Usage

### Running Preprocessing Pipelines

To preprocess data from multiple sensors (force plates, radar, MoCap):

```bash
python preprocessing/scripts/pipelines/main_FP_Preprocessing.py
python preprocessing/scripts/pipelines/main_FP_to_MC_Comparison.py
```

These scripts will load the relevant `.tsv` files, process the data (e.g., calculate angles, velocities), and save the output to the appropriate directory.

### Running Machine Learning Models

To train the CNN-LSTM models on radar and MoCap data:

```bash
python models/train_rdm_cnn_lstm.py
```

This will load the preprocessed data, train the model, and save the results for further analysis.

## Contributing

If you would like to contribute to this project:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Push to your branch and open a Pull Request.

Be sure to fix any broken links or dependencies during the reorganization process. Any contributions to model improvements, pipeline optimization, or adding new features are welcome!

## Contact

**Dan Copeland**  
Email: [dcope@mit.edu](mailto:dcope@mit.edu)  
