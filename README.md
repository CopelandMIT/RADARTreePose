# RADARTreePose

## Project Overview

The RADARTreePose project aims to develop and train machine learning models that assess balance and fall risk using radar data. By leveraging base truth data from motion capture (MOCAP) and force plate measurements, we can create non-invasive, privacy-preserving technologies that monitor and evaluate fall risk. This project has broad implications for improving elderly care and proactive fall prevention.

## Motivation

Falls are a leading cause of injury among older adults, and early detection of balance issues can prevent severe injuries. Traditional methods for balance assessment often involve cameras and other invasive monitoring systems, raising privacy concerns. RADARTreePose aims to address this by using radar technology, which provides a more private and non-intrusive way to measure movement and balance.

## Project Structure

### 1. `conventional_analysis/`
   - Contains scripts and tools for traditional analysis approaches that may complement or validate the machine learning methods.
   - Examples include statistical analysis of force plate data and conventional biomechanics measurements.

### 2. `data_scripts/`
   - Scripts and classes for loading, processing, and organizing various data types (e.g., radar, force plate, MOCAP).
   - Subdirectories:
     - **`data_type_classes/`**: Classes specific to each data type (force plate, MOCAP, radar).
     - **`dataloaders/`**: Custom data loaders for structured access to dataset samples.
     - **`initial_processing_pipelines/`**: Processing pipelines to clean, preprocess, and transform data for model training and evaluation.

### 3. `docs/`
   - Documentation, notes, references, and research protocols related to the project.
   - Organized subfolders include:
     - **`notes/`**: Informal notes and brainstorming sessions.
     - **`paper_drafts/`**: Drafts for research papers and other publications.
     - **`protocols/`**: Established experimental and data processing protocols.
     - **`references/`**: External references and literature.

### 4. `envs/`
   - Environment configuration files for setting up consistent development environments.
   - **`radartreepose_env.yaml`**: Conda environment file listing dependencies required for the project.

### 5. `models/`
   - Code, data, and experiment logs for model development.
   - Subdirectories:
     - **`model_code/`**: Scripts and notebooks for model architecture, training, and utilities.
     - **`model_data/`**: Data files specifically curated for training and evaluating models.
     - **`model_experiments/`**: Experiment records, logs, and checkpoints for model performance tracking.

### 6. `plots/`
   - Visualizations for data analysis, model evaluation, and project presentations.
   - Subdirectories:
     - **`data_visualization/`**: Plots that provide insights into raw and processed data.
     - **`figure_generation/`**: Templates and scripts for generating publication-ready figures.
     - **`model_evaluation/`**: Plots related to model accuracy, confusion matrices, and other evaluation metrics.

### 7. `RADARTreePose_Data/`
   - Centralized repository for all project data, including:
     - **`raw/`**: Raw data directly from experiments, organized by sensor type (force plate, MOCAP, radar).
     - **`processed/`**: Processed data, ready for analysis or model input.
     - **`preprocessed/`**: Intermediate data transformations and cleaning steps.
     - **`metadata/`**: Metadata files containing descriptions, labels, and other context for data files.

### Project Files

- **`.gitignore`**: Specifies files and directories to be ignored by Git, such as environment files and large datasets.
- **`requirements.txt`**: Lists Python dependencies necessary to run the project.

## Goals

1. **Develop Privacy-Preserving Balance Assessment**: Use radar data to predict balance metrics without the need for intrusive monitoring.
2. **Train Models with Ground Truth Data**: Leverage MOCAP and force plate data as ground truth for model training and validation.
3. **Enable Non-Invasive Monitoring**: Create technology that facilitates remote and continuous balance assessment, particularly beneficial for elderly individuals in assisted living settings.

## How to Get Started

1. **Set Up the Environment**:
   - Clone the repository:
     ```bash
     git clone https://github.com/CopelandMIT/RADARTreePose.git
     ```
   - Install dependencies using the provided environment file:
     ```bash
     conda env create -f envs/radartreepose_env.yaml
     conda activate radartreepose
     ```

2. **Data Preparation**:
   - Organize raw data under `RADARTreePose_Data/raw/`, following the structure for different sensor types.
   - Run initial processing pipelines in `data_scripts/initial_processing_pipelines` to clean and preprocess data.

3. **Model Training**:
   - Use scripts in `models/model_code` to define and train models. Adjust configurations and hyperparameters as necessary.

4. **Evaluate and Visualize**:
   - Evaluate models using metrics available in `plots/model_evaluation`.
   - Generate visualizations to validate data and results in `plots/data_visualization`.

## Contributing

This project welcomes contributions. To get started:

1. Create a new branch for your changes.
2. Commit your changes and push them to your branch.
3. Submit a pull request with a clear description of your changes.

## Future Directions

- **Expand to Additional Sensor Types**: Integrate data from additional non-invasive sensors.
- **Automate Data Pipeline**: Improve the processing pipeline for scalability and ease of deployment.
- **Real-World Deployment**: Test and validate models in real-world settings to assess performance in diverse environments.