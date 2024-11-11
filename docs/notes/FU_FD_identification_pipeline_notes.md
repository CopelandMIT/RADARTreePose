# FU/FD Identification Pipeline - Documentation

## Objective

This pipeline is designed to identify and label frames in radar data as part of a "Foot Up" (FU), "Foot Down" (FD), or stability event. Using radar data, alongside ground truth motion capture (MoCap) and force plate data, the pipeline applies processing, preprocessing, and labeling steps to create a dataset that enables model training for balance assessment and fall risk prediction.

---

## Pipeline Overview

The following outlines each stage in the pipeline, the files/scripts involved, and the corresponding data directory structure.

### 1. Raw Data Collection

- **Directory:** `RADARTreePose_Data/raw/`
- **Description:** This directory holds raw data files captured from various sensors:
  - **Radar Data:** The primary data source for FU/FD identification.
  - **MoCap Data:** Provides ground truth for body position.
  - **Force Plate Data:** Captures force changes and stability indicators.
- **Purpose:** These raw data files serve as the starting point for all processing.

### 2. Initial Data Processing

- **Script:** `data_scripts/initial_processing_pipelines/RADAR_RDM_Generation_v1.ipynb`
- **Description:** Processes the raw radar data to generate Range-Doppler data matrices, which form the basis for FU/FD event identification.
- **Steps:**
  1. **Range-Doppler Processing:** Converts radar data into Range-Doppler frames.
     ```python
     processed_data = RADAR_object.range_doppler_processing(dataCubes)
     ```
  2. **Data Sub-selection:** Reduces the data to relevant sub-sections.
     ```python
     sub_selected_processed_data = RADAR_object.sub_select_RADAR_DATA(processed_data)
     ```
  3. **Convert to 3D Array:** Creates a 3D numpy array for compatibility with later stages.
     ```python
     processed_frames_array = np.array(sub_selected_processed_data)
     ```
- **Output Directory:** `RADARTreePose_Data/processed/radar/`
  - The processed radar data is saved here, organized by sensor type.

### 3. Preprocessing: Windowing and Labeling

- **Script:** `data_scripts/preprocessing_pipelines/FUFD_Labeling_and_Windowing_Pipeline_v1.py`
- **Description:** This script takes processed radar data, segments it into windows, and labels each window as "Foot Up," "Foot Down," or stability, based on ground truth data from MoCap and force plate readings.
- **Steps:**
  1. **Windowing:** Splits the radar data into segments, creating smaller analysis windows.
  2. **Labeling:** Assigns FU, FD, or stability labels to each segment, using the MoCap and force plate data as references.
- **Output Directory:** `RADARTreePose_Data/preprocessed/`
  - The windowed and labeled data is saved here, ready for model training and evaluation.

### 4. Model Training and Evaluation

- **Model Architectures:** Defined in `models/model_code/`
  - Key model files include:
    - `COPSpeedPredictionByRDMFrameCNNLSTMModel.py` - For speed prediction during balance activities.
    - `RdmClassifier.py` - Classifies frames into FU, FD, or stability categories.
- **Data Loaders:** Located in `data_scripts/dataloaders/`
  - These scripts load windowed and labeled data, transforming it as necessary for model input.
- **Experiment Configuration and Training:** Configured in `models/model_experiments/`
  - Example scripts set training parameters, initialize model architecture, and execute the training loop.
  
### 5. Visualization and Analysis

- **Visualizations:** Found in `plots/`
  - **Data Visualization:** Visualizes processed radar data with FU, FD, and stability labels for verification.
  - **Model Evaluation:** Visualizes model performance, showing FU, FD predictions, accuracy, and other metrics.

---

## Data Flow Summary

1. **Raw Data Collection**: `RADARTreePose_Data/raw/`
2. **Initial Processing (Range-Doppler Generation)**: `data_scripts/initial_processing_pipelines/RADAR_RDM_Generation_v1.ipynb`
   - Output: `RADARTreePose_Data/processed/`
3. **Windowing and Labeling**: `data_scripts/preprocessing_pipelines/Labeling_and_Windowing_Pipeline.py`
   - Output: `RADARTreePose_Data/preprocessed/`
4. **Model Training & Evaluation**: `models/model_code/`, `models/model_experiments/`
5. **Visualization**: `plots/`

---

## Naming Conventions and Suggestions

For better organization and clarity, here are suggested names for folders, files, and scripts used within this pipeline:

### Folder Names
- `data_scripts`: Stores data processing, preprocessing, and loading scripts.
- `data_type_classes`: Classes for managing specific data types (e.g., Radar, MoCap).
- `initial_processing_pipelines`: Scripts for initial raw data processing.
- `preprocessing_pipelines`: Scripts for windowing, labeling, and transformation.
- `dataloaders`: Scripts to load preprocessed data into models.

### File Names
- **Processing Script**: `RADAR_RDM_Generation_v1.ipynb`
- **Windowing and Labeling Script**: `Labeling_and_Windowing_Pipeline.py`
- **Model Architecture Script**: `RdmClassifier.py`
- **DataLoader Scripts**: `RdmDatasetLoader.py`, `StableRdmDatasetLoader.py`
- **Visualization Scripts**: `data_visualization.ipynb`, `model_evaluation.ipynb`

---

## Goal of the FU/FD Identification Pipeline

The goal of this pipeline is to reliably label radar frames based on specific balance-related events (FU, FD, stability). This setup provides a foundation for building and testing models that assess balance and fall risk in a non-invasive, privacy-preserving way. The pipeline can be adapted and extended for new sensor types or refined event identification in future versions.

This document serves as a guide to understand and replicate the steps of this specific pipeline. Additional pipelines may use similar steps but may differ in data processing, preprocessing, or model configurations.