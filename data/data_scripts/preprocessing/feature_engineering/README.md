### README for `feature_engineering` Directory

#### Purpose
The `feature_engineering` directory contains Jupyter notebooks focused on the transformation and extraction of features from raw data. These features are designed to make the data more meaningful and useful for downstream machine learning models and analysis.

#### Scope
The notebooks in this directory handle specific transformations and engineering tasks for different data types (e.g., force plate, motion capture, radar) to create new variables or refine existing data structures.

#### Usage
- Each notebook is run independently, focusing on feature extraction for a specific data type (e.g., force plate, radar, mocap).
- Outputs from these notebooks are typically saved to an intermediate or preprocessed data directory for further use in model training or analysis pipelines.

#### Directory Structure
- `main_FP_Preprocessing.ipynb`: Processes and extracts features from force plate data.
- `main_MOCAP_Preprocessing.ipynb`: Processes and extracts features from motion capture (mocap) data.
- `main_RADAR_Preprocessing.ipynb`: Processes and extracts features from radar data.
- `main_Time_and_Frame_preprocessing.ipynb`: Aligns and refines data frames by processing timing information.
- `main.ipynb`: A general-purpose notebook, possibly used for running or testing combinations of feature extraction steps.

#### Examples
- Use `main_FP_Preprocessing.ipynb` to process force plate data, extracting relevant metrics or transformations.
- Use `main_RADAR_Preprocessing.ipynb` for radar data, transforming it to produce features like velocity, intensity, etc.

#### Types of Code/Files
- Jupyter notebooks containing code for data transformation, feature extraction, and possibly visualization of features.
- Output files that may be saved as processed data or extracted features in a structured format (e.g., `.csv` or `.npy`).

#### Interfaces With
- **data/dataloaders/**: Dataloaders can access outputs generated here, which may include engineered features used as inputs for model training.
- **pipelines/**: Some feature extraction steps might be included in end-to-end pipelines, especially if certain features are key to training.

#### Gitignore
- `.ipynb_checkpoints/` should be included in `.gitignore`.
- Final output files or intermediate files generated from this directory may be stored in a separate data directory to keep the structure clean.

