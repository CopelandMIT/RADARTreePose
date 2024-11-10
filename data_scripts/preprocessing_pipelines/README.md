### README for `data_scripts/preprocessing`

#### Overview
The `preprocessing` directory contains scripts for **feature engineering**—the process of transforming raw or minimally processed data into structured features for machine learning models. Feature engineering is a crucial step in the data pipeline, as it allows us to extract meaningful patterns and representations from the data that can improve model performance.

The directory is organized into three main subdirectories:

1. **archived_FE_pipelines**: Contains legacy feature engineering pipelines that are no longer in active use. These scripts may include experimental or outdated feature extraction techniques.
2. **in_progress_FE_pipelines**: Contains feature engineering pipelines that are actively being developed and tested. These scripts represent the current iteration of feature extraction and transformation methods.
3. **finalized_FE_pipelines**: Contains feature engineering pipelines that have been validated and finalized. These pipelines produce the polished data used directly in model training.

---

#### Directory Structure

- **`archived_FE_pipelines/`**  
  Contains older or experimental feature engineering scripts. These may include initial versions of transformations or feature extraction approaches that are no longer in active use.

- **`in_progress_FE_pipelines/`**  
  This is the active development area for feature engineering. Here, experimental scripts are tested and adjusted. These pipelines apply transformations, windowing strategies, and feature extraction methods to explore potential improvements in data preparation.

- **`finalized_FE_pipelines/`**  
  This folder contains feature engineering pipelines that have been validated and are considered complete. These scripts output finalized features in a standardized format, ready for model training workflows without further modification.

---

#### Inputs and Outputs

Each feature engineering script follows a structured data flow, where it receives processed input data, applies transformations, and outputs structured feature data for modeling. 

**Inputs**  
- **Source**: Data files come from `data/data_files/intermediate/`, `data/data_files/preprocessed/`, or `data/data_files/metadata/` directories. Inputs are often minimally processed data or time-stamped metadata defining events (e.g., frame times for specific actions).
- **Format**: Typically in CSV, TSV, or NPY formats, these files contain structured sensor data or metadata aligned with events of interest.

**Outputs**  
- **Destination**: Engineered feature files are output to either `data/data_files/preprocessed/` (for finalized features) or `data/data_files/intermediate/` (for data requiring further transformations).
- **Format**: Outputs are often in CSV or NPY formats. These files contain structured windows of data, extracted features, and labels associated with each feature window, formatted to be compatible with data loaders in `data_scripts/dataloaders/`.

---

#### Workflow Summary

1. **Archived Pipelines**:  
   - **Input**: Processed data from `data/data_files/raw` or `data/data_files/intermediate`
   - **Output**: Intermediate data files saved in various formats, often kept for reference only.
   
2. **In-Progress Pipelines**:  
   - **Input**: Data from `data/data_files/raw`, `data/data_files/intermediate`, or `data/data_files/metadata`
   - **Output**: Experimentally engineered features saved in `data/data_files/intermediate` for further validation and adjustments.
   
3. **Finalized Pipelines**:  
   - **Input**: Cleaned or preprocessed data from `data/data_files/intermediate` and `metadata`
   - **Output**: Final, feature-engineered data saved in `data/data_files/preprocessed`, ready for model training.

---

#### Key Files
- **`key_time_to_frames_table_generation.ipynb`**: Generates a table that aligns key times to frame numbers based on the raw data. This table is crucial for synchronizing events across different data sources, such as radar and MOCAP.

Each file applies specific feature engineering transformations to its respective dataset, ensuring the data is ready for subsequent modeling steps.