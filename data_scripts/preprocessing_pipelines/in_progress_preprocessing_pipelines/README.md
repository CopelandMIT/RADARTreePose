### README for `data_scripts/in_progress_preprocessing_pipelines` Directory

#### Purpose
The `in_progress_preprocessing_pipelines` directory contains feature engineering scripts and notebooks that perform crucial preprocessing steps on processed data, preparing it for use in data loaders. These scripts are responsible for tasks like windowing, normalizing, reducing data size, and tagging, which are necessary to make the data compatible with model training workflows.

#### Scope
The scripts and notebooks here are part of an evolving feature engineering process, focusing on refining data for efficient loading and model input preparation. These pipelines are "in-progress," meaning they are being actively developed to improve data quality and accessibility for downstream tasks.

#### Usage
- Run the feature engineering notebooks to prepare data segments by windowing, normalizing, and tagging.
- Each script produces preprocessed outputs that are intended for use in data loading scripts, making it easier to create structured inputs for machine learning models.

#### Directory Structure
- **Windowing**: Segments the data into fixed-size windows, facilitating batch processing and sequence-based model training.
- **Normalization**: Standardizes data values, reducing variance and ensuring consistency across different data samples.
- **Tagging**: Adds labels or tags to data windows based on movement events or other markers, which is essential for supervised learning.

#### Examples
- A notebook here might apply normalization to a radar dataset, creating windows with standardized values, followed by tagging frames based on motion or event markers.
- Another script could create overlapping windows with predefined sizes, cutting down large datasets into manageable pieces for the data loaders.

#### Types of Code/Files
- Jupyter notebooks and Python scripts dedicated to preparing data for model-ready input.
- These scripts generate preprocessed data outputs, which are saved for later use in data loaders.

#### Interfaces With
- **data/dataloaders/**: The preprocessed data produced here is directly consumed by data loaders, which structure it for model training.
- **processed/**: Relies on outputs from the `processed` data directory, taking structured data and applying feature engineering steps to finalize it for training use.

#### Gitignore
- `.ipynb_checkpoints/` should be included in `.gitignore`.
- Pipeline outputs should be saved in structured directories (e.g., `data/preprocessed/`) to keep this directory clean and focused on active scripts. 

