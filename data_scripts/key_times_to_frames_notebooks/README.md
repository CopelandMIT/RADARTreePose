### README for `data_scripts/key_times_to_frames_notebooks` Directory

#### Purpose
The `key_times_to_frames_scripts` directory contains pipelines that streamline the workflow of generating important event markers, start and end times, and frame indexes for different movement events in the dataset.

#### Scope
The scripts and notebooks in this directory provide a unified way to generate and save start/end times, frame indexes, and other key timestamps, enabling efficient data loading and facilitating model training that depends on these markers.

#### Usage
- Run the key time and frame generation pipeline to produce datasets that contain event markers and frame indexes, which can then be used by data loaders.
- Output files from these pipelines should be saved to a designated location (e.g., `data/metadata`) for easy access by downstream processes, including feature extraction, data loading, and model training.

#### Directory Structure
- `key_time_to_frames_table_generation.ipynb`: Generates tables that link key times to frame indexes. This is a crucial step for creating start and end markers for each event or movement in the data.
  
#### Examples
- Run `key_time_to_frames_table_generation.ipynb` to generate a table of timestamps and frame indexes. This table can then be used by data loaders to segment the data according to specific movements or events.

#### Types of Code/Files
- Jupyter notebooks that automate complex workflows involving time and frame alignment.
- Outputs are often metadata tables or timestamp files, providing structured information on event markers across different data types.

#### Interfaces With
- **data/dataloaders/**: The generated metadata tables are essential for data loading, helping to segment data for specific events or movements.
- **feature_engineering/**: Feature extraction scripts can use these markers to focus on specific segments of data.

#### Gitignore
- `.ipynb_checkpoints/` should be included in `.gitignore`.
- The pipeline outputs should be stored in the designated data directory (e.g., `data/metadata`) for easy access, rather than cluttering this directory.
