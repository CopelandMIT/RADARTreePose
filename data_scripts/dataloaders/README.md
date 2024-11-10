# Dataloaders

This folder contains data loading classes designed for different datasets used in this project. Each class here extends PyTorch’s `Dataset` class to manage and preprocess data for model training and evaluation efficiently. These loaders handle specific dataset formats and structures, facilitating batch loading, shuffling, and other utilities.

## Folder Structure

- `dataloaders/`
  - `ByFrameStableRdmDataset.py`: A dataloader for loading stabilized RDM (Radar Data Matrix) data frame by frame.
  - `RdmDataset.py`: A dataloader for RDM data, structured to handle segmented windows of radar captures.
  - `RdmFullCapture.py`: A dataloader for loading complete RDM captures without segmentation, used for full data analysis or model inference.
  - `StableRdmDataset.py`: A dataloader that handles stable RDM data for scenarios requiring reduced variability.

## Dataloader Descriptions

### 1. ByFrameStableRdmDataset
- **Description**: This dataloader handles RDM data in a stabilized frame-by-frame manner, where each frame is loaded individually. Useful for tasks that require precise temporal control over each frame.
- **Functionality**:
  - Loads data one frame at a time.
  - Supports shuffling and batching for frame-level analysis.
  - Prepares the data to ensure stability in temporal sequence.

### 2. RdmDataset
- **Description**: Manages RDM data by dividing it into windows, which helps in segmenting radar captures into chunks for time series analysis or model training.
- **Functionality**:
  - Creates windows of a specified size with optional overlap.
  - Labels data segments based on events or actions captured within each window.
  - Suited for training sequence-based models with radar data.

### 3. RdmFullCapture
- **Description**: Loads entire RDM captures without breaking them into windows, used for full-sequence analysis or model evaluation across the entire capture.
- **Functionality**:
  - Loads complete radar data captures for tasks that do not require segmentation.
  - Maintains original sequence structure for temporal analysis.
  - Ideal for full capture inference models or exploratory data analysis.

### 4. StableRdmDataset
- **Description**: Handles RDM data with a focus on stability, useful for tasks where data consistency and low variability across frames are critical.
- **Functionality**:
  - Loads stable data, minimizing variations that may interfere with analysis.
  - Designed for scenarios where consistent frame quality is essential.

## Usage

Each dataloader can be used with PyTorch’s `DataLoader` class for efficient batching and shuffling during model training and evaluation. Example:

```python
from torch.utils.data import DataLoader
from dataloaders.RdmDataset import RdmDataset

# Initialize the RDM dataset with specified parameters
rdm_dataset = RdmDataset(root_dir="path/to/data", event_csv="path/to/events.csv", included_folders=["folder1", "folder2"])

# Create a PyTorch DataLoader for batching and shuffling
data_loader = DataLoader(rdm_dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch in data_loader:
    data, labels, lengths, metadata = batch
    # Model training or evaluation code
```

## Notes

- These dataloaders are specific to different radar data formats and configurations. Ensure you are using the appropriate dataloader for your dataset.
- Custom parameters (such as window size, overlap, and stability settings) are available in each class to tailor data loading to specific use cases.
- For multi-sensor data collection setups, ensure data format consistency across different sensors before loading.
