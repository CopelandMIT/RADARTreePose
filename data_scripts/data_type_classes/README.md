# Data Type Classes

This folder contains classes responsible for data processing, transformation, and feature extraction specific to various data types used in the project. These classes encapsulate functions to handle data from different sensors and sources, including force plates, motion capture systems, radars, and TSV-based data (e.g., knee angles).

## Folder Structure

- `data_type_classes/`
  - `archived/`: Contains deprecated or experimental classes that are no longer actively used.
  - `FPDataCapture.py`: Handles data processing and feature extraction for Force Plate (FP) data.
  - `MOCAPDataCapture.py`: Manages the processing of Motion Capture (MOCAP) data, including calculating joint angles and other motion features.
  - `RADARDataCapture.py`: Processes radar data, including data extraction and transformation specific to radar sensors.
  - `tsv_processor_knee_angles.py`: Processes TSV data for knee angles, transforming raw sensor data into structured output for further analysis.

## Class Descriptions

- **FPDataCapture**
  - **Description**: This class processes data from Force Plates, a device used to measure forces generated during physical activities.
  - **Functionality**: 
    - Reads raw force plate data.
    - Extracts and calculates relevant features such as force vectors, center of pressure, and balance metrics.

- **MOCAPDataCapture**
  - **Description**: This class is used for processing motion capture data, which includes tracking the movement of body joints.
  - **Functionality**: 
    - Calculates joint angles and velocities.
    - Extracts key motion features for biomechanical analysis.

- **RADARDataCapture**
  - **Description**: Manages radar sensor data processing, converting raw radar captures into structured data suitable for analysis.
  - **Functionality**:
    - Extracts radar features such as distance, speed, and angular data.
    - Converts radar signals into useful measurements for human motion analysis.

- **tsv_processor_knee_angles**
  - **Description**: Processes TSV files containing knee angle data, primarily for gait or postural studies.
  - **Functionality**:
    - Extracts 3D joint angles using coordinates from the TSV files.
    - Calculates angles for right and left knees and other relevant joints.
    - Organizes and outputs the processed data in a structured format.

## Usage

Each class in this folder is designed to be used within data processing pipelines to handle specific sensor data types. Pipelines in the `preprocessing` folder call on these classes to preprocess and transform data for further analysis.

For example:
```python
from data_type_classes.FPDataCapture import FPDataCapture

# Initialize Force Plate Data Capture
fp_data_processor = FPDataCapture(data_path="path/to/fp_data")
processed_data = fp_data_processor.process_data()
```

## Notes

- Ensure that each class is used with compatible data formats, as each class expects data from a specific sensor or source.
- Archived classes are located in the `archived/` folder and are not recommended for active use unless specific functionalities are required.
