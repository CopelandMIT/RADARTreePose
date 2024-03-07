import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import uniform_filter1d

class RdmFullCapture(Dataset):
    def __init__(self, root_dir, event_csv, actuator_csv, included_folders, window_size=100):
        self.data = []
        self.labels = []  # This will store labels for each window
        self.metadata = []  # Store metadata for each window
        self.window_size = window_size
        
        # Load event labels and actuator frames
        self.event_labels_df = pd.read_csv(event_csv)
        self.actuator_df = pd.read_csv(actuator_csv)
        
        # Iterate only over included folders
        for folder_name in included_folders:
            folder_path = os.path.join(root_dir, folder_name)
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('.npy'):
                    filepath = os.path.join(folder_path, file)
                    radar_capture = "_".join(file.split('_')[:-1])  # Extract radar capture name
                    
                    # Ensure radar_capture matches one of the entries in the actuator CSV
                    if self.actuator_df['RADAR_capture'].str.contains(radar_capture).any():
                        rdm_data = np.load(filepath)
                        rdm_data = torch.from_numpy(rdm_data).float()  # Convert numpy array to PyTorch tensor of type float

                        actuator_info = self.actuator_df[self.actuator_df['RADAR_capture'] == radar_capture].iloc[0]
                        actuator_start_frame, actuator_end_frame = actuator_info['RADAR_Start_Frame'] + 150, actuator_info['RADAR_End_Frame']
                        MOCAP_Start_Time = actuator_info['RADAR_Start_Frame']
                        MOCAP_End_Time = actuator_info['MOCAP_End_Time']
                        seconds_per_frame = actuator_info['Seconds_per_Frame']
                        
                        # Create windows, label them, and add metadata
                        labels, goup_ranges, down_ranges = self.label_frames(rdm_data, radar_capture)
                        metadata = {
                            'frame_range': (actuator_start_frame, actuator_end_frame),
                            'MOCAP_time_range' : (MOCAP_Start_Time, MOCAP_Start_Time),
                            'seconds_per_frame': seconds_per_frame,
                            'RADAR_capture': radar_capture,
                            'GOUP_ranges': goup_ranges,
                            'DOWN_ranges': down_ranges,
                            }
                         
                        self.metadata.append(metadata)
                        self.labels.append(labels)

    def label_frames(self, radar_capture):
        num_frames = 1000  # Assuming a fixed size, adjust as necessary
        labels = np.full(num_frames, 2)  # Default to 3 (neither)
        capture_events = self.event_labels_df[self.event_labels_df['RADAR_capture'] == radar_capture]
        
        goup_ranges = []
        down_ranges = []
        
        for _, event in capture_events.iterrows():
            if not pd.isna(event['Start_Frame']) and not pd.isna(event['frame_stable']):
                start = int(event['Start_Frame'])
                end = int(event['frame_stable'])
                labels[start:end] = 0  # GOUP
                goup_ranges.append((start, end))
            if not pd.isna(event['frame_break']) and not pd.isna(event['End_Frame']):
                start = int(event['frame_break'])
                end = int(event['End_Frame'])
                labels[start:end] = 1  # DOWN
                down_ranges.append((start, end))
        
        return labels, goup_ranges, down_ranges
    
    
    def _create_windows(self):
        """
        Creates overlapping windows from the full-length capture data.
        """
        start_points = range(0, len(self.capture_data) - self.window_size + 1, self.window_size - self.overlap)
        return [self.capture_data[start:start + self.window_size] for start in start_points]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float)

    def predict_full_capture(self):
        """
        Evaluates the full-length capture by predicting each window,
        collating the predictions, and then smoothing.
        """
        self.model.eval()
        loader = DataLoader(self, batch_size=1, shuffle=False)
        predictions = []

        with torch.no_grad():
            for window in loader:
                output = self.model(window)
                predictions.append(torch.softmax(output, dim=1).numpy())

        # Collate predictions by averaging overlapping areas
        collated_predictions = self._collate_predictions(predictions)

        # Smooth the collated predictions
        smoothed_predictions = uniform_filter1d(collated_predictions, size=5, axis=0)

        return smoothed_predictions

    def _collate_predictions(self, predictions):
        """
        Averages predictions from overlapping windows.
        """
        full_length = len(self.capture_data)
        collated = np.zeros((full_length, predictions[0].shape[1]))

        count = np.zeros(full_length)
        start = 0
        for pred in predictions:
            end = start + self.window_size
            collated[start:end] += pred.squeeze()
            count[start:end] += 1
            start += self.window_size - self.overlap

        # Avoid division by zero
        count[count == 0] = 1
        collated /= count[:, None]

        return collated

# Example usage:
# Assuming 'model' is your trained model and 'capture_data' is a numpy array of your full-length capture
# full_capture = RdmFullCapture(model, capture_data)
# smoothed_predictions = full_capture.predict_full_capture()
