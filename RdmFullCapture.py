import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch import optim
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

class RdmFullCapture(Dataset):
    def __init__(self, root_dir, event_csv, included_folders, window_size=100):
        self.data = []
        self.full_capture_labels = []
        self.labels = []  # This will store labels for each capture, list of lists
        self.all_metadata = []  # Store metadata for each capture, list of dictionaries
        self.window_size = window_size
        self.current_index = 0
        
        # Load event labels and actuator frames
        self.event_labels_df = pd.read_csv(event_csv)
        
        # Iterate only over included folders
        for folder_name in included_folders:
            folder_path = os.path.join(root_dir, folder_name)
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('.npy'):
                    filepath = os.path.join(folder_path, file)
                    radar_capture = "_".join(file.split('_')[:-1])  # Extract radar capture name
                    
                    channel_number = filepath.split(".")[-2].split("channel")[-1]
                    
                    # Ensure radar_capture matches one of the entries in the actuator CSV
                    if self.event_labels_df['RADAR_capture'].str.contains(radar_capture).any():
                        rdm_data = np.load(filepath)
                        rdm_data = torch.from_numpy(rdm_data).float()  # Convert numpy array to PyTorch tensor of type float

                        actuator_info = self.event_labels_df[self.event_labels_df['RADAR_capture'] == radar_capture].iloc[0]
                        actuator_start_frame, actuator_end_frame = actuator_info['RADAR_Start_Frame'], actuator_info['RADAR_End_Frame']
                        MOCAP_Start_Time = actuator_info['RADAR_Start_Frame']
                        MOCAP_End_Time = actuator_info['MOCAP_End_Time']
                        seconds_per_frame = actuator_info['Seconds_per_Frame']
                        
                        # Create windows, label them, and add metadata
                        labels, goup_ranges, down_ranges = self.label_frames(radar_capture)
                        metadata = {
                            'channel_number': channel_number,
                            'frame_range': (actuator_start_frame, actuator_end_frame),
                            'MOCAP_time_range' : (MOCAP_Start_Time, MOCAP_End_Time),
                            'seconds_per_frame': seconds_per_frame,
                            'RADAR_capture': radar_capture,
                            'GOUP_ranges': goup_ranges,
                            'DOWN_ranges': down_ranges,
                            'window_start_frame': 0,
                            'window_end_frame': 0
                            }
                         
                        self.all_metadata.append(metadata)
                        self.labels.append(labels)
                        self.data.append(rdm_data)

    def label_frames(self, radar_capture):
        num_frames = 1000  # Assuming a fixed size, adjust as necessary
        labels = np.full(num_frames, 2)  # Default to 2 (neither)
        capture_events = self.event_labels_df[self.event_labels_df['RADAR_capture'] == radar_capture]
        
        goup_ranges = []
        down_ranges = []
        
        for _, event in capture_events.iterrows():
            if not pd.isna(event['frame_foot_up']) and not pd.isna(event['frame_stable']):
                start = int(event['frame_foot_up'])+1
                end = int(event['frame_stable'])+1
                labels[start:end] = 0  # GOUP
                goup_ranges.append((start, end))
            if not pd.isna(event['frame_break']) and not pd.isna(event['frame_end']):
                start = int(event['frame_break'])+1
                end = int(event['frame_end'])+1
                labels[start:end] = 1  # DOWN
                down_ranges.append((start, end))
        
        self.full_capture_labels.append(labels)
        
        return labels, goup_ranges, down_ranges
    
    def create_windows_for_capture(self, index, window_size=100, overlap=0):
        """
        Create windows for a specific capture given by index.

        Parameters:
        - index: Index of the capture to process.
        - window_size: The size of each window.
        - overlap: The overlap between consecutive windows.

        Returns:
        - A tuple containing windows, labels for each window, lengths of each window, and metadata.
        """
                
        if index >= len(self.data):
                raise ValueError("Index out of range.")
            
        capture_data, capture_labels, _, capture_metadata = self[index]
        actuator_start_frame = capture_metadata['frame_range'][0]
        actuator_end_frame = capture_metadata['frame_range'][1]
        
        windows_ranges = []
        capture_windows_data = []
        windows_labels_data = []  # Collect labels data
        windows_lengths_tensor = []
        
        num_windows = 1 + (actuator_end_frame - actuator_start_frame - window_size) // (window_size - overlap)

        print(f"Creating windows {num_windows} windows for Radar Capture: {capture_metadata['RADAR_capture']}, channel: {capture_metadata['channel_number']}")

        for w in range(num_windows):
            start = w * (window_size - overlap) + actuator_start_frame  # Adjust for correct overlap
            end = start + window_size
            window_range_dict = {'window_start_frame': start, 'window_end_frame': min(end, actuator_end_frame)}

            if end > actuator_end_frame:
                padding_length = end - actuator_end_frame
                window_data = torch.cat((capture_data[start:actuator_end_frame], torch.zeros(padding_length, *capture_data.shape[1:])), dim=0)
                window_labels = np.pad(capture_labels[start:actuator_end_frame], (0, padding_length), 'constant', constant_values=-1)
            else:
                window_data = torch.tensor(capture_data)[start:end]
                window_labels = capture_labels[start:end]

            capture_windows_data.append(window_data.unsqueeze(0))
            windows_labels_data.append(torch.tensor(window_labels).unsqueeze(0))  # Convert labels to tensor here
            windows_lengths_tensor.append(min(end, actuator_end_frame) - start)
            windows_ranges.append(window_range_dict)

        # Concatenate all windows and labels data after loop
        capture_windows_tensor = torch.cat(capture_windows_data, dim=0)
        windows_labels_tensor = torch.cat(windows_labels_data, dim=0)

        return capture_windows_tensor, windows_labels_tensor, torch.tensor(windows_lengths_tensor, dtype=torch.long), capture_metadata, windows_ranges

    # def predict_on_windows(self, model, windows_tensor, lengths):
    #         model.eval()
    #         predictions = []
            
    #         # Ensure lengths is a tensor
    #         lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    #         windows_tensor = torch.tensor(windows_tensor, dtype=torch.float)

    #         with torch.no_grad():
    #             # Assuming model's forward method signature is like forward(self, x, lengths)
    #             outputs = model(windows_tensor, lengths_tensor)  # Now passing tensors
                
    #             # Correctly flatten output for subsequent operations
    #             outputs_flat = outputs.view(-1, 3)  # Assuming there are 3 classes
                
    #             # Apply softmax to get probabilities
    #             predictions = torch.softmax(outputs_flat, dim=1).numpy()
    #             print(predictions)

    #         return predictions
        
    def predict_on_windows(self, model, windows_tensor, lengths):
            model.eval()
            predictions = []
            
            # Ensure lengths is a tensor
            lengths_tensor = torch.tensor(lengths, dtype=torch.long)
            windows_tensor = torch.tensor(windows_tensor, dtype=torch.float)

            with torch.no_grad():
                # Assuming model's forward method signature is like forward(self, x, lengths)
                outputs = model(windows_tensor, lengths_tensor)  # Now passing tensors
                
                # Correctly flatten output for subsequent operations
                outputs_flat = outputs.view(-1, 3)  # Assuming there are 3 classes
                
                # Apply softmax to get probabilities
                predictions = torch.softmax(outputs_flat, dim=1).numpy()

            return predictions
        
    def aggregate_predictions_sliding_windows(self, predictions, windows_ranges):
        # Assuming predictions is (1100, 3), for 11 windows each covering 100 frames, with 50 frames overlap
        full_length = max(w_range['window_end_frame'] for w_range in windows_ranges) + 1
        num_classes = predictions.shape[1]
        
        # Initialize an array for aggregated predictions across the full sequence
        aggregated_predictions = np.full((full_length, num_classes), -np.inf)
        
        window_length = 100  # Each window covers 100 frames
        slide_step = 0  # Windows slide by 50 frames

        # Process predictions for each window
        for window_idx, window_range in enumerate(windows_ranges):
            start_frame = window_range['window_start_frame']
            end_frame = window_range['window_end_frame']
            window_pred_start_idx = window_idx * window_length - (window_idx * slide_step if window_idx > 0 else 0)

            for frame_offset in range(window_length):
                frame_idx = start_frame + frame_offset
                if frame_idx >= full_length:
                    break  # Beyond the sequence range
                if frame_idx >= end_frame:
                    continue  # Skip if beyond the window's end frame
                
                prediction_idx = window_pred_start_idx + frame_offset
                frame_prediction = predictions[prediction_idx]
                
                # Update aggregated predictions with maximum likelihood
                aggregated_predictions[frame_idx] = np.maximum(aggregated_predictions[frame_idx], frame_prediction)
        
        # Determine class predictions by selecting the class with the highest likelihood for each frame
        class_predictions = np.argmax(aggregated_predictions, axis=1)
        
        return class_predictions
    
    # def aggregate_predictions_max_confidence(self, predictions, windows_ranges):
    #     # Determine the full length of the sequence based on the maximum end frame among windows
    #     full_length = max(window_range['window_end_frame'] for window_range in windows_ranges) + 1  # +1 to account for zero-indexing
        
    #     # Initialize arrays to hold summed probabilities and the sum of likelihoods for normalization
    #     num_classes = predictions.shape[2]
    #     sum_probabilities = np.zeros((full_length, num_classes), dtype=np.float64)
    #     sum_likelihoods = np.zeros(full_length, dtype=np.float64)
        
    #     for prediction, window_range in zip(predictions, windows_ranges):
    #         start_frame = window_range['window_start_frame']
    #         end_frame = window_range['window_end_frame']
            
    #         for frame_idx in range(start_frame, end_frame):
    #             # Calculate relative index within the prediction array
    #             relative_idx = frame_idx - start_frame
                
    #             # Sum the probabilities and likelihoods
    #             likelihoods = prediction[relative_idx]
    #             sum_probabilities[frame_idx] += likelihoods
    #             sum_likelihoods[frame_idx] += np.sum(likelihoods)
        
    #     print(sum_probabilities)
    #     print(sum_probabilities.shape)
    #     # Normalize summed probabilities by the sum of likelihoods to get the average
    #     avg_probabilities = np.divide(sum_probabilities, sum_likelihoods[:, None], out=np.zeros_like(sum_probabilities), where=sum_likelihoods[:, None] != 0)
        
    #     # Determine the class with the highest average probability
    #     class_predictions = np.argmax(avg_probabilities, axis=1)
        
    #     return class_predictions

    # def aggregate_predictions_max_likelihood(self, predictions, windows_ranges):
    #     # Determine the full length of the sequence based on the maximum end frame among windows
    #     full_length = max(range_['window_end_frame'] for range_ in windows_ranges) + 1  # +1 to account for zero-indexing
    #     num_classes = predictions.shape[2]
        
    #     # Initialize an array to hold the class index with the maximum likelihood for each frame
    #     class_predictions = np.full(full_length, 2) 
        
    #     # Initialize an array to track the maximum likelihood for comparison
    #     max_likelihoods = np.full((full_length, num_classes), -np.inf)  # Use -inf to ensure any prediction is greater
        
    #     for prediction, window_range in zip(predictions, windows_ranges):
    #         print('Prediction shape is')
    #         print(predictions[0].shape)

    #         start_frame = window_range['window_start_frame']
    #         end_frame = window_range['window_end_frame']
            
    #         for frame_idx in range(start_frame, end_frame):
    #             # Calculate relative index within the prediction array
    #             relative_idx = frame_idx - start_frame
                
    #             # Update max likelihood and class predictions if current prediction has higher likelihood
    #             for class_idx in range(num_classes):
    #                 if prediction[relative_idx, class_idx] > max_likelihoods[frame_idx, class_idx]:
    #                     max_likelihoods[frame_idx, class_idx] = prediction[relative_idx, class_idx]
    #                     class_predictions[frame_idx] = class_idx
        
    #     return class_predictions

    def calculate_full_length(self, metadata):
        # This method should calculate the full length for aggregated predictions based on the metadata.
        # It could be as simple as the difference between the end and start frame of the capture, plus some padding if needed.
        return metadata['frame_range'][1] - metadata['frame_range'][0] + 1


    def plot_predictions(self, index, smoothed_predictions, capture_name, metadata):
        """
        Plot smoothed predictions against the true labels.
        """
        # Assuming metadata and labels for the index are properly set
        metadata = self.all_metadata[index]
        labels = self.labels[index]

        
        plt.figure(figsize=(20, 5))
        plt.plot(labels, label='True Labels', color='blue')
        plt.plot(smoothed_predictions, label='Predicted', color='red', linestyle='--')
        plt.title(f'Predictions vs. True Labels for {capture_name}')
        plt.xlim([metadata['frame_range'][0] + 20, metadata['frame_range'][1] + 20])
        plt.xlabel('Frame')
        plt.ylabel('Probability')
        plt.legend()
        plt.show()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Assuming self.data is your dataset and self.labels are your labels
        data = self.data[index]
        label = self.labels[index]
        length = len(data)  # Or however you calculate the length of your sequence
        
        # Assuming metadata is stored as a dictionary in self.metadata[index]
        metadata = self.all_metadata[index]

        # metadata could contain: 
        # - 'frame_range': tuple indicating the range of frames, e.g., (350, 450)
        # - 'RADAR_capture': the identifier for the RADAR capture, e.g., '01_MNTRL_RR_V1'
        # - 'GOUP_transition': boolean indicating if there is a GOUP transition within the window
        # - 'DOWN_transition': boolean indicating if there is a DOWN transition within the window

        return data, label, length, metadata
    
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

    @staticmethod
    def collate_fn(batch):
        # Unzip the batch to separate sequences, labels, lengths, and metadata
        sequences, labels, lengths, metadata = zip(*batch)

        # Ensure sequences are tensors and pad them to have the same length
        sequences_padded = pad_sequence([torch.tensor(seq, dtype=torch.float) for seq in sequences], batch_first=True)
        
        # Similarly, pad labels if they are of variable lengths
        labels_padded = pad_sequence([torch.tensor(label, dtype=torch.long) for label in labels], batch_first=True, padding_value=-1)  # Use -1 as an ignore index if labels are of variable lengths

        # Convert lengths to a tensor
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)

        # Since metadata is a list of dictionaries, it doesn't need padding, but you should ensure it's structured correctly for further processing. We'll keep it as a list of dictionaries.
        # No processing needed for metadata as it's not used in tensor operations, but make sure it's correctly structured for any further non-tensor operations

        return sequences_padded, labels_padded, lengths_tensor, metadata
    
    def plot_labels_and_ranges(self, index):
        """
        Plots the labels and frame ranges for a single capture.
        
        Parameters:
        - index: Index of the capture to plot in the dataset.
        """
        if index >= len(self.data):
            print("Index out of range.")
            return
        
        # Assuming metadata and labels for the index are properly set
        metadata = self.all_metadata[index]
        labels = self.labels[index]

        plt.figure(figsize=(20, 5))

        # Plot labels
        plt.plot(labels, label='Labels')

        # Highlight GOUP and DOWN ranges
        # for start, end in metadata['GOUP_ranges']:
        #     plt.axvspan(start-1, end-1, color='green', alpha=0.3, label='GOUP range')

        # for start, end in metadata['DOWN_ranges']:
        #     plt.axvspan(start-1, end-1, color='red', alpha=0.3, label='DOWN range')

        plt.title(f'Labels and Frame Ranges for Capture: {metadata["RADAR_capture"]}')
        plt.xlabel('Frame Index')
        plt.ylabel('Label')
        plt.yticks([0, 1, 2], ['GOUP', 'DOWN', 'NEITHER'])
        plt.xlim([metadata['frame_range'][0], metadata['frame_range'][1]])
        plt.legend()

        plt.show()
        
    # def create_windows(self, window_size=100):
    #     all_windows = []
    #     all_labels = []
    #     all_lengths = []

    #     # Iterate through each full capture data in the dataset
    #     for i in range(len(self)):
    #         capture_data, capture_labels, _, capture_metadata = self[i]
    #         actuator_start_frame = capture_metadata['frame_range'][0]
    #         actuator_end_frame = capture_metadata['frame_range'][1]
            
    #         # Calculate the number of windows
    #         num_windows = ((actuator_end_frame - actuator_start_frame) - window_size) // (window_size // 2) + 1

    #         # Create windows
    #         for w in range(num_windows):
    #             start = w * (window_size // 2) + actuator_start_frame
    #             end = start + window_size
    #             capture_metadata['window_start_frame'] = start
    #             capture_metadata['window_end_frame'] = end

    #             if end > actuator_end_frame:
    #                 capture_metadata['window_end_frame'] = actuator_end_frame
    #                 # Pad the data and labels for windows that exceed the actuator_end_frame
    #                 padding_length = end - actuator_end_frame
    #                 window_data = torch.cat((capture_data[start:actuator_end_frame], torch.zeros(padding_length, *capture_data.shape[1:])))
    #                 window_labels = np.concatenate((capture_labels[start:actuator_end_frame], np.full((padding_length,), -1)))
    #             else:
    #                 window_data = capture_data[start:end]
    #                 window_labels = capture_labels[start:end]

    #             # Convert window_labels to a tensor and add batch dimension
    #             window_labels_tensor = torch.tensor(window_labels).unsqueeze(0)
    #             all_windows.append(window_data.unsqueeze(0))
    #             all_labels.append(window_labels_tensor)
    #             all_lengths.append(window_data.shape[0])
    #             self.all_metadata.append(capture_metadata)

    #     # Combine all windows into a single batch tensor
    #     windows_tensor = torch.cat(all_windows, dim=0)
    #     labels_tensor = torch.cat(all_labels, dim=0)
    #     lengths_tensor = torch.tensor(all_lengths)

    #     return windows_tensor, labels_tensor, lengths_tensor, self.all_metadata
    

# Example usage:
# Assuming 'model' is your trained model and 'capture_data' is a numpy array of your full-length capture
# full_capture = RdmFullCapture(model, capture_data)
# smoothed_predictions = full_capture.predict_full_capture()
