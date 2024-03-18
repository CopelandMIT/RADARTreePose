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
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
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
    
    def create_windows_for_capture(self, index, overlap):
        """
        Create windows for a specific capture given by index.

        Parameters:
        - index: Index of the capture to process.
        - window_size: The size of each window.
        - overlap: The overlap between consecutive windows.

        Returns:
        - A tuple containing windows, labels for each window, lengths of each window, and metadata.
        """
        self.overlap = overlap
        self.current_index = index
                
        if index >= len(self.data):
                raise ValueError("Index out of range.")
            
        capture_data, capture_labels, _, capture_metadata = self[index]
        actuator_start_frame = capture_metadata['frame_range'][0]
        actuator_end_frame = capture_metadata['frame_range'][1]
        
        windows_ranges = []
        capture_windows_data = []
        windows_labels_data = []  # Collect labels data
        windows_lengths_tensor = []
        
        num_windows = 1 + (actuator_end_frame - actuator_start_frame - self.window_size) // (self.window_size - overlap)

        print(f"Creating windows {num_windows} windows for Radar Capture: {capture_metadata['RADAR_capture']}, channel: {capture_metadata['channel_number']}")

        for w in range(num_windows):
            start = w * (self.window_size - overlap) + actuator_start_frame  # Adjust for correct sliding window
            end = start + self.window_size
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
        
    def aggregate_predictions_sliding_windows(self, predictions, windows_ranges, smoothing_window_size=5):
        full_length = max(w_range['window_end_frame'] for w_range in windows_ranges) + 1
        num_classes = predictions.shape[1]
        
        # Initialize an array for the aggregated maximum likelihoods
        aggregated_predictions = np.zeros((full_length, num_classes))
        coverage_count = np.zeros(full_length)  # Track how many times each frame is covered by windows
        
        current_pred_idx = 0  # Track the current index within the flat predictions array
        
        for window_range in windows_ranges:
            start_frame = window_range['window_start_frame']
            end_frame = min(window_range['window_end_frame'], full_length)
            
            for frame_idx in range(start_frame, end_frame):
                # Extract the prediction for the current frame
                frame_prediction = predictions[current_pred_idx]
                current_pred_idx += 1  # Move to the next prediction
                
                # Aggregate by taking the maximum likelihood across overlapping predictions
                aggregated_predictions[frame_idx] = np.maximum(aggregated_predictions[frame_idx], frame_prediction)
                coverage_count[frame_idx] += 1
        
        # Handle frames not covered by any window (if any) to avoid division by zero
        coverage_count[coverage_count == 0] = 1
        
        # Normalize aggregated predictions by the number of windows covering each frame
        aggregated_predictions /= coverage_count[:, None]

        # Let's say 'predictions' is your numpy array with shape (806, 3)
        smoothed_predictions = self.smooth_probabilities(aggregated_predictions)
        
        # Determine class predictions by selecting the class with the highest likelihood for each frame
        class_predictions = np.argmax(smoothed_predictions, axis=1)
        
        class_predictions[-1] = 2
        
        return class_predictions

    def smooth_probabilities(self, probabilities, window_size=7):
        # Check if probabilities array is 2D and has the correct shape
        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError("The probabilities array should be 2D with shape (n, 3).")
        
        # Apply a uniform filter to smooth each class's probability
        smoothed = np.apply_along_axis(lambda m: uniform_filter1d(m, size=window_size), axis=0, arr=probabilities)
        return smoothed

    def plot_predictions_with_time(self, index, smoothed_predictions, capture_name):
        """
        Plot smoothed predictions against the true labels and show time on the secondary x-axis.
        """
        labels = self.labels[index]
        metadata = self.all_metadata[index]
        correction_offset = 0.3

        fig, ax1 = plt.subplots(figsize=(20, 5))

        ax1.plot(labels, label='True Labels', color='blue')
        ax1.plot(smoothed_predictions, label='Predicted', color='red', linestyle='--')
        ax1.set_xlim([metadata['frame_range'][0], metadata['frame_range'][1]])
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Label')
        ax1.legend(loc='upper left')

        frames = np.arange(metadata['frame_range'][0], metadata['frame_range'][1] + 1)
        times = (metadata['MOCAP_time_range'][0] + frames * metadata['seconds_per_frame']) - metadata['frame_range'][0] + correction_offset

        # Dynamically determine tick frequency to avoid zero step size
        tick_frequency = max(1, round(1 / metadata['seconds_per_frame']))
        tick_indices = np.arange(len(frames))[::tick_frequency]
        tick_frames = frames[tick_indices]
        tick_times = times[tick_indices]

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(tick_frames)
        ax2.set_xticklabels(["{:.2f}s".format(time) for time in tick_times], rotation=45)
        ax2.set_xlabel('Time (s)')

        plt.title(f'Predictions vs. True Labels for {capture_name}')
        plt.show()
        
    def find_consecutive_segments(self, predictions=[], min_length=5):
        if not isinstance(predictions,  np.ndarray):
            predictions = self.labels[self.current_index]
        segments = []
        current_segment = []
        last_label = 2

        for i, label in enumerate(predictions):
            if label == last_label and label in [0, 1]:  # GOUP or DOWN
                current_segment.append(i)
            else:
                if len(current_segment) >= min_length:
                    segments.append((current_segment[0], last_label))
                current_segment = [i] if label in [0, 1] else []
            last_label = label

        # Check the last segment
        if len(current_segment) >= min_length:
            segments.append((current_segment[0], last_label))

        return segments
    
    def generate_full_confusion_matrix(self, segments, true_segments, full_predictions, window=12):
        true_labels = self.labels[self.current_index]
        y_pred = []
        y_true = []
        for start, label in segments:
            # Look for the corresponding start in true_labels within a 10-frame window
            for i in range(max(0, start - window), min(len(true_labels), start + window)):
                if true_labels[i] == label:
                    y_pred.append(label)
                    y_true.append(label)
                    break
            else:
                y_pred.append(label)
                y_true.append(2)  # Neither
                
        # Second pass: look for false negatives using a window approach
        window = 10
        for start, label in true_segments:
            # Look for the corresponding start in true_labels within a 10-frame window
            for i in range(max(0, start - window), min(len(true_labels), start + window)):
                if full_predictions[i] == label:
                    # Not a false negative
                    break
            else:
                y_pred.append(full_predictions[i])
                y_true.append(label)  # Neither
            
        return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    def generate_confusion_matrix(self, segments, window=12):
        true_labels = self.labels[self.current_index]
        y_pred = []
        y_true = []
        for start, label in segments:
            # Look for the corresponding start in true_labels within a 10-frame window
            for i in range(max(0, start - window), min(len(true_labels), start + window)):
                if true_labels[i] == label:
                    y_pred.append(label)
                    y_true.append(label)
                    break
            else:
                y_pred.append(label)
                y_true.append(2)  # Neither

        return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    
    def generate_confusion_matrix_with_window_for_false_negatives(self, segments, window=12):
        true_labels = self.labels[self.current_index]
        y_pred = []
        y_true = []
        used_true_labels = []  # Keep track of which true labels have been matched

        # First pass: look for true positives and false positives
        for start, label in segments:
            found_match = False
            for i in range(max(0, start - window), min(len(true_labels), start + window)):
                if true_labels[i] == label and i not in used_true_labels:
                    y_pred.append(label)
                    y_true.append(label)
                    used_true_labels.append(i)
                    found_match = True
                    break
            if not found_match:
                y_pred.append(label)
                y_true.append(2)  # Neither

        print(f"Used true labels are: {used_true_labels}")
        
        # Second pass: look for false negatives using a window approach
        for i, label in enumerate(true_labels):
            # Only consider labels that are GOUP or DOWN and haven't been used
            if label in [0, 1] and i not in used_true_labels:
                # Check if there's a sequence of similar labels within a window
                sequence_found = False
                for j in range(max(0, i - window), min(len(true_labels), i + window)):
                    # If a sequence is detected
                    if true_labels[j] == label:
                        sequence_found = True
                        break

                if sequence_found:
                    # If a sequence of the same event type is found within the window, consider it a false negative
                    y_pred.append(2)  # Neither (predicted)
                    y_true.append(label)  # Actual event type
                else:
                    # If no sequence is found, it's not considered a false negative
                    used_true_labels.append(i)  # Mark as used to avoid re-evaluation

        return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


        
    # def generate_confusion_matrix(self, index, frame_predictions):
    #     metadata = self.all_metadata[index]
    #     segment_boundaries = metadata['GOUP_ranges'] + metadata['DOWN_ranges']
    #     print(f'Segment boundaries: {segment_boundaries}')
    #     # The rest of the frames are considered 'NEITHER', type 2
    #     segment_true_labels = self.labels[index]
        
    #     segment_predictions = []
    #     segment_true_labels_list = []

    #     for segment_range in segment_boundaries:
    #         start_frame, end_frame = segment_range
    #         # Adjust for the actual index in the frame_predictions array
    #         adjusted_start = start_frame
    #         adjusted_end = end_frame 
            
    #         print(segment_range)

    #         # Get the frame-level predictions for this segment
    #         segment_frame_predictions = frame_predictions[adjusted_start:adjusted_end]
    #         print(segment_frame_predictions)
    #         # Aggregate the frame predictions into a segment prediction using the mode
    #         segment_prediction = mode(segment_frame_predictions).mode[0]
    #         segment_predictions.append(segment_prediction)

    #         # Get the true label for the segment from the classified frames
    #         segment_true_label = mode(segment_true_labels[start_frame:end_frame]).mode[0]
    #         segment_true_labels_list.append(segment_true_label)

    #     # Convert lists to arrays
    #     segment_predictions = np.array(segment_predictions)
    #     segment_true_labels = np.array(segment_true_labels_list)

    #     # Now generate the confusion matrix
    #     conf_matrix = confusion_matrix(segment_true_labels, segment_predictions)

    #     return conf_matrix
    
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
        