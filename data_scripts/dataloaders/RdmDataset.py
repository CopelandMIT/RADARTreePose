import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch import optim


class RdmDataset(Dataset):
    def __init__(self, root_dir, event_csv, included_folders, window_size=100, overlap = 90):
        self.data = []
        self.labels = []  # This will store labels for each window
        self.metadata = []  # Store metadata for each window
        self.window_size = window_size
        self.overlap = overlap
        
        # Load event labels and actuator frames
        self.event_labels_df = pd.read_csv(event_csv)
        
        # Iterate only over included folders
        for folder_name in included_folders:
            folder_path = os.path.join(root_dir, folder_name)
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('.npy'):
                    filepath = os.path.join(folder_path, file)
                    radar_capture = "_".join(file.split('_')[:-1])  # Extract radar capture name
                    
                    # Ensure radar_capture matches one of the entries in the actuator CSV
                    if self.event_labels_df['RADAR_capture'].str.contains(radar_capture).any():
                        rdm_data = np.load(filepath)
                        rdm_data = torch.from_numpy(rdm_data).float()  # Convert numpy array to PyTorch tensor of type float

                        actuator_info = self.event_labels_df[self.event_labels_df['RADAR_capture'] == radar_capture].iloc[0]
                        actuator_start_frame, actuator_end_frame = actuator_info['RADAR_Start_Frame'], actuator_info['RADAR_End_Frame']
                        
                        # Create windows, label them, and add metadata
                        self.create_and_label_windows(rdm_data, radar_capture, actuator_start_frame, actuator_end_frame)
    
    def label_frames(self, radar_capture):
        num_frames = 1000  # Assuming a fixed size, adjust as necessary
        labels = np.full(num_frames, 2)  # Default to 3 (neither)
        capture_events = self.event_labels_df[self.event_labels_df['RADAR_capture'] == radar_capture]
        
        goup_ranges = []
        down_ranges = []
        
        for _, event in capture_events.iterrows():
            if not pd.isna(event['frame_foot_up']) and not pd.isna(event['frame_stable']):
                start = int(event['frame_foot_up'])
                end = int(event['frame_stable'])
                labels[start:end] = 0  # GOUP
                goup_ranges.append((start, end))
            if not pd.isna(event['frame_break']) and not pd.isna(event['frame_end']):
                start = int(event['frame_break'])
                end = int(event['frame_end'])
                labels[start:end] = 1  # DOWN
                down_ranges.append((start, end))
        
        return labels, goup_ranges, down_ranges

    # Adjustments to use updated label_frames in create_and_label_windows
    def create_and_label_windows(self, rdm_data, radar_capture, actuator_start_frame, actuator_end_frame):
        num_frames = actuator_end_frame - actuator_start_frame + 1
        labels, goup_ranges, down_ranges = self.label_frames(radar_capture)
        
        for start in range(0, num_frames - self.window_size + 1, (self.window_size - self.overlap)):
            actual_start = start + actuator_start_frame
            actual_end = actual_start + self.window_size
            
            if actual_end > actuator_end_frame:
                break
            
            #TODO remove this break, since we should be padding the ending sequence. 
                
            window_labels = labels[actual_start:actual_end]
            self.data.append(rdm_data[actual_start:actual_end])
            self.labels.append(window_labels)
            
            # Adjust range offsets for metadata
            window_goup_ranges = [(start - actual_start, end - actual_start) for start, end in goup_ranges if start >= actual_start and end <= actual_end]
            window_down_ranges = [(start - actual_start, end - actual_start) for start, end in down_ranges if start >= actual_start and end <= actual_end]
            
            metadata = {
                'frame_range': (actual_start, actual_end),
                'RADAR_capture': radar_capture,
                'GOUP_ranges': window_goup_ranges,
                'DOWN_ranges': window_down_ranges,
            }
            self.metadata.append(metadata)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Assuming self.data is your dataset and self.labels are your labels
        data = self.data[index]
        label = self.labels[index]
        length = len(data)  # Or however you calculate the length of your sequence
        
        # Assuming metadata is stored as a dictionary in self.metadata[index]
        metadata = self.metadata[index]

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
    
