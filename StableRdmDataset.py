import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch import optim
from FPDataCapture import FPDataCapture



class StableRdmDataset(Dataset):
    def __init__(self, root_dir, event_csv, included_folders, label_type = "avg_speed"):
        label_types = ['avg_velocity_squared', 'max_distance_from_centroid', "avg_speed"]
        if label_type not in label_types:
            raise ValueError(f"Invalid lavel type. Expected one of: {label_types}")
        
        self.data = []
        self.labels = []
        self.metadata = []
        self.force_plate_dir = "/Volumes/FourTBLaCie/Yoga_Study_FP_1and2_MNTR"
        self.num_channels = 4
        self.event_labels_df = pd.read_csv(event_csv)
        
        for folder_name in included_folders:
            folder_path = os.path.join(root_dir, folder_name)
            # Filter rows for the current folder
            filtered_df = self.event_labels_df[self.event_labels_df['RADAR_capture'].str.startswith(folder_name)]
            for index, row in filtered_df.iterrows():
                radar_capture = row['RADAR_capture']
                frame_stable = row['frame_stable']
                frame_end = row['frame_end'] if np.isnan(row['frame_break']) else row['frame_break']
                t_stable = row['t_stable']
                t_end = row['t_foot_down'] if np.isnan(row['t_break']) else row['t_break']
                #TODO remove problematic captures! 
                for i in range(self.num_channels):            
                    capture_and_tx = f"{radar_capture}_channel{i+1}_tx{row['tx']}"
                    radar_file_path = os.path.join(folder_path, capture_and_tx + '.npy')  # Assuming .npy format
                    if os.path.exists(radar_file_path):
                        rdm_data = np.load(radar_file_path)

                        self.data.append(rdm_data)
                        metadata = {
                            'RADAR_capture': radar_capture,
                            "tx": row['tx'],
                            'channel': i+1,
                            "n_frames": rdm_data.shape[0],
                            'seconds_per_frame': row['Seconds_per_Frame'],
                            'frame_range': (frame_stable, frame_end),
                            'time_range': (t_stable, t_end)
                        }
                        self.metadata.append(metadata)
                        
                        force_plate_capture = self.create_fp_data_capture(radar_capture)
                        filtered_force_plate_df = force_plate_capture.isolate_rows_by_time(t_stable, t_end)
                        
                        if label_type == 'avg_velocity_squared':
                            label = force_plate_capture.average_velocity_squared(filtered_force_plate_df)
                        elif label_type == 'avg_speed':
                            label = force_plate_capture.average_speed(filtered_force_plate_df)
                        elif label_type == 'max_distance_from_centroid':
                            label = force_plate_capture.maximum_distance_from_centroid(filtered_force_plate_df)
                        
                        self.labels.append(label)

    def create_fp_data_capture(self, radar_capture):
        participant = radar_capture[:2]
        MOCAP_FP_capture_name = radar_capture.replace('_RR_', '_MC_')
        base_file_path = os.path.join(self.force_plate_dir, participant, MOCAP_FP_capture_name + '.tsv')  # Corrected path
        return FPDataCapture(base_file_path=base_file_path, is_foot_always_up=True)
    
    def __len__(self):
        return len(self.data)       
    
    def __getitem__(self, index):
        # Assuming self.data is your dataset and self.labels are your labels
        data = self.data[index]
        label = self.labels[index]

        # Assuming metadata is stored as a dictionary in self.metadata[index]
        metadata = self.metadata[index]

        length = metadata['n_frames']   # Adjust key as necessary

        # metadata could contain: 
        # - 'frame_range': tuple indicating the range of frames, e.g., (350, 450)
        # - 'RADAR_capture': the identifier for the RADAR capture, e.g., '01_MNTRL_RR_V1'
        # - 'GOUP_transition': boolean indicating if there is a GOUP transition within the window
        # - 'DOWN_transition': boolean indicating if there is a DOWN transition within the window

        return data, label, length, metadata

                    
    @staticmethod
    def collate_fn(batch):
        # Implement the collate function as described
        sequences, labels, _, metadata = zip(*batch)  # Adjusted to fit provided batch structure
        sequences_padded = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in sequences], batch_first=True)
        labels_padded = torch.tensor(labels, dtype=torch.float32)  # Assuming labels don't need padding; adjust as necessary
        lengths = [md['n_frames'] for md in metadata]  # Adjust key as necessary

        # Convert lengths to a tensor
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)

        return sequences_padded, labels_padded, lengths_tensor, metadata