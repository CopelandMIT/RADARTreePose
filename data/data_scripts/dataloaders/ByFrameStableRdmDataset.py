import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from balance_assessment_radar.scripts.data_processing.FPDataCapture import FPDataCapture
import numpy as np
from scipy.ndimage import gaussian_filter1d

class ByFrameStableRdmDataset(Dataset):
    def __init__(self, root_dir, event_csv, included_folders):
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
                tx = int(row['tx'])
                is_final_tx = bool(row['is_final_tx'])
                t_stable = row['t_stable']
                t_end = row['t_foot_down'] if pd.isnull(row['t_break']) else row['t_break']
                Seconds_per_Frame = float(row['Seconds_per_Frame'])
                
                # Create force_plate_capture and calculate per-frame COP velocities
                force_plate_capture = self.create_fp_data_capture(radar_capture)
                # Calculate per-frame COP velocities for this transition
                force_plate_capture.calculate_avg_COP_velocity_per_frame()
                # print(force_plate_capture.avg_COP_per_frame_dict)
                # print(f"Keys in avg_COP_per_frame_dict: {list(force_plate_capture.avg_COP_per_frame_dict.keys())}")
                # print(f"Type of keys: {[type(k) for k in force_plate_capture.avg_COP_per_frame_dict.keys()]}")
                if tx in force_plate_capture.avg_COP_per_frame_dict:
                    cop_velocities = force_plate_capture.avg_COP_per_frame_dict[int(tx)]
                else:
                    print(f"tx {tx} not found in avg_COP_per_frame_dict for capture {radar_capture}.")
                    continue

                if cop_velocities is None:
                    print(f"Skipping {radar_capture}, tx {tx} due to missing COP data.")
                    continue  # Skip if unable to calculate COP velocities

                for i in range(self.num_channels):            
                    capture_and_tx = f"{radar_capture}_channel{i+1}_tx{tx}"
                    radar_file_path = os.path.join(folder_path, capture_and_tx + '.npy')  # Assuming .npy format
                    if os.path.exists(radar_file_path):
                        rdm_data = np.load(radar_file_path)
                        # Ensure that the radar data and COP velocities have the same number of frames
                        num_radar_frames = rdm_data.shape[0]
                        num_cop_frames = len(cop_velocities)

                        if num_radar_frames != num_cop_frames:
                            # Resample or interpolate COP velocities to match radar frames
                            cop_velocities_resampled = np.interp(
                                np.linspace(0, num_cop_frames - 1, num_radar_frames),
                                np.arange(num_cop_frames),
                                cop_velocities
                            )
                        else:
                            cop_velocities_resampled = cop_velocities

                        self.data.append(rdm_data)
                        self.labels.append(cop_velocities_resampled)
                        metadata = {
                            'RADAR_capture': radar_capture,
                            'participant_id': radar_capture[:2],
                            'tx': tx,
                            'channel': i+1,
                            'n_frames': num_radar_frames,
                            'seconds_per_frame': Seconds_per_Frame,
                            'time_range': (t_stable, t_end)
                        }
                        self.metadata.append(metadata)
                    else:
                        print(f"Radar file not found: {radar_file_path}")

    def create_fp_data_capture(self, radar_capture):
        participant = radar_capture[:2]
        MOCAP_FP_capture_name = radar_capture.replace('_RR_', '_MC_')
        base_file_path = os.path.join(self.force_plate_dir, participant, MOCAP_FP_capture_name + '.tsv')
        return FPDataCapture(base_file_path=base_file_path, is_foot_always_up=True)
    
    def __len__(self):
        return len(self.data)       
    
    def __getitem__(self, index):
        # Retrieve the radar data and corresponding COP velocities
        data = self.data[index]  # Shape: [seq_len, height, width]
        data = data[:, np.newaxis, :, :]  # Add channel dimension
        
        # Normalize radar data if desired
        data = (data - data.mean()) / data.std()
        
        label = self.labels[index]  # COP velocities
        # Apply Gaussian smoothing
        label_smoothed = self.smooth_cop_velocity_gaussian(label, sigma=2)
        
        # Optionally, scale labels
        label_scaled = (label_smoothed - label_smoothed.mean()) / label_smoothed.std()
        
        metadata = self.metadata[index]
        length = metadata['n_frames']

        return data, label_scaled, length, metadata

    def smooth_cop_velocity_gaussian(self, cop_velocity, sigma=1):
        """
        Apply Gaussian smoothing to COP velocity data.

        Parameters:
        - cop_velocity: numpy array of COP velocities.
        - sigma: Standard deviation for Gaussian kernel.

        Returns:
        - Smoothed COP velocities.
        """
        return gaussian_filter1d(cop_velocity, sigma=sigma)


    @staticmethod
    def collate_fn(batch):
        sequences, labels, lengths, metadata = zip(*batch)
        sequences_padded = pad_sequence(
            [torch.tensor(seq, dtype=torch.float32) for seq in sequences],
            batch_first=True
        )  # Resulting shape: [batch_size, max_seq_len, num_channels, height, width]
        labels_padded = pad_sequence(
            [torch.tensor(lbl, dtype=torch.float32) for lbl in labels],
            batch_first=True
        )  # Resulting shape: [batch_size, max_seq_len]
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)

        return sequences_padded, labels_padded, lengths_tensor, metadata

