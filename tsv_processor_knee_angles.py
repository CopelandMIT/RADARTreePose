import os
import pandas as pd
import numpy as np

class SimplifiedTSVProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def calculate_angular_velocity(self, angles, time_step=0.01):
        return angles.diff() / time_step

    def process_files(self):
        for filename in os.listdir(self.input_folder):
            if "MNTRL" in filename or "MNTRR" in filename and filename.endswith("pos.tsv"):
                self.process_tsv(os.path.join(self.input_folder, filename))

    def process_tsv(self, file_path):
        df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8')
        
        # Select the appropriate knee angle column
        knee_angle_column = 'Knee_R_angle' if 'MNTRL' in file_path else 'Knee_L_angle'
        
        # Ensure the column exists
        if knee_angle_column not in df.columns:
            print(f"Column {knee_angle_column} not found in {file_path}. Skipping this file.")
            return
        
        # Calculate angular velocity for the selected knee angle
        df['Angular_Velocity'] = self.calculate_angular_velocity(df[knee_angle_column])

        # Prepare DataFrame for saving
        output_df = df[['time', knee_angle_column, 'Angular_Velocity']].copy()
        
        # Save the processed data
        output_filename = os.path.basename(file_path).replace('.tsv', '_processed.csv')
        output_file_path = os.path.join(self.output_folder, output_filename)
        output_df.to_csv(output_file_path, index=False)
        print(f"Processed and saved: {output_filename}")

# # Example usage:
# input_folder = '/path/to/input/folder'
# output_folder = '/path/to/output/folder'
# processor = SimplifiedTSVProcessor(input_folder, output_folder)
# processor.process_files()
