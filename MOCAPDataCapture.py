import numpy as np
import pandas as pd
import os
import csv
import re
from datetime import datetime
from scipy.signal import find_peaks, convolve
import matplotlib.pyplot as plt

class MOCAPDataCapture:
    def __init__(self, base_file_path):
        self.base_file_path = base_file_path
        self.sample_frequency = 100
        self.pos_file_path = base_file_path.replace(".tsv", "_pos.tsv")
        self.vel_file_path = base_file_path.replace(".tsv", "_vel.tsv")
        # Pattern to match "/##/" where ## are two digits
        self.participant_pattern = r"/(\d{2})/"
        match = re.search(self.participant_pattern, base_file_path)
        if match:
            self.participant_id = match.group(1)
            print(f"Processing File: {self.base_file_path.split('/')[-1]}")
        else:
            raise ValueError("Participant ID could not be extracted from the base file path.")
        self.position_data = None
        self.velocity_data = None
        self.start_actuator_time = None
        self.end_actuator_time = None
        self.load_and_process_data()

    def load_and_process_data(self):
        """
        Loads and processes position and velocity data from TSV files.

        Args:
            pos_file_path (str): The file path to the position TSV file.
            vel_file_path (str): The file path to the velocity TSV file.
        """
        try:
            # self.position_data = self.process_tsv(self.pos_file_path)
            self.velocity_data = self.process_tsv(self.vel_file_path)
            # print("Position and velocity data loaded and processed.")
            # print(self.position_data)
            # print(self.velocity_data)
        except Exception as e:
            print(f"An error occurred: {e}")
            
    def Thirteen_MNTRL_V2(self):
        return
        
    def process_tsv(self, file_path, save_to_csv=False):
        print(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        with open(file_path, mode='r', newline='') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            # print(tsv_reader)
            first_5_rows_list = []
            remaining_rows_list = []

            try:
                for i, row in enumerate(tsv_reader):
                    if i < 5:
                        first_5_rows_list.append(row)
                    else:
                        if len(row) < 58:
                            row += [''] * (58 - len(row))
                        remaining_rows_list.append(row)
            except Exception as e:
                print(f"An error occurred while processing the file: {e}")
                return


            # Create Header pandas DataFrames from first 5 rows of lists
            df_header = pd.DataFrame(first_5_rows_list).set_index(0)
            try:
                df_header.columns = ["Value"]
            except:
                pass
            # print("Header for data frame")
            # print(df_header)
            
            # Create blank, correct shape pandas DataFrames from remainder of lists
            df = pd.DataFrame(remaining_rows_list)
            # print('New data frame')
            
            # Shift row 6 to the left and remove cell 6,1
            df.iloc[2, 0:-1] = df.iloc[0, 1:].values
            
            #delete empty column
            df = df.iloc[:,:-1]

            # Remove rows 7 and 8 (originally 8 and 9)
            df = df.drop(df.index[0:2])
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            # print('2: New data frame')
            # print(df)
            
            ## Change data types of columns
            df = df.apply(pd.to_numeric, downcast='float')
            
            # Add 'frame', 'time' and participant columns
            df.insert(0, 'frame', range(0, len(df)))
            df.insert(1, 'time', [i * 0.01 for i in range(len(df))])
            df.insert(2,'participant_id', self.participant_id)
            
            # Reset index
            df.reset_index(drop=True, inplace=True)
                  
            if save_to_csv == True:      
                if df.shape[0] != 4000:
                    print(df.shape)
                    raise Exception("DATA Frame is the wrong size!!")
                else:
                    self.output_folder = "/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RADARTreePose/data/csvs"
                    output_file_path = os.path.join(
                        self.output_folder, os.path.splitext(os.path.basename(file_path))[0] + ".csv"
                    )
                    print(f"Saved: {os.path.basename(file_path)}")
                    df.to_csv(output_file_path, index=False, header=True)
            return df
    
    
    def plot_convolution_result(self, actuator_vel_x):
        """
        Plots the convolution result along with a threshold line to help determine an appropriate threshold.

        Args:
            actuator_vel_x (np.array): The actuator velocity data.
        """
        # Generate the template signal
        template = np.concatenate([np.full(102, 50), np.zeros(10), np.full(102, -50)])

        # Convolve the template with the actuator velocity data
        convolution_result = convolve(actuator_vel_x, template, mode='valid')

        # Find local minima in the convolution result
        local_minima_indices, _ = find_peaks(-convolution_result)

        # Define the threshold
        threshold = -4e5

        # Plot the convolution result
        plt.figure(figsize=(12, 6))
        plt.plot(convolution_result, label='Convolution Result')
        
        # Plot the local minima
        plt.plot(local_minima_indices, convolution_result[local_minima_indices], 'rx', label='Local Minima')

        # Plot the threshold line
        plt.axhline(y=threshold, color='g', linestyle='--', label=f'Threshold ({threshold})')

        plt.xlabel('Time Step')
        plt.ylabel('Convolution Value')
        plt.title('Convolution Result with Local Minima and Threshold')
        plt.legend()
        plt.show()
        

    def find_actuator_start_end_direction_changes(self):
        """
        Uses convolution to find the start and end times of transitions in the actuator velocity 
        from around +50 to -50, ensuring that peaks are not within 2 seconds of each other.
        """
        if self.velocity_data is None:
            print("Velocity data not loaded. Please load data before running this function.")
            return

        # Generate the template signal
        template = np.concatenate([np.full(102, 50), np.zeros(10), np.full(102, -50)])

        # Extract the actuator X velocity data
        actuator_vel_x = self.velocity_data['Actuator_vel_X'].to_numpy()

        # Convolve the template with the actuator velocity data
        convolution_result = convolve(actuator_vel_x, template, mode='valid')

        # Find local minima in the convolution result as potential matches
        local_minima_indices, _ = find_peaks(-convolution_result)

        # Threshold for determining a strong match
        threshold = -4e5  # Adjust based on your data's characteristics

        # Filter out matches that don't meet the threshold
        significant_matches = [idx for idx in local_minima_indices if convolution_result[idx] < threshold]

        # Ensure matches are not within 200 indices of each other
        filtered_matches = []
        for match in significant_matches:
            if not filtered_matches:  # If list is empty, add the first match
                filtered_matches.append(match)
            else:
                # Check if current match is more than 200 indices apart from the last added match
                if match - filtered_matches[-1] > 200:
                    filtered_matches.append(match)
                else:
                    # If within 200 indices, keep the one with the more significant peak (lower value in convolution result)
                    if convolution_result[match] < convolution_result[filtered_matches[-1]]:
                        filtered_matches[-1] = match  # Replace the last match with the current one

        if filtered_matches:
            # Set start and end times based on the filtered matches
            self.start_actuator_time = self.velocity_data.iloc[filtered_matches[0]]['time']
            if len(filtered_matches) > 1:
                self.end_actuator_time = self.velocity_data.iloc[filtered_matches[1]]['time']
            print(f"Start actuator time: {self.start_actuator_time}, End actuator time: {self.end_actuator_time}")
        else:
            print("No appropriate transitions found in the Actuator_vel_X data.")