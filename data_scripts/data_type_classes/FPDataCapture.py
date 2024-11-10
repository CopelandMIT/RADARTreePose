import pandas as pd
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import imageio
import os
from scipy.spatial import ConvexHull

class FPDataCapture:
    def __init__(self, base_file_path, is_foot_always_up = False):
        self.headers_f_1 = {}
        self.headers_f_2 = {}
        self.sample_frequency = 1200
        self.base_file_path = base_file_path
        self.radar_to_mocap_conversion_table_path = '/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RADARTreePose/data/csvs/radar_seconds_per_frame_t0.csv' 
        self.event_data_frame_path = "/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RADARTreePose/data/csvs/MOCAP_FP_RADAR_FU_Stable_Break_FD_TIME_FRAMES_v3.csv"
        self.data_f_1 = self.import_data(self.base_file_path.replace(".tsv", "_f_1.tsv"), self.headers_f_1)
        self.data_f_2 = self.import_data(self.base_file_path.replace(".tsv", "_f_2.tsv"), self.headers_f_2)
        if is_foot_always_up:
            if "MNTRL" in self.base_file_path:
                data = self.data_f_2
            elif "MNTRR" in self.base_file_path:
                data = self.data_f_1
            self.data=data        
        self.foot_lift_times = None
        self.foot_down_times = None
        self.foot_lift_frames_after_actuator = None
        self.foot_down_frames_after_actuator = None
        self.RADAR_Capture = self.base_file_path.split('.')[-2].split('/')[-1].replace("MC", "RR")
        self.seconds_per_frame = 0.036352
        self.avg_COP_per_frame_dict = {}

    def import_data(self, file_path, headers_dict):
        # The number of initial lines containing metadata information.
        num_metadata_lines = 26 
        
        # Assuming that the first line of actual data has the correct column headers
        data = pd.read_csv(file_path, delimiter='\t', header=num_metadata_lines)
        
        # Convert data to numeric, handling non-numeric entries
        data = data.apply(pd.to_numeric, errors='coerce')
        
        # Change TIME to time
        data.rename(columns={"TIME": "time"}, inplace=True)

        # Reset the index and column names of the dataframe
        data.reset_index(drop=False, inplace=True)
        column_names = list(data.columns)[1:]
        data.drop(data.columns[-1], axis=1, inplace=True)
        data.columns = column_names
        
        # test_file_path = "/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RADARTreePose/data/csvs/"
         
        # Save the DataFrame to a CSV file
        # csv_file_path = test_file_path + str(file_path.split('/')[-1].replace('.tsv', '.csv'))
        # data.to_csv(csv_file_path, index=False)
        # print(f"Data saved to {csv_file_path}")

        return data
    
    def calculate_avg_COP_velocity_per_frame(self):
        # Initialize the dictionary to store avg COP velocities per tx
        self.avg_COP_per_frame_dict = {}
        
        # Read the event data frame
        event_df = pd.read_csv(self.event_data_frame_path)
        
        # Filter the rows corresponding to the current RADAR capture
        capture_rows = event_df[event_df['RADAR_capture'] == self.RADAR_Capture]
        
        if capture_rows.empty:
            print(f"No matching RADAR_Capture found in event data frame for {self.RADAR_Capture}")
            return

        # Select the appropriate data based on 'MNTRL' or 'MNTRR' in the base file path
        if "MNTRR" in self.base_file_path:
            data = self.data_f_1
        elif "MNTRL" in self.base_file_path:
            data = self.data_f_2
        else:
            print("Base file path does not contain 'MNTRL' or 'MNTRR'.")
            return
        
        # Ensure the necessary columns are present
        required_columns = {'time', 'COP_X', 'COP_Y'}
        if not required_columns.issubset(data.columns):
            print(f"Data is missing required columns: {required_columns - set(data.columns)}")
            return

        # Iterate over each relevant event (transition)
        for idx, row in capture_rows.iterrows():
            tx = int(row['tx'])
            is_final_tx = bool(row['is_final_tx'])

            if pd.isnull(row['t_stable']):
                print(f"Skipping tx {tx} due to NaN value in t_stable")
                continue

            t_start = float(row['t_stable'])

            if is_final_tx:
                # Use t_foot_down instead of t_break for the final transition
                if pd.isnull(row['t_foot_down']):
                    print(f"Skipping tx {tx} due to NaN value in t_foot_down")
                    continue
                t_end = float(row['t_foot_down'])
            else:
                if pd.isnull(row['t_break']):
                    print(f"Skipping tx {tx} due to NaN value in t_break")
                    continue
                t_end = float(row['t_break'])

            Seconds_per_Frame = float(row['Seconds_per_Frame'])
            
            # Initialize list to store avg COP velocities for this transition
            avg_COP_per_frame_list = []

            # Calculate total time and number of frames
            total_time = t_end - t_start
            if total_time <= 0:
                print(f"Invalid time range for tx {tx}")
                continue

            num_frames = int(np.ceil(total_time / Seconds_per_Frame))

            # Loop through each frame in the time range
            for n in range(num_frames):
                t_frame_start = t_start + n * Seconds_per_Frame
                t_frame_end = min(t_frame_start + Seconds_per_Frame, t_end)

                # Extract data for the current frame
                frame_data = data[(data['time'] >= t_frame_start) & (data['time'] < t_frame_end)]
                if frame_data.empty:
                    # Optionally, you can append a zero or continue
                    print(f"No data available for frame {n} in tx {tx}.")
                    continue

                # Calculate displacement
                COP_X = frame_data['COP_X'].values
                COP_Y = frame_data['COP_Y'].values
                delta_COP_X = COP_X[-1] - COP_X[0]
                delta_COP_Y = COP_Y[-1] - COP_Y[0]
                delta_COP = np.sqrt(delta_COP_X**2 + delta_COP_Y**2)

                # Calculate average velocity
                frame_total_time = t_frame_end - t_frame_start
                if frame_total_time == 0:
                    print(f"Total time is zero for frame {n} in tx {tx}.")
                    continue
                avg_velocity = delta_COP / frame_total_time

                # Append the average velocity to the list for this tx
                avg_COP_per_frame_list.append(avg_velocity)

            # Store the list in the dictionary with tx as the key
            self.avg_COP_per_frame_dict[tx] = avg_COP_per_frame_list


    def identify_foot_lift(self):
        # Determine which force plate data to use based on the filename content
        if "MNTRL" in self.base_file_path:
            data = self.data_f_1
        elif "MNTRR" in self.base_file_path:
            data = self.data_f_2
        else:
            raise ValueError("Filename must contain 'MNTRL' or 'MNTRR'")
        
        # Convert 'time' column to numeric for comparison
        data['time'] = pd.to_numeric(data['time'], errors='coerce')

        # Identifying foot lift and put down events
        # A foot lift event is identified when COP_X and COP_Y go from non-zero to zero
        foot_lift_events = data[(data['COP_X'].shift(1) != 0) & (data['COP_Y'].shift(1) != 0) &
                                (data['COP_X'] == 0) & (data['COP_Y'] == 0)]

        # A foot down event is identified when COP_X and COP_Y go from zero to non-zero
        foot_down_events = data[(data['COP_X'].shift(1) == 0) & (data['COP_Y'].shift(1) == 0) &
                                (data['COP_X'] != 0) & (data['COP_Y'] != 0)]

        # Filter out foot lift events that are too close to each other (within 7 seconds)
        filtered_lift_times = []
        for t in data.loc[foot_lift_events.index, 'time']:
            if not filtered_lift_times or t - filtered_lift_times[-1] > 7:
                if t>8:
                    filtered_lift_times.append(t)

        # Filter out foot down events that are too close to foot lift events
        filtered_down_times = []
        for t in data.loc[foot_down_events.index, 'time']:
            if not any(abs(t - lift_time) <= 1 for lift_time in filtered_lift_times):
                filtered_down_times.append(t)

        # Now filter foot down events that are too close to each other (within 7 seconds)
        final_filtered_down_times = []
        for t in filtered_down_times:
            if not final_filtered_down_times or t - final_filtered_down_times[-1] > 7:
                final_filtered_down_times.append(t)

        
        filtered_lift_times, final_filtered_down_times = self.filter_lift_and_down_times(filtered_lift_times, final_filtered_down_times)
        
        # Save the times for foot lift and put down events
        self.foot_lift_times = filtered_lift_times
        self.foot_down_times = final_filtered_down_times
        
        # Returning the times where foot lift and put down events occur
        return self.foot_lift_times, self.foot_down_times
    
    def filter_lift_and_down_times(self, filtered_lift_times, final_filtered_down_times):
        # Ensure alternating sequence of lift and down times, and handle duplicates
        alternating_sequence = []
        last_lift_index = 0
        last_down_index = 0
        max_lifts = 3
        max_downs = 2

        while last_lift_index < len(filtered_lift_times) or last_down_index < len(final_filtered_down_times):
            # Add lift time if not exceeding maximum and if it precedes the corresponding down time
            if last_lift_index < len(filtered_lift_times) and len([t for t in alternating_sequence if "lift" in t]) < max_lifts:
                alternating_sequence.append((filtered_lift_times[last_lift_index], "lift"))
                last_lift_index += 1

            # Add down time if not exceeding maximum and if it follows the corresponding lift time
            if last_down_index < len(final_filtered_down_times) and len([t for t in alternating_sequence if "down" in t]) < max_downs:
                alternating_sequence.append((final_filtered_down_times[last_down_index], "down"))
                last_down_index += 1

            # Remove consecutive duplicates, keeping the first occurrence
            alternating_sequence = [t for i, t in enumerate(alternating_sequence) if i == 0 or t[1] != alternating_sequence[i-1][1]]

            # Break if both lift and down times have reached their maximum count
            if len([t for t in alternating_sequence if "lift" in t]) >= max_lifts and len([t for t in alternating_sequence if "down" in t]) >= max_downs:
                break

        # Extract and separate the filtered lists based on the alternating sequence
        filtered_lift_times = [t[0] for t in alternating_sequence if t[1] == "lift"]
        final_filtered_down_times = [t[0] for t in alternating_sequence if t[1] == "down"]

        return filtered_lift_times, final_filtered_down_times

    
    def convert_force_plate_time_to_frames(self):
        
        # Read in the CSV data into a pandas DataFrame
        df = pd.read_csv(self.radar_to_mocap_conversion_table_path, delimiter=(','))
        
        print(df)
        
        print(self.RADAR_Capture)

        # Find the row that matches the given RADAR capture
        row = df[df['RADAR_capture'] == self.RADAR_Capture].iloc[0]
        
        print(row)
        self.seconds_per_frame = row['Seconds_per_Frame']

        # Calculate the frames after actuator for foot lift and foot down
        self.foot_lift_frames_after_actuator = (np.round((self.foot_lift_times - row['MOCAP_Start_Time']) / row['Seconds_per_Frame']) + row['RADAR_Start_Frame']).astype(int)
        self.foot_down_frames_after_actuator = (np.round((self.foot_down_times - row['MOCAP_Start_Time']) / row['Seconds_per_Frame']) + row['RADAR_Start_Frame']).astype(int)
        
        return
    
    def calculate_rolling_std(self, window_size=100):
        """
        Calculate the rolling standard deviation for the force vectors.

        Parameters:
        window_size (int): The number of samples to include in the rolling window.

        Returns:
        pd.DataFrame: A dataframe with the rolling standard deviation for each force vector.
        """

                # Determine which force plate data to use based on the filename content
        if "MNTRL" in self.base_file_path:
            data = self.data_f_2 
        elif "MNTRR" in self.base_file_path:
            data = self.data_f_1
        else:
            raise ValueError("Filename must contain 'MNTRL' or 'MNTRR'")
        
        print(data.head)
        
        self.rolling_std = data[['Force_X', 'Force_Y', 'Force_Z']].rolling(window=window_size).std()
        return self.rolling_std

    def plot_rolling_std(self):
        """
        Plot the rolling standard deviation for the force vectors.
        """
        if "MNTRL" in self.base_file_path:
            data = self.data_f_2
        elif "MNTRR" in self.base_file_path:
            data = self.data_f_1

        if self.rolling_std is None:
            raise ValueError("Rolling standard deviation has not been calculated. Please run calculate_rolling_std() first.")

        plt.figure(figsize=(14, 7))
        
        # Plot rolling standard deviation for each force vector
        for column in ['Force_X', 'Force_Y', 'Force_Z']:
            plt.plot(data['time'], self.rolling_std[column], label=f'Rolling Std of {column}')
        
        plt.title('Rolling Standard Deviation of Force Vectors')
        plt.xlabel('Time (s)')
        plt.ylabel('Standard Deviation')
        plt.legend()
        plt.show()


    def plot_force_vectors(self):
        """
        Plot the X, Y, Z force values over time.
        """
        # Determine which force plate data to use based on the filename content
        if "MNTRL" in self.base_file_path:
            data = self.data_f_2
        elif "MNTRR" in self.base_file_path:
            data = self.data_f_1
        else:
            raise ValueError("Filename must contain 'MNTRL' or 'MNTRR'")

        plt.figure(figsize=(14, 7))
        
        # Plot force vectors
        for column in ['Force_X', 'Force_Y', 'Force_Z']:
            plt.plot(data['time'], data[column], label=f'{column}')
        
        plt.title('Force Vectors Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend()
        plt.show()
        
    def calculate_and_plot_cop_velocity(self):
        # Determine which force plate data to use based on the filename content
        if "MNTRL" in self.base_file_path:
            data = self.data_f_2
        elif "MNTRR" in self.base_file_path:
            data = self.data_f_1
        else:
            print("Invalid file path")
            return

        # Calculate the time interval between samples
        delta_t = 1 / self.sample_frequency

        # Calculate the differences in COP positions
        dx = np.diff(data['COP_X'])
        dy = np.diff(data['COP_Y'])

        # Calculate the velocity: velocity = sqrt((dx)^2 + (dy)^2) / delta_t
        velocity = np.sqrt(dx**2 + dy**2) / delta_t

        # Calculate the time for each velocity measurement
        # Since velocity is calculated from differences, it has one less point
        time = np.arange(1, len(data['COP_X'])) * delta_t

        # Plot the velocity
        plt.figure(figsize=(10, 6))
        plt.plot(time, velocity, label='COP Velocity')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Velocity (mm/second)')
        plt.title('Center of Pressure (COP) Velocity Over Time')
        plt.legend()
        plt.show()
        
    def generate_cop_trace_gif(self, gif_filename, subsample_factor=50):
        # Determine which force plate data to use based on the filename content
        if "MNTRL" in self.base_file_path:
            data = self.data_f_2
        elif "MNTRR" in self.base_file_path:
            data = self.data_f_1
        else:
            print("Invalid file path")
            return
        
        # Convert foot lift and down times to indices
        foot_lift_indices = [int(time * self.sample_frequency / subsample_factor) for time in self.foot_lift_times]
        foot_down_indices = [int(time * self.sample_frequency / subsample_factor) for time in self.foot_down_times]


        # Ensure data is in the correct format (convert to numpy arrays if they're pandas Series)
        COP_X = np.array(data['COP_X'])
        COP_Y = np.array(data['COP_Y'])
        
        # Center the data by subtracting the mean
        COP_X_centered = COP_X - np.mean(COP_X)
        COP_Y_centered = COP_Y - np.mean(COP_Y)

        # Subsample the centered data
        COP_X_sub = COP_X_centered[::subsample_factor]
        COP_Y_sub = COP_Y_centered[::subsample_factor]
    
        print(f"Length of COP X sub sample is {len(COP_X_sub)}") 

        # Prepare for GIF creation
        filenames = []
        gif_folder = "/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RADARTreePose/data/gifs"
        gif_path = f"{gif_folder}/{gif_filename}"
        
        # Calculate elapsed time for each subsampled point
        elapsed_time_per_point = 1 / self.sample_frequency * subsample_factor
        
        current_label = "Foot Down"  # Track the current label ("Foot Up" or "Foot Down")

        for i in range(len(COP_X_sub)):
            plt.figure(figsize=(6, 6))
            plt.plot(COP_X_sub[:i], COP_Y_sub[:i], 'bo', markersize=2)  # Previous positions in blue
            plt.plot(COP_X_sub[i], COP_Y_sub[i], 'ro', markersize=5)  # Current position in red
            
            # Update current_label based on foot_lift_indices and foot_down_indices
            if i in foot_lift_indices:
                current_label = 'Foot Up'
            elif i in foot_down_indices:
                current_label = 'Foot Down'
            
            # If there's a current label, display it near the top of the chart
            if current_label:
                plt.text(0.5, 0.95, current_label, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, color='green')

            plt.xlim([-25, 25])
            plt.ylim([-45, 45])
            
            # Calculate elapsed time for the title
            elapsed_time = i / self.sample_frequency * subsample_factor
            plt.title(f'Time: {elapsed_time:.2f} s')
            
            # Save plot to a file
            filename = f'{gif_folder}/frame_{i}.png'
            plt.savefig(filename)
            plt.close()
            filenames.append(filename)

        # Create a GIF
        with imageio.get_writer(gif_path, mode='I', duration=40/len(COP_X_sub)) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                
        # Cleanup the temporary frame images
        for filename in filenames:
            os.remove(filename)

        print(f'GIF saved to {gif_path}')
        return gif_path
    
                
    def calculate_cop_x_velocity_between_FUFD(self, start_end_times, number_of_frames, output_folder_path, file_name):
        # Determine which force plate data to use based on the filename content
        data = self.data_f_2 if "MNTRL" in self.base_file_path else self.data_f_1 if "MNTRR" in self.base_file_path else None
        if data is None:
            print("Invalid file path")
            return
        
        # Ensure the output folder exists
        specific_output_folder_path = os.path.join(output_folder_path, file_name[:2])
        os.makedirs(specific_output_folder_path, exist_ok=True)  # Simplified folder creation
          
        # Filter the data to include only the range between start_end_times
        start_time, end_time = start_end_times
        time_series = np.linspace(0, len(data) / self.sample_frequency, len(data), endpoint=False)
        filtered_indices = (time_series >= start_time) & (time_series <= end_time)
        filtered_data = data[filtered_indices]
        
        # Calculate the time interval between samples
        delta_t = 1 / self.sample_frequency

        # Calculate the differences in COP_X positions using shift
        dx = filtered_data['COP_X'] - filtered_data['COP_X'].shift(1)

        # Drop the NaN values resulted from shifting
        dx = dx.dropna()

        # Calculate the velocity in the COP_X direction: velocity_x = dx / delta_t
        velocity_x = dx / delta_t

        # Downsample the velocity to match the desired number_of_frames
        downsampling_factor = max(1, len(velocity_x) // number_of_frames)  # Ensure division by zero is avoided
        velocity_x_downsampled = velocity_x.iloc[::downsampling_factor].head(number_of_frames)

        # Prepare the data for CSV
        velocity_data = {'Velocity_X': velocity_x_downsampled}
        velocity_df = pd.DataFrame(velocity_data).reset_index(drop=True)
        
        print(f'The length of {file_name} is {len(velocity_df)}')
        print(f'The number of frames are {number_of_frames}')

        # # Define the full path for saving the CSV file
        # csv_file_path = os.path.join(specific_output_folder_path, f"{file_name}.csv")
        # # Save the DataFrame to CSV
        # velocity_df.to_csv(csv_file_path, index=False)
        
        # print(f"Velocity data saved to {csv_file_path}")
        
        # Note: Saving a DataFrame to .npy format directly isn't typical since .npy is meant for NumPy arrays
        # If needing to save velocity_df as a NumPy array:
        output_filename = os.path.join(specific_output_folder_path, f"{file_name}.npy")
        np.save(output_filename, velocity_df.to_numpy())
        
        print(f"{file_name} processed and saved as NumPy array to {output_filename}")
        

    def generate_cop_trace_gif_fu_only(self, gif_filename, subsample_factor=50):
        # Determine which force plate data to use based on the filename content
        if "MNTRL" in self.base_file_path:
            data = self.data_f_2
        elif "MNTRR" in self.base_file_path:
            data = self.data_f_1
        else:
            print("Invalid file path")
            return

        # Ensure data is in the correct format (convert to numpy arrays if they're pandas Series)
        COP_X = np.array(data['COP_X'])
        COP_Y = np.array(data['COP_Y'])

        # Center the data by subtracting the mean
        COP_X_centered = COP_X - np.mean(COP_X)
        COP_Y_centered = COP_Y - np.mean(COP_Y)

        # Convert foot lift and down times to indices
        foot_lift_indices = [int(time * self.sample_frequency) for time in self.foot_lift_times]
        # Ensure there is a "foot down" time for each "foot up", or use the length of the dataset for the last "foot up"
        if len(self.foot_down_times) < len(self.foot_lift_times):
            self.foot_down_times.append(len(COP_X) / self.sample_frequency)  # Assume the last "foot up" extends to the end
        foot_down_indices = [int(time * self.sample_frequency) for time in self.foot_down_times]

        # Filtering indices for "Foot Up" periods
        indices_to_plot = []
        for start, end in zip(foot_lift_indices, foot_down_indices):
            # Adjust the range to ensure it does not exceed the length of the dataset
            end_index = min(end, len(COP_X))
            indices_to_plot.extend(range(start, end_index, subsample_factor))

        # Subsample the centered data for "Foot Up" periods
        COP_X_sub = COP_X_centered[indices_to_plot]
        COP_Y_sub = COP_Y_centered[indices_to_plot]

        # Prepare for GIF creation
        gif_folder = "/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RADARTreePose/data/gifs"
        if not os.path.exists(gif_folder):
            os.makedirs(gif_folder)
        gif_path = f"{gif_folder}/{gif_filename}"
        
        filenames = []

        for i in range(len(COP_X_sub)):
            plt.figure(figsize=(6, 6))
            plt.plot(COP_X_sub[:i+1], COP_Y_sub[:i+1], 'bo', markersize=2)  # Previous positions in blue
            plt.plot(COP_X_sub[i], COP_Y_sub[i], 'ro', markersize=5)  # Current position in red

            # Adjust the plot limits
            plt.xlim([-25, 25])
            plt.ylim([-45, 45])
            
            elapsed_time = indices_to_plot[i] / self.sample_frequency
            plt.title(f'Time: {elapsed_time:.2f} s')
            
            # Save each frame
            filename = os.path.join(gif_folder, f'frame_{i}.png')
            plt.savefig(filename)
            plt.close()
            filenames.append(filename)

        # Create and save the GIF
        with imageio.get_writer(gif_path, mode='I', duration=subsample_factor/(self.sample_frequency)) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                
        # Cleanup the temporary frame images
        for filename in filenames:
            os.remove(filename)

        print(f'GIF saved to {gif_path}')
        
        
    def isolate_rows_by_time(self, start_time, end_time):
        """
        Isolates rows in the DataFrame between the specified start and end times.

        Parameters:
        - df: DataFrame containing the data.
        - start_time: The start time as a float.
        - end_time: The end time as a float.

        Returns:
        - A DataFrame containing only the rows within the specified time range.
        """
        # Filter the rows where 'time' is between start_time and end_time
        filtered_df = self.data[(self.data['time'] >= start_time) & (self.data['time'] <= end_time)]
        return filtered_df
    
        
    def enclosing_circle_radius(self, df):
        cop_points = df[['COP_X', 'COP_Y']].values
        hull = ConvexHull(cop_points)
        # scipy.spatial.ConvexHull does not directly provide a min_circle method.
        # Assuming this was intended to calculate the radius of the enclosing circle
        # using the convex hull points. However, this requires a different approach or library,
        # such as using 'miniball' to find the minimum enclosing circle of points.
        # This placeholder shows intent but would need to be replaced with actual implementation.
        center = hull.points.mean(axis=0)  # Placeholder for actual center calculation
        distances = np.sqrt(((hull.points - center) ** 2).sum(axis=1))
        radius = distances.max()
        return radius

    def convex_hull_area(self, df):
        cop_points = df[['COP_X', 'COP_Y']].values
        hull = ConvexHull(cop_points)
        return hull.volume  # For 2D, volume returns the area

    def average_velocity_squared(self, df):
        cop_points = df[['COP_X', 'COP_Y']].values
        velocities = np.diff(cop_points, axis=0)
        velocities_squared = np.sum(velocities ** 2, axis=1)
        avg_vel_squared = np.mean(velocities_squared)
        return avg_vel_squared
    
    def average_speed(self, df):
        cop_points = df[['COP_X', 'COP_Y']].values
        velocities = np.diff(cop_points, axis=0)
        velocities_squared = np.sum(velocities ** 2, axis=1)
        speeds = np.sqrt(velocities_squared)
        avg_speed = np.mean(speeds)
        return avg_speed
    
    def sqrt_average_speed(self, df):
        cop_points = df[['COP_X', 'COP_Y']].values
        velocities = np.diff(cop_points, axis=0)
        velocities_squared = np.sum(velocities ** 2, axis=1)
        sqrt_speeds = np.sqrt(np.sqrt(velocities_squared))
        avg_sqrt_speeds = np.mean(sqrt_speeds)
        return avg_sqrt_speeds

    def maximum_distance_from_centroid(self, df):
        cop_points = df[['COP_X', 'COP_Y']].values
        centroid = np.mean(cop_points, axis=0)
        distances = np.sqrt(np.sum((cop_points - centroid) ** 2, axis=1))
        max_distance = np.max(distances)
        return max_distance
    
    def standard_deviation_from_centroid(self, df):
        # Extract the COP points
        cop_points = df[['COP_X', 'COP_Y']].values
        # Calculate the centroid of the COP points
        centroid = np.mean(cop_points, axis=0)
        # Calculate the distances of each point from the centroid
        distances = np.sqrt(np.sum((cop_points - centroid) ** 2, axis=1))
        # Calculate the standard deviation of the distances
        std_deviation = np.std(distances)
        return std_deviation
        
# Usage example:
# file_path_f_1 = 'path_to_f_1.tsv' # Replace with actual file path
# file_path_f_2 = 'path_to_f_2.tsv' # Replace with actual file path
# fp_data_capture = FPDataCapture(file_path_f_1, file_path_f_2)
# filename_to_check = 'some_filename_containing_MNTRL_or_MNTRR'
# fp_data_capture.identify_foot_lift(filename_to_check)
# Now you can access the times directly
# print(fp_data_capture.foot_lift_times_f_1)
# print(fp_data_capture.foot_down_times_f_1)
# print(fp_data_capture.foot_lift_times_f_2)
# print(fp_data_capture.foot_down_times_f_2)
