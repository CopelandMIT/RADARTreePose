import h5py
import numpy as np
import os
import imageio
from datetime import datetime
from scipy.signal import find_peaks, correlate
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class FMCWRADARDataCapture:
    """
    Class for handling the capture, processing, and saving of FMCW RADAR data.

    This class is designed to load Frequency-Modulated Continuous-Wave (FMCW) RADAR data from a specified HDF5 file,
    process the data into a usable format, and save it as a NumPy file (either .npy or .npz format).

    Attributes:
        file_path (str): Path to the HDF5 file containing the RADAR data.
    """

    def __init__(self, file_path):
        """
        Initializes the FMCWRADARDataCapture class with the specified file path.

        Args:
            file_path (str): Path to the HDF5 file to be loaded and processed.
        """
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        self.file_path = file_path
        self.output_path = self.file_path.replace("_Data", "_Data_NP")
        self.seconds_per_frame = 0.036352
        normalized_actuator_pattern_in_radar_path = '/Users/danielcopeland/Library/Mobile Documents/com~apple~CloudDocs/MIT Masters/DRL/LABx/RADARTreePose/data/filter_patterns/normalized_actuator_pattern_in_radar.npy'
        self.normalized_actuator_pattern_in_radar = np.load(normalized_actuator_pattern_in_radar_path)

    def load_and_save(self, output_path=None, format='npy', save_npy = False):
        if output_path is None:
            output_path = self.output_path 
            output_path = os.path.splitext(output_path)[0]
            
        # Ensure the directory of the output path exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        dataCubes = np.zeros((4, 1000, 128, 256))
        

        # Open the HDF5 file for reading
        with h5py.File(self.file_path, 'r') as file:
            # Extract parameters from the file
            FreqStrt = file['/BrdCfg/FreqStrt'][()]
            FreqStop = file['/BrdCfg/FreqStop'][()]
            N = 256  # Number of samples per chirp, assuming it's given or extracted correctly
            Np = 128  # Number of chirps per frame, assuming it's given or extracted correctly

            # Calculate the effective bandwidth
            B = (FreqStop - FreqStrt) / 284 * 256
            
            # Read the If signal
            If = file['/If'][:]
            NrFrms, Nx = If.shape
            
            # Initialize the dataCubes array to hold the processed data for each channel
            dataCubes = np.zeros((4, NrFrms, N, Np))
            
            # Process each frame
            for frame_idx in range(NrFrms-1):
                # print(f'Processing RD Frame: {frame_idx + 1}')
                for channel_idx in range(4):
                    start_idx = channel_idx * N * Np
                    end_idx = (channel_idx + 1) * N * Np
                    # Correctly reshape and assign the data to the corresponding channel and frame
                    reshaped_data = If[frame_idx, start_idx:end_idx].reshape(Np, N)
                    dataCubes[channel_idx, frame_idx, :, :] = reshaped_data.transpose()  # Transpose operation

            # # Verify by plotting the first channel of the first frame
            # plt.imshow(dataCubes[0, 0, :, :])
            # plt.colorbar()
            # plt.show()
        
            if save_npy:
                # Save data in the specified format
                if format == 'npy':
                    np.save(output_path, dataCubes)
                elif format == 'npz':
                    np.savez(output_path, dataCubes)
                else:
                    raise ValueError("Unsupported format. Use 'npy' or 'npz'.")
            
        return dataCubes

    @staticmethod
    def rawDataToDataCube(rawData, numFrames, numChirpsPerFrame, numSamplesPerChirp, numAntennas):
        # Reshape and rearrange the rawData
        matrixData = rawData.T.reshape(numChirpsPerFrame * numSamplesPerChirp, numFrames * numAntennas)
        dataCubes = np.zeros((numFrames, numChirpsPerFrame, numSamplesPerChirp, numAntennas))

        for frame in range(numFrames):
            for antenna in range(numAntennas):
                chirps = matrixData[:, frame * numAntennas + antenna]
                chirpsMatrix = chirps.reshape(numSamplesPerChirp, numChirpsPerFrame)
                dataCubes[frame, :, :, antenna] = chirpsMatrix.T
                

        return dataCubes.transpose((3,0,1,2))
    
    def range_doppler_processing(self, dataCubes):
        """
        Processes each frame in the dataCube for each channel to generate Range-Doppler Maps.

        Args:
            dataCube (np.ndarray): The raw data cubes to be processed.

        Returns:
            np.ndarray: Array of processed Range-Doppler Maps for each channel.
        """
        n_channels, n_frames, n_bins, n_doppler = dataCubes.shape
        rdm_all_channels = []

        # Define a window function for the range and Doppler dimensions
        range_window = np.hanning(n_bins)
        doppler_window = np.hanning(n_doppler)

        for channel_idx in range(n_channels):
            rdm_list = []
            for frame_idx in range(n_frames):
                # Extract current data for the frame and channel
                current_data = dataCubes[channel_idx, frame_idx, :, :]

                # Apply the Hanning window function
                windowed_data = np.outer(range_window, doppler_window) * current_data

                # Apply 2D FFT and shift
                rdm = np.fft.fft2(windowed_data)
                rdm = np.fft.fftshift(rdm, axes=1)  # Shift along the Doppler axis (second axis in Python)

                # Take the absolute value
                rdm = np.abs(rdm)
                
                # Normalize the data before logarithmic scaling
                rdm_max = np.max(rdm)
                rdm_min = np.min(rdm)
                rdm = (rdm - rdm_min) / (rdm_max - rdm_min + 1e-3)  # Avoid division by zero
                
                # Log scaling - apply log1p for numerical stability
                rdm = np.log1p(rdm)

                # Slice the RDM to remove the mirrored part (keep only one half)
                # Assuming the mirrored part is along the Range axis (axis 0)
                half_index = rdm.shape[0] // 2
                rdm = rdm[:half_index, :]

                # Append to the list for the current channel
                rdm_list.append(rdm)

            # Append the result for the current channel
            rdm_all_channels.append(rdm_list)

        return np.array(rdm_all_channels)
    
        
    def range_doppler_processing_with_phase(self, dataCubes):
        """
        Processes each frame in the dataCube for each channel to generate Range-Doppler Maps.

        Args:
            dataCube (np.ndarray): The raw data cubes to be processed.

        Returns:
            np.ndarray: Array of processed Range-Doppler Maps for each channel, including unwrapped phase information.
        """
        n_channels, n_frames, n_bins, n_doppler = dataCubes.shape
        rdm_all_channels = []

        # Define a window function for the range and Doppler dimensions
        range_window = np.hanning(n_bins)
        doppler_window = np.hanning(n_doppler)

        for channel_idx in range(n_channels):
            rdm_list = []
            for frame_idx in range(n_frames):
                # Extract current data for the frame and channel
                current_data = dataCubes[channel_idx, frame_idx, :, :]

                # Apply the Hanning window function
                windowed_data = np.outer(range_window, doppler_window) * current_data

                # Apply 2D FFT and shift
                rdm_complex = np.fft.fft2(windowed_data)
                rdm_complex = np.fft.fftshift(rdm_complex, axes=1)  # Shift along the Doppler axis

                # Extract phase and apply phase unwrapping along the Doppler axis
                phase_data = np.angle(rdm_complex)
                unwrapped_phase = np.unwrap(phase_data, axis=1)  # Unwrapping along Doppler axis

                # You could now use this unwrapped_phase for velocity calculations
                # For demonstration, let's keep using the magnitude for the RDM visual representation
                rdm = np.abs(rdm_complex)
                
                # Normalize the data before logarithmic scaling
                rdm_max = np.max(rdm)
                rdm_min = np.min(rdm)
                rdm = (rdm - rdm_min) / (rdm_max - rdm_min + 1e-3)  # Avoid division by zero
                
                # Log scaling - apply log1p for numerical stability
                rdm = np.log1p(rdm)

                # Slice the RDM to remove the mirrored part (keep only one half)
                half_index = rdm.shape[0] // 2
                rdm = rdm[:half_index, :]

                # Append to the list for the current channel
                rdm_list.append(rdm)

            # Append the result for the current channel
            rdm_all_channels.append(rdm_list)

        return np.array(rdm_all_channels)
    
    def angle_of_arrival_processing(self, dataCube):
        """
        Processes each frame in the dataCube for each channel to generate Angle of Arrival (AoA) heatmaps.

        Args:
            dataCube (np.ndarray): The raw data cubes to be processed.

        Returns:
            np.ndarray: Array of processed AoA heatmaps for each channel.
        """
        n_channels, n_frames, n_bins, n_elements = dataCube.shape
        aoa_all_channels = []

        # Define a window function for the spatial dimension (assuming ULA)
        spatial_window = np.hanning(n_elements)

        for channel_idx in range(n_channels):
            aoa_list = []
            for frame_idx in range(n_frames):
                # Extract current data for the frame and channel
                current_data = dataCube[channel_idx, frame_idx, :, :]

                # Apply the window function to the antenna elements
                windowed_data = current_data * spatial_window

                # Apply 1D FFT along the spatial dimension (assuming antenna elements are along the last axis)
                aoa_spectrum = np.fft.fft(windowed_data, axis=1)
                aoa_spectrum = np.fft.fftshift(aoa_spectrum, axes=1)  # Shift to center the zero-frequency component

                # Take the absolute value and normalize
                aoa_spectrum = np.abs(aoa_spectrum)
                aoa_spectrum_max = np.max(aoa_spectrum)
                aoa_spectrum_min = np.min(aoa_spectrum)
                aoa_normalized = (aoa_spectrum - aoa_spectrum_min) / (aoa_spectrum_max - aoa_spectrum_min + 1e-3)

                # Append to the list for the current channel
                aoa_list.append(aoa_normalized)

            # Append the result for the current channel
            aoa_all_channels.append(aoa_list)

        return np.array(aoa_all_channels)
    
    
    def generate_actuator_filter(self, dataCubes):
        """
        Processes each frame in the dataCube for each channel to generate Range-Doppler Maps.

        Args:
            dataCube (np.ndarray): The raw data cubes to be processed.

        Returns:
            np.ndarray: Array of processed Range-Doppler Maps for each channel.
        """
        n_channels, n_frames, n_bins, n_doppler = dataCubes.shape
        rdm_all_channels = []

        # Define a window function for the range and Doppler dimensions
        range_window = np.hanning(n_bins)
        doppler_window = np.hanning(n_doppler)

        for channel_idx in range(n_channels):
            rdm_list = []
            for frame_idx in range(n_frames):
                # Extract current data for the frame and channel
                current_data = dataCubes[channel_idx, frame_idx, :, :]

                # Apply the Hanning window function
                windowed_data = np.outer(range_window, doppler_window) * current_data

                # Apply 2D FFT and shift
                rdm = np.fft.fft2(windowed_data)
                rdm = np.fft.fftshift(rdm, axes=1)  # Shift along the Doppler axis (second axis in Python)

                # Take the absolute value
                rdm = np.abs(rdm)
                
                # Normalize the data before logarithmic scaling
                rdm_max = np.max(rdm)
                rdm_min = np.min(rdm)
                rdm = (rdm - rdm_min) / (rdm_max - rdm_min + 1e-3)  # Avoid division by zero
                
                # Log scaling - apply log1p for numerical stability
                rdm = np.log1p(rdm)

                # Slice the RDM to remove the mirrored part (keep only one half)
                # Assuming the mirrored part is along the Range axis (axis 0)
                half_index = rdm.shape[0] // 2
                rdm = rdm[:half_index, :]

                # Append to the list for the current channel
                rdm_list.append(rdm)

            # Append the result for the current channel
            rdm_all_channels.append(rdm_list)

        return np.array(rdm_all_channels)[0,50:105,:,:]
    
    def manual_correlation(self, pattern, data, start_frame, end_frame):
        pattern_normalized = (pattern - np.mean(pattern)) / np.std(pattern)
        data_segment = data[start_frame:end_frame + 1]  # Plus 1 because the upper bound is exclusive
        data_segment_normalized = (data_segment - np.mean(data_segment)) / np.std(data_segment)

        correlation_score = np.sum(pattern_normalized * data_segment_normalized)

        return correlation_score


    def slide_pattern_over_data(self, pattern, data):
        if data.ndim == 4:
            data = data[0,:,:,:]
        
        # Normalize the pattern and data (z-scoring)
        pattern_mean = np.mean(pattern)
        pattern_std = np.std(pattern)
        # Check if standard deviation is zero and handle it
        pattern_normalized = (pattern - pattern_mean) / pattern_std if pattern_std else pattern - pattern_mean
        
        data_mean = np.mean(data, axis=(1, 2), keepdims=True)
        data_std = np.std(data, axis=(1, 2), keepdims=True)
        # Check if standard deviation is zero and handle it
        data_normalized = (data - data_mean) / (data_std + 1e-8)  # Added a small constant to avoid division by zero
        
        # Check for NaN or inf values
        if np.isnan(data_normalized).any() or np.isinf(data_normalized).any():
            raise ValueError("Data contains NaN or infinite values after normalization.")

        # Perform 3D correlation
        match_scores = correlate(data_normalized, pattern_normalized, mode='valid', method='auto')
        
        # Sum over the spatial dimensions to get a single match score per frame position
        match_scores_summed = match_scores.sum(axis=(1, 2))

                
        # Find peaks in the match scores
        peaks, properties = find_peaks(match_scores_summed)

        # Ensure 'peak_heights' is in properties, otherwise get the height of all peaks
        if 'peak_heights' not in properties:
            properties['peak_heights'] = match_scores_summed[peaks]

        # Sort the peaks by their height
        sorted_peaks = np.argsort(properties['peak_heights'])[::-1]

        # Filter out peaks that are within 100 frames of each other
        filtered_peaks = []
        for peak_index in sorted_peaks:
            peak = peaks[peak_index]
            if not any(abs(peak - p) <= 100 for p in filtered_peaks):
                filtered_peaks.append(peak)
            if len(filtered_peaks) == 2:  # Stop when the top two valid peaks are found
                break

                
        # Return the indices of the peaks as well as the match scores for plotting or further analysis
        return filtered_peaks, match_scores_summed
    
    def slide_normalized_actuator_pattern_over_data(self, data):
        if data.ndim == 4:
            data = data[0,:,:,:]

        data_mean = np.mean(data, axis=(1, 2), keepdims=True)
        data_std = np.std(data, axis=(1, 2), keepdims=True)
        # Check if standard deviation is zero and handle it
        data_normalized = (data - data_mean) / (data_std + 1e-8)  # Added a small constant to avoid division by zero
        
        # print(data_normalized.shape)
        # print(self.normalized_actuator_pattern_in_radar.shape)

        # Check for NaN or inf values
        if np.isnan(data_normalized).any() or np.isinf(data_normalized).any():
            raise ValueError("Data contains NaN or infinite values after normalization.")

        # Perform 3D correlation
        match_scores = correlate(data_normalized, self.normalized_actuator_pattern_in_radar, mode='valid', method='auto')
        
        # Sum over the spatial dimensions to get a single match score per frame position
        match_scores_summed = match_scores.sum(axis=(1, 2))

                
        # Find peaks in the match scores
        peaks, properties = find_peaks(match_scores_summed)

        # Ensure 'peak_heights' is in properties, otherwise get the height of all peaks
        if 'peak_heights' not in properties:
            properties['peak_heights'] = match_scores_summed[peaks]

        # Sort the peaks by their height
        sorted_peaks = np.argsort(properties['peak_heights'])[::-1]

        # Filter out peaks that are within 100 frames of each other
        filtered_peaks = []
        for peak_index in sorted_peaks:
            peak = peaks[peak_index]
            if not any(abs(peak - p) <= 100 for p in filtered_peaks):
                filtered_peaks.append(peak)
            if len(filtered_peaks) == 2:  # Stop when the top two valid peaks are found
                filtered_peaks.sort()
                break

                
        # Return the indices of the peaks as well as the match scores for plotting or further analysis
        return filtered_peaks, match_scores_summed



    def plot_match_scores(self, match_scores):
        """
        Plots the match scores to help identify where the best matches are.

        Args:
            match_scores (np.ndarray): Array of match scores.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(match_scores, marker='o', linestyle='-')
        plt.title('Pattern Match Score Across Frames')
        plt.xlabel('Frame Position')
        plt.ylabel('Match Score')
        plt.grid(True)
        plt.show()


    # def create_gif(self, data, gif_filename):
    #     """
    #     Creates a GIF from the provided data and saves it to the 'data/gifs' directory within the current working directory.

    #     Args:
    #         data (np.ndarray): 3D or 4D array containing the image data.
    #         gif_filename (str): Filename for the GIF, without path.
    #         duration (float): Duration of each frame in the GIF.
    #     """
    #     # Define the directory to save GIFs
    #     gif_dir = os.path.join(os.getcwd(), 'data/gifs')
    #     # Create the directory if it doesn't exist
    #     os.makedirs(gif_dir, exist_ok=True)
    #     # Full path for the GIF
    #     gif_path = os.path.join(gif_dir, gif_filename)
        
    #     if data.ndim == 4:
    #         data = data[0,:,:,:]

    #     with imageio.get_writer(gif_path, mode='I', duration=self.seconds_per_frame) as writer:
    #         if data.ndim == 3:  # If data is 3D, treat it as a sequence of 2D frames
    #             for i in range(data.shape[0]):
    #                 # Convert the data to an image (you might need to scale/normalize)
    #                 frame = data[i, :, :].T
    #                 plt.imshow(frame, cmap='gray')  # Use appropriate colormap
    #                 plt.axis('off')  # Turn off axis
    #                 plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0)
    #                 plt.close()
    #                 writer.append_data(imageio.imread('temp_frame.png'))
    #         # elif data.ndim == 4:  # If data is 4D, process each channel separately
    #         #     for channel in range(data.shape[0]):
    #         #         for i in range(data.shape[1]):
    #         #             frame = data[channel, i, :, :].T
    #         #             plt.imshow(frame, cmap='gray')  # Use appropriate colormap
    #         #             plt.axis('off')  # Turn off axis
    #         #             plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0)
    #         #             plt.close()
    #         #             writer.append_data(imageio.imread('temp_frame.png'))

    #     # Remove temporary frame image file
    #     os.remove('temp_frame.png')

    #     return gif_path  # Return the path where the GIF was saved


    # def create_gif(self, data, gif_filename, fp_data_capture):
    #     # Define the directory to save GIFs
    #     gif_dir = os.path.join(os.getcwd(), 'data/gifs')
    #     # Create the directory if it doesn't exist
    #     os.makedirs(gif_dir, exist_ok=True)
    #     # Full path for the GIF
    #     gif_path = os.path.join(gif_dir, gif_filename)
        
    #     if data.ndim == 4:
    #         data = data[0,:,:,:]
        
    #     # Sort the lift and down frames
    #     lift_frames = sorted(fp_data_capture.foot_lift_frames_after_actuator)
    #     down_frames = sorted(fp_data_capture.foot_down_frames_after_actuator)

    #     # Create an iterator for the down frames
    #     down_frames_iter = iter(down_frames)
    #     next_down_frame = next(down_frames_iter, None)

    #     # Initialize the status as 'Foot Down' and prepare for the first 'Foot Up'
    #     status = 'Foot Down'
    #     next_lift_frame = lift_frames.pop(0) if lift_frames else None

    #     # Iterate through each frame to create the GIF
    #     for i, frame_data in enumerate(data):
    #         frame = Image.fromarray(frame_data)
    #         draw = ImageDraw.Draw(frame)

    #         # Check if it's time to change the status
    #         if next_lift_frame is not None and i >= next_lift_frame:
    #             status = 'Foot Up'
    #             next_down_frame = next(down_frames_iter, None)
    #             next_lift_frame = lift_frames.pop(0) if lift_frames else None
    #         elif next_down_frame is not None and i >= next_down_frame:
    #             status = 'Foot Down'

    #         # Annotate the frame with the current status
    #         draw.text((10, 10), status, fill="white")

    #         if i == 0 or status == 'Foot Up':
    #             # Save the first frame or 'Foot Up' frames with longer duration
    #             writer.append_data(np.array(frame), duration=fp_data_capture.seconds_per_frame)
    #         else:
    #             writer.append_data(np.array(frame))

    #     # Remove temporary frame image file
    #     os.remove('temp_frame.png')

    #     return gif_path  # Return the path where the GIF was saved
    
    
    def create_gif(self, data, gif_filename, fp_data_capture):
        """
        Creates a GIF from the provided data and saves it to the 'data/gifs' directory within the current working directory.
        It also prints "Foot Down" initially on the GIF, then changes to "Foot Up" at the frame number for foot up,
        then "Foot Down" at the next foot down, and so on based on the attributes 
        fp_data_capture.foot_lift_frames_after_actuator and fp_data_capture.foot_down_frames_after_actuator.

        Args:
            data (np.ndarray): 3D array containing the image data.
            gif_filename (str): Filename for the GIF, without path.
            fp_data_capture (FPDataCapture): FPDataCapture object with foot lift and down frames attributes.
        """
        # Define the directory to save GIFs
        gif_dir = os.path.join(os.getcwd(), 'data/gifs')
        # Create the directory if it doesn't exist
        os.makedirs(gif_dir, exist_ok=True)
        # Full path for the GIF
        gif_path = os.path.join(gif_dir, gif_filename)
        
        if data.ndim == 4:
            data = data[0,:,:,:]
        
        # Initialize the writer
        with imageio.get_writer(gif_path, mode='I', duration=fp_data_capture.seconds_per_frame) as writer:
            status = 'Foot Down'  # Initial status
            for i in range(data.shape[0]):
                frame = data[i, :, :].T  # Assuming data is 3D, with shape (frames, height, width)
                        # Calculate zero velocity column and quarter distance
                zero_vel = int(frame.shape[0] / 2)
                quarter_distance = int(frame.shape[1] / 4)
                
                # Extracting the first quarter on the x-axis (after transpose)
                first_quarter_frame = frame[:, :quarter_distance]
                
                # Calculate start and end indices for extracting the middle 23 values on the y-axis
                mid_start = zero_vel - 11  # Middle minus half of 23 (to center the 23 values)
                mid_end = zero_vel + 12  # Middle plus half of 23
                
                # Extracting the middle 23 values on the y-axis
                middle_23_values_frame = first_quarter_frame[mid_start:mid_end, :]
                plt.figure(figsize=(4, 4))
                plt.imshow(middle_23_values_frame, cmap='gray')
                plt.text(5, 5, status, color='white', fontsize=20)  # Position text at top-left
                plt.axis('off')
                
                # Check and update the status based on frame number
                if i in fp_data_capture.foot_lift_frames_after_actuator:
                    status = 'Foot Up'
                elif i in fp_data_capture.foot_down_frames_after_actuator:
                    status = 'Foot Down'
                
                # Save the annotated frame to a temporary file
                plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0)
                plt.close()
                
                # Append the frame to the GIF
                writer.append_data(imageio.imread('temp_frame.png'))
        
        # Remove the temporary frame image file
        os.remove('temp_frame.png')
        
        # Return the path where the GIF was saved
        return gif_path

    def process_and_save_channels_tx_separately(self, data, output_folder_path, file_name):
        # Define transition times in seconds
        transition_times = [
            (8.5, 11.5),
            (16.5, 19.5),
            (24.5, 27.5)
        ]
        
        # Calculate time per frame assuming 36 seconds for 1000 frames
        time_per_frame = 36 / 1000
        
        # Calculate frame indices for each transition
        transition_frames = [
            (round(transition_time[0] / time_per_frame), round(transition_time[1] / time_per_frame))
            for transition_time in transition_times
        ]
        
        # Ensure the output folder exists
        specific_output_folder_path = os.path.join(output_folder_path, file_name[:2])
        if not os.path.exists(specific_output_folder_path):
            os.makedirs(specific_output_folder_path)
        
        # Process and save each channel for each transition period
        for channel_idx in range(data.shape[0]):
            for i, (start_frame, end_frame) in enumerate(transition_frames):
                # Initialize an empty list to collect processed frames for the current transition
                processed_frames = []
                
                # Extract and process relevant frames for the current transition
                for frame_idx in range(start_frame, end_frame):
                    frame = data[channel_idx, frame_idx, :, :].T  # Transpose the frame
                    
                    # Calculate zero velocity column and quarter distance
                    zero_vel = int(frame.shape[0] / 2)
                    quarter_distance = int(frame.shape[1] / 4)
                    
                    # Extracting the first quarter on the x-axis (after transpose)
                    first_quarter_frame = frame[:, :quarter_distance]
                    
                    # Calculate start and end indices for extracting the middle 23 values on the y-axis
                    mid_start = zero_vel - 11
                    mid_end = zero_vel + 12
                    
                    # Extract the middle 23 values on the y-axis from the first quarter
                    processed_frame = first_quarter_frame[mid_start:mid_end, :]
                    
                    # Append the processed frame to the list
                    processed_frames.append(processed_frame)
                
                # Convert the list of processed frames into a 3D numpy array
                processed_frames_array = np.array(processed_frames)[:83,:,:]
                
                # Define the output file name for the processed data of the current transition
                output_filename = os.path.join(
                    specific_output_folder_path,
                    f"{file_name}_channel_{channel_idx+1}_tx{i+1}.npy"
                )
                
                # Save the 3D array to a binary file in NumPy .npy format
                np.save(output_filename, processed_frames_array)
        
        print(f"{file_name} processed and saved")
        
    def sub_select_RADAR_DATA(self, data):
        """
        Subselects the middle 23 values on the y-axis from the first quarter of the radar data.

        Parameters:
        - data: A 4D numpy array representing the radar data.

        Returns:
        - A 4D numpy array with the specified segment extracted and limited along the last axis.
        """
        # Initialize an empty list to collect processed frames for the current transition
        processed_data = []
       
        num_channels, num_frames, height, width = data.shape
        
        # Extract and process relevant frames for the current transition
        for channel_idx in range(num_channels):
            processed_frames = []
            for frame_idx in range(num_frames):
                frame = data[channel_idx, frame_idx, :, :].T  # Transpose the frame
                
                # Calculate zero velocity column and quarter distance
                zero_vel = int(data.shape[2] / 2)
                
                # Calculate start and end indices for extracting the middle 23 values on the y-axis
                mid_start = zero_vel - 11
                mid_end = zero_vel + 12
                
                # Extract the middle 23 values on the y-axis, first 13 values along x axis
                processed_frame = frame[mid_start:mid_end, :13]
                        
                # Append the processed frame to the list
                processed_frames.append(processed_frame)
            processed_data.append(processed_frames)     
        
        return np.array(processed_data)
        
        
    def process_and_save_FUFD_RADAR_data(self, data, start_end_frames, output_folder_path, file_name):
        
        start_frame, end_frame = start_end_frames

        # Ensure the output folder exists
        specific_output_folder_path = os.path.join(output_folder_path, file_name[:2])
        if not os.path.exists(specific_output_folder_path):
            os.makedirs(specific_output_folder_path)
        
        processed_frames = data[:,start_frame:end_frame,:,:]
        
        # Convert the list of processed frames into a 3D numpy array
        processed_frames_array = np.array(processed_frames)
        
        print(processed_frames_array.shape)
        
        # Define the output file name for the processed data of the current transition
        output_filename = os.path.join(
            specific_output_folder_path,
            f"{file_name}.npy"
        )
        
        # Save the 3D array to a binary file in NumPy .npy format
        np.save(output_filename, processed_frames_array)
        
        print(f"{file_name} processed and saved")

    def process_and_save_channels_separately(self, data, output_folder_path, file_name):

            # Ensure the output folder exists
            specific_output_folder_path = os.path.join(output_folder_path, file_name[:2])
            if not os.path.exists(specific_output_folder_path):
                os.makedirs(specific_output_folder_path)
            
            # Process and save each channel for each transition period
            for channel_idx in range(data.shape[0]):
                
                # Initialize an empty list to collect processed frames for the current transition
                processed_frames = []
                
                # Extract and process relevant frames for the current transition
                for frame_idx in range(data.shape[1]):
                    frame = data[channel_idx, frame_idx, :, :].T  # Transpose the frame
                    
                    # Calculate zero velocity column and quarter distance
                    zero_vel = int(frame.shape[0] / 2)
                    quarter_distance = int(frame.shape[1] / 4)
                    
                    # Extracting the first quarter on the x-axis (after transpose)
                    first_quarter_frame = frame[:, :quarter_distance]
                    
                    # Calculate start and end indices for extracting the middle 23 values on the y-axis
                    mid_start = zero_vel - 11
                    mid_end = zero_vel + 12
                    
                    # Extract the middle 23 values on the y-axis from the first quarter
                    processed_frame = first_quarter_frame[mid_start:mid_end, :]
                    
                    # Append the processed frame to the list
                    processed_frames.append(processed_frame)
                
                # Convert the list of processed frames into a 3D numpy array
                processed_frames_array = np.array(processed_frames)
                
                # Define the output file name for the processed data of the current transition
                output_filename = os.path.join(
                    specific_output_folder_path,
                    f"{file_name}_channel_{channel_idx+1}.npy"
                )
                
                # Save the 3D array to a binary file in NumPy .npy format
                np.save(output_filename, processed_frames_array)
            
            print(f"{file_name} processed and saved")



