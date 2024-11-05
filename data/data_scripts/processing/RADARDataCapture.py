import h5py
import numpy as np
import os
import imageio
from datetime import datetime
from scipy.signal import find_peaks, correlate, spectrogram, iirnotch, filtfilt, butter
from scipy.fft import fft, fftshift
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
        self.central_frequency = None
        self.bandwidth = None
        self.fs = 1e6

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
            self.bandwidth = (FreqStop - FreqStrt) / 284 * 256
            
            self.central_frequency = (FreqStop+FreqStrt)/2
            
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


    def synthesize_rdm(self, dataCubes, max_range=8, velocity_bins_to_keep=25, antenna_spacing=None, plot=False, save_gif=True, gif_path="test_8_25_rdm_animation.gif"):
        """
        Synthesizes the multi-channel radar data into a series of Range-Doppler Maps (RDM),
        slices the data to focus on the first 8 meters and the middle 25 velocity bins, and saves as a GIF.
        
        Args:
            dataCubes (np.ndarray): Radar data cubes of shape 
                (n_channels, n_frames, n_samples_per_chirp, n_chirps_per_frame).
            max_range (float): Maximum range to include in the RDM (default is 8 meters).
            velocity_bins_to_keep (int): Number of velocity bins to retain around the center (default is 25).
            plot (bool): If True, plots the synthesized RDM for each frame.
            save_gif (bool): If True, saves the RDM frames as a GIF.
            gif_path (str): File path for saving the GIF.

        Returns:
            list: A list of synthesized and sliced RDMs (one per frame), range_bins (1D np.ndarray), velocity_bins (1D np.ndarray).
        """
        n_channels, n_frames, n_samples_per_chirp, n_chirps_per_frame = dataCubes.shape
        c = 3e8  # Speed of light in m/s

        # Perform range FFT (fast time FFT) along the third axis
        range_fft = fft(dataCubes, axis=2)

        # Keep only the first half of the FFT results (positive frequencies)
        range_fft = range_fft[:, :, :n_samples_per_chirp // 2, :]

        # Perform Doppler FFT (slow time FFT) along the fourth axis
        doppler_fft = fftshift(fft(range_fft, axis=3), axes=3)

        # Initialize a list to hold the synthesized and sliced RDMs for each frame and image frames for GIF
        synthesized_rdm_list = []
        img_frames = []

        # Calculate range resolution and velocity resolution
        range_resolution = c / (2 * self.bandwidth)
        velocity_resolution = c / (2 * self.central_frequency * n_chirps_per_frame * (1 / self.fs))

        # Calculate range bins and velocity bins before slicing
        range_bins = np.linspace(0, (n_samples_per_chirp // 2) * range_resolution, n_samples_per_chirp // 2)
        velocity_bins = np.linspace(-n_chirps_per_frame / 2 * velocity_resolution, 
                                    n_chirps_per_frame / 2 * velocity_resolution, n_chirps_per_frame)

        # Determine the index corresponding to the 8-meter range limit
        max_range_idx = np.argmax(range_bins > max_range)

        # Determine the center velocity index and the velocity bin slice
        center_velocity_idx = n_chirps_per_frame // 2
        velocity_slice_start = max(center_velocity_idx - velocity_bins_to_keep // 2, 0)
        velocity_slice_end = min(center_velocity_idx + velocity_bins_to_keep // 2, n_chirps_per_frame)

        # Process each frame individually
        for frame_idx in range(n_frames):
            # Get data for the current frame and average across channels
            rdm_for_frame = np.mean(np.abs(doppler_fft[:, frame_idx, :, :]), axis=0)  # Averaging over channels

            # Slice the RDM to focus on the first 8 meters and the middle velocity bins
            rdm_sliced = rdm_for_frame[:max_range_idx, velocity_slice_start:velocity_slice_end]

            # Append the sliced RDM for the current frame to the list
            synthesized_rdm_list.append(rdm_sliced)

            # Plot the sliced RDM for the current frame
            plt.figure(figsize=(10, 8))
            plt.imshow(20 * np.log10(rdm_sliced + 1e-6), aspect='auto', 
                       extent=[velocity_bins[velocity_slice_start], velocity_bins[velocity_slice_end-1], 
                               range_bins[max_range_idx-1], range_bins[0]],
                       cmap='jet')
            plt.colorbar(label='Magnitude (dB)')
            plt.title(f'Sliced Range-Doppler Map (RDM) - Frame {frame_idx+1}')
            plt.xlabel('Velocity (m/s)')
            plt.ylabel('Range (m)')
            plt.grid(True)

            # Save each plot as an image for GIF
            if save_gif:
                # Save the figure as an image to a temporary file
                plt.savefig(f"frame_{frame_idx}.png", format='png')
                img_frames.append(Image.open(f"frame_{frame_idx}.png"))

            # Close the figure to avoid memory leaks
            plt.close()

        # Save GIF if requested
        if save_gif and img_frames:
            img_frames[0].save(gif_path, save_all=True, append_images=img_frames[1:], duration=200, loop=0)

        return synthesized_rdm_list, range_bins[:max_range_idx], velocity_bins[velocity_slice_start:velocity_slice_end]
    
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
            
    def compute_microdoppler_signature(self, dataCubes, range_bins, nfft=256, fs = 128000 / 36.352, plot=False):
        """
        Compute micro-Doppler signatures at specified range bins.

        Args:
            dataCubes (np.ndarray): Radar data cubes of shape (n_channels, n_frames, n_samples_per_chirp, n_chirps_per_frame)
            range_bins (list or array): List of range bins for which to compute micro-Doppler signatures.
            nfft (int): Number of FFT points for Doppler FFT.
            fs (float): Sampling frequency in Hz.
            plot (bool): If True, plot the micro-Doppler spectrograms.

        Returns:
            dict: Dictionary containing the micro-Doppler spectrograms for each channel and range bin.
        """
        
        # Get data dimensions
        n_channels, n_frames, n_samples_per_chirp, n_chirps_per_frame = dataCubes.shape
    
        # Initialize dictionary to hold results
        microdoppler_signatures = {}

        for channel_idx in range(n_channels):

            microdoppler_signatures[channel_idx] = {}

            # Initialize list to hold range profiles over time
            range_profiles_over_time = []

            # Process data over frames
            for frame_idx in range(n_frames):

                # Extract data for current frame and channel
                current_data = dataCubes[channel_idx, frame_idx, :, :]  # Shape: (n_samples_per_chirp, n_chirps_per_frame)

                # Perform FFT over samples to get range profiles
                range_profiles = np.fft.fft(current_data, axis=0)  # Shape: (n_samples_per_chirp, n_chirps_per_frame)
                range_profiles = np.abs(range_profiles)

                # Append range profiles to the list
                range_profiles_over_time.append(range_profiles)

            # Stack range profiles over time
            range_profiles_over_time = np.concatenate(range_profiles_over_time, axis=1)  # Shape: (n_samples_per_chirp, n_frames * n_chirps_per_frame)

            for range_bin in range_bins:

                # Extract data at the specified range bin over time
                data_at_range_bin = range_profiles_over_time[range_bin, :]  # Shape: (n_frames * n_chirps_per_frame,)

                f, t, Sxx = spectrogram(
                    data_at_range_bin,
                    fs=fs,  # Use the correct sampling frequency
                    window='hann',
                    nperseg=nfft,
                    noverlap=int(nfft * 0.5),
                    nfft=nfft,
                    scaling='spectrum',
                    mode='magnitude'
                )

                # Store the spectrogram
                microdoppler_signatures[channel_idx][range_bin] = (f, t, Sxx)

                if plot:
                    # Plot the spectrogram
                    plt.figure(figsize=(10, 6))
                    plt.pcolormesh(t, f, 20 * np.log10(Sxx + 1e-10), shading='gouraud')
                    plt.title(f'Micro-Doppler Spectrogram - Channel {channel_idx}, Range Bin {range_bin}')
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [s]')
                    plt.colorbar(label='Amplitude [dB]')
                    plt.show()

        return microdoppler_signatures


    def compute_magnitude_variation_by_bin(self, capture_name, tx, dataCubes, range_bins, plot=False):
        """
        Compute and plot the amplitude (magnitude) variations over time at specified range bins, with both 
        notch and low-pass filtering applied.

        Args:
            dataCubes (np.ndarray): Radar data cubes of shape 
                (n_channels, n_frames, n_samples_per_chirp, n_chirps_per_frame).
            range_bins (list or array): List of range bins for which to compute displacement-like magnitude variation.
            plot (bool): If True, plot the magnitude waveform.

        Returns:
            dict: Dictionary containing times and magnitude variations for each channel and range bin.
        """

        def apply_notch_filter(data, freq, fs, Q=50):
            # Design and apply notch filter to remove 27 Hz frequency component
            w0 = freq / (fs / 2)  # Normalize frequency
            b, a = iirnotch(w0, Q)
            filtered_data = filtfilt(b, a, data)
            return filtered_data

        def apply_lowpass_filter(data, cutoff, fs, order=4):
            # Design and apply low-pass filter to remove frequencies above 25 Hz
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            filtered_data = filtfilt(b, a, data)
            return filtered_data

        n_channels, n_frames, n_samples_per_chirp, n_chirps_per_frame = dataCubes.shape
        T_total = n_frames * self.seconds_per_frame  # Total recording time

        displacements = {}

        for channel_idx in range(n_channels):
            displacements[channel_idx] = {}
            magnitude_over_time = []  # To store magnitude data over time

            # Process data over frames
            for frame_idx in range(n_frames):
                current_data = dataCubes[channel_idx, frame_idx, :, :]  # Shape: (n_samples_per_chirp, n_chirps_per_frame)

                for chirp_idx in range(n_chirps_per_frame):
                    current_chirp = current_data[:, chirp_idx]  # Get one chirp (one column)

                    for range_bin in range_bins:
                        if range_bin >= current_chirp.size:
                            print(f"Range bin {range_bin} exceeds the size of the chirp data. Skipping this range bin.")
                            continue

                        # Extract magnitude for the specified range bin (since data is real, no need for complex magnitude)
                        magnitude = np.abs(current_chirp[range_bin])
                        magnitude_over_time.append(magnitude)

            # Stack the magnitude data for analysis
            if magnitude_over_time:
                magnitude_over_time = np.array(magnitude_over_time)  # Shape: (n_frames * n_chirps_per_frame,)

                # Time vector
                N_samples = len(magnitude_over_time)
                t = np.linspace(0, T_total, N_samples)
                fs = 1 / (t[1] - t[0])  # Sampling frequency

                # Apply notch filter at 27 Hz
                magnitude_filtered = apply_notch_filter(magnitude_over_time, 27.0, fs)

                # Apply low-pass filter below 25 Hz
                magnitude_filtered = apply_lowpass_filter(magnitude_filtered, 25.0, fs)

                # Store the magnitude variation data
                displacements[channel_idx][range_bins[0]] = (t, magnitude_filtered)

                if plot:
                    # Plot the magnitude waveform after filtering
                    plt.figure(figsize=(10, 6))
                    plt.plot(t, magnitude_filtered)
                    plt.title(f'Magnitude Variation for {capture_name}, tx {tx} - Channel {channel_idx}, Range Bin {range_bins[0]}')
                    plt.xlabel('Time [s]')
                    plt.ylabel('Magnitude')
                    plt.grid(True)
                    plt.show()
            else:
                print(f"No valid magnitude data to process for channel {channel_idx}.")

        return displacements

    def unwrapped_microdoppler_phase_spectrogram(self, dataCubes, range_bins, nfft=1024, plot=False):
        """
        Compute and plot the unwrapped phase of micro-Doppler signatures at specified range bins.

        Args:
            dataCubes (np.ndarray): Radar data cubes of shape 
                (n_channels, n_frames, n_samples_per_chirp, n_chirps_per_frame).
            range_bins (list or array): List of range bins for which to compute micro-Doppler phases.
            nfft (int): Number of FFT points for Doppler FFT.
            plot (bool): If True, plot the unwrapped phase of the micro-Doppler signatures.

        Returns:
            dict: Dictionary containing frequencies, times, and unwrapped phases for each channel and range bin.
        """

        n_channels, n_frames, n_samples_per_chirp, n_chirps_per_frame = dataCubes.shape
        T_total = n_frames * self.seconds_per_frame  # e.g., 1000 frames * 0.036352 s/frame

        microdoppler_phases = {}

        for channel_idx in range(n_channels):
            microdoppler_phases[channel_idx] = {}
            range_profiles_over_time = []

            # Process data over frames
            for frame_idx in range(n_frames):
                current_data = dataCubes[channel_idx, frame_idx, :, :]  # Shape: (256, 128)
                range_profiles = np.fft.fft(current_data, axis=0)  # Shape: (256, 128)
                range_profiles_over_time.append(range_profiles)  # Each element is (256, 128)

            # Concatenate over time
            range_profiles_over_time = np.concatenate(range_profiles_over_time, axis=1)  # Shape: (256, 128, n_frames)
            range_profiles_over_time = range_profiles_over_time.reshape(256, -1)  # Shape: (256, 128 * n_frames)

            for range_bin in range_bins:
                data_at_range_bin = range_profiles_over_time[range_bin, :]  # Shape: (N_samples,)
                N_samples = len(data_at_range_bin)
                fs = N_samples / T_total

                nperseg = min(nfft, N_samples)
                noverlap = int(nperseg * 0.5)

                f, t, Sxx = spectrogram(
                    data_at_range_bin,
                    fs=fs,
                    window='hann',
                    nperseg=nperseg,
                    noverlap=noverlap,
                    nfft=nperseg,
                    scaling='spectrum',
                    mode='complex'
                )

                # Extract phase and unwrap
                phase = np.angle(Sxx)
                unwrapped_phase = np.unwrap(phase, axis=0)  # Unwrap along frequency axis

                # Store the unwrapped phase
                microdoppler_phases[channel_idx][range_bin] = (f, t, unwrapped_phase)

                if plot:
                    # Plot the unwrapped phase
                    plt.figure(figsize=(10, 6))
                    plt.pcolormesh(t, f, unwrapped_phase, shading='gouraud')
                    plt.title(f'Unwrapped Phase - Channel {channel_idx}, Range Bin {range_bin}')
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [s]')
                    plt.colorbar(label='Phase [radians]')
                    plt.show()

        return microdoppler_phases

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

                # Log scaling - apply log1p for numerical stability
                aoa_normalized_log = np.log1p(aoa_normalized)

                # Append to the list for the current channel
                aoa_list.append(aoa_normalized_log)

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
        
    def sub_select_AoA_DATA(self, data):
        """
        Subselects a central box of 23 rows and 13 columns from the AoA data.

        Parameters:
        - data: A 4D numpy array representing the radar data.

        Returns:
        - A 4D numpy array with the central 23x13 segment extracted from each frame.
        """
        # Initialize an empty list to collect processed frames for the current transition
        processed_data = []
        
        num_channels, num_frames, height, width = data.shape
        
        # Calculate the central starting and ending indices for both dimensions
        central_row_start = (height - 23) // 2
        central_row_end = central_row_start + 23
        central_col_start = (width - 13) // 2
        central_col_end = central_col_start + 13
        
        # Extract and process relevant frames for the current transition
        for channel_idx in range(num_channels):
            processed_frames = []
            for frame_idx in range(num_frames):
                # No need to transpose in this case, as we're selecting based on actual dimensions
                frame = data[channel_idx, frame_idx, :, :]
                
                # Extract the central 23x13 box
                processed_frame = frame[central_row_start:central_row_end, central_col_start:central_col_end]
                        
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

