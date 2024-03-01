import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, LSTM, TimeDistributed, Reshape, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import imageio
import numpy as np
import os
from PIL import Image, ImageDraw



def generate_gif_from_radar_single_channel_larger_image(radar_data, gif_path='radar_visualization_single_channel_large_image2.gif', duration=0.5, scale_factor=10):
    """
    Generates a GIF from RADAR data over time, for a single channel, with enlarged images.
    
    Parameters:
    - radar_data: RADAR data with shape (depth, height, width) for a single channel.
    - gif_path: Path to save the generated GIF.
    - duration: Duration of each frame in the GIF.
    - scale_factor: Factor by which to scale the image size.
    """
    
    depth, height, width = radar_data.shape
    frames = []  # List to hold generated frames

    # Loop through each time step (depth) to generate frames
    for time_step in range(depth):
        # Extract the frame for the current time step
        current_frame = radar_data[time_step, :, :]
        
        # Convert to PIL Image for drawing
        img = Image.fromarray(np.uint8(current_frame * 255))  # Normalize assuming data is in [0, 1]
        
        # Resize the image by the scale factor
        img = img.resize((width * scale_factor, height * scale_factor), Image.NEAREST)
        
        draw = ImageDraw.Draw(img)
        
        # Add frame number to the image
        draw.text((10, 10), f'Frame: {time_step}', fill='white')
        
        frames.append(img)
    
    # Save frames as a GIF
    imageio.mimsave(gif_path, frames, 'GIF', duration=duration)

    print(f'GIF saved at: {gif_path}')


def pad_radar_sequences(sequences, maxlen, padding='post', dtype='float16'):
    """
    Pad 4D RADAR sequences to the same length.

    Parameters:
    - sequences: List of 4D arrays with shape (channels, sequence_length, height, width).
    - maxlen: Maximum length to pad the sequences to.
    - padding: 'pre' or 'post' padding.
    - dtype: Data type of the output array.
    
    Returns:
    - Padded 4D sequences.
    """
    sample_shape = sequences[0].shape  # (channels, sequence_length, height, width)
    num_channels, height, width = sample_shape[0], sample_shape[2], sample_shape[3]
    
    # Initialize the padded array
    padded_sequences = np.zeros((len(sequences), num_channels, maxlen, height, width), dtype=dtype)
    
    for idx, seq in enumerate(sequences):
        # print(f'BEFORE: {seq.shape}')
        sequence_length = seq.shape[1]
        if padding == 'post':
            padded_sequences[idx, :, :sequence_length, :, :] = seq
        elif padding == 'pre':
            padded_sequences[idx, :, -sequence_length:, :, :] = seq
        # print(f'After: {padded_sequences.shape}')
    
    return padded_sequences

def pad_fp_sequences(sequences, maxlen, padding='post', dtype='float16'):
    """
    Pad 1D FP sequences to the same length.

    Parameters:
    - sequences: List of 1D arrays with shape (sequence_length,).
    - maxlen: Maximum length to pad the sequences to.
    - padding: 'pre' or 'post' padding.
    - dtype: Data type of the output array.
    
    Returns:
    - Padded 1D sequences.
    """
    # Initialize the padded array
    padded_sequences = np.zeros((len(sequences), maxlen), dtype=dtype)
    
    for idx, seq in enumerate(sequences):
        sequence_length = len(seq)
        if padding == 'post':
            padded_sequences[idx, :sequence_length] = seq
        elif padding == 'pre':
            padded_sequences[idx, -sequence_length:] = seq
    
    return padded_sequences


def get_files_from_dir(base_path, extension='.npy'):
    """
    Traverse through the directory and its subdirectories 
    to find files with the specified extension.
    """
    files = []
    for root, dirs, files_in_dir in os.walk(base_path):
        for file in files_in_dir:
            if file.endswith(extension):
                files.append(os.path.join(root, file))
    return files

# Function to load and pad the data
def load_and_pad_data(files_fp, files_radar, max_length):
    # Load FP data as float16, pad, and reshape to match the required input shape
    fp_sequences = [np.load(file).astype('float16').flatten() for file in files_fp]
    fp_data_padded = pad_fp_sequences(fp_sequences, maxlen=max_length, padding='post')

    # Load RADAR data as float16 and pad
    radar_sequences = [np.load(file).astype('float16') for file in files_radar]
    radar_data_padded = pad_radar_sequences(radar_sequences, maxlen=max_length, padding='post')

    return fp_data_padded, radar_data_padded


# Define the base paths for force plate (FP) and RADAR data
base_path_fp = '/Volumes/FourTBLaCie/Yoga_Study_FP_FUFD'
base_path_radar = '/Volumes/FourTBLaCie/Yoga_Study_RADAR_4Ch_FUFD'

# Get the list of .npy files in each base path including subfolders
files_fp = get_files_from_dir(base_path_fp)
files_radar = get_files_from_dir(base_path_radar)

# Extract base filenames for matching
fp_filenames = [os.path.basename(fp) for fp in files_fp]
radar_filenames = [os.path.basename(radar) for radar in files_radar]

# Assuming that the file naming convention is consistent and each FP file has a corresponding RADAR file
matched_files = [(fp, radar) for fp, fp_filename in zip(files_fp, fp_filenames) 
                 for radar, radar_filename in zip(files_radar, radar_filenames) 
                 if radar_filename == fp_filename] 

# Calculate max_length based on FP data
max_length = max([np.load(fp).shape[0] for fp, _ in matched_files])

# Assuming max_length is based on RADAR data's time dimension
# You might also need to calculate a separate max_length for FP data if its lengths vary significantly
fp_data_padded, radar_data_padded = load_and_pad_data(
    [fp for fp, _ in matched_files],
    [radar for _, radar in matched_files],
    max_length
)

generate_gif_from_radar_single_channel_larger_image(radar_data_padded[0][1])


import matplotlib.pyplot as plt

# Assuming fp_data_padded is a numpy array with the padded force plate data
# Plot the first sample's time series data
plt.figure(figsize=(12, 6))
plt.plot(fp_data_padded[10], label='COP_X Time Series')
plt.title('COP_X Time Series for the 11th Sample')
plt.xlabel('Time Steps')
plt.ylabel('COP_X Value')
plt.legend()
plt.show()


print(f"Shape of padded FP data: {fp_data_padded.shape}")
print(f"Shape of padded RADAR data: {radar_data_padded.shape}")



# # Load and pad the data
# fp_data_padded = pad_fp_sequences([np.load(fp).flatten() for fp, _ in matched_files], maxlen=max_length, padding='post')
# radar_data_padded = pad_radar_sequences([np.load(radar) for _, radar in matched_files], maxlen=max_length, padding='post')

# Split the pairs into train and test sets
train_fp, test_fp, train_radar, test_radar = train_test_split(fp_data_padded, radar_data_padded, test_size=0.2, random_state=42)

# Output the number of items in train and test sets
print(f"Training items: {len(train_fp)}")
print(f"Testing items: {len(test_fp)}")

def create_model_for_cop_x(input_shape, output_sequence_length):
    input_layer = Input(shape=input_shape)
    
    # Slightly reduced number of filters
    x = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Conv3D(filters=24, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)  # Reduced from 32 to 24
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    
    # Reduced units in the Dense layer
    x = Dense(output_sequence_length * 48, activation='relu')(x)  # Reduced from 64 to 48
    x = Dropout(0.3)(x)  # Reduced dropout rate from 0.5 to 0.3
    
    x = Reshape((output_sequence_length, -1))(x)
    
    # Slightly reduced LSTM units
    x = LSTM(48, return_sequences=True)(x)  # Reduced from 64 to 48
    x = Dropout(0.3)(x)  # Consistent dropout rate reduction
    
    output_layer = TimeDistributed(Dense(1, activation='linear'))(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Optimizer with adjusted learning rate if needed
    optimizer = Adam(learning_rate=0.01)  # Adjust learning rate if necessary
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    return model, [reduce_lr, early_stopping]

# Assuming radar_data is your input RADAR RDMs with shape [num_samples, 4, 151, 23, 13]
# Assuming cop_x_data is your target COP_X data with shape [num_samples, 151, 1]

model, callbacks = create_model_for_cop_x((4, 151, 23, 13), 151)
history = model.fit(
    train_radar, 
    train_fp, 
    epochs=100, 
    batch_size=16, 
    validation_split=0.2, 
    verbose=1, 
    callbacks=callbacks
)

test_loss = model.evaluate(test_radar, test_fp, verbose=1)

print(f"Test loss: {test_loss}")

predictions = model.predict(test_radar)
# Now, you can compare 'predictions' with 'test_fp'

import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
