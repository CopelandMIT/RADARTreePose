import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch import optim

# Define the model
class RdmClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(RdmClassifier, self).__init__()
        self.num_classes = num_classes  # Add this line
        # Define a simple CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Assuming RDMs have a single channel
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),  # This will flatten the output of the convolutional layers
        )
        cnn_output_size = self._get_conv_output_size()

        # Define the LSTM layer
        self.lstm = nn.LSTM(cnn_output_size, hidden_size, batch_first=True)
        
        # Define the fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        x = x.float()  # Ensure input is float type
        batch_size, seq_len, _, _ = x.size()
        # Apply CNN to each RDM in the sequence
        c_out = self.cnn(x.view(batch_size * seq_len, 1, *x.size()[-2:]))
        
        # Reshape for LSTM input
        r_out = c_out.view(batch_size, seq_len, -1)
        
        # Pack the sequence for LSTM
        packed_input = pack_padded_sequence(r_out, lengths, batch_first=True, enforce_sorted=False)
        # Instead of using just the last hidden state
        packed_output, (hidden, cell) = self.lstm(packed_input)
        # Decode the packed output
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Apply the fully connected layer to all time steps
        out = self.fc(lstm_out)
        return out
    
    def _get_conv_output_size(self):
        # We can create a dummy input to calculate the output size after the CNN layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 23, 13)  # Replace with your RDM shape
            dummy_output = self.cnn(dummy_input)
            return dummy_output.size(-1)

