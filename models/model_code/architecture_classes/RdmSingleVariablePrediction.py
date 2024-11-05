import torch
import torch.nn as nn
import torch.nn.functional as F

class RdmCNNLSTMModel(nn.Module):
    def __init__(self, num_channels, hidden_dim, lstm_layers=1, bidirectional=False):
        super(RdmCNNLSTMModel, self).__init__()
        self.num_channels = num_channels

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.cnn_output_size = self._get_conv_output_size()

        # LSTM layers
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=self.cnn_output_size, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        
        # Linear layer for output
        direction_multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_multiplier, 1)  # Predicting a single velocity value
        
    def forward(self, x):
        # Reshape output for LSTM layers
        batch_size, time_steps, height, width = x.shape
        x = x.view(batch_size * time_steps, 1, height, width)
        
        # Apply convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Reshape x back to [batch_size, time_steps, features] for LSTM processing
        x = x.view(batch_size, time_steps, self.cnn_output_size)
        
        # LSTM layers...
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last LSTM layer
        if self.bidirectional:
            lstm_out = lstm_out[:, -1, :]
        else:
            lstm_out = lstm_out[:, -1, :]
        
        # Linear layer
        out = self.fc(lstm_out)
        outputs = torch.squeeze(out)
        return outputs

    def _get_conv_output_size(self):
        # Create a dummy input to pass through the CNN layers to calculate output size
        # Assuming the spatial dimensions of your input radar data are 23x13
        dummy_input = torch.zeros(1, self.num_channels, 23, 13)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        # Multiply the dimensions of the output feature map to get the total feature size
        return x.numel() // x.shape[0]  # Use numel() to get total number of features and divide by batch size (1 in this case)
