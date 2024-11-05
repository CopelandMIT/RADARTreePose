import torch
import torch.nn as nn
import torch.nn.functional as F

class COPSpeedPredictionByRDMFrameCNNLSTMModel(nn.Module):
    def __init__(self, num_channels, hidden_dim, lstm_layers=1, bidirectional=False):
        super(COPSpeedPredictionByRDMFrameCNNLSTMModel, self).__init__()
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
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Linear layer for output
        direction_multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_multiplier, 1)  # Predicting speed per frame

    def forward(self, x):
        # x shape: [batch_size, seq_len, num_channels, height, width]
        batch_size, seq_len, num_channels, height, width = x.shape

        # Reshape for CNN input
        x = x.view(batch_size * seq_len, num_channels, height, width)
        
        # Apply convolutional layers
        x = self.pool(F.relu(self.conv1(x)))  # [batch_size * seq_len, 16, h, w]
        x = self.pool(F.relu(self.conv2(x)))  # [batch_size * seq_len, 32, h', w']
        
        # Flatten and reshape for LSTM input
        x = x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, cnn_output_size]
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim * direction_multiplier]
        
        # Fully connected layer to get speed predictions
        out = self.fc(lstm_out)  # [batch_size, seq_len, 1]
        out = out.squeeze(-1)    # [batch_size, seq_len]
        
        return out  # Output speed for each frame

    def _get_conv_output_size(self):
        # Create a dummy input with the correct number of channels
        dummy_input = torch.zeros(1, self.num_channels, 23, 13)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(1, -1).size(1)
