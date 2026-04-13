import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CrimeNet(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, num_classes=4, dropout=0.5):
        super(CrimeNet, self).__init__()
        # Load ResNet18 with updated weights syntax
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # Remove final classification layer
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()              # Batch, Time, Channels, Height, Width
        x = x.view(-1, c, h, w)               # Combine batch and time for CNN

        with torch.no_grad():                # Freeze CNN during training
            features = self.cnn(x)           # [b*t, 512]

        features = features.view(b, t, -1)    # [b, t, 512]
        lstm_out, _ = self.lstm(features)     # [b, t, hidden_size]
        last_hidden = self.dropout(lstm_out[:, -1, :])  # [b, hidden_size]
        output = self.fc(last_hidden)         # [b, num_classes]
        return output
