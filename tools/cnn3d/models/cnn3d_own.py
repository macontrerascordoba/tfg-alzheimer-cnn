import torch
import torch.nn as nn
import torch.nn.functional as F 

# Define the CNN3D model
class Cnn3DOwn(nn.Module):
    def __init__(self):
        super(Cnn3DOwn, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.dropout = nn.Dropout(p=0.5)
        
        # Calculate the correct size after convolutions and max pooling
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), self.pool,
            self.conv2, self.bn2, nn.ReLU(), self.pool,
            self.conv3, self.bn3, nn.ReLU(), self.pool
        )
        x = torch.randn(1, 1, 150, 150, 125)
        self._to_linear = self.convs(x).view(1, -1).size(1)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x