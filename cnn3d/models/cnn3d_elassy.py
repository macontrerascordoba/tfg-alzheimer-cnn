import torch
import torch.nn as nn
import torch.nn.functional as F

class Cnn3DElAssy(nn.Module):
    def __init__(self):
        super(Cnn3DElAssy, self).__init__()
        
        # First two convolutional layers with 16 filters
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        
        # First max-pooling layer
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        # Next two convolutional layers with 64 filters
        self.conv3 = nn.Conv3d(16, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(64)
        
        # Second max-pooling layer
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        # Final convolutional layer with 256 filters
        self.conv5 = nn.Conv3d(64, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(256)
        
        # Third max-pooling layer
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        # Dropout layer with 20% rate
        self.dropout = nn.Dropout(p=0.2)
        
        # Calculate the flattened feature map size after convolutions and pooling
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), 
            self.conv2, self.bn2, nn.ReLU(), self.pool1,
            self.conv3, self.bn3, nn.ReLU(), 
            self.conv4, self.bn4, nn.ReLU(), self.pool2,
            self.conv5, self.bn5, nn.ReLU(), self.pool3
        )
        
        # Assuming input size is (1, 150, 150, 125)
        x = torch.randn(1, 1, 150, 150, 125)
        self._to_linear = self.convs(x).view(1, -1).size(1)
        
        # Fully connected layer
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

