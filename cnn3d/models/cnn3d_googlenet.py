import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception3D(nn.Module):
    def __init__(self, in_channels, out1x1, out3x3_reduce, out3x3, out5x5_reduce, out5x5, out_pool_proj):
        super(Inception3D, self).__init__()
        
        self.branch1x1 = nn.Sequential(
            nn.Conv3d(in_channels, out1x1, kernel_size=1),
            nn.BatchNorm3d(out1x1),
            nn.ReLU(True)
        )
        
        self.branch3x3 = nn.Sequential(
            nn.Conv3d(in_channels, out3x3_reduce, kernel_size=1),
            nn.BatchNorm3d(out3x3_reduce),
            nn.ReLU(True),
            nn.Conv3d(out3x3_reduce, out3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(out3x3),
            nn.ReLU(True)
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv3d(in_channels, out5x5_reduce, kernel_size=1),
            nn.BatchNorm3d(out5x5_reduce),
            nn.ReLU(True),
            nn.Conv3d(out5x5_reduce, out5x5, kernel_size=5, padding=2),
            nn.BatchNorm3d(out5x5),
            nn.ReLU(True)
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_pool_proj, kernel_size=1),
            nn.BatchNorm3d(out_pool_proj),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class Cnn3DGoogleNet(nn.Module):
    def __init__(self, num_classes=3):
        super(Cnn3DGoogleNet, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv3d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv3d(64, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(192)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.inception3a = Inception3D(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception3D(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = Inception3D(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception3D(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception3D(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception3D(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception3D(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = Inception3D(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception3D(832, 384, 192, 384, 48, 128, 128)
        
        self.pool5 = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn2(self.conv3(x)))
        x = self.pool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.pool5(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = self.fc(x)
        return x