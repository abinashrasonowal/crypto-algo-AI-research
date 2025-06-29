import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNMultiTF(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNMultiTF, self).__init__()

        # Input shape: [batch, channels=3, T, F]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global avg pooling
        self.fc = nn.Linear(32, num_classes)  # Output logits for 3 classes

    def forward(self, x):
        # x shape: [batch, 3, T, F]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # shape: [batch, 32, 1, 1]
        x = x.view(x.size(0), -1)  # flatten to [batch, 32]
        x = self.fc(x)  # shape: [batch, 3]
        return x