"""
Simple CNN for testing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as fn


class Net(nn.Module):
    def __init__(self, bands, classes):
        super(Net, self).__init__()
        self.bands = bands
        self.classes = classes

        self.conv1 = nn.Conv2d(bands, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(774400, 224)
        self.fc2 = nn.Linear(224, classes)

    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = fn.relu(x)

        x = self.conv2(x)
        x = fn.relu(x)

        # Run max pooling over x
        x = fn.max_pool2d(x, 2)
        # Pass data through dropout1
        x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through ``fc1``
        x = self.fc1(x)
        x = fn.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # Apply softmax to x
        output = fn.log_softmax(x, dim=1)
        return output
