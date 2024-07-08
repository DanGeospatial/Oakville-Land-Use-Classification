"""
Simple CNN for testing.
"""

import torch.nn as nn
import torch.nn.functional as fn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 3,1,1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 12,3,1,1)
        self.conv3 = nn.Conv2d(12,24,3,1,1)
        self.drop = nn.Dropout2d(p=0.2)
        self.fc = nn.Linear(in_features=64 * 64 * 24, out_features=3)

    def forward(self, x):
        x = fn.relu(self.pool(self.conv1(x)))
        x = fn.relu(self.pool(self.conv2(x)))
        x = fn.relu(self.drop(self.conv3(x)))
        x = fn.dropout(x, training=self.training)
        x = x.view(-1, 64 * 64 * 24)
        x = self.fc(x)
        return fn.log_softmax(x, dim=1)
