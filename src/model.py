import torch.nn as nn


class DefectClassifier(nn.Module):
    def __init__(self):
        super(DefectClassifier, self).__init__()

        self.conv1 =  nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3,padding=1)
        self.relu =  nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=16, out_channels = 32, kernel_size=3,padding=1)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(in_features=32 * 56 * 56, out_features=6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x