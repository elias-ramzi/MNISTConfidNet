import torch.nn as nn
import torch.nn.functional as F


class SmallConvNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(SmallConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.maxpool(out)
        out = self.dropout1(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)

        pred = self.fc2(out)
        return pred, out
