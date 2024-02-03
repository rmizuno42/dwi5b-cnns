import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, nb_channel=100, nb_class=8):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(nb_channel, 32, 3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv3d(32, 32, 3, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 32, 3, stride=1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm3d(32)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(32, nb_class)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc1(x)
        return x

    def forward_features(self, x):
        x = F.max_pool3d(F.relu(self.conv1(x)), (1, 2, 2))
        x = F.max_pool3d(F.relu(self.bn2(self.conv2(x))), (1, 2, 2))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
