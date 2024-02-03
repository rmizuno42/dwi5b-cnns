import torch.nn as nn
import torch.nn.functional as F


class MultiOutputLinear(nn.Module):
    def __init__(self, in_features, out_features_list):
        super().__init__()
        self.out_features_list = out_features_list
        self.linears = nn.ModuleList(
            [nn.Linear(in_features, out_features) for out_features in out_features_list]
        )

    def forward(self, x):
        return [linear(x) for linear in self.linears]


class Net(nn.Module):
    def __init__(self, nb_channel=100, nb_class_list=[8]):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(nb_channel, 32, 3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(32)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = MultiOutputLinear(32, nb_class_list)

    def forward(self, x):
        x = self.forward_features(x)
        x_out = self.fc1(x)
        return x_out

    def forward_features(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
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
