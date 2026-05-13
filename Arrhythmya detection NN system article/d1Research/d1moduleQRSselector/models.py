import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_default_device("cuda")

class QRSnet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(64,40)
        self.L2 = nn.Linear(40, 20)
        self.L3 = nn.Linear(20, 1)

        self.N1 = nn.BatchNorm1d(40)

    def forward(self,x):
        h = self.L1(x)
        h = self.N1(h)
        h = F.relu(h)
        h = self.L2(h)
        h = F.relu(h)
        h = self.L3(h)
        h = F.sigmoid(h)

        return h


class QRSnet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1_1 = nn.Conv1d(1,4, kernel_size=4,stride=2, padding=1)
        self.C1_2 = nn.Conv1d(1,8, kernel_size=8,stride=4, padding=2)
        self.C1_3 = nn.Conv1d(1,16, kernel_size=16,stride=8, padding=4)

        self.C2_1 = nn.Conv1d(4,32, kernel_size=16,stride=8, padding=4)
        self.C2_2 = nn.Conv1d(8,16, kernel_size=8,stride=4, padding=2)
        self.C2_3 = nn.Conv1d(16,8, kernel_size=4,stride=2, padding=1)


        self.C2 = nn.Conv1d(30, 10, kernel_size=16, stride=8, padding=4)

        self.L1 = nn.Linear(224,100)
        self.L2 = nn.Linear(100, 30)
        self.L3 = nn.Linear(30, 1)

        self.F1 = nn.Flatten()

        self.N1 = nn.BatchNorm1d(224)
    def forward(self,x):
        # заводиться через раз
        h1_1 = self.C1_1(x)
        h1_1 = F.relu(h1_1)

        h2_1 = self.C2_1(h1_1)
        h2_1 = F.relu(h2_1)
        # print(h1_1.size())
        # print(h2_1.size())

        h1_2 = self.C1_2(x)
        h1_2 = F.relu(h1_2)

        h2_2 = self.C2_2(h1_2)
        h2_2 = F.relu(h2_2)
        # print(h1_2.size())
        # print(h2_2.size())

        h1_3 = self.C1_3(x)
        h1_3 = F.relu(h1_3)

        h2_3 = self.C2_3(h1_3)
        h2_3 = F.relu(h2_3)
        # print(h1_3.size())
        # print(h2_3.size())

        combined_features = torch.cat((h2_1, h2_2, h2_3), dim=1)

        h = self.F1(combined_features)
        h = self.N1(h)
        h = F.relu(h)
        h = self.L1(h)
        h = F.relu(h)
        h = self.L2(h)
        h = F.relu(h)
        h = self.L3(h)
        h = F.sigmoid(h)


        return h



