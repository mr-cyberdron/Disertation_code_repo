import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from FILES_processing_lib import scandir

class SawtoothActivation(nn.Module):
    def __init__(self, a1=1.0, a2=1.0):
        """
        Конструктор для функции активации с обучаемыми параметрами наклона.

        Args:
            a1 (float): Начальное значение наклона в области x < -1.
            a2 (float): Начальное значение наклона в области x > 1.
        """
        super(SawtoothActivation, self).__init__()
        # Определяем обучаемые параметры a1 и a2
        self.a1 = nn.Parameter(torch.tensor(a1, dtype=torch.float32))
        self.a2 = nn.Parameter(torch.tensor(a2, dtype=torch.float32))

    def forward(self, x):
        """
        Вычисление функции активации.

        Args:
            x (torch.Tensor): Входной тензор.

        Returns:
            torch.Tensor: Выходной тензор.
        """
        # Область x < -1
        left = torch.where(x < 0, torch.sigmoid(self.a1) * x * (-1), torch.zeros_like(x))

        # Область -1 <= x <= 1
        middle = torch.where((x >= 0) & (x <= 1), x, torch.zeros_like(x))

        # Область x > 1
        right = torch.where(x > 1, torch.sigmoid(self.a2) *(-1) * (x - 1) + 1, torch.zeros_like(x))

        return left+middle+right


class SiameseNetwork1D0(nn.Module):
    def __init__(self):
        super(SiameseNetwork1D0, self).__init__()

        AF = SawtoothActivation(a1=0.5, a2=-0.7)

        self.act1 = copy.deepcopy(AF)
        self.act2 = copy.deepcopy(AF)
        self.act3 = copy.deepcopy(AF)



        self.fc1 = nn.Linear(500, 300)



        self.fc2 = nn.Linear(300, 100)


        self.fc3 = nn.Linear(100, 128)


        self.fc4 = nn.Linear(128, 1)


    def forward_once(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        x = self.act3(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        diff = torch.abs(output1 - output2)

        similarity_score = torch.sigmoid(self.fc4(diff))

        return similarity_score



class SiameseNetwork1D1(nn.Module):
    def __init__(self):
        super(SiameseNetwork1D1, self).__init__()

        AF = SawtoothActivation(a1=0.5, a2=-0.7)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=5, stride=3, padding=0)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=10, stride=5, padding=0)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=50, stride=20, padding=0)
        # self.fc1 = nn.Linear(64 * 5491, 256)

        self.bn0 = nn.BatchNorm1d(1)
        # self.bn1 = nn.BatchNorm1d(1)
        # self.bn2 = nn.BatchNorm1d(1)
        # self.bn3 = nn.BatchNorm1d(1)

        self.act1 = copy.deepcopy(AF)
        self.act2 = copy.deepcopy(AF)
        self.act3 = copy.deepcopy(AF)
        self.act4 = copy.deepcopy(AF)
        self.act5 = copy.deepcopy(AF)
        self.act6 = copy.deepcopy(AF)

        self.dropout = nn.Dropout(p=0.8)


        self.fc1 = nn.Linear(142, 100)



        self.fc2 = nn.Linear(100, 80)


        self.fc3 = nn.Linear(80, 128)


        self.fc4 = nn.Linear(128, 1)


    def forward_once(self, x):
        # x = F.relu(self.conv1(x))
        # x = x.view(x.size(0), -1)

        x = self.pool1(x)

        x1 = self.conv1(x)
        x1 = self.act4(x1)
        # x1 = self.bn1(x1)
        x1 = x1.view(x1.size()[0], 1, -1)


        x2 = self.conv2(x)
        x2 = self.act5(x2)
        # x2 = self.bn2(x2)
        x2 = x2.view(x2.size()[0], 1, -1)



        x3 = self.conv3(x)
        x3 = self.act6(x3)
        # x3 = self.bn3(x3)

        x3 = x3.view(x3.size()[0], 1, -1)


        out = torch.cat([x1, x2, x3], dim=2)

        x = out

        x = self.bn0(x)
        #
        # x = torch.relu(self.conv1(x))


        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        x = self.act3(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        diff = torch.abs(output1 - output2)

        similarity_score = torch.sigmoid(self.fc4(diff))

        return similarity_score

class SiameseNetwork1D2(nn.Module):
    def __init__(self):
        super(SiameseNetwork1D2, self).__init__()

        self.lstm = nn.LSTM(input_size=55, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)

        self.fc1 = nn.Linear(128 , 256)  # 2 * hidden_size due to bidirectionality
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 1)

    def forward_once(self, x):
        # Reshape to [batch_size, seq_len, feature_size]
        x = x.view(x.size(0), 100, -1)

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Use the output of the last time step
        x = x[:, -1, :]  # Shape: [batch_size, 128 * 2]

        # Fully connected layers
        x = self.fc1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.act2(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        diff = torch.abs(output1 - output2)
        similarity_score = torch.sigmoid(self.fc3(diff))

        return similarity_score


class SiameseNetwork1D3(nn.Module):
    def __init__(self):
        super(SiameseNetwork1D3, self).__init__()
        self.DRaval = True
        AF = SawtoothActivation(a1=0.5, a2=-0.7)
        self.act1 = copy.deepcopy(AF)
        self.act2 = copy.deepcopy(AF)
        self.act3 = copy.deepcopy(AF)
        self.act4 = copy.deepcopy(AF)
        self.act5 = copy.deepcopy(AF)
        self.act6 = copy.deepcopy(AF)
        self.act7 = copy.deepcopy(AF)

        self.conv1 = nn.Conv1d(1,10,3,1)
        self.conv2 = nn.Conv1d(10, 20, 10, 5)
        self.conv3 = nn.Conv1d(20, 30, 20, 5)

        self.fc1 = nn.Linear(480, 450)
        self.fc2 = nn.Linear(450, 420)
        self.fc3 = nn.Linear(420, 410)
        self.fc4 = nn.Linear(500, 1)
        self.fc5 = nn.Linear(410, 400)


        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.1)

        self.bn0 = nn.BatchNorm1d(1)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(10)

        self.pool1 = nn.AdaptiveMaxPool1d(1000)
        self.conv21 = nn.Conv1d(1,10,50,10)

        self.fc21 = nn.Linear(960, 400)
        self.fc22 = nn.Linear(400, 200)
        self.fc23 = nn.Linear(200, 100)

        self.act21 = copy.deepcopy(AF)
        self.act22 = copy.deepcopy(AF)
        self.act23 = copy.deepcopy(AF)
        self.act24 = copy.deepcopy(AF)


        self.fc31 = nn.Linear(500, 200)
        self.fc32 = nn.Linear(200, 500)

        self.act31 = copy.deepcopy(AF)
        self.act32 = copy.deepcopy(AF)



    def forward_once(self, x):
        x1 = x[:, :, :500]

        x1 = self.conv1(x1)
        x1 = self.act1(x1)
        x1 = self.conv2(x1)
        x1 = self.act2(x1)
        x1 = self.conv3(x1)
        x1 = self.act3(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = x1.unsqueeze(1)

        x1 = self.bn1(x1)

        if self.DRaval:
            x1 = self.drop1(x1)

        x1 = self.fc1(x1)
        x1 = self.act4(x1)
        x1 = self.fc2(x1)
        x1 = self.act5(x1)

        x1 = self.fc3(x1)
        x1 = self.act6(x1)
        x1 = self.fc5(x1)
        x1 = self.act7(x1)

        x2 = x[:, :, 500:]

        x2 = self.pool1(x2)


        x2 = self.conv21(x2)
        x2 = self.act21(x2)
        # x2 = self.bn2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = x2.unsqueeze(1)

        if self.DRaval:
            x2 = self.drop2(x2)

        x2 = self.fc21(x2)
        x2 = self.act22(x2)

        x2 = self.fc22(x2)
        x2 = self.act23(x2)

        x2 = self.fc23(x2)
        x2 = self.act24(x2)

        out = torch.cat([x1, x2], dim=2)

        out = self.fc31(out)
        out = self.act31(out)

        out = self.fc32(out)
        out = self.act32(out)

        return out

    def forward(self, input1, input2, dropout = True):
        self.DRaval = dropout
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        diff = torch.abs(output1 - output2)

        similarity_score = torch.sigmoid(self.fc4(diff))

        return similarity_score
