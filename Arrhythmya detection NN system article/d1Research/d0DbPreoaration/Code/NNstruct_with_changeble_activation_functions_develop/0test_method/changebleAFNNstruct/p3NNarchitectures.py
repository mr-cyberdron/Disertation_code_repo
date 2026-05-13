import copy

import torch.nn as nn
from p2ActivationFunctions import CustomSigmoid, CustomSigmoidForAllNeurons,CustomPolynom,CustomPolynom2,CustomPReLU, CustomPiecewiseActivation, SawtoothActivation, ShiftReLU


class SimpleNetWithBN(nn.Module):
    #https://www.researchgate.net/figure/Neural-network-architecture-for-MNIST-classification_fig2_382379965
    def __init__(self):
        super().__init__()
        # AF = CustomSigmoid(initial_a=1.0, initial_b=0.0)
        # AF = ShiftReLU()
        # AF = nn.ReLU()
        # AF = CustomPolynom()
        # AF = nn.Sigmoid()
        # AF = CustomPReLU()
        # AF = CustomPiecewiseActivation()
        AF = SawtoothActivation(a1=0.5, a2=-0.7)

        self.layer1 = nn.Linear(500, 400)
        # AF = CustomSigmoidForAllNeurons(512)
        self.act1 = copy.deepcopy(AF)

        #----------------------useless layers------------#
        # #1
        # self.layer1_a = nn.Linear(512, 512)
        # self.act1_a = copy.deepcopy(AF)
        #
        # self.layer1_b = nn.Linear(512, 512)
        # self.act1_b = copy.deepcopy(AF)
        #
        # self.layer1_с = nn.Linear(512, 512)
        # self.act1_с = copy.deepcopy(AF)

        #2
        self.layer1_a = nn.Linear(400, 256)
        # AF = CustomSigmoidForAllNeurons(512)
        self.act1_a = copy.deepcopy(AF)

        self.layer1_b = nn.Linear(200, 200)
        # AF = CustomSigmoidForAllNeurons(20)
        self.act1_b = copy.deepcopy(AF)

        self.layer1_c = nn.Linear(200, 200)
        # AF = CustomSigmoidForAllNeurons(512)
        self.act1_c = copy.deepcopy(AF)

        self.layer1_d = nn.Linear(200, 400)
        # AF = CustomSigmoidForAllNeurons(512)
        self.act1_d = copy.deepcopy(AF)

        # # 3
        # self.layer1_a = nn.Linear(512, 5)
        # self.act1_a = copy.deepcopy(AF)
        #
        # self.layer1_b = nn.Linear(5, 5)
        # self.act1_b = copy.deepcopy(AF)
        #
        # self.layer1_с = nn.Linear(5, 512)
        # self.act1_с = copy.deepcopy(AF)




        #-----------------------------------------------#

        self.layer2 = nn.Linear(400, 256)
        self.bn1 = nn.BatchNorm1d(256)
        # AF = CustomSigmoidForAllNeurons(256)
        self.act2 = copy.deepcopy(AF)

        self.layer3 = nn.Linear(256, 128)
        # AF = CustomSigmoidForAllNeurons(128)
        self.act3 = copy.deepcopy(AF)

        self.dropout = nn.Dropout(p=0.3)
        self.layer4 = nn.Linear(128, 1)
        self.act4 = copy.deepcopy(AF)


    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)

        # ----------------------useless layers------------#
        x = self.layer1_a(x)
        x = self.act1_a(x)

        # x = self.layer1_b(x)
        # x = self.act1_b(x)
        #
        # x = self.layer1_c(x)
        # x = self.act1_c(x)
        #
        # x = self.layer1_d(x)
        # x = self.act1_d(x)
        #------------------------------------------------#

        # x = self.layer2(x)
        # x = self.act2(x)
        x = self.bn1(x)
        x = self.layer3(x)
        x = self.act3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.act4(x)
        return x


class SimpleNet(nn.Module):
    # https://www.researchgate.net/figure/Neural-network-architecture-for-MNIST-classification_fig2_382379965
    def __init__(self):
        super().__init__()
        # AF = CustomSigmoid(initial_a=1.0, initial_b=0.0)
        # AF = nn.ReLU()
        # AF = ShiftReLU()
        # AF = CustomPolynom()
        # AF = CustomPolynom2()
        # AF = CustomPolynom()
        # AF = nn.Sigmoid()
        # AF = CustomPReLU()
        # AF = CustomPiecewiseActivation()
        AF = SawtoothActivation()

        self.layer1 = nn.Linear(500, 400)
        # AF = CustomSigmoidForAllNeurons(512)
        self.act1 = copy.deepcopy(AF)

        self.layer2 = nn.Linear(400, 256)
        self.bn1 = nn.BatchNorm1d(256)
        # AF = CustomSigmoidForAllNeurons(256)
        self.act2 = copy.deepcopy(AF)

        self.layer3 = nn.Linear(256, 128)
        # AF = CustomSigmoidForAllNeurons(128)
        self.act3 = copy.deepcopy(AF)

        self.dropout = nn.Dropout(p=0.3)
        self.layer4 = nn.Linear(128, 1)
        self.act4 = copy.deepcopy(AF)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.bn1(x)
        x = self.layer3(x)
        x = self.act3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.act4(x)
        return x


class TestNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 512)
        self.act1 = CustomSigmoid(initial_a=1.0, initial_b=0.0)

        self.layer1_a = nn.Linear(512, 512)
        self.act1_a = CustomSigmoid(initial_a=1.0, initial_b=0.0)
        self.layer1_b = nn.Linear(512, 512)
        self.act1_b = CustomSigmoid(initial_a=1.0, initial_b=0.0)
        self.layer1_c = nn.Linear(512, 512)
        self.act1_c = CustomSigmoid(initial_a=1.0, initial_b=0.0)

        # Добавление нелинейности и batch normalization
        self.layer2 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act2 = CustomSigmoid(initial_a=0.5, initial_b=-0.5)

        self.layer3 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.layer4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.act1(self.layer1(x))

        x = self.act1_a(self.layer1_a(x))
        x = self.act1_b(self.layer1_b(x))
        x = self.act1_c(self.layer1_c(x))

        x = self.bn1(self.act2(self.layer2(x)))
        x = self.dropout(self.layer3(x))
        x = self.layer4(x)
        return x