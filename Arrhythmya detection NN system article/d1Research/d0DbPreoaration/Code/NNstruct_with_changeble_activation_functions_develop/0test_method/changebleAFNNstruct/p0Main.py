import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import torch.optim as optim
from p1LoadData import loadMNIST,showTrainTestSizes
from p3NNarchitectures import TestNet1, SimpleNet,SimpleNetWithBN
from ptools import plotActivations,plot_layer_sigmoids,testACC,trainNNepoch,CountExecutionTime, calc_der2,calc_layers_AF_deviation_cofs
from p2ActivationFunctions import CustomSigmoid,CustomPolynom, CustomPiecewiseActivation
import numpy as np

#---------------
batch_size = 50
n_epochs = 5
optimizer_learning_rate = 0.001
#---------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

#----------------------------------------data1------------------#
X_train, X_test, y_train, Y_test = loadMNIST()
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:500]
y_test = Y_test[:500]
showTrainTestSizes(X_train, X_test, y_train, Y_test)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
#---------------------------------------------------------------#

#-----------------------------------------data2-----------------#
# print('load npy')
# X_train = torch.tensor(np.load('trainDAta/X_train.npy'), dtype=torch.float32)
# Y_train = torch.tensor(np.load('trainDAta/Y_train.npy'), dtype=torch.float32)
# X_test = torch.tensor(np.load('trainDAta/X_test.npy'), dtype=torch.float32)
# Y_test = torch.tensor(np.load('trainDAta/Y_test.npy'), dtype=torch.float32)

# X_train = torch.tensor(np.load('trainDAta/X_train2.npy'), dtype=torch.float32)
# Y_train = torch.tensor(np.load('trainDAta/Y_train2.npy'), dtype=torch.float32)
# X_test = torch.tensor(np.load('trainDAta/X_test2.npy'), dtype=torch.float32)
# Y_test = torch.tensor(np.load('trainDAta/Y_test2.npy'), dtype=torch.float32)



# train_data = TensorDataset(X_train, Y_train)
# train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
#---------------------------------------------------------------#



# model = TestNet1()
# model = SimpleNet()
model = SimpleNetWithBN()

optimizer = optim.Adam(model.parameters(), lr=optimizer_learning_rate) #, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.BCELoss()
# loss_fn = nn.SmoothL1Loss()

if __name__ == "__main__":
    t1 = CountExecutionTime()
    t1.start()

    for epoch in range(n_epochs):
        model = trainNNepoch(model,train_loader,loss_fn,optimizer, device = device)
        # print(model.act1_a.a, model.act1_a.b)
        testACC(model, X_test, Y_test, epoch)

    t1.printTime()

    # plot_layer_sigmoids(model.act1, 'layer1')
    # plot_layer_sigmoids(model.act1_a, 'layer1_a')
    # plot_layer_sigmoids(model.act1_b, 'layer1_b')
    # plot_layer_sigmoids(model.act1_c, 'layer1_c')
    # plot_layer_sigmoids(model.act2, 'layer2')
    # plot_layer_sigmoids(model.act3, 'layer3')






    # calc_layers_AF_deviation_cofs([
    #     [model.act1.a.detach().numpy(), model.act1.b.detach().numpy()],
    #     [model.act2.a.detach().numpy(), model.act2.b.detach().numpy()],
    #     [model.act3.a.detach().numpy(), model.act3.b.detach().numpy()]
    #
    # ], initial_cofs_mass= [1.0, 0.0])

    calc_layers_AF_deviation_cofs([
        [model.act1.a.detach().numpy(), model.act1.b.detach().numpy()],

        [model.act1_a.a.detach().numpy(), model.act1_a.b.detach().numpy()],
        [model.act1_b.a.detach().numpy(), model.act1_b.b.detach().numpy()],
        [model.act1_c.a.detach().numpy(), model.act1_c.b.detach().numpy()],

        [model.act2.a.detach().numpy(), model.act2.b.detach().numpy()],
        [model.act3.a.detach().numpy(), model.act3.b.detach().numpy()]

    ], initial_cofs_mass=[1.0, 0.0])


    # calc_layers_AF_deviation_cofs([
    #     [model.act1.a1.detach().numpy(), model.act1.b1.detach().numpy(),model.act1.a2.detach().numpy(),model.act1.b2.detach().numpy(),model.act1.a3.detach().numpy(),model.act1.b3.detach().numpy(),model.act1.t1.detach().numpy(),model.act1.t2.detach().numpy()],
    #     [model.act2.a1.detach().numpy(), model.act2.b1.detach().numpy(), model.act2.a2.detach().numpy(),
    #      model.act2.b2.detach().numpy(), model.act2.a3.detach().numpy(), model.act2.b3.detach().numpy(),
    #      model.act2.t1.detach().numpy(), model.act2.t2.detach().numpy()],
    #     [model.act3.a1.detach().numpy(), model.act3.b1.detach().numpy(), model.act3.a2.detach().numpy(),
    #      model.act3.b2.detach().numpy(), model.act3.a3.detach().numpy(), model.act3.b3.detach().numpy(),
    #      model.act3.t1.detach().numpy(), model.act3.t2.detach().numpy()],
    #
    # ], initial_cofs_mass=[1.0, 0.0, 5.0, 0.0, 1.0, 0.0, -1.0, 1.0])

    # calc_layers_AF_deviation_cofs([
    #     [model.act1.a1.detach().numpy(), model.act1.b1.detach().numpy(), model.act1.a2.detach().numpy(),
    #      model.act1.b2.detach().numpy(), model.act1.a3.detach().numpy(), model.act1.b3.detach().numpy(),
    #      model.act1.t1.detach().numpy(), model.act1.t2.detach().numpy()],
    #
    #     [model.act1_a.a1.detach().numpy(), model.act1_a.b1.detach().numpy(), model.act1_a.a2.detach().numpy(),
    #      model.act1_a.b2.detach().numpy(), model.act1_a.a3.detach().numpy(), model.act1_a.b3.detach().numpy(),
    #      model.act1_a.t1.detach().numpy(), model.act1_a.t2.detach().numpy()],
    #     [model.act1_b.a1.detach().numpy(), model.act1_b.b1.detach().numpy(), model.act1_b.a2.detach().numpy(),
    #      model.act1_b.b2.detach().numpy(), model.act1_b.a3.detach().numpy(), model.act1_b.b3.detach().numpy(),
    #      model.act1_b.t1.detach().numpy(), model.act1_b.t2.detach().numpy()],
    #     [model.act1_c.a1.detach().numpy(), model.act1_c.b1.detach().numpy(), model.act1_c.a2.detach().numpy(),
    #      model.act1_c.b2.detach().numpy(), model.act1_c.a3.detach().numpy(), model.act1_c.b3.detach().numpy(),
    #      model.act1_c.t1.detach().numpy(), model.act1_c.t2.detach().numpy()],
    #
    #     [model.act2.a1.detach().numpy(), model.act2.b1.detach().numpy(), model.act2.a2.detach().numpy(),
    #      model.act2.b2.detach().numpy(), model.act2.a3.detach().numpy(), model.act2.b3.detach().numpy(),
    #      model.act2.t1.detach().numpy(), model.act2.t2.detach().numpy()],
    #     [model.act3.a1.detach().numpy(), model.act3.b1.detach().numpy(), model.act3.a2.detach().numpy(),
    #      model.act3.b2.detach().numpy(), model.act3.a3.detach().numpy(), model.act3.b3.detach().numpy(),
    #      model.act3.t1.detach().numpy(), model.act3.t2.detach().numpy()],
    #
    # ], initial_cofs_mass=[1.0, 0.0, 5.0, 0.0, 1.0, 0.0, -1.0, 1.0])

    # plotActivations(model.act1,
    #
    #                 model.act1_a, model.act1_b, model.act1_c,
    #
    #                 model.act2, model.act3)

    # plotActivations(model.act1,
    #                 model.act2, model.act3)

    # plotActivations(model.act1,
    #                 model.act2, model.act3)




