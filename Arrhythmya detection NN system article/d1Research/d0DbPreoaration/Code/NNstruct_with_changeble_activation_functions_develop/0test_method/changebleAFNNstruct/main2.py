import matplotlib.pyplot as plt
import pylab as pl
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import torch.optim as optim
from p1LoadData import loadMNIST,showTrainTestSizes
from p3NNarchitectures import TestNet1, SimpleNet,SimpleNetWithBN
from ptools import plotActivations,plot_layer_sigmoids,testACC, test_metrics,trainNNepoch,CountExecutionTime, calc_der2,calc_layers_AF_deviation_cofs
from p2ActivationFunctions import CustomSigmoid,CustomPolynom, CustomPiecewiseActivation,ShiftReLU,SawtoothActivation
import numpy as np

#---------------
batch_size = 30 #20 30
n_epochs = 50  #90 8
optimizer_learning_rate = 0.001 #0.001
#---------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.tensor(np.load('trainDAta/X_train2.npy'), dtype=torch.float32)
Y_train = torch.tensor(np.load('trainDAta/Y_train2.npy'), dtype=torch.float32)
X_test = torch.tensor(np.load('trainDAta/X_test2.npy'), dtype=torch.float32)
Y_test = torch.tensor(np.load('trainDAta/Y_test2.npy'), dtype=torch.float32)


# X_train = torch.tensor(np.load('FINAL_NET/test2data/LAP/X_Train.npy'), dtype=torch.float32)
# Y_train = torch.tensor(np.load('FINAL_NET/test2data/LAP/Y_Train.npy'), dtype=torch.float32)
# X_test = torch.tensor(np.load('FINAL_NET/test2data/LAP/X_Test.npy'), dtype=torch.float32)
# Y_test = torch.tensor(np.load('FINAL_NET/test2data/LAP/Y_Test.npy'), dtype=torch.float32)
#
#
# X_train = torch.tensor(np.load('FINAL_NET/test2data/LVP/X_Train.npy'), dtype=torch.float32)
# Y_train = torch.tensor(np.load('FINAL_NET/test2data/LVP/Y_Train.npy'), dtype=torch.float32)
# X_test = torch.tensor(np.load('FINAL_NET/test2data/LVP/X_Test.npy'), dtype=torch.float32)
# Y_test = torch.tensor(np.load('FINAL_NET/test2data/LVP/Y_Test.npy'), dtype=torch.float32)
#
# X_train = X_train[1700:]
# Y_train = Y_train[1700:]

# print(Y_train)
# n = 2500
#
# for i in range(n, len(Y_train)):
#     print(Y_train[i])
#     pl.figure()
#     plt.plot(X_train[i])
#     plt.show()

# X_test = X_test[1:]
# Y_test = Y_test[1:]

train_data = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

def train_one_epoch(model, dataloader, criterion, optimizer):
    print('epoch')
    model.train()
    total_loss = 0
    counter = 0
    for input1, label in dataloader:
        counter+=1
        # print(f'{counter}/{len(dataloader)}')
        input1, label = input1.to(device),  label.to(device)
        optimizer.zero_grad()

        similarity_score = model(input1)


        loss = criterion(similarity_score.squeeze(), label)  # добавляем измерение для label

        # loss = criterion(input1.squeeze(), input2.squeeze(), label)  # добавляем измерение для label

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# model = SimpleNet()
model = SimpleNetWithBN()
criterion = nn.SmoothL1Loss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=optimizer_learning_rate)

epoch_loss_mass = []

for epoch in range(n_epochs):
    model.to(device)
    avg_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    epoch_loss_mass.append(avg_loss)
    test_metrics(model, X_test, Y_test, epoch)
    print(f"Эпоха [{epoch+1}/{n_epochs}], Средняя потеря: {avg_loss:.4f}")

    calc_layers_AF_deviation_cofs([
            [model.act1.a1.detach().numpy(),model.act1.a2.detach().numpy()],
    [model.act1_a.a1.detach().numpy(),model.act1_a.a2.detach().numpy()],
    # [model.act1_b.a1.detach().numpy(),model.act1_b.a2.detach().numpy()],
    # [model.act1_c.a1.detach().numpy(),model.act1_c.a2.detach().numpy()],
    # [model.act1_d.a1.detach().numpy(),model.act1_d.a2.detach().numpy()],
    #         [model.act2.a1.detach().numpy(),model.act2.a2.detach().numpy()],
            [model.act3.a1.detach().numpy(),model.act3.a2.detach().numpy()],
            [model.act4.a1.detach().numpy(),model.act4.a2.detach().numpy()]

        ], initial_cofs_mass= [0.5, -0.7])




# np.save('testCurve/PiecwiceActivations.npy',np.array(epoch_loss_mass))


# plotActivations(model.act1,
#                     model.act2, model.act3, model.act4,fnames=['Функція активації шару 1','Функція активації шару 2','Функція активації шару 3',
#                                                                'Функція активації шару 4'], af= SawtoothActivation())

# plotActivations(model.act1, model.act1_a, model.act1_b, model.act1_c, model.act1_d,
#                     model.act2, model.act3, model.act4,fnames=['Сигмоїда шару 1','Сигмоїда шару 1а','Сигмоїда шару 1б',
#                                                                'Сигмоїда шару 1в','Сигмоїда шару 1г','Сигмоїда шару 2','Сигмоїда шару 3',
#                                                                'Сигмоїда шару 4'], af=CustomSigmoid(initial_a=1.0, initial_b=0.0))

print('calc_cofs')
# calc_layers_AF_deviation_cofs([
#         [model.act1.a.detach().numpy(), model.act1.b.detach().numpy()],
# [model.act1_a.a.detach().numpy(), model.act1_a.b.detach().numpy()],
# [model.act1_b.a.detach().numpy(), model.act1_b.b.detach().numpy()],
# [model.act1_c.a.detach().numpy(), model.act1_c.b.detach().numpy()],
# [model.act1_d.a.detach().numpy(), model.act1_d.b.detach().numpy()],
#         [model.act2.a.detach().numpy(), model.act2.b.detach().numpy()],
#         [model.act3.a.detach().numpy(), model.act3.b.detach().numpy()]
#
#     ], initial_cofs_mass= [1.0, 0.0])


# calc_layers_AF_deviation_cofs([
#         [model.act1.a1.detach().numpy(),model.act1.a2.detach().numpy(), model.act1.b.detach().numpy()],
# [model.act1_a.a1.detach().numpy(),model.act1_a.a2.detach().numpy(), model.act1_a.b.detach().numpy()],
# [model.act1_b.a1.detach().numpy(),model.act1_b.a2.detach().numpy(), model.act1_b.b.detach().numpy()],
# [model.act1_c.a1.detach().numpy(),model.act1_c.a2.detach().numpy(), model.act1_c.b.detach().numpy()],
# [model.act1_d.a1.detach().numpy(),model.act1_d.a2.detach().numpy(), model.act1_d.b.detach().numpy()],
#         [model.act2.a1.detach().numpy(),model.act2.a2.detach().numpy(), model.act2.b.detach().numpy()],
#         [model.act3.a1.detach().numpy(),model.act3.a2.detach().numpy(), model.act3.b.detach().numpy()],
#         [model.act4.a1.detach().numpy(),model.act4.a2.detach().numpy(), model.act4.b.detach().numpy()]
#
#     ], initial_cofs_mass= [1.0, 0.25, 0.0])


# calc_layers_AF_deviation_cofs([
#         [model.act1.a1.detach().numpy(),model.act1.a2.detach().numpy()],
# [model.act1_a.a1.detach().numpy(),model.act1_a.a2.detach().numpy()],
# [model.act1_b.a1.detach().numpy(),model.act1_b.a2.detach().numpy()],
# [model.act1_c.a1.detach().numpy(),model.act1_c.a2.detach().numpy()],
# [model.act1_d.a1.detach().numpy(),model.act1_d.a2.detach().numpy()],
#         [model.act2.a1.detach().numpy(),model.act2.a2.detach().numpy()],
#         [model.act3.a1.detach().numpy(),model.act3.a2.detach().numpy()],
#         [model.act4.a1.detach().numpy(),model.act4.a2.detach().numpy()]
#
#     ], initial_cofs_mass= [1.0, 0.25])




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
# [model.act1_d.a1.detach().numpy(), model.act1_d.b1.detach().numpy(), model.act1_d.a2.detach().numpy(),
#      model.act1_d.b2.detach().numpy(), model.act1_d.a3.detach().numpy(), model.act1_d.b3.detach().numpy(),
#      model.act1_d.t1.detach().numpy(), model.act1_d.t2.detach().numpy()],
#
#     [model.act2.a1.detach().numpy(), model.act2.b1.detach().numpy(), model.act2.a2.detach().numpy(),
#      model.act2.b2.detach().numpy(), model.act2.a3.detach().numpy(), model.act2.b3.detach().numpy(),
#      model.act2.t1.detach().numpy(), model.act2.t2.detach().numpy()],
#     [model.act3.a1.detach().numpy(), model.act3.b1.detach().numpy(), model.act3.a2.detach().numpy(),
#      model.act3.b2.detach().numpy(), model.act3.a3.detach().numpy(), model.act3.b3.detach().numpy(),
#      model.act3.t1.detach().numpy(), model.act3.t2.detach().numpy()],
# [model.act4.a1.detach().numpy(), model.act4.b1.detach().numpy(), model.act4.a2.detach().numpy(),
#      model.act4.b2.detach().numpy(), model.act4.a3.detach().numpy(), model.act4.b3.detach().numpy(),
#      model.act4.t1.detach().numpy(), model.act4.t2.detach().numpy()],
#
# ], initial_cofs_mass=[1.0, 0.0, 5.0, 0.0, 1.0, 0.0, -1.0, 1.0])


# calc_layers_AF_deviation_cofs([
#         [model.act1.a1.detach().numpy(),model.act1.a2.detach().numpy()],
# [model.act1_a.a1.detach().numpy(),model.act1_a.a2.detach().numpy()],
# [model.act1_b.a1.detach().numpy(),model.act1_b.a2.detach().numpy()],
# [model.act1_c.a1.detach().numpy(),model.act1_c.a2.detach().numpy()],
# [model.act1_d.a1.detach().numpy(),model.act1_d.a2.detach().numpy()],
#         [model.act2.a1.detach().numpy(),model.act2.a2.detach().numpy()],
#         [model.act3.a1.detach().numpy(),model.act3.a2.detach().numpy()],
#         [model.act4.a1.detach().numpy(),model.act4.a2.detach().numpy()]
#
#     ], initial_cofs_mass= [0.5, -0.7])


print('done')



