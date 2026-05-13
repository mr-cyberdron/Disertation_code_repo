import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch

import matplotlib.pyplot as plt
from torch import optim
import numpy as np
from sklearn.model_selection import train_test_split
import pywt
from tqdm import tqdm
import models
from FILES_processing_lib import scandir

print(torch.cuda.is_available())
print(torch.version.cuda)

counting_base = 'GPU'
if counting_base == 'GPU':
    torch.set_default_device("cuda")

def cwt_spectrum(input_data, fs):

    wavelet = 'morl'
    scales = np.arange(1, 17)

    # Вычисляем CWT
    coefficients, frequencies = pywt.cwt(input_data, scales, wavelet, sampling_period=1 / fs)

    # # Построение графика CWT
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(input_data)
    # plt.subplot(2, 1, 2)
    # plt.imshow((np.abs(coefficients)), aspect='auto', extent=[0, 1, 1, len(scales)], cmap='jet', interpolation='bilinear')
    # plt.colorbar(label='Амплітуда')
    # plt.ylabel('Маштаб')
    # plt.xlabel('Час (секунди)')
    # plt.title('Continuous Wavelet Transform (CWT)')
    # plt.show()
    return coefficients
#
def test_model(modell, test_loaderr):
    print('test_model')
    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0
    for dataa, labelss in test_loaderr:

        if counting_base == 'GPU':
            dataa = dataa.float().cuda()
            labelss = labelss.float().cuda()
        else:
            dataa = dataa.float()
            labelss = labelss.float()

        # dataa = dataa.unsqueeze(1)
        outputt = modell(dataa).squeeze()
        for out_val, lable_val in zip(outputt, labelss):
            if int(round(float(out_val))) == int(lable_val) == 1:
                tp_count += 1 #tp
            elif int(round(float(out_val))) == int(lable_val) == 0:
                tn_count +=1 #tn
            elif int(round(float(out_val))) != int(lable_val) and int(lable_val) == 0:
                fp_count +=1 #fp!
            elif int(round(float(out_val))) != int(lable_val) and int(lable_val) == 1:
                fn_count +=1 #fn
            else:
                raise ValueError("what the situation no fp no fn")

    acc = (tp_count+tn_count) / (tp_count+tn_count+fp_count+fn_count)
    print('Accuracy: ' + str(acc))
    fpr = (fp_count)/(fp_count+tn_count)
    print(f'FPR: {fpr}')
    return acc

def shortage_xy(x_mass, y_mass, target_mass_len):
    print('shortage xy')
    assert len(x_mass) == len(y_mass)
    permuted_indices = np.random.permutation(np.shape(x_mass)[0])
    indices_sample = np.random.choice(permuted_indices, size=target_mass_len, replace=False)
    x_mass_to_return = x_mass[indices_sample]
    y_mass_to_return = y_mass[indices_sample]
    assert len(x_mass_to_return) == len(y_mass_to_return)
    return x_mass_to_return, y_mass_to_return

class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # pokazatel_stepeni = ((2*4*torch.sum(torch.abs((targets-inputs))))/targets.size()[0])
        x = (targets - inputs)*-1
        pokazatel_stepeni = torch.sum (torch.abs(torch.maximum(0.05*x,x)))
        # print('pp')
        # print(pokazatel_stepeni)
        loss_body = (torch.sum(torch.abs(targets - inputs)))
        # print('pp')
        # print(loss_body)
        # print(pokazatel_stepeni)
        loss =loss_body*pokazatel_stepeni

        return loss

batch_size = 100
epochs = 200

weights_path = 'model_weights.pth'

print('load data')
np_data = np.load('D:/Bases/1QRS_noise_base/prepared/QRS_Noise_prep.npz')
X_mass = np_data['x']
Y_mass = np_data['y']

X_mass,Y_mass = shortage_xy(X_mass,Y_mass,100000)

# fs = 500
# print('calc CWT')
# new_X_mass = []
# for x_item in X_mass:
#     new_X_mass.append(cwt_spectrum(x_item,fs))
# X_mass=new_X_mass



X_train, X_test, y_train, y_test = train_test_split(X_mass, Y_mass, train_size=0.95, shuffle=True, random_state=14)

if counting_base == 'GPU':
    trainloader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), shuffle=False, batch_size=batch_size,
                                              generator=torch.Generator(device='cuda')
                                              )
    testloader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), shuffle=False, batch_size=batch_size,
                                             generator=torch.Generator(device='cuda')
                                             )
else:
    trainloader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), shuffle=False, batch_size=batch_size,
                                              )
    testloader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), shuffle=False, batch_size=batch_size,
                                             )

input_size = np.shape(trainloader.dataset[0][0])[0]

model = models.QRSnet1()
# model = models.QRSnet2()

files_in_dir = scandir('./',ext='pth')
if weights_path in files_in_dir:
    model.load_state_dict(torch.load(weights_path))
    model.eval()

if counting_base == 'GPU':
    model.cuda()
print(model)
#criterion = nn.BCELoss() #Бінарна крос ентропія (перехресна втрата ентропії)
#criterion = nn.L1Loss()
criterion = CustomLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005,)

for e in range(epochs):
    running_loss = 0
    total_iterations = len(trainloader)
    progress_bar = tqdm(total=total_iterations, desc=f"epoch {e+1}/{epochs}")
    for data, labels in trainloader:
        progress_bar.update(1)
        if counting_base == 'GPU':
            data = data.float().cuda()
            labels = labels.float().cuda()
        else:
            data = data.float()
            labels = labels.float()
        optimizer.zero_grad()
        # data = data.unsqueeze(1)
        output = model(data).squeeze()
        loss = criterion(output, labels)
        # backward propagation
        loss.backward()
        # update the gradient to new gradients
        optimizer.step()
        running_loss += loss.item()
    else:
        print("Training loss: ", (running_loss / len(trainloader)))
        torch.save(model.state_dict(), weights_path)
        progress_bar.close()

        if (e+1)%1 == 0:
            test_model(model,testloader)




true_count = 0
false_count = 0
for images, labels in testloader:
    labels = labels.cuda()
    images = images.float().unsqueeze(1).cuda()
    output = model(images).squeeze()
    for out_val, lable_val in zip(output, labels):
        if int(round(float(out_val))) == int(lable_val):
            true_count +=1
        else:
            false_count+=1

print(true_count)
print(false_count)
acc = true_count/(true_count+false_count)
print('Accuracy: '+ str(acc))