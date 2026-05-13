import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from FILES_processing_lib import scandir
from FinalNetArchitectures import SiameseNetwork1D1, SiameseNetwork1D2, SiameseNetwork1D0, SiameseNetwork1D3
from ptools import calc_layers_AF_deviation_cofs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")





# Основной цикл обучения
num_epochs = 500 #500
batch_size = 200 #200

model = SiameseNetwork1D3().to(device)
# criterion = nn.BCELoss()
# criterion = nn.CosineEmbeddingLoss()
# criterion = nn.MarginRankingLoss(margin=1.0)
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.SmoothL1Loss()

optimizer = optim.Adam(model.parameters(), lr=0.001) #0.001)

print('load npy')
X1 = np.load('FinalNetData/TrainDataForFinalNET/TrainData/X1_train.npy')
X2 = np.load('FinalNetData/TrainDataForFinalNET/TrainData/X2_train.npy')
Y = np.load('FinalNetData/TrainDataForFinalNET/TrainData/y_train.npy')

# for i in range(6650, len(X1), 100):
#     print(i)
#     plt.figure(figsize=(10, 6))
#
#     # Графік 1
#     plt.subplot(2, 1, 1)
#     plt.plot(X1[i], color='blue', label='Сигнал для класифікації')
#     plt.title("Сигнали що подаються на входи нейронної мережі", fontsize=14)
#     plt.xlabel("Семпли", fontsize=12)
#     plt.ylabel("Амплітуда", fontsize=12)
#     plt.grid(True)
#     plt.legend()
#
#     # Графік 2
#     plt.subplot(2, 1, 2)
#     plt.plot(-1 * np.array(X2[i]), color='blue', label='Вхідний вектор ознак')
#
#     plt.xlabel("Семпли", fontsize=12)
#     plt.ylabel("Амплітуда", fontsize=12)
#     plt.grid(True)
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#


print('load tensor')
X1_tensor = torch.tensor(X1, dtype=torch.float32).unsqueeze(1).to(device)  # добавляем измерение канала
X2_tensor = torch.tensor(X2, dtype=torch.float32).unsqueeze(1).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

# X1_tensor = X1_tensor[:, :, :500]
# X2_tensor = X2_tensor[:, :, :500]


dataset = TensorDataset(X1_tensor, X2_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



def train_one_epoch(model, dataloader, criterion, optimizer):
    print('epoch')
    model.train()
    total_loss = 0
    counter = 0
    for input1, input2, label in dataloader:
        counter+=1
        # print(f'{counter}/{len(dataloader)}')
        input1, input2, label = input1.to(device), input2.to(device), label.to(device)
        optimizer.zero_grad()

        similarity_score = model(input1, input2)
        loss = criterion(similarity_score.squeeze(), label)  # добавляем измерение для label

        # loss = criterion(input1.squeeze(), input2.squeeze(), label)  # добавляем измерение для label

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def compute_metrics(predictions, targets):
    """
    Вычисляет TP, TN, FP, FN, чувствительность, специфичность и точность.

    :param predictions: Тензор с предсказаниями модели (float), shape [N]
    :param targets: Тензор с эталонными значениями (0 или 1), shape [N]
    :return: Словарь с метриками
    """
    # Округляем предсказания до ближайшего целого (0 или 1)
    predictions = torch.round(predictions)

    # Вычисляем TP, TN, FP, FN
    TP = ((predictions == 1.0) & (targets == 1.0)).sum().item()
    TN = ((predictions == 0.0) & (targets == 0.0)).sum().item()
    FP = ((predictions == 1.0) & (targets == 0.0)).sum().item()
    FN = ((predictions == 0.0) & (targets == 1.0)).sum().item()

    # Рассчитываем метрики
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    # Возвращаем метрики в словаре
    ret_dict =  {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "Accuracy": round(accuracy, 4)
    }

    print(ret_dict)

    return ret_dict


for epoch in range(num_epochs):
    avg_loss = train_one_epoch(model, dataloader, criterion, optimizer)
    print(f"Эпоха [{epoch+1}/{num_epochs}], Средняя потеря: {avg_loss:.4f}")



test_dir = './FinalNetData/TrainDataForFinalNET/TestData/'
test_pathologies_dirs = scandir(test_dir)

total_acc = []
for pat_code in test_pathologies_dirs:
    print(pat_code)
    X1_test = np.load(test_dir+pat_code+'/'+pat_code+'_X1_test.npy')
    X2_test = np.load(test_dir +pat_code+'/'+ pat_code + '_X2_test.npy')
    y_test = np.load(test_dir +pat_code+'/'+ pat_code + '_y_test.npy')

    X1_tensor_test = torch.tensor(X1_test, dtype=torch.float32).unsqueeze(1).to(device)  # добавляем измерение канала
    X2_tensor_test = torch.tensor(X2_test, dtype=torch.float32).unsqueeze(1).to(device)
    Y_tensor_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # X1_tensor_test = X1_tensor_test[:, :, :500]
    # X2_tensor_test = X2_tensor_test[:, :, :500]
    # model.eval()
    y_pred = model(X1_tensor_test, X2_tensor_test,  dropout = False).squeeze()

    res_dict = compute_metrics(y_pred,Y_tensor_test)

    acc = res_dict["Accuracy"]
    total_acc.append(acc)

print('total_Acc:')
print(np.mean(total_acc))

# model.cpu()
calc_layers_AF_deviation_cofs([
        [model.act1.a1.detach().numpy(),model.act1.a2.detach().numpy()],
        [model.act2.a1.detach().numpy(),model.act2.a2.detach().numpy()],
        [model.act3.a1.detach().numpy(),model.act3.a2.detach().numpy()],
        [model.act4.a1.detach().numpy(),model.act4.a2.detach().numpy()],
        [model.act5.a1.detach().numpy(),model.act5.a2.detach().numpy()],
        [model.act6.a1.detach().numpy(),model.act6.a2.detach().numpy()],
        [model.act7.a1.detach().numpy(),model.act7.a2.detach().numpy()],
        [model.act21.a1.detach().numpy(),model.act21.a2.detach().numpy()],
        [model.act22.a1.detach().numpy(),model.act22.a2.detach().numpy()],
        [model.act23.a1.detach().numpy(),model.act23.a2.detach().numpy()],
        [model.act24.a1.detach().numpy(),model.act24.a2.detach().numpy()]

    ], initial_cofs_mass= [0.5, -0.7])










def calculate_metrics_and_plot(tp, tn, fp, fn):
    # Сумма всех случаев
    total = tp + tn + fp + fn

    # Пересчет в проценты
    tp_percent = (tp / total) * 100
    tn_percent = (tn / total) * 100
    fp_percent = (fp / total) * 100
    fn_percent = (fn / total) * 100

    # Расчет метрик
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    accuracy = (tp + tn) / total

    # Вывод результатов
    print(f"TP: {tp_percent:.2f}%")
    print(f"TN: {tn_percent:.2f}%")
    print(f"FP: {fp_percent:.2f}%")
    print(f"FN: {fn_percent:.2f}%")
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Accuracy: {accuracy:.2f}")









