import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from FILES_processing_lib import scandir

pat_dict_enum = {'SVTAC': 0, 'PSVT': 1, 'BIGU': 2, 'RVH': 3, 'ANEUR': 4, 'LPFB': 5, 'WPW': 6,
                 'LMI': 7, 'LPR': 8, 'RAO/RAE': 9, 'ISCIN': 10, 'ISCIL': 11, 'ISCAS': 12,
                 'IPLMI': 13, 'ALMI': 14, 'LNGQT': 15, 'SBRAD': 16, 'CRBBB': 17, 'LAO/LAE': 18,
                 'PAC': 19, 'SVARR': 20, 'CLBBB': 21, 'ISCAL': 22, 'AMI': 23, 'ILMI': 24, 'VCLVH': 25,
                 'STACH': 26, '1AVB': 27, 'IRBBB': 28, 'ISC/': 29, 'PVC': 30, 'LAP': 31, 'LVP': 32,
                 'AFIB': 33, 'SARRH': 34, 'LAFB': 35, 'LVH': 36, 'ASMI': 37, 'NORM': 38, 'IMI': 39}
# Класс с ограничением нормы для свёрточных слоёв
class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv1dWithConstraint, self).forward(x)

# Нейросеть
class ECGNet1D(nn.Module):
    def __init__(self, input_length=5500, n_classes=40, dropout_rate=0.5, kernel_length=64, F1=8, D=2, F2=16):
        super(ECGNet1D, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.input_length = input_length
        self.n_classes = n_classes
        self.kernel_length = kernel_length
        self.dropout_rate = dropout_rate

        # Начальные блоки
        self.block1 = nn.Sequential(
            nn.Conv1d(1, self.F1, self.kernel_length, stride=1, padding=self.kernel_length // 2, bias=False),
            nn.BatchNorm1d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv1dWithConstraint(self.F1, self.F1 * self.D, kernel_size=1, groups=self.F1, max_norm=1, bias=False),
            nn.BatchNorm1d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=self.dropout_rate)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(self.F1 * self.D, self.F1 * self.D, kernel_size=self.kernel_length, stride=1,
                      padding=self.kernel_length // 2, groups=self.F1 * self.D, bias=False),
            nn.Conv1d(self.F1 * self.D, self.F2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=self.dropout_rate)
        )

        # Расчёт размера выходного слоя
        self.flat_size = self._calculate_flat_size(input_length)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, self.n_classes, bias=False),
            nn.Sigmoid()
        )

    def _calculate_flat_size(self, input_length):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_length)
            x = self.block1(x)
            x = self.block2(x)
            return x.numel()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)  # Преобразование в вектор
        x = self.classifier(x)
        return x

# Данные
input_length = 5500  # Длина входного сигнала
n_classes = 40  # Количество выходных классов


X_train = np.load('./FinalNetData/TrainDataForClassicNET/X_Train.npy')
y_train = np.load('./FinalNetData/TrainDataForClassicNET/Y_Train.npy')

X_test = np.load('./FinalNetData/TrainDataForClassicNET/X_Test.npy')
y_test = np.load('./FinalNetData/TrainDataForClassicNET/Y_Test.npy')


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test,dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Создание DataLoader
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=100, shuffle=True)

# Инициализация модели, функции потерь и оптимизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECGNet1D(input_length=input_length, n_classes=n_classes).to(device)
criterion = nn.BCELoss()  # Для многоклассовой бинарной классификации
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
epochs = 100
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        X_batch = X_batch.unsqueeze(1)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# Оценка модели
model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    X_test = X_test.unsqueeze(1)
    y_pred = model(X_test)

    # for a, b in zip(list(y_test), list(y_pred)):
    #     print(a)
    #     input(np.round(b.cpu()))
    #     input('ss')

    accuracy = ((y_pred > 0.5) == y_test).float().mean().item()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


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




test_dir = './FinalNetData/TrainDataForFinalNET/TestData/'
test_pathologies_dirs = scandir(test_dir)

acc_mass = []
for pat_code in test_pathologies_dirs:
    print(pat_code)
    X1_test = np.load(test_dir+pat_code+'/'+pat_code+'_X1_test.npy')
    X2_test = np.load(test_dir +pat_code+'/'+ pat_code + '_X2_test.npy')
    y_test = np.load(test_dir +pat_code+'/'+ pat_code + '_y_test.npy')


    X1_tensor_test = torch.tensor(X1_test, dtype=torch.float32).unsqueeze(1).to(device)  # добавляем измерение канала
    X2_tensor_test = torch.tensor(X2_test, dtype=torch.float32).unsqueeze(1).to(device)
    Y_tensor_test = torch.tensor(y_test, dtype=torch.float32).to(device)


    y_pred = model(X1_tensor_test)

    # for i in range(len(Y_tensor_test)):
    #     print(Y_tensor_test[i])
    #     print(np.round(y_pred.cpu().detach().numpy())[i])
    #
    #
    # print(y_pred.size())



    y_pred_reformat = []
    for y_line in y_pred.cpu().detach().numpy():
        # print(np.round(y_line))
        # print(Y_tensor_test)
        # input('ss')
        pat_val = y_line[pat_dict_enum[pat_code.replace('_','/')]]
        y_pred_reformat.append(np.round(pat_val))

    y_pred_reformat = torch.tensor(y_pred_reformat, dtype=torch.float32).unsqueeze(1).to(device).squeeze()


    res_dict = compute_metrics( y_pred_reformat, Y_tensor_test)

    acc = res_dict["Accuracy"]
    acc_mass.append(acc)

print(np.mean(acc_mass))