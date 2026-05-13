import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from FILES_processing_lib import scandir
import torch
import torch.nn as nn

class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv1dWithConstraint, self).forward(x)

class ECGNet1D(nn.Module):
    def __init__(self, in_channels, input_length=5000, n_classes=40, dropout_rate=0.5, kernel_length=64, F1=8, D=2, F2=16):
        super(ECGNet1D, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.input_length = input_length
        self.n_classes = n_classes
        self.kernel_length = kernel_length
        self.dropout_rate = dropout_rate

        # Универсальный вход: количество каналов задается параметром `in_channels`
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, self.F1, self.kernel_length, stride=1, padding=self.kernel_length // 2, bias=False),
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

        # Динамический расчет размера входа для классификатора
        self.flat_size = self._calculate_flat_size(in_channels, input_length)

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, self.n_classes, bias=False),
            nn.Sigmoid()
        )

    def _calculate_flat_size(self, in_channels, input_length):
        """Динамически вычисляет размер выхода перед подачей в fully-connected слой"""
        with torch.no_grad():
            x = torch.zeros(1, in_channels, input_length)  # Универсальный вход
            x = self.block1(x)
            x = self.block2(x)
            return x.numel()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)  # Преобразуем в вектор
        x = self.classifier(x)
        return x


import numpy as np


def calculate_class_accuracy(y_pred, y_true, class_labels):
    """
    Рассчитывает точность классификации отдельно для каждого класса.

    :param y_pred: numpy.array формы (N, C) — предсказания модели (one-hot).
    :param y_true: numpy.array формы (N, C) — реальные метки (one-hot).
    :param class_labels: словарь с соответствием классов и их one-hot кодировкой, например: {'SVTAC': [1, 0], 'PSVT': [0, 1]}

    :return: словарь с точностью определения каждого класса.
    """
    # Преобразуем one-hot в индексы классов
    y_pred_labels = np.argmax(y_pred, axis=1)  # Индексы предсказанных классов
    y_true_labels = np.argmax(y_true, axis=1)  # Индексы истинных классов

    # Создаем словарь с именами классов и их индексами
    label_to_index = {tuple(v): i for i, v in enumerate(class_labels.values())}
    index_to_label = {i: k for k, i in label_to_index.items()}


    class_accuracy = {}

    for class_name, one_hot_vector in class_labels.items():
        class_index = label_to_index[tuple(one_hot_vector)]  # Получаем индекс класса
        # Создаем маску для выборки примеров этого класса
        class_mask = (y_true_labels == class_index)

        # Если в данных нет примеров данного класса, устанавливаем None
        if np.sum(class_mask) == 0:
            class_accuracy[class_name] = None
            continue

        # Подсчитываем верные предсказания для этого класса
        correct_predictions = np.sum((y_pred_labels == class_index) & class_mask)
        total_samples = np.sum(class_mask)

        # Точность = верные предсказания / всего примеров этого класса
        class_accuracy[class_name] = correct_predictions / total_samples

    return class_accuracy





def CodeOnehot(classes:dict):
    # Список всех уникальных классов
    class_list = list(classes.keys())
    # Создаем one-hot encoding словарь
    one_hot_dict = {cls: np.eye(len(class_list), dtype=int)[i].tolist() for i, cls in enumerate(class_list)}
    return one_hot_dict

test_slice_conf_sorted = {
 'SVTAC': [None,20],
 'PSVT': [None,20],
 'BIGU': [None,20],
 'RVH': [None,20],
 'ANEUR': [None,20],
'LPFB': [None,25],
 'WPW': [None,25],
 'LMI': [None,25],
 'LPR': [None,30],
 'RAO/RAE': [None,30],
 'ISCIN': [None,30],
 'ISCIL': [None,30],
 'ISCAS': [None,30],
 'IPLMI': [None,40],
 'ALMI': [None,50],
 'LNGQT': [None,50],
 'SBRAD': [None,60],
 'CRBBB': [None,60],
 'LAO/LAE': [None,60],
 'PAC': [None,70],
 'SVARR': [None,70],
 'CLBBB': [None,70],
 'ISCAL': [None,70],
 'AMI': [None,70],
 'ILMI': [None,70],
 'VCLVH': [None,75],
 'STACH': [None,100],
 '1AVB': [None,100],
 'IRBBB': [None,100],
 'ISC_': [None,100],
 'PVC': [None,110],
 'LAP': [None,120],
 'LVP': [None,120],
 'AFIB': [None,120],
 'SARRH': [None,120],
 'LAFB': [None,130],
 'LVH': [None,150],
 'ASMI': [None,160],
 'NORM': [2000,200],
 'IMI': [None,200]
}

test_slice_onehot_codded = CodeOnehot(test_slice_conf_sorted)


X_train = np.load('./PREPARED_data/ECGnet/Encoded_Data/X_train.npy')
y_train = np.load('./PREPARED_data/ECGnet/Encoded_Data/Y_train.npy')

X_test = np.load('./PREPARED_data/ECGnet/Encoded_Data/X_test.npy')
y_test = np.load('./PREPARED_data/ECGnet/Encoded_Data/Y_test.npy')

input_length = np.shape(X_train)[2]  # Длина входного сигнала
ch_length = np.shape(X_train)[1]
n_classes = np.shape(y_train)[1]  # Количество выходных классов



X_train = torch.tensor(X_train, dtype=torch.float32)
print(f'in_size:{X_train.size()}')
y_train = torch.tensor(y_train, dtype=torch.float32)


X_test = torch.tensor(X_test,dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Создание DataLoader
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=50, shuffle=True)

# Инициализация модели, функции потерь и оптимизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECGNet1D(in_channels=ch_length ,input_length=input_length, n_classes=n_classes).to(device)
criterion = nn.BCELoss()  # Для многоклассовой бинарной классификации
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
epochs = 100
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

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

# Оценка модели
model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    y_pred = model(X_test)

    accuracy = ((y_pred > 0.5) == y_test).float().mean().item()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    acc_res = calculate_class_accuracy(y_pred.cpu().numpy(),y_test.cpu().numpy(), test_slice_onehot_codded)
    for key, value in acc_res.items():
        print(f"{key}: {value:.2f}")







