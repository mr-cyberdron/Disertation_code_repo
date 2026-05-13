import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from FILES_processing_lib import scandir
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

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



# Слой с ограничением max-norm (если ты его уже определял отдельно)
class Conv1dWithConstraint2(nn.Conv1d):
    def __init__(self, *args, max_norm=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = self.weight.renorm(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)

class ECGNet1D2(nn.Module):
    def __init__(self, in_channels=1, input_length=1300, n_classes=40, dropout_rate=0.5, kernel_length=64, F1=8, D=2, F2=16):
        super(ECGNet1D2, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.input_length = input_length
        self.n_classes = n_classes
        self.kernel_length = kernel_length
        self.dropout_rate = dropout_rate

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

        # Авторасчет размера выходного тензора
        self.flat_size = self._calculate_flat_size(in_channels, input_length)

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, self.n_classes, bias=False),
            nn.Sigmoid()
        )

    def _calculate_flat_size(self, in_channels, input_length):
        with torch.no_grad():
            x = torch.zeros(1, in_channels, input_length)
            x = self.block1(x)
            x = self.block2(x)
            return x.numel()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LinearNet(nn.Module):
    def __init__(self, in_channels, input_length=5000, n_classes=40, dropout_rate=0.0):
        super(LinearNet, self).__init__()
        self.input_length = input_length
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.in_channels = in_channels

        # Calculate the input size for the fully connected network
        # Input size = in_channels * input_length (e.g., for in_channels=5 and input_length=2500, input_size=12500)
        self.input_size = in_channels * input_length

        # Define the fully connected architecture as per the image
        # Input: 12,500 (in_channels * input_length should match this, adjust if needed)
        # Hidden Layer 1: 6,000 neurons, ReLU
        # Hidden Layer 2: 2,000 neurons, ReLU
        # Hidden Layer 3: 500 neurons, Sigmoid
        # Output Layer: 15 neurons (we'll adjust to n_classes)

        self.fc_network = nn.Sequential(
            # Input layer to first hidden layer (12,500 -> 6,000)
            nn.Linear(self.input_size, 6000, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),

            # First hidden layer to second hidden layer (6,000 -> 2,000)
            nn.Linear(6000, 2000, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),

            # Second hidden layer to third hidden layer (2,000 -> 500)
            nn.Linear(2000, 500, bias=True),
            nn.Sigmoid(),
            nn.Dropout(p=self.dropout_rate),

            # Third hidden layer to output layer (500 -> n_classes)
            nn.Linear(500, self.n_classes, bias=True)
            # No activation here; activation can be applied based on the task (e.g., Sigmoid for binary classification)
        )

    def forward(self, x):
        # Flatten the input: (batch_size, in_channels, input_length) -> (batch_size, in_channels * input_length)
        x = x.view(x.size(0), -1)

        # Pass through the fully connected network
        x = self.fc_network(x)
        return x


def CodeOnehot(classes:dict):
    # Список всех уникальных классов
    class_list = list(classes.keys())
    # Создаем one-hot encoding словарь
    one_hot_dict = {cls: np.eye(len(class_list), dtype=int)[i].tolist() for i, cls in enumerate(class_list)}
    return one_hot_dict

test_slice_conf_sorted = {
 # 'SVTAC': [None,20],
 # 'PSVT': [None,20],
 # 'BIGU': [None,20],
 # 'RVH': [None,20],
 #    'ANEUR': [None,20],
    'LPFB': [None,25], #0.84 0.88
    'WPW': [None,25], #0.6 0.8
    'LMI': [None,25], #0.67 0.75
 # 'LPR': [None,30],
 # 'RAO/RAE': [None,30],
 'ISCIN': [None,30], #0.67 0.8
 'ISCIL': [None,30], #0.23 0.6
    'ISCAS': [None,30], #0.43 0.67
    'IPLMI': [None,40],#0.18  0.63
    # 'ALMI': [None,50],#0 0.52
 #    'LNGQT': [None,50],
 # 'SBRAD': [None,60],
 # 'CRBBB': [None,60],
 # 'LAO/LAE': [None,60],
 # 'PAC': [None,70],
 #    'SVARR': [None,70],
    'CLBBB': [None,70],#0.89 0.92
    'ISCAL': [None,70],#0.5 0.74
    'AMI': [None,70],#0.37 0.63
 # 'ILMI': [None,70],#0.07 0.57
 # 'VCLVH': [None,75],
 # 'STACH': [None,100],
 # '1AVB': [None,100],
 #    'IRBBB': [None,100],
 #    'ISC_': [None,100],
 #    'PVC': [None,110],
    'LAP': [None,120],#0.86 0.92
 'LVP': [None,120],#0.04 0.93
 # 'AFIB': [None,120],
 # 'SARRH': [None,120],
 # 'LAFB': [None,130],
    'LVH': [None,150],#0.38 0.65
    # 'ASMI': [None,160],#0.12 0.37
    'NORM': [2000,160],#0.73 0.88
    'IMI': [None,160]#0.27 0.53
}


test_slice_onehot_codded = CodeOnehot(test_slice_conf_sorted)

# path = 'SplineDecoder'
# X_train = np.load(f'./ECGnet/{path}/X_train.npy')
# X_train = X_train[:, np.newaxis, :]
# y_train = np.load(f'./ECGnet/{path}/Y_train.npy')
# X_test = np.load(f'./ECGnet/{path}/X_test.npy')
# X_test = X_test[:, np.newaxis, :]
# y_test = np.load(f'./ECGnet/{path}/Y_test.npy')
#

path = 'SplineDecoder'
X_train = np.load(f'./ECGnet/{path}/X_train.npy')
X_train = X_train[:,:,0:500]
y_train = np.load(f'./ECGnet/{path}/Y_train.npy')
X_test = np.load(f'./ECGnet/{path}/X_test.npy')
X_test = X_test[:,:,0:500]
y_test = np.load(f'./ECGnet/{path}/Y_test.npy')
# X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=0.5, random_state=42,shuffle=True)


# X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=0.2, random_state=42,shuffle=True)
# _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=0.99, random_state=42,shuffle=True)

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.8, random_state=42,shuffle=True)

input_length = np.shape(X_train)[2]  # Длина входного сигнала
print(input_length)
ch_length = np.shape(X_train)[1]
n_classes = np.shape(y_train)[1]  # Количество выходных классов
print(n_classes)


X_train = torch.tensor(X_train, dtype=torch.float32)
print(X_train.size())
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test,dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Создание DataLoader
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=30, shuffle=True)

# Инициализация модели, функции потерь и оптимизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ECGNet1D(in_channels=ch_length ,input_length=input_length, n_classes=n_classes).to(device)
model = LinearNet(in_channels=ch_length ,input_length=input_length, n_classes=n_classes).to(device)

# criterion = nn.BCELoss()  # Для многоклассовой бинарной классификации
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
epochs = 50#50 #800
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)


        # loss = criterion(y_pred, y_batch)

        new_y_batch =  torch.argmax(y_batch, dim=1)
        loss = criterion(y_pred, new_y_batch)


        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

def clean_fom_lead1(keys_Vec, labels_lev):
    new_keys_mass = []
    for vec, lev_lab in zip(keys_Vec, labels_lev):
        vec[lev_lab] = 0.0
        new_keys_mass.append(vec)
    return np.array(new_keys_mass)


def calculate_class_accuracy(y_pred, y_true, class_labels):

    y_pred_to_analyse = copy.deepcopy(y_pred)

    y_pred_labels_lev1 = np.argmax(y_pred_to_analyse, axis=1)  # Индексы предсказанных классов
    y_true_labels_lev12 = np.argmax(y_true, axis=1)  # Индексы истинных классов

    y_pred_to_analyse = clean_fom_lead1(y_pred_to_analyse, y_pred_labels_lev1)

    y_pred_labels_lev2 = np.argmax(y_pred_to_analyse, axis=1)  # Индексы предсказанных классов


    total_counter = 0
    true_lev1 = 0
    false_lev1 = 0
    true_lev2 = 0
    false_lev2 = 0
    clas_res_dict = {pat_key:{'l1':0,'l2':0,'tc':0} for pat_key in class_labels.keys()}
    for pred_lev1, pred_lev2, true_lev12 in zip(y_pred_labels_lev1,  y_pred_labels_lev2, y_true_labels_lev12):
        total_counter +=1
        clas_res_dict[list(class_labels.keys())[true_lev12]]['tc'] += 1

        if true_lev12 == pred_lev1:
            true_lev1+=1
            clas_res_dict[list(class_labels.keys())[true_lev12]]['l1']+=1
        else:
            false_lev1+=1

        if true_lev12 == pred_lev2:
            true_lev2+=1
            clas_res_dict[list(class_labels.keys())[true_lev12]]['l2'] += 1
        else:
            false_lev2+=1

    l1_accuracy = true_lev1/total_counter
    l2_accuracy = true_lev2/total_counter
    l1_l2_accuracy = (true_lev1+true_lev2)/total_counter

    print(f"l1_accuracy: {l1_accuracy:.4f}, l2_accuracy: {l2_accuracy:.4f}, l1+l2_accuracy: {l1_l2_accuracy:.4f}")

    l1_ac_mass = []
    l2_ac_mass = []
    l1_l2_ac_mass = []
    for c_n, v_d in zip(clas_res_dict.keys(),clas_res_dict.values()):
        print(c_n)
        if v_d['tc'] == 0:
            print('No_obs')
        else:
            c_l1_ac = v_d['l1']/v_d['tc']
            c_l2_ac = v_d['l2']/v_d['tc']
            c_l1_l2_ac = (v_d['l1']+v_d['l2']) / v_d['tc']

            l1_ac_mass.append(c_l1_ac)
            l2_ac_mass.append(c_l2_ac)
            l1_l2_ac_mass.append(c_l1_l2_ac)

            print(
                f"{c_n}_l1_acc: {c_l1_ac:.4f}, {c_n}_l2_acc: {c_l2_ac:.4f}, {c_n}_l1+l2_acc: {c_l1_l2_ac:.4f}")
    c_l1_ac_mean = float(np.mean(l1_ac_mass))
    c_l2_ac_mean = float(np.mean(l2_ac_mass))
    c_l1_l2_ac_mean =float(np.mean(l1_l2_ac_mass))
    print(f"Classes avg acc: l1:{c_l1_ac_mean:.4f}, l2:{c_l2_ac_mean:.4f}, l1+l2:{c_l1_l2_ac_mean:.4f}")

def calculate_class_accuracy2(y_pred, y_true, class_labels):
    y_pred_to_analyse = copy.deepcopy(y_pred)
    y_pred_labels_lev1 = np.argmax(y_pred_to_analyse, axis=1)  # Индексы предсказанных классов для l1
    y_true_labels_lev12 = np.argmax(y_true, axis=1)  # Индексы истинных классов

    y_pred_to_analyse = clean_fom_lead1(y_pred_to_analyse, y_pred_labels_lev1)
    y_pred_labels_lev2 = np.argmax(y_pred_to_analyse, axis=1)  # Индексы предсказанных классов для l2

    total_counter = 0
    true_lev1 = 0
    false_lev1 = 0
    true_lev2 = 0
    false_lev2 = 0

    # Словарь для хранения результатов по классам
    clas_res_dict = {pat_key: {'l1': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'tc': 0},
                               'l2': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'tc': 0}}
                     for pat_key in class_labels.keys()}

    # Подсчет TP, FP, TN, FN для каждого класса
    for pred_lev1, pred_lev2, true_lev12 in zip(y_pred_labels_lev1, y_pred_labels_lev2, y_true_labels_lev12):
        total_counter += 1
        true_class = list(class_labels.keys())[true_lev12]
        clas_res_dict[true_class]['l1']['tc'] += 1
        clas_res_dict[true_class]['l2']['tc'] += 1

        # Уровень l1
        if true_lev12 == pred_lev1:
            true_lev1 += 1
            clas_res_dict[true_class]['l1']['TP'] += 1
            # Увеличиваем TN для всех остальных классов
            for other_class in class_labels.keys():
                if other_class != true_class:
                    clas_res_dict[other_class]['l1']['TN'] += 1
        else:
            false_lev1 += 1
            clas_res_dict[true_class]['l1']['FN'] += 1
            pred_class_l1 = list(class_labels.keys())[pred_lev1]
            clas_res_dict[pred_class_l1]['l1']['FP'] += 1
            # Увеличиваем TN для всех классов, кроме истинного и предсказанного
            for other_class in class_labels.keys():
                if other_class != true_class and other_class != pred_class_l1:
                    clas_res_dict[other_class]['l1']['TN'] += 1

        # Уровень l2
        if true_lev12 == pred_lev2:
            true_lev2 += 1
            clas_res_dict[true_class]['l2']['TP'] += 1
            # Увеличиваем TN для всех остальных классов
            for other_class in class_labels.keys():
                if other_class != true_class:
                    clas_res_dict[other_class]['l2']['TN'] += 1
        else:
            false_lev2 += 1
            clas_res_dict[true_class]['l2']['FN'] += 1
            pred_class_l2 = list(class_labels.keys())[pred_lev2]
            clas_res_dict[pred_class_l2]['l2']['FP'] += 1
            # Увеличиваем TN для всех классов, кроме истинного и предсказанного
            for other_class in class_labels.keys():
                if other_class != true_class and other_class != pred_class_l2:
                    clas_res_dict[other_class]['l2']['TN'] += 1

    # Вычисление общей точности
    l1_accuracy = true_lev1 / total_counter
    l2_accuracy = true_lev2 / total_counter
    l1_l2_accuracy = (true_lev1 + true_lev2) / total_counter

    print(f"l1_accuracy: {l1_accuracy:.4f}, l2_accuracy: {l2_accuracy:.4f}, l1+l2_accuracy: {l1_l2_accuracy:.4f}")

    # Списки для хранения метрик по классам
    l1_sensitivity_mass = []
    l1_specificity_mass = []
    l2_sensitivity_mass = []
    l2_specificity_mass = []
    l1_ac_mass = []
    l2_ac_mass = []
    l1_l2_ac_mass = []

    # Вычисление метрик для каждого класса
    for c_n, v_d in clas_res_dict.items():
        print(c_n)
        if v_d['l1']['tc'] == 0:
            print('No_obs')
            continue

        # Точность для l1 и l2
        c_l1_ac = v_d['l1']['TP'] / v_d['l1']['tc']
        c_l2_ac = v_d['l2']['TP'] / v_d['l2']['tc']
        c_l1_l2_ac = (v_d['l1']['TP'] + v_d['l2']['TP']) / v_d['l1']['tc']

        # Чувствительность и специфичность для l1
        l1_sensitivity = v_d['l1']['TP'] / (v_d['l1']['TP'] + v_d['l1']['FN']) if (v_d['l1']['TP'] + v_d['l1']['FN']) > 0 else 0
        l1_specificity = v_d['l1']['TN'] / (v_d['l1']['TN'] + v_d['l1']['FP']) if (v_d['l1']['TN'] + v_d['l1']['FP']) > 0 else 0

        # Чувствительность и специфичность для l2
        l2_sensitivity = v_d['l2']['TP'] / (v_d['l2']['TP'] + v_d['l2']['FN']) if (v_d['l2']['TP'] + v_d['l2']['FN']) > 0 else 0
        l2_specificity = v_d['l2']['TN'] / (v_d['l2']['TN'] + v_d['l2']['FP']) if (v_d['l2']['TN'] + v_d['l2']['FP']) > 0 else 0

        # Добавление в списки
        l1_ac_mass.append(c_l1_ac)
        l2_ac_mass.append(c_l2_ac)
        l1_l2_ac_mass.append(c_l1_l2_ac)
        l1_sensitivity_mass.append(l1_sensitivity)
        l1_specificity_mass.append(l1_specificity)
        l2_sensitivity_mass.append(l2_sensitivity)
        l2_specificity_mass.append(l2_specificity)

        print(f"{c_n}_l1_acc: {c_l1_ac:.4f}, {c_n}_l2_acc: {c_l2_ac:.4f}, {c_n}_l1+l2_acc: {c_l1_l2_ac:.4f}")
        print(f"{c_n}_l1_sensitivity: {l1_sensitivity:.4f}, {c_n}_l1_specificity: {l1_specificity:.4f}")
        print(f"{c_n}_l2_sensitivity: {l2_sensitivity:.4f}, {c_n}_l2_specificity: {l2_specificity:.4f}")

    # Средние значения метрик по классам
    c_l1_ac_mean = float(np.mean(l1_ac_mass)) if l1_ac_mass else 0
    c_l2_ac_mean = float(np.mean(l2_ac_mass)) if l2_ac_mass else 0
    c_l1_l2_ac_mean = float(np.mean(l1_l2_ac_mass)) if l1_l2_ac_mass else 0
    c_l1_sensitivity_mean = float(np.mean(l1_sensitivity_mass)) if l1_sensitivity_mass else 0
    c_l1_specificity_mean = float(np.mean(l1_specificity_mass)) if l1_specificity_mass else 0
    c_l2_sensitivity_mean = float(np.mean(l2_sensitivity_mass)) if l2_sensitivity_mass else 0
    c_l2_specificity_mean = float(np.mean(l2_specificity_mass)) if l2_specificity_mass else 0

    print(f"Classes avg acc: l1:{c_l1_ac_mean:.4f}, l2:{c_l2_ac_mean:.4f}, l1+l2:{c_l1_l2_ac_mean:.4f}")
    print(f"Classes avg sensitivity: l1:{c_l1_sensitivity_mean:.4f}, l2:{c_l2_sensitivity_mean:.4f}")
    print(f"Classes avg specificity: l1:{c_l1_specificity_mean:.4f}, l2:{c_l2_specificity_mean:.4f}")



# Оценка модели
model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    y_pred = model(X_test)

    # calculate_class_accuracy(y_pred.cpu().numpy(),y_test.cpu().numpy(), test_slice_onehot_codded)
    calculate_class_accuracy2(y_pred.cpu().numpy(), y_test.cpu().numpy(), test_slice_onehot_codded)








