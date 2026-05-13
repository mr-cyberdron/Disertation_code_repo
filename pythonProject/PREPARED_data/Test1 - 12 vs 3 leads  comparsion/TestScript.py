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
 #    'LPFB': [None,25],
 #    'WPW': [None,25],
 #    'LMI': [None,25],
 # 'LPR': [None,30],
 # 'RAO/RAE': [None,30],
 # 'ISCIN': [None,30],
 # 'ISCIL': [None,30],
 #    'ISCAS': [None,30],
 #    'IPLMI': [None,40],
 #    'ALMI': [None,50],
 #    'LNGQT': [None,50],
 # 'SBRAD': [None,60],
 # 'CRBBB': [None,60],
 # 'LAO/LAE': [None,60],
 # 'PAC': [None,70],
 #    'SVARR': [None,70],
 #    'CLBBB': [None,70],
 #    'ISCAL': [None,70],
 #    'AMI': [None,70],
 # 'ILMI': [None,70],
 # 'VCLVH': [None,75],
 # 'STACH': [None,100],
 # '1AVB': [None,100],
 #    'IRBBB': [None,100],
 #    'ISC_': [None,100],
 #    'PVC': [None,110],
 #    'LAP': [None,120],
 # 'LVP': [None,120],
 # 'AFIB': [None,120],
 # 'SARRH': [None,120],
 # 'LAFB': [None,130],
    'LVH': [None,150],
    'ASMI': [None,160],
    'NORM': [2000,200],
    'IMI': [None,200]
}

test_slice_onehot_codded = CodeOnehot(test_slice_conf_sorted)

# X_train = np.load('./3 leads/X_train.npy')
# y_train = np.load('./3 leads/Y_train.npy')
#
# X_test = np.load('./3 leads/X_test.npy')
# y_test = np.load('./3 leads/Y_test.npy')

# X_train = np.load('./12 leads/X_train.npy')
# y_train = np.load('./12 leads/Y_train.npy')
#
# X_test = np.load('./12 leads/X_test.npy')
# y_test = np.load('./12 leads/Y_test.npy')

X_train = np.load('./test/X_train.npy')
y_train = np.load('./test/Y_train.npy')

X_test = np.load('./test/X_test.npy')
y_test = np.load('./test/Y_test.npy')

input_length = np.shape(X_train)[2]  # Длина входного сигнала
ch_length = np.shape(X_train)[1]
n_classes = np.shape(y_train)[1]  # Количество выходных классов


# X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=0.99, random_state=42,shuffle=True)
# _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=0.99, random_state=42,shuffle=True)

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.8, random_state=42,shuffle=True)



X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test,dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Создание DataLoader
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=30, shuffle=True)

# Инициализация модели, функции потерь и оптимизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECGNet1D(in_channels=ch_length ,input_length=input_length, n_classes=n_classes).to(device)
# criterion = nn.BCELoss()  # Для многоклассовой бинарной классификации
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
epochs = 150
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




# Оценка модели
model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    y_pred = model(X_test)



    calculate_class_accuracy(y_pred.cpu().numpy(),y_test.cpu().numpy(), test_slice_onehot_codded)








