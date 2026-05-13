import copy

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Arch import SimpleECG_Autoencoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import Arch3
import torch.nn.functional as F
import random
from matplotlib import rcParams


def plot_vectors(model, ecg_data, sample_idx=0):

    # Преобразуем данные и выбираем одну запись
    ecg_tensor = torch.from_numpy(ecg_data).float()
    device = next(model.parameters()).device
    sample = ecg_tensor[sample_idx:sample_idx + 1].to(device)  # [1, 3, 5500]

    # Получаем выход и латентное представление
    model.eval()
    with torch.no_grad():
        output, latent, logits = model(sample)
        # output, latent= model(sample)
        # output, mu, log_var, latent = model(sample)

    # Преобразуем в numpy
    input_data = sample.cpu().numpy()[0]  # [3, 5500]
    output_data = output.cpu().numpy()[0]  # [3, 5500]
    latent_data = latent.cpu().numpy()[0]  # [1000]



    # Первый график: вход и выход для первого отведения
    plt.figure(figsize=(12, 4))
    plt.subplot(3,2,1)
    plt.plot(input_data[0], label='Input (Lead 1)', alpha=0.7)
    plt.subplot(3, 2, 2)
    plt.plot(output_data[0], label='Output (Lead 1)', alpha=0.7)
    plt.subplot(3, 2, 3)
    plt.plot(input_data[1], label='Input (Lead 2)', alpha=0.7)
    plt.subplot(3, 2, 4)
    plt.plot(output_data[1], label='Output (Lead 2)', alpha=0.7)
    plt.subplot(3, 2, 5)
    plt.plot(input_data[2], label='Input (Lead 3)', alpha=0.7)
    plt.subplot(3, 2, 6)
    plt.plot(output_data[2], label='Output (Lead 3)', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Второй график: все входные каналы и латентное представление
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(4,1,i+1)
        plt.plot(input_data[i], label=f'Input Lead {i + 1}', alpha=0.7)
    plt.subplot(4, 1, 4)
    plt.plot(latent_data, label='Latent', alpha=0.7)
    plt.title('Input Channels and Latent Representation')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_leads_mse_stepwise(vector1, vector2):
    """
    vector1: torch.Tensor of shape (batch_size, 3, sequence_length) - например (200, 3, 5500)
    vector2: torch.Tensor of shape (batch_size, sequence_length) - например (200, 1000)
    """
    batch_size = vector1.size(0)

    # Обрезаем до минимальной длины
    min_length = min(vector1.size(2), vector2.size(1))
    min_length = 500
    vector1 = vector1[:, :, :min_length]
    vector2 = vector2[:, :min_length]

    # Инициализируем MSE loss без редукции, чтобы получить loss для каждого элемента батча
    # mse_loss = torch.nn.MSELoss(reduction='none')
    mse_loss = torch.nn.MSELoss(reduction='none')

    lead_losses = mse_loss(vector1.mean(dim = 1), vector2).mean(dim=1)  # shape: (batch_size)


    # Усредняем все значения
    total_mean = lead_losses.mean()

    return total_mean


def margin_loss(logits, target, margin=1.0):
    correct_logits = logits[range(len(logits)), target]
    max_incorrect_logits = logits.clone()
    max_incorrect_logits[range(len(logits)), target] = -float('inf')
    top_wrong_logits = max_incorrect_logits.max(dim=1)[0]
    loss = torch.clamp(margin - (correct_logits - top_wrong_logits), min=0)
    return loss.mean()

def focal_loss(logits, targets_onehot, gamma=2.0):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    focal_weight = torch.pow(1.0 - probs, gamma)
    loss = -targets_onehot * focal_weight * log_probs
    return loss.sum(dim=1).mean()

def confidence_penalty_loss(logits, targets_onehot, beta=0.1):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
    xent = F.cross_entropy(logits, torch.argmax(targets_onehot, dim=1))
    return xent - beta * entropy.mean()

def logit_separation_loss(logits, targets, margin=1.0):
    batch_size = logits.size(0)
    true_logits = logits[torch.arange(batch_size), targets]
    mask = torch.ones_like(logits).bool()
    mask[torch.arange(batch_size), targets] = False
    wrong_logits = logits[mask].view(batch_size, -1)
    max_wrong_logits, _ = wrong_logits.max(dim=1)
    return F.relu(margin - (true_logits - max_wrong_logits)).mean()

def custom_contrastive_loss(logits, targets_onehot, margin=2.0):
    probs = F.softmax(logits, dim=1)
    true_class_probs = torch.sum(probs * targets_onehot, dim=1)
    contrast = probs - targets_onehot
    contrast_penalty = torch.sum(torch.clamp(margin - true_class_probs.unsqueeze(1) + contrast, min=0)**2, dim=1)
    return contrast_penalty.mean()

class CompositeLoss(nn.Module):
    def __init__(self, MSE_alpha=1.0, CE_beta=1.0):
        super(CompositeLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()  # Потеря для классификации
        self.alpha = MSE_alpha  # Вес для косинусного сходства
        self.beta = CE_beta  # Вес для Cross-Entropy


    def forward(self, output, target, class_logits, labels, latent):
        recon_loss = self.mse_loss(output, target)  # [200]
        # class_loss = self.ce_loss(class_logits, labels)  # Скаляр


        log_probs = F.log_softmax(class_logits, dim=1)
        target_probs = F.one_hot(labels.to(torch.int64), num_classes=class_logits.size(1)).float()
        # criterion = torch.nn.KLDivLoss(reduction='batchmean')
        # class_loss = criterion(log_probs,target_probs)

        # class_loss = margin_loss(log_probs,labels)
        #
        class_loss = focal_loss(log_probs, target_probs)

        # class_loss = confidence_penalty_loss(log_probs, target_probs)

        # class_loss = custom_contrastive_loss(log_probs, target_probs)

        # Комбинируем потери
        total_loss = self.alpha * recon_loss + self.beta * class_loss #+class_loss1+class_loss2

        return total_loss, recon_loss, class_loss  # Все скаляры


# Функция потерь VAE
def vae_loss(reconstructed_x, x, mu, log_var):
    """Сумма reconstruction loss и KL-дивергенции"""
    # Reconstruction loss (MSE или BCE в зависимости от данных)
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')

    # KL-дивергенция: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_div

def train_model(ecg_data, ecg_classes, epochs=20, batch_size=32, logit_len = 1000, input_len=5500):
    # Преобразуем данные в тензоры
    ecg_data = torch.from_numpy(ecg_data).float()  # [1363, 3, 5500]
    ecg_classes = torch.from_numpy(ecg_classes).long()  # [1363], метки классов

    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Инициализация модели (укажите правильное число классов)
    num_classes = len(np.unique(ecg_classes))  # Автоматически определяем число классов
    # model = SimpleECG_Autoencoder(num_classes=num_classes, logit_len=logit_len, input_len=input_len).to(device)
    model = Arch3.SymmetricAutoencoderWithClassifier2().to(device)
    # Оптимизатор и критерий
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CompositeLoss(MSE_alpha=0.9, CE_beta=0.1)  # Исправлены имена аргументов
    # criterion = CompositeLoss(MSE_alpha=0.9, CE_beta=0.01)  # Исправлены имена аргументов



    # criterion = nn.MSELoss()

    # Создаем DataLoader с данными и метками
    dataset = TensorDataset(ecg_data, ecg_classes)  # Добавляем метки в датасет
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    recon_loss_mass =  []
    clss_loss_mass = []
    recon_loss_mass2 = []
    clss_loss_mass2 = []
    # Цикл обучения
    for epoch in range(epochs):
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_class_loss = 0
        num_batches = 0

        # Итерация по батчам
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)  # [batch_size, 3, 5500]
            batch_labels = batch_labels.to(device)  # [batch_size]

            optimizer.zero_grad()

            # Прямой проход
            # output, latent, logits = model(batch_data)
            output, latent, logits = model(batch_data)


            # Вычисление композитного лосса
            total_loss, recon_loss, class_loss = criterion(output, batch_data, logits, batch_labels, latent)
            recon_loss_mass.append(recon_loss.cpu().detach().numpy())
            clss_loss_mass.append(class_loss.cpu().detach().numpy())

            # total_loss = criterion(output,batch_data)
            # recon_loss = 0
            # class_loss = 0

            # Обратное распространение
            total_loss.backward()
            optimizer.step()

            # Суммируем потери для статистики
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_class_loss += class_loss.item()
            num_batches += 1

        # Вычисляем средние потери за эпоху
        avg_total_loss = epoch_total_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_class_loss = epoch_class_loss / num_batches

        recon_loss_mass2.append(np.average(recon_loss_mass))
        clss_loss_mass2.append(np.average(clss_loss_mass))

        # Выводим статистику
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Total Loss: {avg_total_loss:.4f}, "
              f"Recon Loss: {avg_recon_loss:.4f}, "
              f"Class Loss: {avg_class_loss:.4f}")

    np.save('./recon_loss_mass2.npy', np.array(recon_loss_mass2))
    np.save('./class_loss_mass2.npy', np.array(clss_loss_mass2))


    return model



# Использование
# ecg_data = np.load('./AutoencoderData/X_train.npy') #(1363, 3, 5500)
# ecg_classes = np.load('./AutoencoderData/Y_train.npy')

# ecg_data = np.load('../PREPARED_data/ECGnet/XYZ_DATA/X_train.npy') #(1363, 3, 5500)
# ecg_classes = np.load('../PREPARED_data/ECGnet/XYZ_DATA/Y_train.npy')

ecg_data = np.load('../PREPARED_data/ECGnet/lead_12_data/X_train.npy') #(1363, 3, 5500)
print(np.shape(ecg_data))
ecg_classes = np.load('../PREPARED_data/ECGnet/lead_12_data/Y_train.npy')


# ecg_data = np.load('../PREPARED_data/ECGnet/SplineDecoder/X_train.npy') #(1363, 3, 5500)
# print(np.shape(ecg_data))
# ecg_classes = np.load('../PREPARED_data/ECGnet/SplineDecoder/Y_train.npy')


ecg_classes = np.argmax(ecg_classes,axis=1)
# ecg_classes = pd.read_csv('ECG_params_df_clusterised.csv')['Cluster'].to_numpy()

# ecg_data, _, ecg_classes, _ = train_test_split(ecg_data, ecg_classes, train_size=0.3, random_state=42,shuffle=True)

# ecg_data = ecg_data[:, :, 500:]
# model = train_model(ecg_data, ecg_classes, epochs=100, batch_size=200, logit_len=6000,input_len=5000)
# torch.save(model.state_dict(), "ECG_rec_weights.pth")
# torch.save(model.state_dict(), "Spline_decoder_ECG_rec_weights.pth")
# torch.save(model.state_dict(), "Flat_linear_decoder_ECG_rec_weights.pth")


ecg_data = ecg_data[:, :, 0:500]
model = train_model(ecg_data, ecg_classes, epochs=500, batch_size=30, logit_len=64,input_len=500)
torch.save(model.state_dict(), "Spline_decoder_AvC_weights.pth")


# ecg_data = ecg_data[:, :, 0:5000]
# model = train_model(ecg_data, ecg_classes, epochs=120, batch_size=50, logit_len=64,input_len=500)
# torch.save(model.state_dict(), "Spline_decoder_AvC_weights.pth")




# plot_vectors(model, ecg_data, sample_idx=0)



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


def CodeOnehot(classes:dict):
    # Список всех уникальных классов
    class_list = list(classes.keys())
    # Создаем one-hot encoding словарь
    one_hot_dict = {cls: np.eye(len(class_list), dtype=int)[i].tolist() for i, cls in enumerate(class_list)}
    return one_hot_dict

test_slice_onehot_codded = CodeOnehot(test_slice_conf_sorted)

def calculate_class_accuracy(y_pred, y_true, class_labels):
    def clean_fom_lead1(keys_Vec, labels_lev):
        new_keys_mass = []
        for vec, lev_lab in zip(keys_Vec, labels_lev):
            vec[lev_lab] = 0.0
            new_keys_mass.append(vec)
        return np.array(new_keys_mass)

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



def plot_autoenc_comp(init_sig, rec_sig, latent_vec):
    init_sig = init_sig[0,0,:]
    rec_sig = rec_sig[0,0, :]
    latent_vec = latent_vec[0]

    # Настройки шрифта
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['legend.fontsize'] = 12
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11

    # Построение графиков
    fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=False)
    # fig.suptitle("Компоненти розкладу ЕКГ сигналу", fontsize=16)


    axes[0].plot(init_sig, label=f'Ориг. сигнал (1/12)', linewidth=1.5)
    axes[0].set_ylabel("Амп. [мВ]")
    axes[0].grid(True, linestyle='--', linewidth=0.5)
    axes[0].legend(loc='upper right')
    axes[0].tick_params(axis='x', which='both', labelbottom=True)


    axes[1].plot(rec_sig, label=f'Реконструкція (1/12)', linewidth=1.5)
    axes[1].set_ylabel("Амп. [мВ]")
    axes[1].grid(True, linestyle='--', linewidth=0.5)
    axes[1].legend(loc='upper right')
    axes[1].tick_params(axis='x', which='both', labelbottom=True)

    axes[2].plot(latent_vec, label=f'Скорочений вектор ознак', linewidth=1.5)
    axes[2].set_ylabel("Амп.")
    axes[2].grid(True, linestyle='--', linewidth=0.5)
    axes[2].legend(loc='upper right')
    axes[2].tick_params(axis='x', which='both', labelbottom=True)
    axes[2].set_xlim(0,240)

    axes[0].set_title("Результати роботи автоенкодера", fontsize=16)
    axes[-1].set_xlabel("Семпли")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def test_clasifier():

    # X_test = np.load('../PREPARED_data/ECGnet/XYZ_DATA/X_test.npy')
    # X_test = X_test[:, :, 0:500]
    # y_test = np.load('../PREPARED_data/ECGnet/XYZ_DATA/Y_test.npy')

    X_test = np.load('../PREPARED_data/ECGnet/lead_12_data/X_test.npy')
    X_test = X_test[:, :, 0:500]
    y_test = np.load('../PREPARED_data/ECGnet/lead_12_data/Y_test.npy')

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.float32)
    # Оценка модели

    model.to('cpu')
    model.eval()
    with torch.no_grad():
        X_test = X_test.to('cpu')
        y_test = y_test.to('cpu')

        output, latent, y_pred = model(X_test)

        mse_loss = nn.MSELoss()
        recon_loss = mse_loss(output, X_test)
        print(f'rec_loss:{recon_loss}')


        plot_autoenc_comp(X_test,output, latent)



        calculate_class_accuracy(y_pred.cpu().numpy(), y_test.cpu().numpy(), test_slice_onehot_codded)

test_clasifier()




