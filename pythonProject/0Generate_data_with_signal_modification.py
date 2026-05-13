import copy
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb_no_ann,write_wfdb
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
import neurokit2 as nk
from sklearn.decomposition import PCA
from SigPrep import calcResVector,CardAveragingPlusDownsampling,generate_encoder_features, calc_avg_card_for_ecg
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import resample


def filter_ecg(sig,fs):
    return AnalogFilterDesign(sig,fs).bp(order=5, cutoff=[1, 60]).zerophaze().butter().filtration()
def filter50(sig,fs):
    return AnalogFilterDesign(sig, fs).zerophaze().notch(cutoff=50, quality_factor=50).filtration()
def filter_ecg2(sig_mass,fs):
    new_mass = []
    for sig in sig_mass:
        filt_Sig = AnalogFilterDesign(sig,fs).bp(order=5, cutoff=[0.5, 150]).zerophaze().butter().filtration()
        filt_Sig = filter50(filt_Sig,fs)
        new_mass.append(filt_Sig)
    return np.array(new_mass)
def calc_avg_card(signal,fs, qrs_poss):
    card_boundaries_sec = 0.5
    card_boundaries_samples = int(round(card_boundaries_sec*fs))
    fragms_mass = []
    for pos in qrs_poss:
        pos_from = pos-card_boundaries_samples
        pos_to = pos+card_boundaries_samples
        target_fragm_len = pos_to-pos_from
        signal_fragm = signal[pos_from:pos_to]
        if len(signal_fragm)== target_fragm_len:
            fragms_mass.append(signal_fragm)
    avg_card = np.array(fragms_mass).mean(axis=0)
    return avg_card
def transform_12_lead_ECG_to_XYZ(input_ecg_mass,lead_names):
    leads_12 = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    assert lead_names == leads_12, "lead names should be " +str(leads_12)
    transformation_matrix_12 = np.array([
        [0.632, 0.235, -0.397, -0.434, 0.515, -0.081, -0.515, 0.044, 0.882, 1.213, 2.125, 0.831],
        [-0.235, 1.066, 1.301, -0.415, -0.768, 1.184, 0.157, 0.164, 0.098, 0.127, 0.127, 0.076],
        [0.059, -0.132, -0.191, 0.037, 0.125, -0.162, -0.917, -1.387, -1.277, -0.604, -0.086, 0.230]
    ])
    ECG_transformed = np.dot(transformation_matrix_12, input_ecg_mass)
    return ECG_transformed
def compute_resultant_vector(xyz_signals):
    X, Y, Z = xyz_signals
    return np.sqrt(X**2 + Y**2 + Z**2)
def normalize_signal(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
def fix_length(signal, target_length):
    current_length = len(signal)
    if current_length < target_length:
        return np.pad(signal, (0, target_length - current_length), mode='constant')
    else:
        return signal[:target_length]
def generate_avg_card(input_sig,fs):
    sig = filter50(input_sig, fs)
    filtered_sig = filter_ecg(sig, fs)
    stat, sig_peaks = nk.ecg_process(filtered_sig, sampling_rate=fs, method='neurokit')  # kalidas2017
    detected_peaks = sig_peaks['ECG_R_Peaks']
    avg_card = calc_avg_card(sig, fs, detected_peaks)
    return avg_card
def CodeOnehot(classes:dict):
    # Список всех уникальных классов
    class_list = list(classes.keys())
    # Создаем one-hot encoding словарь
    one_hot_dict = {cls: np.eye(len(class_list), dtype=int)[i].tolist() for i, cls in enumerate(class_list)}
    return one_hot_dict
def pca_sum_first_5(X):
    pca = PCA(n_components=5)
    pca.fit(X)  # Обучаем PCA
    components = pca.components_  # (5, 5000)
    # Суммируем первые 5 главных компонент
    summed_components = np.sum(components, axis=0)  # (5000,)
    return summed_components
def generate_char_vector(input_array):
    char_vector_res = []
    for i in range(np.shape(input_array)[1]):
        lead_slice = input_array[:,i,:]
        pca_res_lead = pca_sum_first_5(lead_slice)
        char_vector_res.append(pca_res_lead)

    generated_vector = np.stack(char_vector_res)
    return generated_vector
def create_char_vectors (input_data,n_vectors = 20, out_len = 100):
    prepared_char_vectors = []
    for i in range(n_vectors):
        char_slice = input_data.sample(n= 100).reset_index(drop=True)
        char_slice_mass = np.stack(char_slice['Signals_prep_mass'])
        char_vector = generate_char_vector(char_slice_mass)
        prepared_char_vectors.append(char_vector)
    resulted_char_vectors = []
    for j in range(out_len):
        picked_char_vecttor = random.choice(prepared_char_vectors)
        resulted_char_vectors.append(picked_char_vecttor)

    return np.stack(resulted_char_vectors)
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def perform_pca_and_plot(signal, fs=500):
    """
    Выполняет PCA над 12-канальным ЭКГ сигналом размерности (12, 500)
    и отображает 3 главные компоненты на отдельных субплотах со временем по оси X.

    Параметры:
    signal : np.ndarray
        Массив размерности (12, 500)
    fs : int
        Частота дискретизации сигнала (Гц)
    """
    if signal.shape[0] != 12:
        raise ValueError("Ожидается сигнал размерности (12, 500)")

    # Настройки шрифта
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['legend.fontsize'] = 12
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11

    signal_T = signal.T  # (500, 12)
    time = np.arange(signal_T.shape[0]) / fs  # в секундах

    # PCA
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(signal_T)

    # Построение графиков
    fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
    # fig.suptitle("Компоненти розкладу ЕКГ сигналу", fontsize=16)

    for i in range(3):
        axes[i].plot(time, principal_components[:, i], label=f'Компонент {i + 1}', linewidth=1.5)
        axes[i].set_ylabel("Амп. [мВ]")
        axes[i].grid(True, linestyle='--', linewidth=0.5)
        axes[i].legend(loc='upper right')
        axes[i].tick_params(axis='x', which='both', labelbottom=True)

    axes[0].set_title("Компоненти розкладу ЕКГ сигналу", fontsize=16)
    axes[-1].set_xlabel("Час [сек]")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return principal_components



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

db_initial_path = "D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
metadata_file = "D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/metadata_normalised_LP_atached.csv"

data_prep_dict = {
    'Signals_prep_mass' : [],
    'pat_codes_mass' : [],
    'train_test_class' : []
}
metadata_Db = pd.read_csv(metadata_file)
metadata_to_process = copy.deepcopy(metadata_Db)
conter = 0
for target_pat in list(test_slice_conf_sorted.keys()):
    conter+=1
    print(f'{conter}/{len(list(test_slice_conf_sorted.keys()))}')
    db_slice_with_pat = metadata_to_process[metadata_to_process['scp_codes'].str.contains(target_pat, case=False, na=False)].drop(columns=['Unnamed: 0'], errors='ignore')
    if test_slice_conf_sorted[target_pat][0] is not None:
        db_slice_with_pat = db_slice_with_pat.sample(n=test_slice_conf_sorted[target_pat][0], random_state=42)
    local_counter = 0
    for i,row in db_slice_with_pat.iterrows():
        local_counter+=1
        print(test_slice_conf_sorted[target_pat][1])
        print(f'{conter}/{len(list(test_slice_conf_sorted.keys()))} ({local_counter})')
        if local_counter>300:
            break

        # if local_counter>500:
        #     break

        filepath = db_initial_path+row['filename_hr']
        signals, re_format, fs, units, adc_gain, baseline, \
            coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb_no_ann(filepath)

        # -----------------------------------------DataPrep-----------------------------------------------------
        # 0)--No changes--
        signals = filter_ecg2(signals, fs)
        Signals_Prepared = signals


        # perform_pca_and_plot(Signals_Prepared2)

        # print(np.shape(Signals_Prepared))
        # 1)--Resulting Vector Calc--
        # Signals_Prepared = calcResVector(Signals_Prepared)
        # 2)--CalcXYZECG--
        # Signals_Prepared = transform_12_lead_ECG_to_XYZ(Signals_Prepared,['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])

        # 3)--ReduceLeadsNumbber to  I,II,V1,V4,V6--
        # leads_12 = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        # target_leads_mass = [leads_12.index('I'),leads_12.index('II'),leads_12.index('V1'),leads_12.index('V4'),leads_12.index('V6')]
        # Signals_Prepared = Signals_Prepared[target_leads_mass, :]

        # 4)--Card Averaging Plus DownSampling--
        # Signals_Prepared = CardAveragingPlusDownsampling(Signals_Prepared,500)
        # Добавить даунсемплинг
        # 5)--autoencoder With attention--

        # Signals_Prepared = transform_12_lead_ECG_to_XYZ(Signals_Prepared,
        #                                                 ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4',
        #                                                  'V5', 'V6'])

        try:
            Signals_Prepared2 = calc_avg_card_for_ecg(Signals_Prepared, 500)
        except:
            continue


        def apply_pca(signal, n_components=3):
            signal_T = signal.T

            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(signal_T)

            # Транспонируем обратно: (components, samples)
            return transformed.T

        Signals_Prepared2 =apply_pca(Signals_Prepared2)
        # Signals_Prepared2 = calcResVector(Signals_Prepared2)


        # def resample_signal(signal, original_fs, target_fs):
        #     num_channels, num_samples = signal.shape
        #     target_num_samples = int(num_samples * target_fs / original_fs)
        #
        #     resampled_signal = resample(signal, target_num_samples, axis=1)
        #     return resampled_signal


        # Signals_Prepared2 = resample_signal(Signals_Prepared2,500,75)




        # xyzECG_encoded = generate_encoder_features(Signals_Prepared,
        #                                                     model_path='./TrainAutoencoder/Spline_decoder_ECG_rec_weights.pth',
        #                                                     num_classes=4, logit_len=6000)

        # avgCard_encoded = generate_encoder_features(Signals_Prepared2,
        #                                             model_path='./TrainAutoencoder/Spline_decoder_AvC_weights.pth',
        #                                             num_classes=15, logit_len=64)

        # Signals_Prepared = np.concatenate([avgCard_encoded, xyzECG_encoded], axis=0)
        Signals_Prepared = Signals_Prepared2
        # Signals_Prepared = avgCard_encoded
        # for autoencoder training:
        # Signals_Prepared = transform_12_lead_ECG_to_XYZ(Signals_Prepared,['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
        # try:
        #     Signals_Prepared2 = CardAveragingPlusDownsampling(Signals_Prepared, 500)[:,0:500]
        # except:
        #     print('skipped')
        #     continue
        # Signals_Prepared = np.concatenate([Signals_Prepared2,Signals_Prepared],axis=1)



        # ------------------------------------------------------------------------------------------------------
        data_prep_dict['Signals_prep_mass'].append(np.array(Signals_Prepared))
        data_prep_dict['pat_codes_mass'].append(target_pat)
        if local_counter>test_slice_conf_sorted[target_pat][1]:
            data_prep_dict['train_test_class'].append('Train')
        else:
            data_prep_dict['train_test_class'].append('Test')
    prep_data_len = len(data_prep_dict['Signals_prep_mass'])
    print(f'{target_pat}:{prep_data_len}')

signal_prep_df = pd.DataFrame(data=data_prep_dict)

Train_df = signal_prep_df[signal_prep_df['train_test_class']=='Train']
Test_df = signal_prep_df[signal_prep_df['train_test_class']=='Test']

# -----------------ECGNET data--------------
X_ecgnet_train = np.stack(Train_df['Signals_prep_mass'].to_numpy())
X_ecgnet_test = np.stack(Test_df['Signals_prep_mass'].to_numpy())

Y_ecgnet_train = np.stack(Train_df['pat_codes_mass'].map(test_slice_onehot_codded).to_numpy())
Y_ecgnet_test = np.stack(Test_df['pat_codes_mass'].map(test_slice_onehot_codded).to_numpy())

# np.save('./PREPARED_data/ECGnet/X_train.npy',X_ecgnet_train)
# np.save('./PREPARED_data/ECGnet/X_test.npy',X_ecgnet_test)
# np.save('./PREPARED_data/ECGnet/Y_train.npy',Y_ecgnet_train)
# np.save('./PREPARED_data/ECGnet/Y_test.npy',Y_ecgnet_test)

data_p = './PREPARED_data/ECGnet/SplineDecoder'
# data_p = './PREPARED_data/ECGnet/SplineDecoder'
# data_p = './PREPARED_data/ECGnet/lead_12_data'
# data_p = './PREPARED_data/ECGnet/XYZ_DATA'

# data_p = './PREPARED_data/ECGnet/FlatLinearDecoder'


np.save(f'{data_p}/X_train.npy',X_ecgnet_train)
np.save(f'{data_p}/X_test.npy',X_ecgnet_test)
np.save(f'{data_p}/Y_train.npy',Y_ecgnet_train)
np.save(f'{data_p}/Y_test.npy',Y_ecgnet_test)
exit()
# ------------------SiamseNet data---------------

X_train = []
X1_train = []

X_test = []
X1_test = []

Y_train = []
Y_test = []

pat_codes = np.unique(signal_prep_df['pat_codes_mass'].to_numpy())
print('Create_siamse_dataframe:')
counter2 = 0
for code in pat_codes:
    print(code)
    counter2+=1
    print(f'{counter2}/{len(pat_codes)}')
    train_code_slice = Train_df[Train_df['pat_codes_mass'] == code]
    test_code_slice = Test_df[Test_df['pat_codes_mass'] == code]
    not_code_slice = Train_df[Train_df['pat_codes_mass'] != code]

    true_char_vect = create_char_vectors(train_code_slice, out_len=len(train_code_slice)+len(test_code_slice))
    false_char_vect = create_char_vectors(not_code_slice, out_len=len(train_code_slice) + len(test_code_slice))

    X_train_part = np.concatenate([np.stack(train_code_slice['Signals_prep_mass']),
                                  np.stack(train_code_slice['Signals_prep_mass'])],axis=0)

    X1_train_part = np.concatenate([true_char_vect[0:len(train_code_slice),:,:],
                                    false_char_vect[0:len(train_code_slice),:,:]],axis=0)
    # 1- true 0 - false
    Y_train_part = np.concatenate([np.ones(len(train_code_slice)),
                                   np.zeros(len(train_code_slice))],axis=0)

    X_test_part = np.concatenate([np.stack(test_code_slice['Signals_prep_mass']),
                                  np.stack(test_code_slice['Signals_prep_mass'])],axis=0)
    X1_test_part = np.concatenate([true_char_vect[len(train_code_slice):,:,:],
                                    false_char_vect[len(train_code_slice):,:,:]],axis=0)
    Y_test_part = np.concatenate([np.ones(len(test_code_slice)),
                                   np.zeros(len(test_code_slice))],axis=0)

    floder_name = code.replace('/','_')
    store_path = f'./PREPARED_data/TestData/{floder_name}/'
    create_folder_if_not_exists(store_path)
    np.save(store_path+'X_test.npy', X_test_part)
    np.save(store_path + 'X1_test.npy', X1_test_part)
    np.save(store_path + 'Y_test.npy', Y_test_part)

    if len(X_train)>0:
        X_train = np.concatenate([X_train,X_train_part],axis=0)
    else:
        X_train = X_train_part

    if len(X_test)>0:
        X_test = np.concatenate([X_test,X_test_part],axis=0)
    else:
        X_test = X_test_part

    if len(X1_train)>0:
        X1_train = np.concatenate([X1_train,X1_train_part],axis=0)
    else:
        X1_train = X1_train_part

    if len(X1_test) > 0:
        X1_test = np.concatenate([X1_test, X1_test_part], axis=0)
    else:
        X1_test = X1_test_part

    if len(Y_train) > 0:
        Y_train = np.concatenate([Y_train, Y_train_part], axis=0)
    else:
        Y_train = Y_train_part

    if len(Y_test) > 0:
        Y_test = np.concatenate([Y_test, Y_test_part], axis=0)
    else:
        Y_test = Y_test_part

np.save('./PREPARED_data/SiamseNet/X_train.npy',X_train)
np.save('./PREPARED_data/SiamseNet/X1_train.npy',X1_train)
np.save('./PREPARED_data/SiamseNet/X_test.npy',X_test)
np.save('./PREPARED_data/SiamseNet/X1_test.npy',X1_test)
np.save('./PREPARED_data/SiamseNet/Y_train.npy',Y_train)
np.save('./PREPARED_data/SiamseNet/Y_test.npy',Y_test)

