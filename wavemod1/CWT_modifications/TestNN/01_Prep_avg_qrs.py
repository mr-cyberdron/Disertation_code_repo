import matplotlib.pyplot as plt
import wfdb
import os
import numpy as np
from scipy.signal import resample
from Dataformats_tools.EDF import save_edf

#data https://www.kaggle.com/datasets/bjoernjostein/normalsinusdataset?resource=download-directory

def resample_signal(signal, original_fs, target_fs):
    """
    Изменяет частоту дискретизации сигнала.

    :param signal: массив сигнала, который нужно изменить.
    :param original_fs: исходная частота дискретизации сигнала.
    :param target_fs: целевая частота дискретизации.
    :return: сигнал с измененной частотой дискретизации.
    """
    # Расчет количества отсчетов в целевом сигнале
    num_samples = int(len(signal) * target_fs / original_fs)

    # Изменение частоты дискретизации
    resampled_signal = resample(signal, num_samples)

    return resampled_signal


files_path = 'D:/Bases/only_sinus/'
bigfiles_store = 'D:/Bases/only_sinus/biger100/'


hea_files = [file for file in os.listdir(files_path) if file.endswith('.' + 'hea')]
counter = 0
c2 = 0
for file in hea_files:
    c2+=1
    print(f'{c2}/{len(hea_files)}')
    try:
        record = wfdb.rdrecord(files_path+file.replace('.hea',''))
        signals = record.p_signal
        signals = signals.T
        fs = record.fs
        siglen = len(signals[0])/fs
        if siglen> 100:
            counter+=1
            counter_2 = 0
            for signal in signals:
                counter_2+=1
                sig_prepared = resample_signal(signal,fs,500)*1000*1000
                save_edf(np.array([sig_prepared]),500,p=bigfiles_store+f'_{counter}_{counter_2}.edf')


    except:
        pass

print(counter)