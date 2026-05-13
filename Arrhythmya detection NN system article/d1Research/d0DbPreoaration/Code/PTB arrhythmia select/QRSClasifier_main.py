from models import QRSnet1,QRSnet2
import torch
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
import numpy as np
from scipy.signal import resample
import neurokit2 as nk
import matplotlib.pyplot as plt

def normalize_ecg_signals(signals,fs, peaks_pos):
    norm_sigmas = []
    for sig in signals:
        sig = sig - np.mean(sig)
        sig_max = get_ecg_max_amp(sig,fs,peaks_pos)
        sig_normalised = sig/sig_max
        norm_sigmas.append(sig_normalised)
    return np.array(norm_sigmas)

def get_ecg_max_amp(signal,fs,peaks_pos):
    boundaries = np.ceil(0.05*fs)
    peaks_amps_mass = []
    for ps in peaks_pos:
        b_backward = int(ps-boundaries)
        b_forward = int(ps+boundaries)
        if b_backward>0:
            local_max_abs = np.max(np.abs(signal[b_backward:b_forward]))
            peaks_amps_mass.append(local_max_abs)
    return np.mean(peaks_amps_mass)

def prep_qrs_data(sigs, fs, anot_pos,anot_names,allowed_beat_types): # preprocessing for neurel network
    signals = filter_signals(sigs,fs, type='hp') #baseline removal
    signals = normalize_ecg_signals(signals,fs,anot_pos)
    qrs_samples = select_qrs_samples(signals,fs,anot_pos,anot_names, allowed_beat_types)
    return qrs_samples

def select_qrs_samples(signals,fs,annotations_pos,qrs_annotations, allowed_annotations):
    boundaries_sec = 0.5
    boundaries_samp = boundaries_sec*fs
    qrs_samples_mass = []
    for pos,anot in zip(annotations_pos,qrs_annotations):
        from_sample = int(pos-boundaries_samp)
        to_samp = int(pos+boundaries_samp)
        if from_sample >=0 and anot in allowed_annotations:
            signals_fragm = signals[:,from_sample:to_samp]
            for sig_lead in signals_fragm:
                qrs_samples_mass.append(prepQRS(sig_lead))

    return np.array(qrs_samples_mass)

def prepQRS(qrs_fragm):
    desired_length = 64
    qrs_fragm = resample(qrs_fragm, desired_length)
    return qrs_fragm

def filter_signals(signals,fs,freq_hp = 0.5,freq_lp = 60, type = 'bp'):
    new_sigmas = []
    for singal in signals:
        filtered_signal = None
        if type == 'bp':
            filtered_signal = AnalogFilterDesign(singal, fs).bp(order=5, cutoff=[freq_hp,freq_lp]).zerophaze().butter().filtration()
        if type == 'hp':
            filtered_signal = AnalogFilterDesign(singal, fs).hp(order=5, cutoff=freq_hp).zerophaze().butter().filtration()
        if type == 'lp':
            filtered_signal = AnalogFilterDesign(singal, fs).lp(order=5, cutoff=freq_lp).zerophaze().butter().filtration()
        new_sigmas.append(filtered_signal)
    return np.array(new_sigmas)

def generate_Rpeaks_anot(peaks_pos):
    qrs = []
    anot = []
    for pos in peaks_pos:
        qrs.append('N')
        anot.append('')
    return np.array(qrs), np.array(anot)

def filter_ecg(sig,fs):
    return AnalogFilterDesign(sig,fs).bp(order=5, cutoff=[1, 60]).zerophaze().butter().filtration()

def calc_grops_procentage(res_mass):
    unique_res = np.unique(res_mass)
    res_counter_dict = {}
    for res_item in unique_res:
        counter_val = res_mass.count(res_item)
        res_counter_dict[res_item] = np.round((counter_val/len(res_mass))*100,0)
    return res_counter_dict




def classify_qrs_by_nn(qrs_samples):
    similarity_treshold = 0.95
    qrs_samples = torch.tensor([qrs_samples]).float()
    model_path = './baclup QRSnn2/model_weights.pth'
    model = QRSnet2()
    # model_path = './model_backup_QRSnn1/model_weights.pth'
    # model = QRSnet1()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    qrs_samples = qrs_samples.unsqueeze(1) #OPTIONAL
    model_res = model(qrs_samples).squeeze()
    res_item = model_res.item()
    if res_item>similarity_treshold:
        return 1
    else:
        return 0


def ProcessQRSqual(signal, fs):
    allowed_beat_types = ["N", "V", "A", "F", "a", "E", "R",
                          "/", "j", "L", "f", "!", "S", "r", "e", "n"]
    filtered_sig = filter_ecg(signal, fs)
    stat, sig_peaks = nk.ecg_peaks(filtered_sig, sampling_rate=fs, method="neurokit")  # kalidas2017
    detected_peaks = sig_peaks['ECG_R_Peaks']
    f_qrs, _ = generate_Rpeaks_anot(detected_peaks)
    prep_qrs = prep_qrs_data([signal], fs, detected_peaks, f_qrs, allowed_beat_types)

    clasif_res_mass = []
    counter = 1
    for qrs_samp, pos, anot in zip(prep_qrs, detected_peaks, f_qrs):
        # print(f'{counter}/{len(f_qrs)}')
        counter+=1
        classif_res = classify_qrs_by_nn(qrs_samp)
        clasif_res_mass.append(classif_res)
        # if classif_res == 1:
        #     plt.figure()
        #     plt.plot(qrs_samp)
        #     plt.title(str(classif_res))
        #     plt.show()
    qual_dict = calc_grops_procentage(clasif_res_mass)
    try:
        qrs_qual = qual_dict[1]
    except:
        qrs_qual = 0.0
    return qrs_qual




if __name__ == "__main__":
    file = './poore_ecg_signal/p1.npz'
    file_data = np.load(file)
    signal = file_data['signals'][0]
    fs = file_data['fs']
    good_qual_procentage = ProcessQRSqual(signal, fs)
    print(f'Signal_good_qual_procentage: {good_qual_procentage}')