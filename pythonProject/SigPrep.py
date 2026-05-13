import matplotlib.pyplot as plt
import numpy as np
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
import neurokit2 as nk
from scipy.signal import resample
import torch
from TrainAutoencoder.Arch import SimpleECG_Autoencoder

def filter_ecg(sig,fs):
    return AnalogFilterDesign(sig,fs).bp(order=5, cutoff=[1, 60]).zerophaze().butter().filtration()
def calc_avg_card(signal,fs, qrs_poss,card_boundaries_sec = 0.5):
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

def Remove_avg_card(input_sig,fs, avg_card,detected_peaks, card_boundaries_sec = 0.5):
    card_boundaries_samples = int(round(card_boundaries_sec*fs))
    new_sig = np.zeros(len(input_sig))
    for pos in detected_peaks:
        pos_from = pos-card_boundaries_samples
        pos_to = pos+card_boundaries_samples
        target_fragm_len = pos_to-pos_from
        signal_fragm = input_sig[pos_from:pos_to]
        if len(signal_fragm)== target_fragm_len:
            removing_result = signal_fragm-avg_card
            new_sig[pos_from:pos_to] = removing_result

    return new_sig

def generate_avg_card(input_sig,fs, detected_peaks):
    avg_card = calc_avg_card(input_sig, fs, detected_peaks)
    return avg_card
def downsample_signal(signal: np.ndarray, fs1: int, fs2: int) -> np.ndarray:
    if fs2 >= fs1:
        raise ValueError("fs2 должна быть меньше fs1 для даунсемплинга.")
    num_samples = int(len(signal) * fs2 / fs1)  # Вычисляем новое количество точек
    downsampled_signal = resample(signal, num_samples)  # Даунсемплируем
    return downsampled_signal

def calcResVector(ecgleads):
    resulting_vector = np.expand_dims(np.sqrt(np.sum(ecgleads ** 2, axis=0)),axis=0)
    return resulting_vector

def CardAveragingPlusDownsampling(ecgleads, fs):
    res_mass = []

    sig = ecgleads[1,:]
    filtered_sig = filter_ecg(sig, fs)
    stat, sig_peaks = nk.ecg_process(filtered_sig, sampling_rate=fs, method='neurokit')  # kalidas2017
    detected_peaks = sig_peaks['ECG_R_Peaks']

    for lead in ecgleads:
        lead_avg_calc = generate_avg_card(lead,fs, detected_peaks)
        lead_avg_card_removed = Remove_avg_card(lead, fs,lead_avg_calc, detected_peaks)
        sig_downsampled = downsample_signal(lead_avg_card_removed,fs,64)
        result_lead_transformed = np.concatenate([lead_avg_calc, sig_downsampled],axis=0)
        res_mass.append(result_lead_transformed)
    return np.stack(res_mass)


def calc_avg_card_for_ecg(ecgleads, fs):
    res_mass = []

    sig = ecgleads[1, :]
    filtered_sig = filter_ecg(sig, fs)
    stat, sig_peaks = nk.ecg_process(filtered_sig, sampling_rate=fs, method='neurokit')  # kalidas2017
    detected_peaks = sig_peaks['ECG_R_Peaks']

    for lead in ecgleads:
        lead_avg_calc = generate_avg_card(lead, fs, detected_peaks)
        res_mass.append(lead_avg_calc)
    return np.stack(res_mass)


def generate_encoder_features(input_sig, model_path, num_classes = 3, logit_len = 300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def use_encoder(input_ecg, model):
        model.eval()
        ecg_data = torch.from_numpy(input_ecg).float().unsqueeze(0).to(device)  # [3, 5500]
        output, latent, logits = model(ecg_data)
        # output, latent = model(ecg_data)
        # output, mu, log_var, latent = model(ecg_data)
        return latent[0].detach().cpu().numpy()

    inp_len = np.shape(input_sig[2])[0]
    # model = SimpleECG_Autoencoder(num_classes=num_classes, logit_len=logit_len, input_len=inp_len).to(device)
    from TrainAutoencoder import Arch3
    model = Arch3.SymmetricAutoencoderWithClassifier2().to(device)

    model.load_state_dict(torch.load(model_path))
    return use_encoder(input_sig,model)


