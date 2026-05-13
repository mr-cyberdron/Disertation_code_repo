import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from Artifitial_signal_creation import AnalogFilterDesign
from collections import Counter
import torch
import pandas as pd
from scipy.signal import resample
import random
import copy
from fastdtw import fastdtw
from Artifitial_signal_creation import simulate_ecg_with_VLP_ALP
from Artifitial_signal_creation import sigTotest
import matplotlib.pyplot as plt
# from CWT_basis import cwt,icwt,morlet_wavelet,scale_to_frequency
from CWT_mod1 import cwt,icwt,morlet_wavelet,scale_to_frequency
import numpy as np
from AnalogFilters import AnalogFilterDesign

def mount_LP_to_ecg(signal,fs, qrs_peaks_pos, LP_sample,LP_start_from_qrs_sec = 0.035):
    signal_to_change = copy.deepcopy(signal)
    LP_start_from_qrs_samp = np.ceil(LP_start_from_qrs_sec*fs)
    for peak_pos in qrs_peaks_pos:
        LP_start = int(peak_pos+LP_start_from_qrs_samp)
        LP_stop = int(peak_pos+LP_start_from_qrs_samp+len(LP_sample))
        if LP_stop<len(signal_to_change):
            signal_to_change[LP_start:LP_stop] = signal_to_change[LP_start:LP_stop]+(LP_sample)
        # peak_to_plot_from = max(0, peak_pos-500)
        # peak_to_plot_to = min(peak_pos + 500, len(signal))
        # plot_fragm = signal_to_change[peak_to_plot_from:peak_to_plot_to]
        # print(np.shape(plot_fragm))
        # plt.plot(plot_fragm)
        # plt.show()
    return signal_to_change


def generate_LP_signal(lp_signal_fs):
    #LP duration = 0.3 sec
    max_len = 2500
    sig_frag_csv = pd.read_csv('artif_samples.csv')

    sigpart1 = list(sig_frag_csv["1"].to_numpy()[0:max_len])
    sigpart2 = list(sig_frag_csv["2"].to_numpy()[0:max_len])
    sigpart3 = list(sig_frag_csv["3"].to_numpy()[0:max_len])
    sigpart4 = list(sig_frag_csv["4"].to_numpy()[0:max_len])

    lp_dur_sec = 0.3
    fs = 10000
    iter_num = 20

    parts_mass = [sigpart1, sigpart2, sigpart3, sigpart4]
    empty_sig = np.zeros(int(lp_dur_sec * fs))

    old_sig = empty_sig
    new_sig = []

    for i in range(iter_num):
        shift = int(np.round(random.random() * len(empty_sig)))
        part = random.choice(parts_mass)
        new_empty_sig = np.zeros(int(lp_dur_sec * fs))
        new_empty_sig[shift:min(len(empty_sig) - 1, shift + max_len)] = part[0:min(len(empty_sig) - 1,
                                                                                   shift + max_len) - shift]
        new_sig = old_sig + new_empty_sig
        old_sig = new_sig

    noize_sig = np.random.normal(0, 0.01, int(lp_dur_sec * fs))
    new_sig = new_sig + noize_sig

    new_sig = AnalogFilterDesign(new_sig, fs).lp(order=5, cutoff=250).zerophaze().bessel().filtration()
    new_sig = AnalogFilterDesign(new_sig, fs).hp(order=5, cutoff=1).zerophaze().bessel().filtration()

    new_sig = new_sig - np.mean(new_sig)
    # Target sampling frequency
    target_sampling_frequency = lp_signal_fs

    length_in_samp = lp_dur_sec * fs
    resampled_new_sig = resample(new_sig, int(length_in_samp * target_sampling_frequency / fs))

    return resampled_new_sig

def get_qrs_annot(pos,types,anot, allowed):
    new_pos = []
    new_types = []
    new_anot = []

    for p,t,a in zip(pos,types,anot):
        if t in allowed:
            new_pos.append(p)
            new_types.append(t)
            new_anot.append(a)
    return np.array(new_pos), np.array(new_types),np.array(new_anot)


def read_np_sample(data_path):
    file_data = np.load(data_path)
    signal = file_data['signals']
    fs = file_data['fs']
    etalon_pos = file_data['pos']
    qrs_annotations = file_data['qrs']
    events_anot = file_data['events']
    etalon_pos,qrs_annotations,events_anot = get_qrs_annot(etalon_pos, qrs_annotations,events_anot, allowed_beat_types)
    return signal, fs,etalon_pos,qrs_annotations,events_anot

def filter_ecg(sig,fs):
    return AnalogFilterDesign(sig,fs).bp(order=5, cutoff=[1, 60]).zerophaze().butter().filtration()

def filter_LP(sig,fs):
    return AnalogFilterDesign(sig,fs).lp(order=5, cutoff=150).zerophaze().butter().filtration()


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

def add_noise(signal, snr_dB):
    # Рассчитываем мощность сигнала в линейной форме
    power_signal = np.mean(np.abs(signal) ** 2)

    # Рассчитываем мощность шума в децибелах
    snr = 10 ** (snr_dB / 10.0)
    power_noise = power_signal / snr

    # Генерируем случайный шум с нужной мощностью
    noise = np.random.normal(0, np.sqrt(power_noise), signal.shape)

    # Добавляем шум к сигналу
    signal_with_noise = signal + noise

    return signal_with_noise

def check_LP_in_avg_sig(sig_fragm, fs,Lp_frag, LP_startpos):
    lp_startpos_samp = np.ceil(LP_startpos*fs)
    Lp_sample_from = int(np.ceil(len(sig_fragm)/2)+lp_startpos_samp)
    Lp_sample_to = int(Lp_sample_from+len(Lp_frag))
    LP_sigpart = sig_fragm[Lp_sample_from:Lp_sample_to]
    LP_sigpart = AnalogFilterDesign(LP_sigpart,fs).hp(order=3, cutoff=18).zerophaze().butter().filtration()

    # distance, path = fastdtw(Lp_frag,LP_sigpart)
    distance = np.corrcoef(Lp_frag,LP_sigpart)[0][1]
    print(distance)
    print(len(Lp_frag))
    print(len(LP_sigpart))
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(Lp_frag)
    # plt.subplot(2,1,2)
    # plt.plot(LP_sigpart)
    # plt.show()
    return distance

def generate_log_scale_system(freq_from, freq_to, N_cofs,fs, w0):
    scale_from = scale_to_frequency([freq_to],w0=w0,fs=fs)[0]
    scale_to = scale_to_frequency([freq_from],w0=w0,fs=fs)[0]

    print(f"""
    log_scale system:
    scale from: {scale_from}
    scale to: {scale_to}
    """)

    N_vect = np.array(list(range(N_cofs)))

    r = (scale_to/scale_from)**(1/(N_cofs-1))
    s_mass = []
    for k in N_vect:
        s_val = scale_from*r**k
        s_mass.append(s_val)

    return s_mass

sample_data_path = "./test_signals/"
allowed_beat_types = ["N","V","A","F","a","E","R",
                      "/","j","L","f","!","S","r","e","n"]
def check_LA_components_selection():
    file = 'p8_500.npz'
    signal, fs, etalon_pos, qrs_annotations, events_anot = read_np_sample(sample_data_path + file)

    signal = signal[0]

    Lp_signal = generate_LP_signal(fs)*0.25#0.08

    Lp_signal = filter_LP(Lp_signal,fs)
    Lp_signal = Lp_signal[0:50]
    lp_startpos = 0.035
    ecg_with_lp = mount_LP_to_ecg(signal,fs,etalon_pos,Lp_signal, LP_start_from_qrs_sec = lp_startpos)

    ecg_with_lp = add_noise(ecg_with_lp,130)

    filtered_sig = filter_ecg(ecg_with_lp, fs)
    stat, sig_peaks = nk.ecg_peaks(filtered_sig, sampling_rate=fs)  # kalidas2017
    detected_peaks = sig_peaks['ECG_R_Peaks']

    # plt.figure()
    # print('card_detector')
    # avg_card_detector = calc_avg_card(ecg_with_lp,fs,detected_peaks)

    # corr_coef_card_detector = check_LP_in_avg_sig(avg_card_detector, fs, Lp_signal, lp_startpos)
    # print(f'corr_coef:{corr_coef_card_detector}')
    # x = np.array(list(range(len(avg_card_detector))))/fs
    # plt.plot(x, avg_card_detector)
    # plt.xlabel('Time[sec]')
    # plt.ylabel('Amplitude[mV]')
    # plt.show(block = False)

    plt.figure()
    print('etalon_qrs')
    avg_card_etalon = calc_avg_card(ecg_with_lp,fs,etalon_pos)
    corr_coef_compared_etalon = check_LP_in_avg_sig(avg_card_etalon,fs,Lp_signal,lp_startpos)
    print(f'corr_coef:{corr_coef_compared_etalon}')
    x = np.array(list(range(len(avg_card_etalon))))/fs
    plt.plot(x, avg_card_etalon)
    plt.xlabel('Time[sec]')
    plt.ylabel('Amplitude[mV]')
    plt.show(block = False)

    w0 = 6
    sig = avg_card_etalon
    scales = generate_log_scale_system(40, 200, 40, fs, w0)
    t = np.array(list(range(len(sig)))) / fs

    cwt_coefficients = cwt(sig, scales, morlet_wavelet, dt=1, fs=fs, w0=w0, plot_wavelets_spectrum=False)
    reconstructed_signal = icwt(cwt_coefficients, scales, morlet_wavelet, dt=1, ds=1)

    corr_coef_compared_etalon = check_LP_in_avg_sig(reconstructed_signal, fs, Lp_signal, lp_startpos)

    fig, axs = plt.subplots(3, 1, figsize=(7, 5), gridspec_kw={'height_ratios': [2, 1, 1]})

    axs[1].plot(t, sig, label='Original Signal')
    axs[1].legend()
    axs[1].set_ylabel('Amplitude [mV]')

    # start_time = 0.093
    # end_time = 0.1267
    # start_time2 = 0.2484
    # end_time2 = 0.2807
    # red_part = copy.deepcopy(signal)
    # indices_to_keep = np.r_[np.round(start_time * fs).astype(int):np.round(end_time * fs).astype(int),
    #                   np.round(start_time2 * fs).astype(int):np.round(end_time2 * fs).astype(
    #                       int)]  # объединяем диапазоны, Python использует включительные диапазоны для срезов
    # red_part[~np.in1d(np.arange(red_part.size), indices_to_keep)] = np.nan  # заменяем все, что вне диапазонов
    #
    # axs[1].plot(t, red_part, c='r')

    freqs = scale_to_frequency(scales, w0, fs)
    X, Y = np.meshgrid(t, freqs)
    im = axs[0].pcolormesh(X, Y, np.abs(cwt_coefficients), cmap='viridis')  # Используем цветовую карту 'viridis'
    plt.colorbar(im, ax=axs[0])
    plt.colorbar(im, ax=axs[1])
    axs[0].set_ylabel('Frequency [Hz]')
    axs[0].set_title('Wavelet transform')

    axs[2].plot(t, reconstructed_signal, label='Reconstructed Signal', color='orange')
    axs[2].legend()
    plt.colorbar(im, ax=axs[2])
    axs[2].set_xlabel('time [s]')
    axs[2].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


check_LA_components_selection()
