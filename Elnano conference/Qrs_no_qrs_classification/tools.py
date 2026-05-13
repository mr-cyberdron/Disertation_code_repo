import random

import numpy as np

import Plottermaan_lib
from FILES_processing_lib import scandir
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy import signal

def prep_qrs_data(sigs, fs, anot_pos,anot_names,allowed_beat_types): # preprocessing for neurel network
    signals = filter_signals(sigs,fs, type='hp') #baseline removal
    signals = normalize_ecg_signals(signals,fs,anot_pos)
    qrs_samples = select_qrs_samples(signals,fs,anot_pos,anot_names, allowed_beat_types)
    return qrs_samples

def get_wfdb_names(files_path):
    def remove_format(fnames_mass):
        cleaned_fnames = []
        for fname  in fnames_mass:
            fmt = fname.split('.')[-1]
            new_fname = fname.replace(f'.{fmt}','')
            cleaned_fnames.append(new_fname)
        return cleaned_fnames
    hea_files = remove_format(scandir(files_path,'.hea'))
    dat_files = remove_format(scandir(files_path,'.dat'))
    atr_files = remove_format(scandir(files_path,'.atr'))
    wfdb_names = []
    for file in hea_files:
        if file in dat_files and file in atr_files:
            wfdb_names.append(file)
    return wfdb_names

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

def normalize_ecg_signals(signals,fs, peaks_pos):
    norm_sigmas = []
    for sig in signals:
        sig = sig - np.mean(sig)
        sig_max = get_ecg_max_amp(sig,fs,peaks_pos)
        sig_normalised = sig/sig_max
        norm_sigmas.append(sig_normalised)
    return np.array(norm_sigmas)

def prepQRS(qrs_fragm):
    desired_length = 64
    qrs_fragm = resample(qrs_fragm, desired_length)
    return qrs_fragm

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

def generate_fake_qrs(signal_mass, fs, step_sec_from = 0.4, step_sec_to = 0.6):
    sample_mass = []
    sample_types = []

    signal_samples_len = np.shape(signal_mass)[1]
    samples_sequence = list(range(signal_samples_len))
    max_step_sample = round(step_sec_to*fs)
    qrs_pos_old = 0
    while qrs_pos_old+max_step_sample < signal_samples_len:
        pos_step = random.uniform(step_sec_from,step_sec_to)
        qrs_pos_total = qrs_pos_old+round(pos_step*fs)
        total_sample = samples_sequence[qrs_pos_total]
        qrs_pos_old = qrs_pos_total
        sample_mass.append(total_sample)
        sample_types.append('N')
    return  np.array(sample_mass), np.array(sample_types)


def resample_signal_to_fs(original_signal, original_fs, target_fs):
    # Calculate the resampling factor
    resample_factor = target_fs / original_fs
    # Use the scipy resample function
    resampled_signal = signal.resample(original_signal, int(len(original_signal) * resample_factor))

    return resampled_signal


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





