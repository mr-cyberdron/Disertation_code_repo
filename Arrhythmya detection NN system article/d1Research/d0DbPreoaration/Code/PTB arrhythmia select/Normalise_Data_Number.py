import copy

import numpy as np
import pandas as pd
from ClassificationScheme import data_normalisation_scheme
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb_no_ann,write_wfdb
import matplotlib.pyplot as plt
from QRSClasifier_main import ProcessQRSqual,filter_ecg
import neurokit2 as nk
import random
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
from tools import plotmultileadSig
from scipy.signal import resample


def reduceTest(wfdbPath):
    low_qual_treshold = 92
    low_qual_num_treshold = 2
    avg_qual_treshold = 92

    signals, re_format, fs, units, adc_gain, baseline, \
        coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb_no_ann(wfdbPath)
    qual_proc_list = []
    for lead_signal in signals:
        try:
            qual_proc = ProcessQRSqual(lead_signal,fs)
        except:
            qual_proc = 0
        qual_proc_list.append(qual_proc)

    avg_qual = np.mean(qual_proc_list)
    qual_proc_list = np.array(qual_proc_list)
    low_qual_sihnals_num = np.sum(qual_proc_list<low_qual_treshold)



    if avg_qual<avg_qual_treshold:
        return True
    if low_qual_sihnals_num>=low_qual_num_treshold:
        return True
    else:
        return False

def detect_R_peaks(sig, fs):
    filtered_sig = filter_ecg(sig, fs)
    stat, sig_peaks = nk.ecg_peaks(filtered_sig, sampling_rate=fs, method="neurokit")  # kalidas2017
    detected_peaks = sig_peaks['ECG_R_Peaks']
    return detected_peaks

def avgRpeakAmp(sig,fs):
    peaks = detect_R_peaks(sig,fs)
    peaks_amp = sig[peaks]
    return np.mean(peaks_amp)


def ECGmassFilt(ecg_sigs, fs):
    new_sig_mass = []
    for ecg_lead_sig in ecg_sigs:
        new_sig_mass.append(filter_ecg(ecg_lead_sig,fs))
    return new_sig_mass


def add_noise_parts_with_snr(signal, snr_from, snr_to):
    noise_parts_number = random.randint(2, 5)
    snr_ranges = np.linspace(snr_from,snr_to,noise_parts_number)
    noise_signal_cuts = np.round(np.linspace(0,len(signal),noise_parts_number+1))
    signal_ranges = []
    for i in range(noise_parts_number):
        signal_ranges.append([noise_signal_cuts[i],noise_signal_cuts[i+1]])
    noise_line = np.zeros(len(signal))

    for sig_range in signal_ranges:
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (random.choice(snr_ranges) / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)
        noise_part_shape = sig_range[1]-sig_range[0]
        noise = np.random.normal(0, noise_std, size=int(noise_part_shape))
        noise_line[int(sig_range[0]):int(sig_range[1])] = noise

    noisy_signal = signal+noise_line

    return noisy_signal

def addIsolineDrift(signal, fs):
    R_amp = avgRpeakAmp(signal, fs)
    noise = np.random.normal(0, abs(R_amp), size=len(signal))
    noise_filtered = AnalogFilterDesign(noise, fs).bp(order=5, cutoff=[0.01,1]).zerophaze().butter().filtration()
    signalIsolined = signal+noise_filtered
    return signalIsolined

def addBreathNoise(signal,fs,RespRate):
    dur = len(signal)/fs
    rsp7 = nk.rsp_simulate(duration=int(np.round(dur)), respiratory_rate=RespRate, method="breathmetrics")
    resampled_resp = resample(np.array(rsp7), int(len(signal)))
    resped_sig = signal+resampled_resp
    return resped_sig

def addMioNoise(signal,fs):
    dur = len(signal) / fs
    emg5 = nk.emg_simulate(duration=int(np.round(dur)), burst_number=random.randint(1, 4), burst_duration=random.randint(2, 10)/10)
    resampled_mio = resample(np.array(emg5), int(len(signal)))*np.std(signal)*0.2
    sig_mio_noised = signal+resampled_mio
    return sig_mio_noised


def expandSig(filePath, metadataDbToNormalize, degree):
    signals, re_format, fs, units, adc_gain, baseline, \
        coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb_no_ann(filePath)
    filtered_signals = ECGmassFilt(signals,fs)
    def expand12lead(sig12lead, fs, brethRate):
        new12lead_mass = []
        for leadSig in sig12lead:
            randomNoisedSig = add_noise_parts_with_snr(leadSig, 18,40)
            drifted_sig = addIsolineDrift(randomNoisedSig,fs)
            brethaddSig = addBreathNoise(drifted_sig,fs, brethRate)
            mioNoisedSig = addMioNoise(brethaddSig,fs)
            new12lead_mass.append(mioNoisedSig)
        return new12lead_mass

    counter1 = 0
    for deg in range(degree):
        counter1+=1
        expand_part = expand12lead(filtered_signals,fs, random.randint(10, 15))
        filePath_to_index = '/'.join(filePath.split('/')[-3:])
        metadata_line = metadataDbToNormalize[metadataDbToNormalize['filename_hr'] == filePath_to_index]
        new_rec_name =filePath_to_index.split('/')[-1]+f'_{counter1}'
        new_fpath = 'expandSig/'+new_rec_name
        full_fpath_to_save  = dbPath+'expandSig/'
        new_metadata_line = copy.deepcopy(metadata_line)
        new_metadata_line['filename_hr'] = new_fpath
        new_metadata_line['filename_lr'] = None
        try:
            write_wfdb(new_rec_name, np.array(expand_part).T, fs, re_format, units,None,
                       None,coments, lead_names,p=full_fpath_to_save)
            metadataDbToNormalize = pd.concat([metadataDbToNormalize,new_metadata_line], ignore_index=True)
        except:
            print('WFDB write exeption')
    return metadataDbToNormalize













dbPath = "D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
metadataPath = dbPath+'ptbxl_database.csv'

metadataDb = pd.read_csv(metadataPath)

metadataDbToNormalize = copy.deepcopy(metadataDb)

extension_data = pd.DataFrame(columns=metadataDb.columns)

reduce_counter_dict = {}
for i, row in metadataDb.iterrows():
    print(f'{i}/{len(metadataDb)}')
    file_path = dbPath + row['filename_hr']
    scp_codes_for_row = list(eval(row['scp_codes']).keys())
    for scp_code in scp_codes_for_row:
        if scp_code in list(data_normalisation_scheme.keys()):
            normalisation_vector = data_normalisation_scheme[scp_code]
            normalisation_vector_mode = normalisation_vector['mode']
            normalisation_vector_degree = normalisation_vector['degree']
            if normalisation_vector_mode == 'reduce':
                test_result = reduceTest(file_path)
                if test_result == False:
                    test_result = not bool(row['validated_by_human'])
                test_result = reduceTest(file_path)
                if test_result:
                    file_path_to_index = '/'.join(file_path.split('/')[-3:])
                    reduce_index = metadataDbToNormalize[metadataDbToNormalize['filename_hr'] == file_path_to_index].index
                    metadataDbToNormalize = metadataDbToNormalize.drop(reduce_index)
                    try:
                        reduce_counter_dict[scp_code]+=1
                    except:
                        reduce_counter_dict[scp_code] = 1

            if normalisation_vector_mode == 'expand':
                metadataDbToNormalize = expandSig(file_path, metadataDbToNormalize, normalisation_vector_degree)

print(reduce_counter_dict)

metadataDbToNormalize.to_csv('Normalised_PTB_XL_metadata.csv')



