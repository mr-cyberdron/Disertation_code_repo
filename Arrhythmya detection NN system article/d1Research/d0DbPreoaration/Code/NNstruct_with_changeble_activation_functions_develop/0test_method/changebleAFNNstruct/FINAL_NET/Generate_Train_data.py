import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb_no_ann,write_wfdb
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
import neurokit2 as nk
from scipy.signal import resample

def filter_ecg2(sig_mass,fs):
    new_mass = []
    for sig in sig_mass:
        filt_Sig = AnalogFilterDesign(sig,fs).bp(order=5, cutoff=[1, 150]).zerophaze().butter().filtration()
        new_mass.append(filt_Sig)
    return np.array(new_mass)

def filter_ecg(sig,fs):
    return AnalogFilterDesign(sig,fs).bp(order=5, cutoff=[1, 60]).zerophaze().butter().filtration()

def filter50(sig,fs):
    return AnalogFilterDesign(sig, fs).zerophaze().notch(cutoff=50, quality_factor=50).filtration()

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

db_initial_path = "D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
metadata_file = "D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/metadata_normalised_LP_atached.csv"

metadata_Db = pd.read_csv(metadata_file)

fpath_mass = []
pat_list_mass = []
validated_mass = []

counter = 0
for i, metadata_line in metadata_Db.iterrows():
    input(metadata_line)
    counter+=1
    print(f'{counter}/{len(metadata_Db)}')
    filepath = db_initial_path+metadata_line['filename_hr']
    signals, re_format, fs, units, adc_gain, baseline, \
            coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb_no_ann(filepath)
    signals = filter_ecg2(signals,fs)
    XYZ_ECG = transform_12_lead_ECG_to_XYZ(signals,lead_names)
    ECG_R_VECTOR = compute_resultant_vector(XYZ_ECG)
    ECG_R_VECTOR_NORMALISED = normalize_signal(ECG_R_VECTOR)
    AVG_CARD_from_R_VECTOR = generate_avg_card(ECG_R_VECTOR_NORMALISED,fs)
    TrainVector = np.concatenate((AVG_CARD_from_R_VECTOR,ECG_R_VECTOR_NORMALISED))
    TrainVector = fix_length(TrainVector, 5500)
    fname = './FinalNetData/TransformedData/'+str(counter)+'_fs500.npy'
    np.save(fname, TrainVector)
    fpath_mass.append(fname)
    pat_list_mass.append(metadata_line['scp_codes'])
    validated_mass.append(metadata_line['validated_by_human'])


pd.DataFrame(data={'fpath':fpath_mass,'pathologies':pat_list_mass, 'doctor_validated':validated_mass}).to_csv('./FinalNetData/TransformedData/metadata.csv')




