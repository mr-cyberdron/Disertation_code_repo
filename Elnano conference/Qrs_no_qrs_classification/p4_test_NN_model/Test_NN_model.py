import matplotlib.pyplot as plt
import numpy as np
from Qrs_no_qrs_classification import tools
from FILES_processing_lib import scandir
from Plottermaan_lib import masplot
from Dataformats_tools.WFDB.plot_wfdb_data import plot_events_annotated_ECG
import neurokit2 as nk
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
from collections import Counter
from Qrs_no_qrs_classification.p3_NN_creation.models import QRSnet1,QRSnet2
import torch
import pandas as pd
from scipy.signal import resample
import random
import copy
from fastdtw import fastdtw


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

def generate_fake_anot(peaks_pos):
    qrs = []
    anot = []
    for pos in peaks_pos:
        qrs.append('N')
        anot.append('')
    return np.array(qrs), np.array(anot)

def check_QRS_detection_quality(sig, etalon_qrs,detected_qrs, fs, true_detection_range_sec = 0.1):
    true_detection_range_samples = np.ceil(true_detection_range_sec*fs)

    et_pos_true_count_mass = []
    for det_pos in detected_qrs:
        pos_from = max(0,det_pos-true_detection_range_samples)
        pos_to = min(len(sig),det_pos+true_detection_range_samples)
        et_pos_true_count = 0
        for et_pos in etalon_qrs:
            if pos_from<=et_pos<=pos_to:
                et_pos_true_count = et_pos_true_count+1
                # plt.plot(sig)
                # plt.axvline(pos_from, color = 'blue')
                # plt.axvline(pos_to, color='blue')
                # plt.axvline(et_pos, color='red')
                # plt.axvline(det_pos, color='orange')
                # plt.xlim([(det_pos - (fs/2)),(det_pos + (fs/2))])
                # plt.show()
        et_pos_true_count_mass.append(et_pos_true_count)

    count_true_detection = Counter(et_pos_true_count_mass)
    print(count_true_detection)
    print(f'True QRS percent = {round(count_true_detection[1]/len(detected_qrs)*100,2)}%')

def classify_qrs_by_nn(qrs_samples):
    similarity_treshold = 0.95
    qrs_samples = torch.tensor([qrs_samples]).float()
    # model_path = './baclup QRSnn2/model_weights.pth'
    # model = QRSnet2()
    model_path = './model_backup_QRSnn1/model_weights.pth'
    model = QRSnet1()

    model.load_state_dict(torch.load(model_path))
    model.eval()
    # qrs_samples = qrs_samples.unsqueeze(1)
    model_res = model(qrs_samples).squeeze()
    res_item = model_res.item()
    if res_item>similarity_treshold:
        return 1
    else:
        return 0

def filter_qrs_annotations_by_NN(signal, fs, detected_qrs_pos, QRS_anot, allowed_beat_types):
        print('NN filter')
        new_qrs_pos = []
        new_qrs_anot = []
        prep_qrs = tools.prep_qrs_data([signal],fs,detected_qrs_pos,QRS_anot,allowed_beat_types)
        for qrs_samp, pos, anot in zip(prep_qrs,detected_qrs_pos,QRS_anot):
            classif_res = classify_qrs_by_nn(qrs_samp)
            if classif_res == 1:
                new_qrs_pos.append(pos)
                new_qrs_anot.append(anot)
        return np.array(new_qrs_pos), np.array(new_qrs_anot)


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

def check_LP_in_avg_sig(sig_fragm, fs,Lp_frag, LP_startpos):
    lp_startpos_samp = np.ceil(LP_startpos*fs)
    Lp_sample_from = int(np.ceil(len(sig_fragm)/2)+lp_startpos_samp)
    Lp_sample_to = int(Lp_sample_from+len(Lp_frag))
    LP_sigpart = sig_fragm[Lp_sample_from:Lp_sample_to]
    LP_sigpart = AnalogFilterDesign(LP_sigpart,fs).hp(order=3, cutoff=18).zerophaze().butter().filtration()

    distance, path = fastdtw(Lp_frag,LP_sigpart)
    # distance = np.corrcoef(Lp_frag,LP_sigpart)[0][1]
    print(distance)
    print(len(Lp_frag))
    print(len(LP_sigpart))
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(Lp_frag)
    plt.subplot(2,1,2)
    plt.plot(LP_sigpart)
    plt.show()
    return distance

sample_data_path = './poore_ecg_signal/'

allowed_beat_types = ["N","V","A","F","a","E","R",
                      "/","j","L","f","!","S","r","e","n"]

true_decision_range = 0.1

sample_files = scandir(sample_data_path,ext='.npz')
print(sample_files)

def check_qrs_detection_quality():
    file = 'p1.npz'
    signal, fs, etalon_pos, qrs_annotations, events_anot = read_np_sample(sample_data_path+file)
    # plot_events_annotated_ECG([signal],fs,etalon_pos,qrs_annotations,events_anot,['lead'])
    filtered_sig = filter_ecg(signal,fs)
    stat, sig_peaks = nk.ecg_peaks(filtered_sig,sampling_rate=fs,method='pantompkins1985') #kalidas2017
    detected_peaks = sig_peaks['ECG_R_Peaks']
    f_qrs, f_anot = generate_fake_anot(detected_peaks)
    # plot_events_annotated_ECG([signal], fs, detected_peaks, f_qrs, f_anot, ['lead'])
    check_QRS_detection_quality(signal,etalon_pos,detected_peaks,fs,true_detection_range_sec=true_decision_range)
    print('removing_false_Qrs')
    filtered_qrs_pos, filtered_qrs_anot = filter_qrs_annotations_by_NN(signal,fs,detected_peaks,
                                                                                           f_qrs,allowed_beat_types)
    check_QRS_detection_quality(signal,etalon_pos,filtered_qrs_pos,fs, true_detection_range_sec=true_decision_range)

    print('check etalon')
    check_QRS_detection_quality(signal, etalon_pos, etalon_pos, fs, true_detection_range_sec=true_decision_range)

def check_LA_components_selection():
    file = 'p8_500.npz'
    signal, fs, etalon_pos, qrs_annotations, events_anot = read_np_sample(sample_data_path + file)
    signal = signal[0]

    Lp_signal = generate_LP_signal(fs)*0.2#0.08
    Lp_signal = Lp_signal[0:50]
    lp_startpos = 0.035
    ecg_with_lp = mount_LP_to_ecg(signal,fs,etalon_pos,Lp_signal, LP_start_from_qrs_sec = lp_startpos)

    ecg_with_lp = tools.add_noise(ecg_with_lp,30)

    filtered_sig = filter_ecg(ecg_with_lp, fs)
    stat, sig_peaks = nk.ecg_peaks(filtered_sig, sampling_rate=fs, method='pantompkins1985')  # kalidas2017
    detected_peaks = sig_peaks['ECG_R_Peaks']
    f_qrs, f_anot = generate_fake_anot(detected_peaks)
    print('removing_false_Qrs')

    filtered_qrs_pos, filtered_qrs_anot = filter_qrs_annotations_by_NN(ecg_with_lp, fs, detected_peaks,
                                                                       f_qrs, allowed_beat_types)

    print('card_detector')
    avg_card_detector = calc_avg_card(ecg_with_lp,fs,detected_peaks)
    corr_coef_card_detector = check_LP_in_avg_sig(avg_card_detector, fs, Lp_signal, lp_startpos)
    print(f'corr_coef:{corr_coef_card_detector}')
    x = np.array(list(range(len(avg_card_detector))))/fs
    plt.plot(x, avg_card_detector)
    plt.xlabel('Time[sec]')
    plt.ylabel('Amplitude[mV]')
    plt.show()

    print('card_detector_cleaned')
    avg_card_detector_cleaned = calc_avg_card(ecg_with_lp, fs, filtered_qrs_pos)
    corr_coef_card_detector_cleaned = check_LP_in_avg_sig(avg_card_detector_cleaned, fs, Lp_signal, lp_startpos)
    print(f'corr_coef:{corr_coef_card_detector_cleaned}')
    plt.plot(x,avg_card_detector_cleaned)
    plt.xlabel('Time[sec]')
    plt.ylabel('Amplitude[mV]')
    plt.show()

    print('etalon_qrs')
    avg_card_etalon = calc_avg_card(ecg_with_lp,fs,etalon_pos)
    corr_coef_compared_etalon = check_LP_in_avg_sig(avg_card_etalon,fs,Lp_signal,lp_startpos)
    print(f'corr_coef:{corr_coef_compared_etalon}')
    plt.plot(x, avg_card_etalon)
    plt.xlabel('Time[sec]')
    plt.ylabel('Amplitude[mV]')
    plt.show()


check_qrs_detection_quality()
check_LA_components_selection()

